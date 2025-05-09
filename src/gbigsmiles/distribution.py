# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details
"""
This module defines base classes for handling stochastic generation based on
various statistical distributions.
"""
from abc import abstractmethod
from typing import Any, ClassVar, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from scipy import integrate, special, stats

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .core import BigSMILESbase
from .exception import UnknownDistribution
from .util import RememberAdd, get_global_rng

_T = TypeVar("_T", bound="StochasticDistribution")
_S = TypeVar("_S", bound="StochasticGeneration")


class StochasticGeneration(BigSMILESbase):
    """
    Base class for stochastic generation components in BigSMILES.
    """

    pass


class StochasticDistribution(StochasticGeneration):
    """
    Base class for stochastic distributions used in BigSMILES.

    Subclasses should implement specific distributions and register themselves
    in the `_known_distributions` class attribute.
    """

    _known_distributions: ClassVar[List[Type["StochasticDistribution"]]] = list()
    _distribution: Optional[stats.rv_discrete] = None

    def __init__(self, children: List[Any]):
        """
        Initializes a StochasticDistribution object.

        Args:
            children (List[Any]): List of parsed child elements.
        """
        super().__init__(children)

    def __bool__(self) -> bool:
        """
        Returns True if a statistical distribution is associated with this object.
        """
        return self._distribution is not None

    @classmethod
    def make(cls: Type[_T], text: str) -> _T:
        """
        Creates a specific StochasticDistribution subclass instance from a text representation.

        It iterates through the registered `_known_distributions` and attempts
        to create an instance if the distribution's token name (snake case)
        is found in the input text.

        Args:
            text (str): The textual representation of the stochastic distribution.

        Returns:
            _T: An instance of the appropriate StochasticDistribution subclass.

        Raises:
            UnknownDistribution: If no known distribution's token name is found in the text.
        """
        for known_distr in cls._known_distributions:
            if known_distr.token_name_snake_case in text:
                return known_distr.make(text)
        raise UnknownDistribution(text)

    def draw_mw(self, rng: Optional[np.random.Generator] = None, **kwargs: Any) -> Any:
        """
        Draws a sample from the molecular weight distribution.

        Args:
            rng (Optional[np.random.Generator]): Numpy random number generator for sampling.
                                                 If None, the global RNG is used.
            **kwargs (Any): Keyword arguments to pass to the distribution's sampling method.

        Returns:
            Any: A sample drawn from the distribution.

        Raises:
            NotImplementedError: If the `_distribution` attribute is None.
        """
        if self._distribution is None:
            raise NotImplementedError

        if rng is None:
            rng = get_global_rng()
        return max(0, float(self._distribution.rvs(random_state=rng, **kwargs)))

    def prob_mw(self, mw: Union[float, "RememberAdd"], **kwargs: Any) -> float:
        """
        Calculates the probability (PMF or CDF difference) for a given molecular weight.

        Args:
            mw (Union[float, RememberAdd]): The molecular weight to calculate the probability for.
                                           If a RememberAdd object, calculates the probability
                                           within the range defined by its previous and current values.
            **kwargs (Any): Keyword arguments to pass to the distribution's probability method.

        Returns:
            float: The probability of the given molecular weight(s).

        Raises:
            NotImplementedError: If the `_distribution` attribute is None.
        """
        if self._distribution is None:
            raise NotImplementedError

        if isinstance(mw, RememberAdd):
            return self._distribution.cdf(mw.value, **kwargs) - self._distribution.cdf(mw.previous, **kwargs)

        if hasattr(self._distribution, "pdf"):
            return self._distribution.pdf(mw, **kwargs)
        if hasattr(self._distribution, "pmf"):
            return self._distribution.pmf(k=int(mw), **kwargs)
        raise NotImplementedError

    @classmethod
    def _default_serialize(cls: Type["StochasticDistribution"], n: int) -> Tuple[float, ...]:
        """
        Internal helper method to create a tuple of default serialization values (-1.0).

        Args:
            n (int): The number of default values to generate.

        Returns:
            Tuple[float, ...]: A tuple containing n -1.0 values.
        """
        return tuple((-1.0 for _ in range(n)))

    @classmethod
    def default_serialize(cls: Type["StochasticDistribution"]) -> Tuple[float, ...]:
        """
        Returns the default serialization vector for this distribution type (an empty tuple).
        """
        return cls._default_serialize(0)

    @classmethod
    def get_empty_serial_vector(cls: Type["StochasticDistribution"]) -> List[float]:
        """
        Returns an empty serialization vector with the correct length to hold
        the default serialization of all known stochastic distributions.
        """
        vector: List[float] = []
        for distr_type in cls._known_distributions:
            vector += list(distr_type.default_serialize())
        return vector

    def get_serial_vector(self) -> List[float]:
        """
        Returns the serialization vector for this specific stochastic distribution instance.

        The vector contains the serialized parameters of this instance, with default
        serialization values for other known distribution types.
        """
        vector: List[float] = []
        for distr_type in self._known_distributions:
            if type(self) is distr_type:
                vector += list(self.serialize())
            else:
                vector += list(distr_type.default_serialize())
        return vector

    @classmethod
    def from_serial_vector(cls: Type[_T], vector: List[float]) -> Optional[_T]:
        """
        Creates a StochasticDistribution instance from a serialization vector.

        It iterates through known distributions, extracts the corresponding
        segment from the vector, and if it's not the default serialization,
        creates an instance of that distribution with the deserialized parameters.

        Args:
            vector (List[float]): The serialization vector.

        Returns:
            Optional[_T]: An instance of a StochasticDistribution subclass if
                           the vector contains non-default serialization for one,
                           otherwise None.

        Raises:
            ValueError: If the vector contains non-default serialization for more
                        than one known distribution.
        """
        candidates: List[Tuple[float, ...]] = []
        type_candidates: List[Type[_T]] = []
        for distr_type in cls._known_distributions:
            default_serial = distr_type.default_serialize()
            given_serial = tuple((vector.pop(0) for _ in default_serial))
            if default_serial != given_serial:
                candidates.append(given_serial)
                type_candidates.append(distr_type)

        if not candidates:
            return None

        if len(candidates) != 1:
            raise ValueError("The passed vector did not contain only one candidate for the distribution.")
        distr_type = type_candidates[0]
        params = candidates[0]
        return distr_type.make(distr_type.token_name_snake_case + str(params))

    @abstractmethod
    def serialize(self) -> Tuple[float, ...]:
        """
        Abstract method to serialize the parameters of this distribution into a tuple of floats.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Abstract class method to return the default serialization (e.g., a tuple of -1.0s)
        representing the absence of this distribution's parameters in a serial vector.
        """
        raise NotImplementedError


class FlorySchulz(StochasticDistribution):
    """
    Flory-Schulz distribution of molecular weights for geometrically distributed chain lengths.

    :math:`W_a(N) = a^2 N (1-a)^{N-1}`

    where :math:`0<a<1` is the experimentally determined constant of remaining monomers and :math:`N` is the chain length.

    The textual representation of this distribution is: `flory_schulz(a)`
    """

    class flory_schulz_gen(stats.rv_continuous):
        """Flory Schulz distribution."""

        def __init__(self, fls_a, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fls_a = fls_a
            self.discrete_function = lambda k: self.fls_a**2 * k * (1 - self.fls_a) ** (k - 1)
            self.norm = integrate.quad(self.discrete_function, 0, np.inf)[0]

        def _pdf(self, fls_k):
            return self.discrete_function(fls_k) / self.norm

    _fls_a: Optional[float] = None

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        """
        Creates a FlorySchulz instance from its textual representation.

        Args:
            text (str): The textual representation, e.g., 'flory_schulz(0.9)'.

        Returns:
            Self: A FlorySchulz instance.
        """
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def __init__(self, children: List[Any]):
        """
        Initialization of Flory-Schulz distribution object.

        Args:
            children (List[Any]): List of parsed children, expected to contain the 'a' parameter as a float.
        """
        super().__init__(children)

        fls_a: Optional[float] = None
        for child in self._children:
            if isinstance(child, float):
                fls_a = child

        if not 0 < fls_a < 1:
            raise RuntimeError(f"The Flory-Schulz distribution needs an a parameter between 0, and 1. But got {fls_a}.")

        self._fls_a = fls_a
        self._distribution = self.flory_schulz_gen(name="Flory-Schulz", fls_a=self._fls_a, a=0)

    def generate_string(self, extension: bool) -> str:
        """
        Generates the textual representation of the Flory-Schulz distribution.

        Args:
            extension (bool): Whether to include the '|' delimiters.

        Returns:
            str: The textual representation, e.g., '|flory_schulz(0.9)|'.
        """
        if extension:
            return f"|flory_schulz({self._fls_a})|"
        return ""

    @property
    def generable(self) -> bool:
        """
        Returns True if the distribution is initialized (i.e., the 'a' parameter is set).
        """
        return self._distribution is not None

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Returns the default serialization for FlorySchulz (a tuple with one -1.0).
        """
        return cls._default_serialize(1)

    def serialize(self) -> Tuple[float, ...]:
        """
        Serializes the 'a' parameter of the FlorySchulz distribution.
        """
        return (self._fls_a,)

    def draw_mw(self, rng=None):
        return super().draw_mw(rng=rng)

    def prob_mw(self, mw):
        return super().prob_mw(mw)


StochasticDistribution._known_distributions.append(FlorySchulz)


class SchulzZimm(StochasticDistribution):
    r"""
    Schulz-Zimm distribution of molecular weights.

    :math:`P(M) = \frac{z^{z+1}}{\Gamma(z+1)} \left(\frac{M}{M_n}\right)^{z-1} \frac{1}{M_n} \exp\left(-\frac{zM}{M_n}\right)`
    :math:`z = \frac{M_n}{M_w - M_n}`

    where :math:`\Gamma` is the Gamma function, :math:`M_w` is the weight-average
    molecular weight, and :math:`M_n` is the number-average molecular weight.
    P. C. Hiemenz, T. P. Lodge, Polymer Chemistry, CRC Press, Boca Raton, FL 2007.

    The textual representation of this distribution is: `schulz_zimm(Mw, Mn)`
    """

    class schulz_zimm_gen(stats.rv_continuous):
        """Flory Schulz distribution."""

        def __init__(self, z, Mn, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.z = z
            self.Mn = Mn
            self.prefactor = self.z ** (self.z + 1) / special.gamma(self.z + 1)
            self.discrete_function = lambda M: self.prefactor * (M ** (self.z - 1) / self.Mn**self.z) * np.exp(-self.z * M / self.Mn)
            self.norm = integrate.quad(self.discrete_function, 0, np.inf)[0]

        # def _pmf(self, M, z, Mn):
        #     return z ** (z + 1) / special.gamma(z + 1, dtype=np.float64) * M ** (z - 1) / Mn**z * np.exp(-z * M / Mn)

        def _pdf(self, M):
            return self.discrete_function(M) / self.norm

    _Mw: Optional[float] = None
    _Mn: Optional[float] = None
    _z: Optional[float] = None

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        """
        Creates a SchulzZimm instance from its textual representation.

        Args:
            text (str): The textual representation, e.g., 'schulz_zimm(1000, 500)'.

        Returns:
            Self: A SchulzZimm instance.
        """
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def __init__(self, children: List[Any]):
        """
        Initialization of Schulz-Zimm distribution object.

        Args:
            children (List[Any]): List of parsed children, expected to contain Mw and Mn as floats.
        """
        super().__init__(children)

        numbers: List[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(child)

        self._Mw, self._Mn = numbers
        self._z = self._Mn / (self._Mw - self._Mn) if self._Mw > self._Mn else None
        self._distribution = self.schulz_zimm_gen(name="Schulz-Zimm", z=self._z, Mn=self._Mn, a=0)

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Returns the default serialization for SchulzZimm (a tuple with two -1.0s).
        """
        return cls._default_serialize(2)

    def serialize(self) -> Tuple[float, ...]:
        """
        Serializes the Mw and Mn parameters of the SchulzZimm distribution.
        """
        return (self._Mw, self._Mn)

    def generate_string(self, extension: bool) -> str:
        """
        Generates the textual representation of the Schulz-Zimm distribution.

        Args:
            extension (bool): Whether to include the '|' delimiters.

        Returns:
            str: The textual representation, e.g., '|schulz_zimm(1000, 500)|'.
        """
        if extension:
            return f"|schulz_zimm({self._Mw}, {self._Mn})|"
        return ""

    @property
    def generable(self) -> bool:
        """
        Returns True if the distribution is initialized (i.e., Mw and Mn are set and valid).
        """
        return self._distribution is not None and self._z is not None

    def draw_mw(self, rng: Optional[np.random.Generator] = None) -> Any:
        """
        Draws a sample from the Schulz-Zimm distribution.
        """
        return super().draw_mw(rng=rng)

    def prob_mw(self, mw: Union[float, "RememberAdd"]) -> float:
        """
        Calculates the probability for a given molecular weight using the Schulz-Zimm distribution.
        """
        return super().prob_mw(mw)


StochasticDistribution._known_distributions.append(SchulzZimm)


class Gauss(StochasticDistribution):
    r"""
    Gauss (Normal) distribution of molecular weights.

    :math:`G(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2} \left(\frac{x-\mu}{\sigma}\right)^2\right)`

    where :math:`\mu` is the mean and :math:`\sigma` is the standard deviation.
    The textual representation is: `gauss(mu, sigma)`
    """

    _mu: Optional[float] = None
    _sigma: Optional[float] = None

    def __init__(self, children: List[Any]):
        """
        Initialization of Gaussian distribution object.

        Args:
            children (List[Any]): List of parsed children, expected to contain mean (mu) and
                                 standard deviation (sigma) as floats.
        """
        super().__init__(children)

        numbers: List[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(child)

        self._mu, self._sigma = numbers
        self._distribution = stats.norm(loc=self._mu, scale=self._sigma)

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Returns the default serialization for Gauss (a tuple with two -1.0s).
        """
        return cls._default_serialize(2)

    def serialize(self) -> Tuple[float, ...]:
        """
        Serializes the mean (mu) and standard deviation (sigma) of the Gauss distribution.
        """
        return (self._mu, self._sigma)

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        """
        Creates a Gauss instance from its textual representation.

        Args:
            text (str): The textual representation, e.g., 'gauss(100, 10)'.

        Returns:
            Self: A Gauss instance.
        """
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def generate_string(self, extension: bool) -> str:
        """
        Generates the textual representation of the Gauss distribution.

        Args:
            extension (bool): Whether to include the '|' delimiters.

        Returns:
            str: The textual representation, e.g., '|gauss(100, 10)|'.
        """
        if extension:
            return f"|gauss({self._mu}, {self._sigma})|"
        return ""

    @property
    def generable(self) -> bool:
        """
        Returns True if the distribution is initialized (i.e., mu and sigma are set).
        """
        return self._distribution is not None

    def prob_mw(self, mw: Union[float, "RememberAdd"]) -> float:
        """
        Calculates the probability density for a given molecular weight using the Gauss distribution.

        Args:
            mw (Union[float, RememberAdd]): The molecular weight to calculate the probability for.
                                           If a RememberAdd object, this method might not be directly
                                           meaningful for a continuous distribution.

        Returns:
            float: The probability density at the given molecular weight.
        """
        if self._sigma is not None and self._sigma < 1e-6 and self._mu is not None and abs(self._mu - mw) < 1e-6:
            return 1.0
        return super().prob_mw(mw)


StochasticDistribution._known_distributions.append(Gauss)


class Uniform(StochasticDistribution):
    """
    Uniform distribution of different lengths, usually useful for short chains.

    The textual representation of this distribution is: `uniform(low, high)`
    """

    _low: Optional[float] = None
    _high: Optional[float] = None

    def __init__(self, children: List[Any]):
        """
        Initialization of Uniform distribution object.

        Args:
            children (List[Any]): List of parsed children, expected to contain the lower (low) and
                                 upper (high) bounds as floats.
        """
        super().__init__(children)

        numbers: List[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(child)

        self._low, self._high = numbers
        self._distribution = stats.uniform(loc=self._low, scale=(self._high - self._low) if self._low is not None and self._high is not None else 0)

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Returns the default serialization for Uniform (a tuple with two -1.0s).
        """
        return cls._default_serialize(2)

    def serialize(self) -> Tuple[float, ...]:
        """
        Serializes the lower (low) and upper (high) bounds of the Uniform distribution.
        """
        return (self._low, self._high)

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        """
        Creates a Uniform instance from its textual representation.

        Args:
            text (str): The textual representation, e.g., 'uniform(1, 5)'.

        Returns:
            Self: A Uniform instance.
        """
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def generate_string(self, extension: bool) -> str:
        """
        Generates the textual representation of the Uniform distribution.

        Args:
            extension (bool): Whether to include the '|' delimiters.

        Returns:
            str: The textual representation, e.g., '|uniform(1, 5)|'.
        """
        if extension:
            return f"|uniform({self._low}, {self._high})|"
        return ""

    @property
    def generable(self) -> bool:
        """
        Returns True if the distribution is initialized (i.e., low and high are set).
        """
        return self._distribution is not None


StochasticDistribution._known_distributions.append(Uniform)


class LogNormal(StochasticDistribution):
    r"""
    LogNormal distribution of molecular weights.

    :math:`f(x; S, \sigma) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln x - S)^2}{2\sigma^2}\right)`

    where :math:`S` is the shape parameter and :math:`\sigma` is the scale parameter.
    In the context of the original code, it seems :math:`M_n` (number average MW)
    and :math:`D` (polydispersity) are used as parameters. The provided PDF in the
    original docstring doesn't directly match the standard log-normal PDF.
    Assuming the original intent was to use :math:`M_n` and :math:`D`:

    The textual representation of this distribution is: `log_normal(Mn, D)`
    """

    class log_normal_gen(stats.rv_continuous):
        """Log-Normal distribution (parameterized by Mn and D)."""

        def _pdf(self, m, Mn, D):
            prefactor = 1 / (m * np.sqrt(2 * np.pi * np.log(D)))
            value = prefactor * np.exp(-((np.log(m / Mn) + np.log(D) / 2) ** 2) / (2 * np.log(D)))
            return value

        def _get_support(self, Mn: float, D: float) -> Tuple[float, float]:
            """Returns the support of the distribution."""
            return (0, np.inf)

    _M: Optional[float] = None  # Assuming this corresponds to Mn
    _D: Optional[float] = None  # Assuming this corresponds to D

    def __init__(self, children: List[Any]):
        """
        Initialization of LogNormal distribution object.

        Args:
            children (List[Any]): List of parsed children, expected to contain Mn and D as floats.
        """
        super().__init__(children)

        numbers: List[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(child)

        self._M, self._D = numbers
        if self._M is not None and self._D is not None and self._D > 0:
            self._distribution = self.log_normal_gen(name="Log-Normal")
        else:
            self._distribution = None

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Returns the default serialization for LogNormal (a tuple with two -1.0s).
        """
        return cls._default_serialize(2)

    def serialize(self) -> Tuple[float, ...]:
        """
        Serializes the Mn and D parameters of the LogNormal distribution.
        """
        return (self._M, self._D)

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        """
        Creates a LogNormal instance from its textual representation.

        Args:
            text (str): The textual representation, e.g., 'log_normal(500, 1.1)'.

        Returns:
            Self: A LogNormal instance.
        """
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def generate_string(self, extension: bool) -> str:
        """
        Generates the textual representation of the LogNormal distribution.

        Args:
            extension (bool): Whether to include the '|' delimiters.

        Returns:
            str: The textual representation, e.g., '|log_normal(500, 1.1)|'.
        """
        if extension:
            return f"|log_normal({self._M}, {self._D})|"
        return ""

    @property
    def generable(self) -> bool:
        """
        Returns True if the distribution is initialized (i.e., Mn and D are set and valid).
        """
        return self._distribution is not None

    def draw_mw(self, rng: Optional[np.random.Generator] = None) -> Any:
        """
        Draws a sample from the LogNormal distribution.
        """
        return super().draw_mw(rng=rng, Mn=self._M, D=self._D)

    def prob_mw(self, mw: Union[float, "RememberAdd"]) -> float:
        """
        Calculates the probability density for a given molecular weight using the LogNormal distribution.

        Args:
            mw (Union[float, RememberAdd]): The molecular weight to calculate the probability for.
                                           If a RememberAdd object, this method might not be directly
                                           meaningful for a continuous distribution.

        Returns:
            float: The probability density at the given molecular weight.
        """
        return super().prob_mw(mw, Mn=self._M, D=self._D)


StochasticDistribution._known_distributions.append(LogNormal)


class Poisson(StochasticDistribution):
    """
    Poisson distribution of molecular weights for chain lengths.
    Flory, P. J. Molecular size distribution in ethylene oxide polymers. Journal of the American chemical society 1940, 62, 1561–1565.

    The textual representation of this distribution is: `poisson(N)`
    """

    _N: Optional[float] = None  # Mean number of repeating units

    def __init__(self, children: List[Any]):
        """
        Initialization of Poisson distribution object.

        Args:
            children (List[Any]): List of parsed children, expected to contain the mean (N) as a float.
        """
        super().__init__(children)
        N: Optional[float] = None
        for child in self._children:
            if isinstance(child, float):
                N = child

        self._N = N
        self._distribution = stats.poisson(mu=self._N)

    @classmethod
    def default_serialize(cls) -> Tuple[float, ...]:
        """
        Returns the default serialization for Poisson (a tuple with one -1.0).
        """
        return cls._default_serialize(1)

    def serialize(self) -> Tuple[float, ...]:
        """
        Serializes the mean (N) of the Poisson distribution.
        """
        return (self._N,)

    @classmethod
    def make(cls: Type[Self], text: str) -> Self:
        """
        Creates a Poisson instance from its textual representation.

        Args:
            text (str): The textual representation, e.g., 'poisson(10)'.

        Returns:
            Self: A Poisson instance.
        """
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def generate_string(self, extension: bool) -> str:
        """
        Generates the textual representation of the Poisson distribution.

        Args:
            extension (bool): Whether to include the '|' delimiters.

        Returns:
            str: The textual representation, e.g., '|poisson(10)|'.
        """
        if extension:
            return f"|poisson({self._N})|"
        return ""

    @property
    def generable(self) -> bool:
        """
        Returns True if the distribution is initialized (i.e., N is set).
        """
        return self._distribution is not None


StochasticDistribution._known_distributions.append(Poisson)

## Gervasio: Define and implement your new distribution here. Don't forget StochasticDistribution._known_distributions.append(...) since that ensure that the stochastic vector includes your new distribution as well
