# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import numpy as np
from scipy import special, stats

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .core import BigSMILESbase
from .exception import UnknownDistribution
from .util import RememberAdd, get_global_rng


class StochasticGeneration(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)


class StochasticDistribution(StochasticGeneration):
    _known_distributions: set = set()

    def __init__(self, children: list):
        super().__init__(children)
        self._distribution: stats.rv_discrete | None = None

    @classmethod
    def make(cls, text: str) -> Self:
        for known_distr in cls._known_distributions:
            if known_distr.token_name_snake_case in text:
                return known_distr.make(text)
        raise UnknownDistribution(text)

    def draw_mw(self, rng=None, **kwargs):
        """
        Draw a sample from the molecular weight distribution.

        Arguments:
        ---------
        rng: numpy.random.Generator
             Numpy random number generator for the generation of numbers.
        kwargs: dict
             Keyword arguments to deliver parameters for the distribution in questions

        """
        if self._distribution is None:
            raise NotImplementedError

        if rng is None:
            rng = get_global_rng()
        return self._distribution.rvs(random_state=rng, **kwargs)

    def prob_mw(self, mw, **kwargs):
        """
        Calculate the probability that this mw was from this distribution.

        Arguments:
        ---------
        mw: float
             Integer heavy atom molecular weight.
        kwargs: dict
             Keyword arguments to deliver parameters for the distribution in questions

        """
        if self._distribution is None:
            raise NotImplementedError

        if isinstance(mw, RememberAdd):
            return self._distribution.cdf(mw.value, **kwargs) - self._distribution.cdf(
                mw.previous, **kwargs
            )

        if hasattr(self._distribution, "pdf"):
            return self._distribution.pdf(mw, **kwargs)
        if hasattr(self._distribution, "pmf"):
            return self._distribution.pmf(k=int(mw), **kwargs)
        raise NotImplementedError


class FlorySchulz(StochasticDistribution):
    """
    Flory-Schulz distribution of molecular weights for geometrically distributed chain lengths.

    :math:`W_a(N) = a^2 N (1-a)^M`

    where :math:`0<a<1` is the experimentally determined constant of remaining monomers and :math:`k` is the chain length.

    The textual representation of this distribution is: `flory_schulz(a)`
    """

    class flory_schulz_gen(stats.rv_discrete):
        """Flory Schulz distribution."""

        def _pmf(self, k, a):
            return a**2 * k * (1 - a) ** (k - 1)

    @classmethod
    def make(cls, text: str) -> Self:
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def __init__(self, children: list):
        """
        Initialization of Flory-Schulz distribution object.

        Arguments:
        ---------
        children: list[lark.Token]
            List of parsed children

        """
        super().__init__(children)

        self._distribution = self.flory_schulz_gen(name="Flory-Schulz")

        a: float | None = None
        for child in self._children:
            if isinstance(child, float):
                a = child

        self._a = a

    def generate_string(self, extension):
        if extension:
            return f"|flory_schulz({self._a})|"
        return ""

    @property
    def generable(self):
        return self._distribution is not None

    def draw_mw(self, rng=None):
        return super().draw_mw(rng=rng, a=self._a)

    def prob_mw(self, mw):
        return super().prob_mw(mw=mw, a=self._a)


StochasticDistribution._known_distributions.add(FlorySchulz)


class SchulzZimm(StochasticDistribution):
    r"""
    Schulz-Zimm distribution of molecular weights for geometrically distributed chain lengths.

    :math:`P_{M_w,M_n}(M) = z^{z+1}/\\Gamma(z+1) M^{z-1}/M_n^z \\exp(-zM/M_n)`
    :math:`z(M_w, M_n) = M_n/(M_w-M_n)`

    where :math:`\\Gamma` is the Gamma function, and :math:`M_w` weight-average molecular weight and `M_n` is the number average molecular weight.
        P. C. Hiemenz, T. P. Lodge, Polymer Chemistry, CRC Press, Boca Raton, FL 2007.

    The textual representation of this distribution is: `schulz_zimm(Mw, Mn)`
    """

    class schulz_zimm_gen(stats.rv_discrete):
        """Flory Schulz distribution."""

        def _pmf(self, M, z, Mn):
            return z ** (z + 1) / special.gamma(z + 1) * M ** (z - 1) / Mn**z * np.exp(-z * M / Mn)

    @classmethod
    def make(cls, text: str) -> Self:
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def __init__(self, children: list):
        """
        Initialization of Schulz-Zimm distribution object.

        Arguments:
        ---------
        children: list[lark.Token]
            List of parsed children

        """
        super().__init__(children)

        numbers: list[float] = []

        for child in self._children:
            if isinstance(child, float):
                numbers.append(child)

        self._Mw, self._Mn = numbers
        self._z = self._Mn / (self._Mw - self._Mn)
        self._distribution = self.schulz_zimm_gen(name="Schulz-Zimm")

    def generate_string(self, extension):
        if extension:
            return f"|schulz_zimm{self._Mw, self._Mn}|"
        return ""

    @property
    def generable(self):
        return True

    def draw_mw(self, rng=None):
        return super().draw_mw(rng=rng, z=self._z, Mn=self._Mn)

    def prob_mw(self, mw):
        return super().draw_mw(z=self._z, Mn=self._Mn)


StochasticDistribution._known_distributions.add(SchulzZimm)


class Gauss(StochasticDistribution):
    r"""
    Gauss distribution of molecular weights for geometrically distributed chain lengths.

    :math:`G_{\\sigma,\\mu}(N) = 1/\\sqrt{\\sigma^2 2\\pi} \\exp(-1/2 (x-\\mu^2)/\\sigma`

    where :math:`\\mu` is the mean and :math:`\\sigma^2` the variance.

    The textual representation of this distribution is: `gauss(\\mu, \\sigma)`
    """

    def __init__(self, children: list):
        """
        Initialization of Gaussian distribution object.

        Arguments:
        ---------
        children: list[lark.Token]
            List of parsed children

        """
        super().__init__(children)

        numbers: list[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(child)

        self._mu, self._sigma = numbers
        self._distribution = stats.norm(loc=self._mu, scale=self._sigma)

    @classmethod
    def make(cls, text: str) -> Self:
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def generate_string(self, extension):
        if extension:
            return f"|gauss({self._mu}, {self._sigma})|"
        return ""

    @property
    def generable(self):
        return True

    def prob_mw(self, mw):
        if self._sigma < 1e-6 and abs(self._mu - mw) < 1e-6:
            return 1.0
        return super().prob_mw(mw)


StochasticDistribution._known_distributions.add(Gauss)


class Uniform(StochasticDistribution):
    """
    Uniform distribution of different lengths, usually useful for short chains.

    The textual representation of this distribution is: `uniform(low, high)`
    """

    def __init__(self, children):
        """
        Initialization of Uniform distribution object.

        Arguments:
        ---------
        children: list[lark.Token]
            List of parsed children

        """
        super().__init__(children)

        numbers: list[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(child)

        self._low, self._high = numbers
        self._distribution = stats.uniform(loc=self._low, scale=(self._high - self._low))

    @classmethod
    def make(cls, text: str) -> Self:
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def generate_string(self, extension):
        if extension:
            return f"|uniform({self._low}, {self._high})|"
        return ""

    @property
    def generable(self):
        return True


StochasticDistribution._known_distributions.add(Uniform)


class LogNormal(StochasticDistribution):
    r"""
    LogNormal distribution of molecular weights for narrowly distributed chain lengths.

    :math:`\\mathrm{PDF}(m_{w,i}) =\\frac{1}{m_{w,i}\\sqrt{2\\pi \\ln(\\text{\\DH})}} \\exp\\left(-\\frac{\\left(\\ln\\left(\\frac{m_{w,i}}{M_n}\\right)+\\frac{\\text{\\DH}}{2}\\right)^2}{2\\sigma^2}\\right)`

    where :math:`\\text{\\DH}` is the poly dispersity, and and :math:`M_n` is the number average molecular weight.
           Walsh, D. J.; Wade, M. A.; Rogers, S. A.; Guironnet, D. Challenges of Size-Exclusion Chromatography for the Analysis of Bottlebrush Polymers. Macromolecules 2020, 53, 8610–8620.

    The textual representation of this distribution is: `log_normal(Mn, D)`
    """

    class log_normal_gen(stats.rv_continuous):
        """Log-Normal distribution."""

        def _pdf(self, m, M, D):
            prefactor = 1 / (m * np.sqrt(2 * np.pi * np.log(D)))
            value = prefactor * np.exp(-((np.log(m / M) + np.log(D) / 2) ** 2) / (2 * np.log(D)))
            return value

        def _get_support(self, M, D):
            return (0, np.inf)

    def __init__(self, children):
        """
        Initialization of LogNormal distribution object.

        Arguments:
        ---------
        children: list[lark.Token]
            List of parsed children

        """
        super().__init__(children)

        numbers: list[float] = []
        for child in self._children:
            if isinstance(child, float):
                numbers.append(child)

        self._M, self._D = numbers
        self._distribution = self.log_normal_gen(name="Log-Normal")

    @classmethod
    def make(cls, text: str) -> Self:
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def generate_string(self, extension):
        if extension:
            return f"|log_normal({self._M}, {self._D})|"
        return ""

    @property
    def generable(self):
        return True

    def draw_mw(self, rng=None):
        return super().draw_mw(rng=rng, M=self._M, D=self._D)

    def prob_mw(self, mw):
        return super().prob_mw(mw, M=self._M, D=self._D)


StochasticDistribution._known_distributions.add(LogNormal)


class Poisson(StochasticDistribution):
    """
    Poisson distribution of molecular weights for chain lengths.
    Flory, P. J. Molecular size distribution in ethylene oxide polymers. Journal of the American chemical society 1940, 62, 1561–1565.

    The textual representation of this distribution is: `poisson(N)`
    """

    def __init__(self, children: list):
        """
        Initialization of Poisson distribution object.

        Arguments:
        ---------
        children: list[lark.Token]
            List of parsed children

        """
        super().__init__(children)
        N: float | None = None
        for child in self._children:
            if isinstance(child, float):
                N = child

        self._N = N
        self._distribution = stats.poisson(mu=self._N)

    @classmethod
    def make(cls, text: str) -> Self:
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def generate_string(self, extension):
        if extension:
            return f"|poisson({self._N})|"
        return ""

    @property
    def generable(self):
        return True


StochasticDistribution._known_distributions.add(Poisson)
