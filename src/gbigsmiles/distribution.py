# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from ast import literal_eval as make_tuple

import numpy as np
from scipy import special, stats

import gbigsmiles

from .core import _GLOBAL_RNG, BigSMILESbase


def get_distribution(distribution_text):
    if "flory_schulz" in distribution_text:
        return FlorySchulz(distribution_text)
    if "gauss" in distribution_text:
        return Gauss(distribution_text)
    if "uniform" in distribution_text:
        return Uniform(distribution_text)
    if "schulz_zimm" in distribution_text:
        return SchulzZimm(distribution_text)
    if "log_normal" in distribution_text:
        return LogNormal(distribution_text)
    if "poisson" in distribution_text:
        return Poisson(distribution_text)
    raise RuntimeError(f"Unknown distribution type {distribution_text}.")


class Distribution(BigSMILESbase):
    """
    Generic class to generate molecular weight numbers.
    """

    def __init__(self, raw_text):
        """
        Initialize the generic distribution.

        Arguments:
        ---------
        raw_text: str
             Text representation of the distribution. Example: `flory_schulz(0.01)`

        """
        self._raw_text = raw_text.strip("| \t\n")
        self._distribution = None

    def draw_mw(self, rng=None):
        """
        Draw a sample from the molecular weight distribution.

        Arguments:
        ---------
        rng: numpy.random.Generator
             Numpy random number generator for the generation of numbers.

        """
        if self._distribution is None:
            raise NotImplementedError

        if rng is None:
            rng = _GLOBAL_RNG
        return self._distribution.rvs(random_state=rng)

    def prob_mw(self, mw):
        """
        Calculate the probability that this mw was from this distribution.

        Arguments:
        ---------
        mw: float
             Integer heavy atom molecular weight.

        """
        if self._distribution is None:
            raise NotImplementedError

        if isinstance(mw, gbigsmiles.mol_prob.RememberAdd):
            return self._distribution.cdf(mw.value) - self._distribution.cdf(mw.previous)

        return self._distribution.pdf(mw)


class FlorySchulz(Distribution):
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

    def __init__(self, raw_text):
        """
        Initialization of Flory-Schulz distribution object.

        Arguments:
        ---------
        raw_text: str
             Text representation of the distribution.
             Has to start with `flory_schulz`.

        """
        super().__init__(raw_text)

        if not self._raw_text.startswith("flory_schulz"):
            raise RuntimeError(
                f"Attempt to initialize Flory-Schulz distribution from text '{raw_text}' that does not start with 'flory_schulz'"
            )

        self._a = float(make_tuple(self._raw_text[len("flory_schulz") :]))
        self._distribution = self.flory_schulz_gen(name="Flory-Schulz")

    def generate_string(self, extension):
        if extension:
            return f"|flory_schulz({self._a})|"
        return ""

    @property
    def generable(self):
        return True

    def draw_mw(self, rng=None):
        if rng is None:
            rng = _GLOBAL_RNG
        return self._distribution.rvs(a=self._a, random_state=rng)

    def prob_mw(self, mw):
        if isinstance(mw, gbigsmiles.mol_prob.RememberAdd):
            return self._distribution.cdf(mw.value, a=self._a) - self._distribution.cdf(
                mw.previous, a=self._a
            )
        return self._distribution.pmf(int(mw), a=self._a)


class SchulzZimm(Distribution):
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

    def __init__(self, raw_text):
        """
        Initialization of Schulz-Zimm distribution object.

        Arguments:
        ---------
        raw_text: str
             Text representation of the distribution.
             Has to start with `schulz_zimm`.

        """
        super().__init__(raw_text)

        if not self._raw_text.startswith("schulz_zimm"):
            raise RuntimeError(
                f"Attempt to initialize Schulz-Zimm distribution from text '{raw_text}' that does not start with 'schulz_zimm'"
            )

        self._Mw, self._Mn = make_tuple(self._raw_text[len("schulz_zimm") :])
        self._Mw = float(self._Mw)
        self._Mn = float(self._Mn)
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
        if rng is None:
            rng = _GLOBAL_RNG
        return self._distribution.rvs(z=self._z, Mn=self._Mn, random_state=rng)

    def prob_mw(self, mw):
        if isinstance(mw, gbigsmiles.mol_prob.RememberAdd):
            return self._distribution.cdf(
                mw.value, z=self._z, Mn=self._Mn
            ) - self._distribution.cdf(mw.previous, z=self._z, Mn=self._Mn)
        return self._distribution.pmf(
            int(mw),
            z=self._z,
            Mn=self._Mn,
        )


class Gauss(Distribution):
    r"""
    Gauss distribution of molecular weights for geometrically distributed chain lengths.

    :math:`G_{\\sigma,\\mu}(N) = 1/\\sqrt{\\sigma^2 2\\pi} \\exp(-1/2 (x-\\mu^2)/\\sigma`

    where :math:`\\mu` is the mean and :math:`\\sigma^2` the variance.

    The textual representation of this distribution is: `gauss(\\mu, \\sigma)`
    """

    def __init__(self, raw_text):
        """
        Initialization of Gaussian distribution object.

        Arguments:
        ---------
        raw_text: str
             Text representation of the distribution.
             Has to start with `gauss`.

        """
        super().__init__(raw_text)

        if not self._raw_text.startswith("gauss"):
            raise RuntimeError(
                f"Attempt to initialize Gaussian distribution from text '{raw_text}' that does not start with 'gauss'"
            )

        self._mu, self._sigma = make_tuple(self._raw_text[len("gauss") :])
        self._mu = float(self._mu)
        self._sigma = float(self._sigma)
        self._distribution = stats.norm(loc=self._mu, scale=self._sigma)

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


class Uniform(Distribution):
    """
    Uniform distribution of different lengths, usually useful for short chains.

    The textual representation of this distribution is: `uniform(low, high)`
    """

    def __init__(self, raw_text):
        """
        Initialization of Uniform distribution object.

        Arguments:
        ---------
        raw_text: str
             Text representation of the distribution.
             Has to start with `gauss`.

        """
        super().__init__(raw_text)

        if not self._raw_text.startswith("uniform"):
            raise RuntimeError(
                f"Attempt to initialize Uniform distribution from text '{raw_text}' that does not start with 'uniform'"
            )

        self._low, self._high = make_tuple(self._raw_text[len("uniform") :])
        self._low = int(self._low)
        self._high = int(self._high)
        self._distribution = stats.uniform(loc=self._low, scale=(self._high - self._low))

    def generate_string(self, extension):
        if extension:
            return f"|uniform({self._low}, {self._high})|"
        return ""

    @property
    def generable(self):
        return True


class LogNormal(Distribution):
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

    def __init__(self, raw_text):
        """
        Initialization of LogNormal distribution object.

        Arguments:
        ---------
        raw_text: str
             Text representation of the distribution.
             Has to start with `log_normal`.

        """
        super().__init__(raw_text)

        if not self._raw_text.startswith("log_normal"):
            raise RuntimeError(
                f"Attempt to initialize LogNormal distribution from text '{raw_text}' that does not start with 'log_normal'"
            )

        self._M, self._D = make_tuple(self._raw_text[len("log_normal") :])
        self._M = float(self._M)
        self._D = float(self._D)

        self._distribution = self.log_normal_gen(name="Log-Normal")

    def generate_string(self, extension):
        if extension:
            return f"|log_normal({self._M}, {self._D})|"
        return ""

    @property
    def generable(self):
        return True

    def draw_mw(self, rng=None):
        if rng is None:
            rng = _GLOBAL_RNG
        return self._distribution.rvs(M=self._M, D=self._D, random_state=rng)

    def prob_mw(self, mw):
        if isinstance(mw, gbigsmiles.mol_prob.RememberAdd):
            return self._distribution.cdf(mw.value, M=self._M, D=self._D) - self._distribution.cdf(
                mw.previous, M=self._M, D=self._D
            )

        return self._distribution.pdf(mw, M=self._M, D=self._D)


class Poisson(Distribution):
    """
    Poisson distribution of molecular weights for chain lengths.
    Flory, P. J. Molecular size distribution in ethylene oxide polymers. Journal of the American chemical society 1940, 62, 1561–1565.

    The textual representation of this distribution is: `poisson(N)`
    """

    def __init__(self, raw_text):
        """
        Initialization of Poisson distribution object.

        Arguments:
        ---------
        raw_text: str
             Text representation of the distribution.
             Has to start with `poisson`.

        """
        super().__init__(raw_text)

        if not self._raw_text.startswith("poisson"):
            raise RuntimeError(
                f"Attempt to initialize Poisson distribution from text '{raw_text}' that does not start with 'poisson'"
            )

        self._N = float(self._raw_text[len("poisson") + 1 : -1])
        self._distribution = stats.poisson(mu=self._N)

    def generate_string(self, extension):
        if extension:
            return f"|poisson({self._N})|"
        return ""

    @property
    def generable(self):
        return True

    def prob_mw(self, mw):
        try:
            return super().prob_mw(mw)
        except AttributeError:
            return self._distribution.pmf(int(mw))
