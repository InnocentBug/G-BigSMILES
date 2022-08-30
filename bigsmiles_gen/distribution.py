# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from abc import abstractmethod
from ast import literal_eval as make_tuple

from scipy import stats

from .core import _GLOBAL_RNG, BigSMILESbase


def get_distribution(distribution_text):
    if "flory_schulz" in distribution_text:
        return FlorySchulz(distribution_text)
    if "gauss" in distribution_text:
        return Gauss(distribution_text)
    raise RuntimeError(f"Unknown distribution type {distribution_text}.")


class Distribution(BigSMILESbase):
    """
    Generic class to generate molecular weight numbers.
    """

    def __init__(self, raw_text):
        """
        Initialize the generic distribution.

        Arguments:
        ----------
        raw_text: str
             Text represenation of the distribution. Example: `flory_schulz(0.01)`
        """
        self._raw_text = raw_text.strip("| \t\n")

    @abstractmethod
    def draw_mw(self, rng=None):
        """
        Draw a sample from the molecular weight distribution.
        Arguments:
        ----------
        rng: numpy.random.Generator
             Numpy random number generator for the generation of numbers.
        """
        pass


class FlorySchulz(Distribution):
    """
    Flory-Schulz distribution of molecular weights for geometrically distributed chain lengths.

    :math:`W_a(N) = a^2 N (1-a)^M`

    where :math:`0<a<1` is the experimentally determined constant of remaining monomers and :math:`k` is the chain length.

    The textual representation of this distribution is: `flory_schulz(a)`
    """

    class flory_schulz_gen(stats.rv_discrete):
        "Flory Schulz distribution"

        def _pmf(self, k, a):
            return a**2 * k * (1 - a) ** (k - 1)

    def __init__(self, raw_text):
        """
        Initialization of Flory-Schulz distribution object.

        Arguments:
        ----------
        raw_text: str
             Text represenation of the distribution.
             Has to start with `flory_schulz`.
        """
        super().__init__(raw_text)

        if not self._raw_text.startswith("flory_schulz"):
            raise RuntimeError(
                f"Attemp to initlize Flory-Schulz distribution from text '{raw_text}' that does not start with 'flory_schulz'"
            )

        self._a = float(make_tuple(self._raw_text[len("flory_schulz") :]))
        self._flory_schulz = self.flory_schulz_gen(name="Flory-Schulz")

    def draw_mw(self, rng=None):
        if rng is None:
            rng = _GLOBAL_RNG
        return self._flory_schulz.rvs(self._a, random_state=rng)

    def generate_string(self, extension):
        if extension:
            return f"|flory_schulz({self._a})|"
        return ""

    @property
    def generatable(self):
        return True


class Gauss(Distribution):
    """
    Gauss distribution of molecular weights for geometrically distributed chain lengths.

    :math:`G_{\\sigma,\\mu}(N) = 1/\\sqrt{\\sigma^2 2\\pi} \\exp(-1/2 (x-\\mu^2)/\\sigma`

    where :math:`\\mu` is the mean and :math:`\\sigma^2` the variance.

    The textual representation of this distribution is: `gauss(\\mu, \\sigma)`
    """

    def __init__(self, raw_text):
        """
        Initialization of Gaussian distribution object.

        Arguments:
        ----------
        raw_text: str
             Text represenation of the distribution.
             Has to start with `gauss`.
        """
        super().__init__(raw_text)

        if not self._raw_text.startswith("gauss"):
            raise RuntimeError(
                f"Attempt to initlize Gaussian distribution from text '{raw_text}' that does not start with 'gauss'"
            )

        self._mu, self._sigma = make_tuple(self._raw_text[len("gauss") :])
        self._mu = float(self._mu)
        self._sigma = float(self._sigma)

    def draw_mw(self, rng=None):
        if rng is None:
            rng = _GLOBAL_RNG
        mw = int(round(rng.normal(self._mu, self._sigma)))
        if mw < 0:
            mw = 0
        return mw

    def generate_string(self, extension):
        if extension:
            return f"|gauss({self._mu}, {self._sigma})|"
        return ""

    @property
    def generatable(self):
        return True
