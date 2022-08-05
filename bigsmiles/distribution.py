# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from abc import ABC, abstractmethod
from scipy.special import lambertw

class Distribution(ABC):
    """
    Generic class to generate molecular weight numbers.
    """
    def __init__(self, raw_text, rng):
        """
        Initialize the generic distribution.

        Arguments:
        ----------
        raw_text: str
             Text represenation of the distribution.

        rng: numpy.random.Generator
             Numpy random number generator for the generation of numbers.
        """
        self._raw_text = raw_text.strip("| \t\n")
        self._rng = rng

    @abstractmethod
    def draw_mw(self):
        pass

class FlorySchulz(Distribution):
    """
    Flory-Schulz distribution of molecular weights for geometrically distributed chain lengths.

    :math:`W_a(N) = a^2 N (1-a)^M`

    where :math:`0<a<1` is the experimentally determined constant of remaining monomers and :math:`k` is the chain length.

    The textual representation of this distribution is: `flory_schulz(a)`
    """
    def __init__(self, raw_text, rng):
        """
        Initialization of Flory-Schulz distribution object.

        Arguments:
        ----------
        raw_text: str
             Text represenation of the distribution.
             Has to start with `flory_schulz`.

        rng: numpy.random.Generator
             Numpy random number generator for the generation of numbers.
        """
        super().__init__(raw_text, rng)

        if not raw_text.startswith("flory_schulz"):
            raise RuntimeError("Attemp to initlize Flory-Schulz distribution from text '{raw_text}' that does not start with 'flory_schulz'")

        self._a = float(eval(self._raw_text[len("flory_schulz"):]))

    def draw_mw(self, k=0):
        def invert(a, y, k):
            return (lambertw(-( (1-a)**(1/a) * (y-1) * np.log(1-a) )/a))/np.log(1-a) - 1/a
        y = self._rng.uniform(0, 1)
        return invert(self._a, y, k)
