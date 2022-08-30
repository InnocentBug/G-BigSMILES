# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import numpy as np
from .core import BigSMILESbase

class System(BigSMILESbase):
    """
    Entire system representation in extended bigSMILES.
    """

    def __init__(self, big_smiles_ext):
        """
        Construction of an entire system (mixtures).

        Arguments:
        ----------
        big_smiles_ext: str
           text representation

        seed: int
           Seed for the random number generation.
        """
        self._raw_text = big_smiles_ext.strip()

        self._elements = []
        i = 0
        while i < len(self._raw_text):
            if self._raw_text.find("{") >= 0:
                token_bundle = self._raw_text[i, self._raw_text.find(i, "{")]
                while
            else:


    @property
    def n_objects(self):
        return len(self._stochastic)
