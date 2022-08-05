# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import numpy as np

class System:
    """
    Entire system representation in extended bigSMILES.
    """
    def __init__(self, big_smiles_ext, seed=None):
        """
        Construction of an entire system (mixtures).

        Arguments:
        ----------
        big_smiles_ext: str
           text representation

        seed: int
           Seed for the random number generation.
        """
        self._rng = np.random.default_generator(seed)
        self._stochastic = []
        self._distribution []

    @property
    def n_objects(self):
        return len(self._stochastic)
