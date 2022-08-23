# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from abc import ABC, abstractmethod

import numpy as np

_GLOBAL_RNG = np.random.default_rng()


class BigSMILESbase(ABC):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def pure_big_smiles(self):
        pass

    @property
    @abstractmethod
    def generatable(self):
        pass
