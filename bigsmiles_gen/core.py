# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from abc import ABC, abstractmethod

import numpy as np

_GLOBAL_RNG = np.random.default_rng()


class BigSMILESbase(ABC):
    def __str__(self):
        return self.generate_string(True)

    @abstractmethod
    def generate_string(self, extension: bool):
        pass

    @property
    @abstractmethod
    def generatable(self):
        pass
