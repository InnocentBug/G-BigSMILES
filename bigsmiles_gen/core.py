# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from abc import ABC, abstractmethod
from warnings import warn

import numpy as np

_GLOBAL_RNG = np.random.default_rng()


class BigSMILESbase(ABC):

    bond_descriptors = []

    def __str__(self):
        return self.generate_string(True)

    @abstractmethod
    def generate_string(self, extension: bool):
        pass

    @property
    @abstractmethod
    def generable(self):
        pass

    def generate(self, prefix=None, rng=_GLOBAL_RNG):
        if not self.generable:
            raise RuntimeError("Attempt to generate a non-generable molecule.")
        if prefix:
            if len(prefix.bond_descriptors) != 1:
                raise RuntimeError(
                    f"Prefixes for generating Mols must have exactly one open bond descriptor found {len(prefix.bond_descriptors)}."
                )


def get_compatible_bond_descriptor_ids(bond_descriptors, bond):
    compatible_idx = []
    for i, other in enumerate(bond_descriptors):
        if bond is None or bond.is_compatible(other):
            compatible_idx.append(i)
    return np.asarray(compatible_idx, dtype=int)


def choose_compatible_weight(bond_descriptors, bond, rng):
    weights = []
    compatible_idx = get_compatible_bond_descriptor_ids(bond_descriptors, bond)
    for i in compatible_idx:
        weights.append(bond_descriptors[i].weight)
    weights = np.asarray(weights)
    weights /= np.sum(weights)

    try:
        idx = rng.choice(compatible_idx, p=weights)
    except ValueError as exc:
        warn(
            f"Cannot choose compatible bonds, available bonds {len(compatible_idx)}, sum of weights {np.sum(weights)}."
        )
        raise exc

    return idx
