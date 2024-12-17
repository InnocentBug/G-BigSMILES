# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details


from .bond import BondDescriptor
from .core import BigSMILESbase

# stochastic_object: "{" WS_INLINE* terminal_bond_descriptor WS_INLINE* smiles WS_INLINE* _monomer_list*


class StochasticObject(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)

        self._repeat_residues: list = []
        self._termination_residues: list = []
        self._left_terminal_bond_d: None | BondDescriptor = None
        self._right_terminal_bond_d: None | BondDescriptor = None

        self._generation: None = None

        # Parse info


class Stochastic(StochasticObject):
    """Deprecated with the grammar based G-BigSMILES, use StochasticObject instead."""

    pass
