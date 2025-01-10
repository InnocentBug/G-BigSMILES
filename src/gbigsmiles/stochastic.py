# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from .big_smiles import BigSmilesMolecule
from .bond import BondDescriptor, TerminalBondDescriptor
from .core import BigSMILESbase
from .distribution import StochasticGeneration


class StochasticObject(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)

        self._repeat_residues: list = []
        self._termination_residues: list = []
        self._left_terminal_bond_d: None | BondDescriptor = None
        self._right_terminal_bond_d: None | BondDescriptor = None

        self._generation: StochasticGeneration | None = None

        # Parse info
        termination_separator_found = False
        for child in self._children:

            if isinstance(child, TerminalBondDescriptor):
                if self._left_terminal_bond_d is None:
                    self._left_terminal_bond_d = child
                else:
                    if self._right_terminal_bond_d is not None:
                        raise ValueError(f"{self}, {self._children}, {self._right_terminal_bond_d}")
                    self._right_terminal_bond_d = child

            if str(child) == ";":
                termination_separator_found = True

            if isinstance(child, BigSmilesMolecule):
                if not termination_separator_found:
                    self._repeat_residues.append(child)
                else:
                    self._termination_residues.append(child)

            if isinstance(child, StochasticGeneration):
                self._generation = child

    def generate_string(self, extension: bool):
        string = "{" + self._left_terminal_bond_d.generate_string(extension) + " "
        if len(self._repeat_residues) > 0:
            string += self._repeat_residues[0].generate_string(extension)
            for residue in self._repeat_residues[1:]:
                string += ", " + residue.generate_string(extension)

        if len(self._termination_residues) > 0:
            string += "; " + self._termination_residues[0].generate_string(extension)
            for residue in self._termination_residues[1:]:
                string += ", " + residue.generate_string(extension)

        string += " " + self._right_terminal_bond_d.generate_string(extension) + "}"

        if self._generation:
            string += self._generation.generate_string(extension)

        return string


"""Deprecated with the grammar based G-BigSMILES, use StochasticObject instead."""
Stochastic = StochasticObject
