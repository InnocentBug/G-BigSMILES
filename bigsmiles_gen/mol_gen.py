# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy

from rdkit import Chem


class MolGen:
    """
    Wrapper class to hold incompletely generated molecules.
    It mainly holds the incompletely generated rdkit.mol,
    as well as a list of not completed bond_descriptors.
    """

    def __init__(self, token):
        """
        Generate a new incompletely generated molecule from a token.

        Arguments:
        ----------
        token: SmilesToken
           Token, that starts a new molecule generation.
        """
        if not token.generable:
            raise RuntimeError(f"Attempting to generate token {str(token)}, which isn't generable.")
        self.bond_descriptors = copy.deepcopy(token.bond_descriptors)
        self.mol = Chem.MolFromSmiles(token.generate_smiles_fragment())

    @property
    def fully_generated(self):
        """
        Is this molecule fully generated?
        """
        return len(self.bond_descriptors) == 0

    def attach_other(self, self_bond_idx: int, other, other_bond_idx: int):
        """
        Combine two molecules and store the result in this one.

        Arguments:
        ----------
        self_bond_idx: int
           Bond descriptor index of this molecule, that is going to bind.

        other: MolGen
           Other half generated molecules, that binds to this one.

        other_bond_idx: int
           Index of the BondDescriptor in the other molecule, that is going to bind.
        """

        self.get_mol()
        self.get_mol()

        if self_bond_idx >= len(self.bond_descriptors):
            raise RuntimeError(f"Invalid bond descriptor id {self_bond_idx} (self).")
        if other_bond_idx >= len(other.bond_descriptors):
            raise RuntimeError(f"Invalid bond descriptor id {other_bond_idx} (other).")

        current_atom_number = len(self.mol.GetAtoms())
        other_bond_descriptors = copy.deepcopy(other.bond_descriptors)

        if not other_bond_descriptors[other_bond_idx].is_compatible(
            self.bond_descriptors[self_bond_idx]
        ):
            raise RuntimeError(
                f"Unable to attach mols, because bond descriptor {str(other_bond_descriptors[other_bond_idx])}"
                " is incompatible with {str(self.bond_descriptors[self_bond_idx])}."
            )
        for bd in other_bond_descriptors:
            bd.atom_bonding_to += current_atom_number

        new_mol = Chem.CombineMols(self.mol, other.mol)
        new_mol = Chem.EditableMol(new_mol)

        new_mol.AddBond(
            self.bond_descriptors[self_bond_idx].atom_bonding_to,
            other_bond_descriptors[other_bond_idx].atom_bonding_to,
            self.bond_descriptors[self_bond_idx].bond_type,
        )

        # Remove bond descriptors from list, as they have reacted now.
        del self.bond_descriptors[self_bond_idx]
        del other_bond_descriptors[other_bond_idx]

        self.bond_descriptors += other_bond_descriptors
        self.mol = new_mol.GetMol()

        return self

    def get_mol(self):
        """
        Obtain a sanitized copy of the generated (so far) generated molecule.
        """

        mol = copy.deepcopy(self.mol)
        Chem.SanitizeMol(mol)
        return mol

    @property
    def smiles(self):
        """
        Get SMILES of the (so far) generated molecule.
        """
        mol = self.get_mol()
        return Chem.MolToSmiles(mol)
