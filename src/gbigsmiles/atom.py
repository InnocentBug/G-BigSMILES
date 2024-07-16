# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from rdkit import Chem, RDLogger

from .core import BigSMILESbase


class Atom(BigSMILESbase):
    """
    A single atom representation in a SMILES fragment.

    """

    def __init__(self, big_smiles_ext):
        """
        Initialize atoms.

        Arguments:
        ---------
        big_smiles_ext: str
           string to describe a SMILES atom.

        """
        text = big_smiles_ext.strip()
        self._raw_text = text

        if len(text) == 1:
            text = text.upper()

        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)

        mol = Chem.MolFromSmiles(text)
        if mol is None:
            raise RuntimeError(f"Invalid atom string {text} not valid SMILES.")
        atoms_list = mol.GetAtoms()
        if len(atoms_list) != 1:
            raise RuntimeError("Not exactly one atom in the SMILES string.")

    def generate_string(self, extension):
        return self._raw_text

    @property
    def generable(self):
        return True
