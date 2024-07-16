# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import numpy as np
import rdkit.Chem.rdchem as rc

from .core import BigSMILESbase


def _create_compatible_bond_text(bond):
    compatible_symbol = "$"
    if "<" in str(bond):
        compatible_symbol = "<"
    if ">" in str(bond):
        compatible_symbol = ">"
    bond_string = f"{bond.preceding_characters}[{compatible_symbol}{bond.descriptor_id}]"
    return bond_string


class BondDescriptor(BigSMILESbase):
    """
    Bond descriptor of the bigSMILES notation.
    """

    def __init__(self, big_smiles_ext, descr_num, preceding_characters, atom_bonding_to):
        """
        Construction of a bond descriptor.

        Arguments:
        ---------
        big_smiles_ext: str
           text representation of a bond descriptor. Example: `[$0]`

        descr_num: int
           Position of bond description in the line notation of the stochastic object.
           Ensure that it starts with `0` and is strictly monotonic increasing during a stochastic object.

        preceding_characters: str
           Characters preceding the bond descriptor, that is not an atom.
           These characters are used to identify the type of bond, any stereochemistry etc.
           If no further characters are provided, it is assumed to be a single bond without stereochemistry specification.

        atom_bonding_to:
           Index of the atom this bond descriptor is bonding to

        """
        self._raw_text = big_smiles_ext

        self.descriptor = ""
        self.descriptor_id = ""
        self.descriptor_num = int(descr_num)
        self.weight = 1.0
        self.transitions = None
        self.preceding_characters = preceding_characters
        self.bond_type = rc.BondType.UNSPECIFIED
        self.bond_stereo = rc.BondStereo.STEREOANY
        if self._raw_text == "[]":
            return

        if len(preceding_characters) == 0:
            self.preceding_characters = self._raw_text[: self._raw_text.find("[")]
            self._raw_text = self._raw_text[self._raw_text.find("[") :]

        self.atom_bonding_to = atom_bonding_to
        if self.atom_bonding_to is not None:
            self.atom_bonding_to = int(self.atom_bonding_to)

        if self._raw_text[0] != "[" or self._raw_text[-1] != "]":
            raise RuntimeError(f"Bond descriptor {self._raw_text} does not start and end with []")
        if self._raw_text[1] not in ("$", "<", ">"):
            raise RuntimeError(
                f"Bond descriptor {self._raw_text} does not have '$<>' as its second character"
            )
        self.descriptor = self._raw_text[1]

        id_end = -1
        if "|" in self._raw_text:
            id_end = self._raw_text.find("|")
        id_str = self._raw_text[2:id_end]
        if "[" in id_str or "]" in id_str:
            raise RuntimeError("Nested bond descriptors not supported.")
        self.descriptor_id = ""
        if len(id_str) > 0:
            self.descriptor_id = int(id_str.strip())

        self.weight = 1.0
        self.transitions = None
        if "|" in self._raw_text:
            if self._raw_text.count("|") != 2:
                raise RuntimeError(f"Invalid number of '|' in bond descriptor {self._raw_text}")
            weight_string = self._raw_text[self._raw_text.find("|") : self._raw_text.rfind("|")]
            weight_string = weight_string.strip("|")
            weight_list = [float(w) for w in weight_string.split()]
            if len(weight_list) == 1:
                self.weight = weight_list[0]
            else:
                self.transitions = np.asarray(weight_list)
                self.weight = self.transitions.sum()

        self.preceding_characters = preceding_characters
        self.bond_type = rc.BondType.SINGLE
        if "=" in self.preceding_characters:
            self.bond_type = rc.BondType.DOUBLE
        if "#" in self.preceding_characters:
            self.bond_type = rc.BondType.TRIPLE
        if "$" in self.preceding_characters:
            self.bond_type = rc.BondType.QUADRUPLE
        if ":" in self.preceding_characters:
            self.bond_type = rc.BondType.ONEANDAHALF

        self.bond_stereo = rc.BondStereo.STEREOANY
        if (
            "@" in self.preceding_characters
            or "/" in self.preceding_characters
            or "\\" in self.preceding_characters
        ):
            raise RuntimeError("Stereochemistry not implemented yet.")

    def is_compatible(self, other):
        if self.bond_type != other.bond_type:
            return False
        if self.descriptor_id != other.descriptor_id:
            return False
        if self.descriptor == "" or other.descriptor == "":
            return False
        if self.descriptor == "$" and other.descriptor == "$":
            return True
        if self.descriptor == "<" and other.descriptor == ">":
            return True
        if self.descriptor == ">" and other.descriptor == "<":
            return True
        return False

    def generate_string(self, extension):
        string = ""
        string += f"[{self.descriptor}{self.descriptor_id}"
        if extension and self.weight != 1.0:
            string += "|"
            if self.transitions is None:
                string += f"{self.weight}"
            else:
                for t in self.transitions:
                    string += f"{t} "
                string = string[:-1]
            string += "|"
        string += "]"
        return string.strip()

    @property
    def generable(self):
        return self.weight >= 0
