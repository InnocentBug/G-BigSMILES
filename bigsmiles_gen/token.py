# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from .bond import BondDescriptor


class SmilesToken:
    """
    SMILES fragment including the bond descriptors, that make up the monomers and end groups.
    This also includes the weight of this particular monomer in the generation.
    Example:  '[$]CC(c1ccccc1)[$]|0.25|'
    """

    def __init__(self, big_smiles_ext, bond_id_offset):
        """
        Construct the element.

        Arguments:
        ----------
        big_smiles_ext: str
           Text that describes the smiles token.

        bond_id_offset: int
           Number of bond descriptors in the stochastic objects before this token.
        """

        bond_id_offset = int(bond_id_offset)
        if bond_id_offset < 0:
            raise RuntimeError(f"bond_id_offset {bond_id_offset} is not positive.")
        self._raw_text = big_smiles_ext

        self.strip_smiles = ""
        self.bond_descriptors = []
        self.descriptor_pos = []

        i = 0
        while self._raw_text.find("[", i) >= 0 and i <= len(self._raw_text):
            self.strip_smiles += self._raw_text[i : self._raw_text.find("[", i)]
            i = self._raw_text.find("[", i)
            if self._raw_text.find("]", i) < 0:
                raise RuntimeError(
                    "Bond descriptor {self._raw_text} has an opening bond descriptor '[', but no closing counter part ']'."
                )
            bond_text = self._raw_text[i : self._raw_text.find("]", i) + 1]

            while i > 0 and self._raw_text[i] in r".-=#$:/\@":
                i -= 1
            preceding_char = self._raw_text[i : self._raw_text.find("[", i)]
            bond = BondDescriptor(
                bond_text, bond_id_offset + len(self.bond_descriptors), preceding_char
            )
            self.descriptor_pos.append(len(self.strip_smiles))
            self.bond_descriptors.append(bond)
            i = self._raw_text.find("]", i) + 1

        self.strip_smiles += self._raw_text[i:]
        self.strip_smiles = self.strip_smiles.strip()

        self.weight = None
        if self.strip_smiles.find("|") >= 0:
            self.weight = float(self.strip_smiles[self.strip_smiles.find("|") + 1 : -1])
            self.strip_smiles = self.strip_smiles[: self.strip_smiles.find("|")]
            if self.weight < 0 or self.weight > 1:
                raise RuntimeError(
                    f"Invalid weight {self.weight} not in [0,1] for bond descriptor {self._raw_text}"
                )

    def _construct_string(self, weight):
        string = ""
        if len(self.bond_descriptors) == 0:
            string = self.strip_smiles
        else:
            string += self.strip_smiles[: self.descriptor_pos[0]]
            for i in range(len(self.descriptor_pos) - 1):
                start = self.descriptor_pos[i]
                end = self.descriptor_pos[i + 1]
                if weight:
                    string += str(self.bond_descriptors[i])
                else:
                    string += self.bond_descriptors[i].pure_big_smiles()
                string += self.strip_smiles[start:end]
            if weight:
                string += str(self.bond_descriptors[-1])
            else:
                string += self.bond_descriptors[-1].pure_big_smiles()
            string += self.strip_smiles[self.descriptor_pos[-1] :]

        if weight and self.weight is not None:
            string += f"|{self.weight}|"

        return string

    def __str__(self):
        return self._construct_string(True)

    def pure_big_smiles(self):
        return self._construct_string(False)

    @property
    def generatable(self):
        for bond in self.bond_descriptors:
            if not bond.generatable:
                return False
        return True
