# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from .atom import Atom
from .bond import BondDescriptor
from .core import _GLOBAL_RNG, BigSMILESbase, choose_compatible_weight
from .mol_gen import MolGen

_SMILES_DOUBLE_LETTER_ATOM = ("Cl", "Br")

_SMILES_SINGLE_LETTER_ATOM = (
    "B",
    "C",
    "N",
    "O",
    "P",
    "S",
    "F",
    "I",
    "c",
    "n",
    "s",
    "p",
    "o",
)


def _push_pop_atom_branch(string, atom_to_bond):
    # Remember atoms before branch opening, to bind to correct atom
    for _ in range(string.count("(")):
        atom_to_bond.append(atom_to_bond[-1])
    for _ in range(string.count(")")):
        atom_to_bond.pop(-1)
    return atom_to_bond


class SmilesToken(BigSMILESbase):
    """
    SMILES fragment including the bond descriptors, that make up the monomers and end groups.
    This also includes the weight of this particular monomer in the generation.
    Example:  '[$]CC(c1ccccc1)[$]'.
    """

    def __init__(self, big_smiles_ext, bond_id_offset, res_id):
        """
        Construct the element.

        Arguments:
        ---------
        big_smiles_ext: str
           Text that describes the smiles token.

        bond_id_offset: int
           Number of bond descriptors in the stochastic objects before this token.

        res_id: int
           Residue ID this token is a part of.

        """
        self.res_id = res_id
        bond_id_offset = int(bond_id_offset)
        if bond_id_offset < 0:
            raise RuntimeError(f"bond_id_offset {bond_id_offset} is not positive.")

        self._raw_text = big_smiles_ext.strip()
        if big_smiles_ext.count("(") != big_smiles_ext.count(")"):
            raise RuntimeError(
                f"Token {big_smiles_ext} has unbalanced branches, this is not supported."
            )

        elements = []
        current_string = self._raw_text

        sub_string = ""
        total_atom_number = 0
        while len(current_string) > 0:
            if len(current_string) > 1 and current_string[:2] in _SMILES_DOUBLE_LETTER_ATOM:
                atom = Atom(current_string[:2])
                total_atom_number += 1
                if len(sub_string) != 0:
                    elements.append(sub_string)
                    sub_string = ""
                elements.append(atom)
                current_string = current_string[2:]
                continue

            if current_string[0] in _SMILES_SINGLE_LETTER_ATOM:
                atom = Atom(current_string[0])
                total_atom_number += 1
                if len(sub_string) != 0:
                    elements.append(sub_string)
                    sub_string = ""
                elements.append(atom)
                current_string = current_string[1:]
                continue

            if current_string[0] == "[":
                if current_string.find("]") < 0:
                    raise RuntimeError(
                        f"Token {self._raw_text} has opening '[' but not closing ']'"
                    )
                token = current_string[: current_string.find("]") + 1]
                current_string = current_string[current_string.find("]") + 1 :]
                # Bond descriptor
                if "$" in token or "<" in token or ">" in token:
                    sub_string += token
                else:
                    atom = Atom(token)
                    total_atom_number += 1
                    if len(sub_string) != 0:
                        elements.append(sub_string)
                        sub_string = ""
                    elements.append(atom)
                continue
            sub_string += current_string[0]
            current_string = current_string[1:]
        if len(sub_string) > 0:
            elements.append(sub_string)

        atoms = []
        atom_to_bond = [-1]
        bond_descriptors = []
        element_counter = 0
        while element_counter < len(elements):
            element = elements[element_counter]
            if isinstance(element, Atom):
                atoms.append(element)
                atom_to_bond[-1] = len(atoms) - 1
            elif not isinstance(elements[element_counter], BondDescriptor):
                if "$" in element or "<" in element or ">" in element:
                    if element.find("[") < 0:
                        raise RuntimeError(f"Malformed token found '[' {element}")
                    if element.find("]") <= 0:
                        raise RuntimeError(f"Malformed token ']' found {element}")

                    elementA = element[: element.find("[")]
                    bond_text = element[element.find("[") : element.find("]") + 1]
                    elementB = element[element.find("]") + 1 :]
                    atom_to_bond = _push_pop_atom_branch(elementA, atom_to_bond)

                    if "." not in elementA:
                        atom_bonding_to = atom_to_bond[-1]
                        if atom_bonding_to < 0:
                            atom_bonding_to = 0
                    else:
                        raise RuntimeError(
                            f"Token {big_smiles_ext} not implemented: bond descriptors with a . before them."
                        )
                    # Ensure that bigSMILES always only binds to one atom.
                    # BondDescriptor is first
                    if element_counter != 0:
                        # BondDescriptor is last
                        if element_counter != len(elements) - 1:
                            # Closing branch
                            if ")" not in elementB:
                                # Explicit non-bond after
                                if "." not in elementB:
                                    raise RuntimeError(
                                        f"Token {big_smiles_ext} appears to be invalid, as bond descriptors bond more than one atom."
                                    )
                    first_half = elements[:element_counter]
                    second_half = elements[element_counter + 1 :]

                    elements = first_half

                    if len(elementA) > 0:
                        elements.append(elementA)

                    preceding_characters = elementA
                    if "(" in preceding_characters:
                        preceding_characters = preceding_characters[
                            preceding_characters.find("(") + 1 :
                        ]
                    if ")" in elementB:
                        preceding_characters += elementB[: preceding_characters.find(")")]
                    else:
                        preceding_characters += elementB

                    bond = BondDescriptor(
                        bond_text,
                        len(bond_descriptors) + bond_id_offset,
                        preceding_characters,
                        atom_bonding_to,
                    )
                    elements.append(bond)
                    bond_descriptors.append(bond)
                    if len(elementB) > 0:
                        elements.append(elementB)

                    elements += second_half
                else:
                    atom_to_bond = _push_pop_atom_branch(element, atom_to_bond)

            element_counter += 1

        self.elements = elements
        self.atoms = atoms
        self.bond_descriptors = bond_descriptors

    def generate_string(self, extension):
        string = ""
        for element in self.elements:
            if isinstance(element, str):
                string += element
            else:
                string += element.generate_string(extension)
        return string.strip()

    def generate_smiles_fragment(self):
        string = ""
        for element in self.elements:
            element_string = ""
            if isinstance(element, str):
                element_string += element
            if isinstance(element, Atom):
                element_string += element.generate_string(False)
            if isinstance(element, BondDescriptor):
                # Bond descriptors indicate a missing atom, so no connection between existing atoms
                element_string += "."
            if len(element_string) <= 0:
                raise RuntimeError("expected as non-empty string")

            string += element_string
        # Remove empty branches
        string = string.replace("(.)", "")
        # Remove no-bond before branch end
        string = string.replace(".)", ")")

        string = string.strip(".")
        return string

    @property
    def residues(self):
        return [self]

    @property
    def generable(self):
        for bond in self.bond_descriptors:
            if not bond.generable:
                return False
        return True

    def generate(self, prefix=None, rng=_GLOBAL_RNG):
        super().generate(prefix, rng)

        my_mol = MolGen(self)
        if prefix:
            my_idx = choose_compatible_weight(
                my_mol.bond_descriptors, prefix.bond_descriptors[0], rng
            )

            my_mol = prefix.attach_other(0, my_mol, my_idx)

        return my_mol
