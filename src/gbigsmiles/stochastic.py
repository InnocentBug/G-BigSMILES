# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy
from warnings import warn

from rdkit.Chem import Descriptors as rdDescriptors

from .bond import BondDescriptor, _create_compatible_bond_text
from .core import _GLOBAL_RNG, BigSMILESbase, choose_compatible_weight
from .distribution import get_distribution
from .mol_gen import MolGen
from .token import SmilesToken


class Stochastic(BigSMILESbase):
    """
    Stochastic object parsing for extended bigSMILES.

    ## Note: Empty stochastic objects, that only contain a single terminal bond descriptor are not supported.
    """

    def __init__(self, big_smiles_ext, res_id_prefix):
        """
        Constructor, taking a extended bigSMILES string for generation.

        Arguments:
        ---------
        big_smiles_ext: str
          text representation of bigSMILES stochastic object.
        res_id_prefix: int
          How many residues are preceding this one in the G-BigSMILES string.

        """

        self._raw_text = big_smiles_ext.strip()
        self._generable = True
        if self._raw_text[0] != "{":
            raise RuntimeError(
                "Stochastic object '" + self._raw_text + "' does not start with '{'."
            )
        if self._raw_text.rfind("}") < 0:
            raise RuntimeError("Stochastic object '" + self._raw_text + "' does not end with '}'.")

        middle_text = self._raw_text[1 : self._raw_text.rfind("}")]
        if middle_text[middle_text.find("]") + 1] == "}":
            raise RuntimeError(
                f"Empty stochastic object {middle_text} that have only a single terminal bond descriptor are not supported."
            )
        # Left terminal bond descriptor.
        if middle_text.find("]", 1) <= 0:
            raise RuntimeError(f"Unterminated left terminal bond descriptor in {middle_text}.")
        bond_text = middle_text[middle_text.find("[") : middle_text.find("]", 1) + 1]
        preceding_characters = middle_text[: middle_text.find("[")]
        self.bond_descriptors = []
        bond = BondDescriptor(bond_text, len(self.bond_descriptors), preceding_characters, None)
        self.left_terminal = bond

        # Right terminal bond descriptor
        i = middle_text.rfind("[")
        right_bond_text = middle_text[i : middle_text.find("]", middle_text.rfind("[")) + 1]
        while i > 0 and middle_text[i] in r".-=#$:/\@":
            i -= 1
        right_preceding_char = middle_text[i : middle_text.find("[", i)]

        if ";" in middle_text:
            repeat_unit_text = middle_text[middle_text.find("]", 1) + 1 : middle_text.find(";")]
            end_group_text = middle_text[middle_text.find(";") + 1 : middle_text.rfind("[")]
        else:
            repeat_unit_text = middle_text[middle_text.find("]", 1) + 1 : middle_text.rfind("[")]
            end_group_text = ""

        self.repeat_tokens = []
        self.repeat_bonds = []
        self.repeat_bond_token_idx = []
        res_id_counter = 0
        for ru in repeat_unit_text.split(","):
            ru = ru.strip()
            if len(ru) > 0:
                token = SmilesToken(ru, len(self.bond_descriptors), res_id_prefix + res_id_counter)
                res_id_counter += 1
                self.repeat_tokens.append(token)
                self.bond_descriptors += token.bond_descriptors
                self.repeat_bonds += token.bond_descriptors
                for _ in range(len(token.bond_descriptors)):
                    self.repeat_bond_token_idx.append(len(self.repeat_tokens) - 1)

        self.end_tokens = []
        self.end_bonds = []
        self.end_bond_token_idx = []
        for eg in end_group_text.split(","):
            eg = eg.strip()
            if len(eg) > 0:
                token = SmilesToken(eg, len(self.bond_descriptors), res_id_prefix + res_id_counter)
                res_id_counter += 1
                self.end_tokens.append(token)
                self.bond_descriptors += token.bond_descriptors
                self.end_bonds += token.bond_descriptors
                for _ in range(len(token.bond_descriptors)):
                    self.end_bond_token_idx.append(len(self.end_tokens) - 1)

        right_terminal_token = BondDescriptor(
            right_bond_text, len(self.bond_descriptors), right_preceding_char, None
        )
        self.right_terminal = right_terminal_token

        end_text = self._raw_text[self._raw_text.find("}") + 1 :]
        if end_text.find(".|") >= 0:
            distribution_text = end_text[: end_text.find(".|")].strip()
        else:
            distribution_text = end_text.strip()

        self.distribution = None
        if len(distribution_text) > 1:
            self.distribution = get_distribution(distribution_text)
        self._validate()

    def _validate(self):
        for bd in self.bond_descriptors:
            if bd.transitions is not None and len(bd.transitions) != len(self.bond_descriptors):
                raise RuntimeError(
                    f"Invalid transition length in bond descriptor {len(bd.transitions)} but the stochastic element has only {len(self.bond_descriptors)} descriptors."
                )

        if not len(self.bond_descriptors) == len(self.end_bonds) + len(self.repeat_bonds):
            raise RuntimeError(
                f"Length of bond descriptors {len(self.bond_descriptors)} is incompatible with individual bonds counted {len(self.end_bonds)} {len(self.repeat_bonds)}."
            )

    @property
    def generable(self):
        for bond in self.bond_descriptors:
            if not bond.generable:
                return False
        for token in self.repeat_tokens + self.end_tokens:
            if not token.generable:
                return False
        if self.distribution is None:
            return False
        if not self.distribution.generable:
            return False

        return self._generable

    def generate_string(self, extension):
        string = "{"
        string += self.left_terminal.generate_string(extension)
        for token in self.repeat_tokens:
            string += token.generate_string(extension) + ", "
        string = string[:-2]
        if len(self.end_tokens) > 0:
            string += "; "
            for token in self.end_tokens:
                string += token.generate_string(extension) + ", "
            string = string[:-2]
        string += self.right_terminal.generate_string(extension)
        string += "}"
        if self.distribution:
            string += self.distribution.generate_string(extension)

        return string.strip()

    def generate(self, prefix=None, rng=_GLOBAL_RNG):
        def get_start():
            my_mol = prefix
            if my_mol is None:
                # Ensure, that really no prefix is expected.
                if str(self.left_terminal) != "[]":
                    raise RuntimeError(
                        "Generating stochastic object without prefix,"
                        " but the first terminal bond descriptor is not '[]',"
                        " so a prefix is expected."
                    )
                # Find a random end group to start
                try:
                    end_bond_idx = choose_compatible_weight(self.end_bonds, None, rng)
                except ValueError as exc:
                    warn("Unable to pick an end group bond to start generating.", stacklevel=1)
                    raise exc
                start_token = self.end_tokens[self.end_bond_token_idx[end_bond_idx]]
                if len(start_token.bond_descriptors) != 1:
                    raise RuntimeError("Single bond descriptor expected here.")
                my_mol = MolGen(start_token)
            else:
                # Ensure prefix is compatible with terminal bond descriptor.
                if len(prefix.bond_descriptors) != 1:
                    raise RuntimeError("Single bond descriptor expected here.")
                if prefix.bond_descriptors[0].generate_string(
                    False
                ) != self.left_terminal.generate_string(False):
                    raise RuntimeError(
                        "The open bond descriptor of the prefix is not compatible"
                        " with the left terminal bond descriptor of the stochastic object."
                    )
            return my_mol

        def generate_repeat_units_and_finalize(my_mol):
            def add_repeat_unit(my_mol):
                starting_bond_idx = choose_compatible_weight(my_mol.bond_descriptors, None, rng)
                starting_bond = my_mol.bond_descriptors[starting_bond_idx]

                connecting_bond_idx = choose_compatible_weight(
                    self.repeat_bonds, starting_bond, rng
                )

                # Handle explicit transition probability
                if starting_bond.transitions is not None:
                    prob = starting_bond.transitions / starting_bond.weight
                    connecting_bond_idx = rng.choice(range(len(prob)), p=prob)

                # Find the new token and bond
                if connecting_bond_idx < len(self.repeat_bonds):
                    token = self.repeat_tokens[self.repeat_bond_token_idx[connecting_bond_idx]]
                    connecting_bond = self.repeat_bonds[connecting_bond_idx]
                else:  # In case of transition probabilities, we explicitly handle end tokens
                    connecting_bond_idx -= len(self.repeat_bonds)
                    connecting_bond = self.end_bonds[connecting_bond_idx]
                    token = self.end_tokens[self.end_bond_token_idx[connecting_bond_idx]]

                # Find the bond index in the new token
                connecting_bond_idx = token.bond_descriptors.index(connecting_bond)
                new_mol = MolGen(token)

                my_mol = my_mol.attach_other(starting_bond_idx, new_mol, connecting_bond_idx)

                return my_mol

            starting_mol_weight = rdDescriptors.HeavyAtomMolWt(my_mol.mol)
            target_mol_weight = self.distribution.draw_mw(rng)
            while True:
                my_mol = add_repeat_unit(my_mol)
                # Prematurely end if no more open bonds available
                if len(my_mol.bond_descriptors) == 0:
                    warn(
                        f"Premature end of generation of {str(self)} because no more open bond descriptors found.",
                        stacklevel=1,
                    )
                    finalized_my_mol = my_mol
                    break
                # End prematurely if transitions set (bc they can instate end groups)
                # and only one bond descriptor is present for the terminal end
                # if (
                #     starting_bond.transitions is not None
                #     and str(self.right_terminal) != "[]"
                #     and len(my_mol.bond_descriptors) == 1
                # ):
                #     warn(
                #         f"Premature end of generation of {str(self)} with transitions specified"
                #         " and only a single bond descriptor open for a required terminal.",
                #         stacklevel=1,
                #     )
                #     finalized_my_mol = my_mol
                #     break

                finalized_my_mol = finalize_mol(copy.deepcopy(my_mol))
                if (
                    rdDescriptors.HeavyAtomMolWt(my_mol.mol) - starting_mol_weight
                    > target_mol_weight
                ):
                    break

            return finalized_my_mol

        def finalize_mol(my_mol):
            terminal_bond = None
            if str(self.right_terminal) != "[]":
                # Invert compatibility
                invert_text = _create_compatible_bond_text(self.right_terminal)
                invert_terminal = BondDescriptor(invert_text, 0, "", None)
                terminal_bond_idx = choose_compatible_weight(
                    my_mol.bond_descriptors, invert_terminal, rng
                )
                terminal_bond = my_mol.bond_descriptors[terminal_bond_idx]
                # Remove this bond descriptor from molecule temporarily
                # such that it doesn't get reacted with end group.
                del my_mol.bond_descriptors[terminal_bond_idx]

            # stochastic generation complete, now starting end group termination
            while len(my_mol.bond_descriptors) > 0:
                starting_bond_idx = choose_compatible_weight(my_mol.bond_descriptors, None, rng)
                starting_bond = my_mol.bond_descriptors[starting_bond_idx]
                connecting_bond_idx = choose_compatible_weight(self.end_bonds, starting_bond, rng)

                token = self.end_tokens[self.end_bond_token_idx[connecting_bond_idx]]
                connecting_bond = self.end_bonds[connecting_bond_idx]

                # Find the bond index in the new token
                connecting_bond_idx = token.bond_descriptors.index(connecting_bond)
                new_mol = MolGen(token)

                my_mol = my_mol.attach_other(starting_bond_idx, new_mol, connecting_bond_idx)

            # Reinsert final terminal bond descriptor
            if terminal_bond:
                my_mol.bond_descriptors.append(terminal_bond)

            return my_mol

        super().generate(prefix, rng)

        my_mol = get_start()
        my_mol = generate_repeat_units_and_finalize(my_mol)

        return my_mol

    @property
    def residues(self):
        residues = []
        for token in self.repeat_tokens:
            residues += token.residues
        for token in self.end_tokens:
            residues += token.residues
        return residues
