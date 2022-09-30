# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy

from .bond import _create_compatible_bond_text
from .core import _GLOBAL_RNG, BigSMILESbase
from .mixture import Mixture
from .stochastic import Stochastic
from .token import SmilesToken


class Molecule(BigSMILESbase):
    """
    Part of an extended bigSMILES description hat contains up to exactly one mixture description.
    """

    def __init__(self, big_smiles_ext):
        """
        Construction of a molecule to make up mixtures.

        Arguments:
        ----------
        big_smiles_ext: str
           text representation
        """
        self._raw_text = big_smiles_ext.strip()

        self._elements = []
        stochastic_text = copy.copy(self._raw_text)

        self.mixture = None
        # TODO: find and verify non-extension non-bonds '.'
        # without confusing them with floating point numbers
        if stochastic_text.find(".|") >= 0:
            start = stochastic_text.find(".|")
            end = stochastic_text.find("|", start + 3) + 1
            mixture_text = stochastic_text[start:end]
            end_text = stochastic_text[end:].strip()
            if len(end_text) > 0:
                raise RuntimeError(
                    f"Molecule {stochastic_text} does not end with a mixture descriptor '.|'."
                )
            stochastic_text = stochastic_text[:start]
            self.mixture = Mixture(mixture_text)

        while stochastic_text.find("{") >= 0:
            pre_token = stochastic_text[: stochastic_text.find("{")].strip()
            pre_stochastic = None
            if len(pre_token) > 0:
                pre_stochastic = SmilesToken(pre_token, 0)
                # Find the connecting terminal bond descriptor of previous element.
                if len(self._elements) > 0:
                    # Get expected terminal bond descriptor
                    if isinstance(self._elements[-1], Stochastic):
                        other_bd = self._elements[-1].right_terminal
                    else:
                        other_bd = self._elements[-1].bond_descriptors[-1]
                    if len(pre_stochastic.bond_descriptors) > 0:
                        found_compatible = False
                        for bd in pre_stochastic.bond_descriptors[0]:
                            if bd.is_compatible(other_bd):
                                found_compatible = True
                        if not found_compatible:
                            raise RuntimeError(
                                f"Token {pre_token} only has incompatible bond descriptors with previous element {str(self._elements[-1])}."
                            )
                    # Since this isn't standard, we add a bond descriptor here.
                    else:
                        bond_string = _create_compatible_bond_text(other_bd)
                        pre_token = bond_string + pre_token
                        pre_stochastic = SmilesToken(pre_token, 0)

            stochastic_text = stochastic_text[stochastic_text.find("{") :].strip()
            end_pos = stochastic_text.find("}") + 1
            if end_pos < 0:
                raise RuntimeError(
                    f"System {stochastic_text} contains an opening '{' for a stochastic object, but no closing '}'."
                )
            # Find distribution extension
            if stochastic_text[end_pos] == "|":
                end_pos = stochastic_text.find("|", end_pos + 2) + 1
            stochastic = Stochastic(stochastic_text[:end_pos])
            if pre_stochastic:
                min_expected_bond_descriptors = 2
                if len(self._elements) == 0:
                    min_expected_bond_descriptors = 1
                if len(pre_stochastic.bond_descriptors) < min_expected_bond_descriptors:
                    # Attach a compatible bond descriptor automatically
                    other_bd = stochastic.left_terminal
                    bond_text = _create_compatible_bond_text(other_bd)
                    pre_token += bond_text
                    pre_stochastic = SmilesToken(pre_token, 0)
                self._elements.append(pre_stochastic)
            self._elements.append(stochastic)

            stochastic_text = stochastic_text[end_pos:].strip()

        if len(stochastic_text) > 0:
            token = SmilesToken(stochastic_text, 0)
            if len(self._elements) > 0 and len(token.bond_descriptors) == 0:
                if isinstance(self._elements[-1], Stochastic):
                    bond_text = _create_compatible_bond_text(self._elements[-1].right_terminal)
                else:
                    bond_text = _create_compatible_bond_text(
                        self._elements[-1].bond_descriptors[-1]
                    )
                token = SmilesToken(bond_text + stochastic_text, 0)
            self._elements.append(token)

    @property
    def generable(self):
        if self.mixture is not None:
            if not self.mixture.generable:
                return False

        for ele in self._elements:
            if not ele.generable:
                return False

        return True

    def generate_string(self, extension):
        string = ""
        for ele in self._elements:
            string += ele.generate_string(extension)
        if self.mixture:
            string += self.mixture.generate_string(extension)
        return string

    def generate(self, prefix=None, rng=_GLOBAL_RNG):
        my_mol = prefix
        for element in self._elements:
            my_mol = element.generate(my_mol, rng)

        return my_mol
