# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from warnings import warn

from .core import BigSMILESbase
from .bond import BondDescriptor
from .distribution import get_distribution
from .mixture import Mixture
from .token import SmilesToken


def _adjust_weight(text, token_list):
    if len(token_list) == 0:
        return True
    num_unspecified_weights = 0
    total_weight = 0
    for token in token_list:
        if token.weight is not None:
            total_weight += token.weight
        else:
            num_unspecified_weights += 1
    if num_unspecified_weights > 1:
        warn(
            f"Stochastic object {text} has too many missing weight of the tokens to be generative."
        )
        return False

    if num_unspecified_weights == 1:
        for token in token_list:
            if token.weight is None:
                weight = 1 - total_weight
                if weight < 0 or weight > 1:
                    warn(
                        "The missing weight of added to {repeat_unit_text} stochastic object is invalid. Check the weights of the smiles tokens."
                    )
                    return False
                else:
                    token.weight = weight
                    total_weight += weight
                    break
    if abs(total_weight - 1) > 1e-6:
        warn(f"Stochastic object {text} has invalid total repeat unit weight.")
        return False
    return True



class Stochastic(BigSMILESbase):
    """
    Stoachstic object parsing for extended bigSMILES.

    ## Note: Empty stochastic objects, that only contain a single terminal bond descriptor are not supported.
    """

    def __init__(self, big_smiles_ext):
        """
        Constructor, taking a extended bigSMILES string for generation.

        Arguments:
        ----------
        big_smiles_ext: str
          text representation of bigSMILES stochastic object.
        """

        self._raw_text = big_smiles_ext.strip()
        self._generatable = True
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
        bond_text = middle_text[middle_text.find("[") : middle_text.find("]", 1)+1]
        preceding_characters = middle_text[: middle_text.find("[")]
        self.bond_descriptors = []
        bond = BondDescriptor(bond_text, len(self.bond_descriptors), preceding_characters)
        self.bond_descriptors.append(bond)

        # Right terminal bond descriptor
        i = middle_text.rfind("[")
        right_bond_text = middle_text[i : middle_text.find("]", middle_text.rfind("["))+1]
        while i > 0 and middle_text[i] in r".-=#$:/\@":
            i -= 1
        right_preceding_char = middle_text[i : middle_text.find("[", i)]

        if ";" in middle_text:
            repeat_unit_text = middle_text[middle_text.find("]", 1) : middle_text.find(";")]
            end_group_text = middle_text[middle_text.find(";") : middle_text.rfind("[")]
        else:
            repeat_unit_text = middle_text[middle_text.find("]", 1)+1 : middle_text.rfind("[")]
            end_group_text = ""

        self.repeat_tokens = []
        for ru in repeat_unit_text.split(","):
            ru = ru.strip()
            token = SmilesToken(ru, len(self.bond_descriptors))
            self.repeat_tokens.append(token)
            self.bond_descriptors += token.bond_descriptors
        self._generatable = _adjust_weight(repeat_unit_text, self.repeat_tokens)

        self.end_tokens = []
        for eg in end_group_text.split(","):
            eg = eg.strip()
            if len(eg) > 0:
                token = SmilesToken(eg, len(self.bond_descriptors))
                self.end_tokens.append(token)
                self.bond_descriptors += token.bond_descriptors
        self._generatable = _adjust_weight(end_group_text, self.repeat_tokens)

        right_terminal_token = BondDescriptor(
            right_bond_text, len(self.bond_descriptors), right_preceding_char
        )
        self.bond_descriptors.append(right_terminal_token)

        end_text = self._raw_text[self._raw_text.find("}")+1 :]
        if end_text.find(".|") >= 0:
            distribution_text = end_text[: end_text.find(".|")].strip()
            mixture_text = end_text[end_text.find(".|") :].strip()
        else:
            distribution_text = end_text.strip()
            mixture_text = ""

        self.distribution = None
        if len(distribution_text) > 1:
            self.distribution = get_distribution(distribution_text)

        self.mixture = None
        if len(mixture_text) > 1:
            self.mixture = Mixture(mixture_text)

    @property
    def generatable(self):
        for bond in self.bond_descriptors:
            if not bond.generatable:
                return False
        for token in self.repeat_tokens + self.end_tokens:
            if not token.generatable:
                return False

        return self._generatable

    def generate_string(self, extension):
        string = "{"
        string += self.bond_descriptors[0].generate_string(extension)
        for token in self.repeat_tokens:
            string += token.generate_string(extension)+", "
        string = string[:-2]
        if len(self.end_tokens) > 0:
            string += "; "
            for token in self.end_tokens:
                string += token.generate_string(extension) + ", "
            string = string[:-2]
        string += self.bond_descriptors[-1].generate_string(extension)
        string += "}"
        if self.distribution:
            string += self.distribution.generate_string(extension)
        if self.mixture:
            string += self.mixture.generate_string(extension)

        return string
