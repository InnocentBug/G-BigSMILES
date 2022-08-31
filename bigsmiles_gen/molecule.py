# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy

from .core import BigSMILESbase
from .mixture import Mixture
from .stochastic import Stochastic
from .token import SmilesToken


class Molecule(BigSMILESbase):
    """
    Part of an extended bigSMILES description hat contains up to exactly one mixture description.
    """

    def __init__(self, big_smiles_ext):
        """
        Construction of a molecue to make up mixtures.

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
            if len(pre_token) > 0:
                token = SmilesToken(pre_token, 0)
                self._elements.append(token)
            stochastic_text = stochastic_text[stochastic_text.find("{") :].strip()
            end_pos = stochastic_text.find("}") + 1
            if end_pos < 0:
                raise RuntimeError(
                    f"System {stochastic_text} contains an openining '{' for a stochastic object, but no closing '}'."
                )
            # Find distribution extension
            if stochastic_text[end_pos] == "|":
                end_pos = stochastic_text.find("|", end_pos + 2) + 1
            self._elements.append(Stochastic(stochastic_text[:end_pos]))

            stochastic_text = stochastic_text[end_pos:].strip()

        if len(stochastic_text) > 0:
            token = SmilesToken(stochastic_text, 0)
            self._elements.append(token)

    @property
    def generatable(self):
        if self.mixture is not None:
            if not self.mixture.generatable:
                return False

        for ele in self._elements:
            if not ele.generatable:
                return False

        return True

    def generate_string(self, extension):
        string = ""
        for ele in self._elements:
            string += ele.generate_string(extension)
        if self.mixture:
            string += self.mixture.generate_string(extension)
        return string
