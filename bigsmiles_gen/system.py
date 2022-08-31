# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy

from .core import BigSMILESbase
from .stochastic import Stochastic
from .token import SmilesToken


def _estimate_system_molecular_weight(elements):
    estimated_weights = []
    fractions = {}
    absolutes = {}
    n_


class System(BigSMILESbase):
    """
    Entire system representation in extended bigSMILES.
    """

    def __init__(self, big_smiles_ext):
        """
        Construction of an entire system (mixtures).

        Arguments:
        ----------
        big_smiles_ext: str
           text representation

        seed: int
           Seed for the random number generation.
        """
        self._raw_text = big_smiles_ext.strip()

        self._elements = []
        stochastic_text = copy.copy(self._raw_text)
        while stochastic_text.find("{") >= 0:
            pre_token = stochastic_text[: stochastic_text.find("{")].strip()
            if len(pre_token) > 0:
                token = SmilesToken(pre_token, 0)
                self._elements.append(token)
            stochastic_text = stochastic_text[stochastic_text.find("{") :].strip()
            end_pos = stochastic_text.find("}")
            if end_pos < 0:
                raise RuntimeError(
                    f"System {stochastic_text} contains an openining '{' for a stochastic object, but no closing '}'."
                )
            # Find distribution extension
            if stochastic_text[end_pos + 1] == "|":
                end_pos = stochastic_text.find("|", end_pos + 2)
            # Find mixture extension
            if stochastic_text[end_pos + 1 :].startswith(".|"):
                end_pos = stochastic_text.find("|", end_pos + 3)
            self._elements.append(Stochastic(stochastic_text[:end_pos]))

            stochastic_text = stochastic_text[end_pos:].strip()
        if len(stochastic_text) > 0:
            token = SmilesToken(stochastic_text, 0)
            self._elements.append(token)
