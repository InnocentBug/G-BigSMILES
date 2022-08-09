# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details


class Stochastic:
    """
    Stoachstic object parsing for extended bigSMILES.
    """

    def __init__(self, big_smiles_ext):
        """
        Constructor, taking a extended bigSMILES string for generation.

        Arguments:
        ----------
        big_smiles_ext: str
          text representation of bigSMILES stochastic object.
        """
