# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details


class BondDescriptor:
    """
    Bond descriptor of the bigSMILES notation.
    """

    def __init__(self, big_smiles_ext, descr_id, preceding_characters):
        """
        Construction of a bond descriptor.

        Arguments:
        ----------
        big_smiles_ext: str
           text representation of a bond descriptor. Example: `[$0]`

        descr_id: int
           Position of bond description in the line notation of the stoachstic object.
           Ensure that it starts with `0` and is strictly monotonic increasing during a stochastic object.

        preceding_characters: str
           Characters preceding the bond descriptor, that is not an atom.
           These characters are used to idendify the type of bond, any stereochemistry etc.
           If no further characters are provided, it is assumed to be a single bond without stereochemistry specification.
        """
        self._raw_text = big_smiles_ext
        if self._raw_text[0] != "[" or self._raw_text[-1] != "]":
            raise RuntimeError(f"Bond descriptor {self._raw_text} does not start and end with []")
