# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details
import lark
from .parser import _GLOBAL_PARSER
from .atom import AromaticSymbol, Atom, AtomSymbol, Isotope


class GBigSMILESTransformer(lark.Transformer):
    def atom(self, children):
        return Atom(text=None, children=children)

    def isotope(self, children):
        return Isotope(children=children)

    def atom_symbol(self, children):
        return AtomSymbol(children=children)

    def aromatic_symbol(self, children):
        return AromaticSymbol(children=children)

    def bond_descriptor(self, children):
        # Remove the square_brackets
        children = children[1:-1]

        children[0]

        # return BondDescriptor(
        print(children)

    def NUMBER(self, children):
        return float(children)

    def INT(self, children):
        return int(children)

    def WS_INLINE(self, children):
        return lark.DISCARD

    # def big_smiles_fragment_definition(self, chil


#    "{[][<]N=Cc(cc1)ccc1C=N[13C@OH1H2+1:3]CC[Si](C)(C)O{[<][>][Si](C)(C)O[<][>]}[Si](C)(C)CCC[>][]}"
