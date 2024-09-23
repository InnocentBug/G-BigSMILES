# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details
import lark


class GBigSMILESTransformer(lark.Transformer):
    def atom(self, children):
        from .atom import Atom

        a = Atom(children)
        return a

    def bracket_atom(self, children):
        from .atom import BracketAtom

        return BracketAtom(children)

    def chiral(self, children):
        from .atom import Chiral

        return Chiral(children)

    def h_count(self, children):
        from .atom import HCount

        return HCount(children)

    def atom_charge(self, children):
        from .atom import AtomCharge

        return AtomCharge(children)

    def atom_class(self, children):
        from .atom import AtomClass

        return AtomClass(children)

    def isotope(self, children):
        from .atom import Isotope

        return Isotope(children)

    def atom_symbol(self, children):
        from .atom import AtomSymbol

        return AtomSymbol(children)

    def aromatic_symbol(self, children):
        from .atom import AromaticSymbol

        return AromaticSymbol(children)

    def aliphatic_organic(self, children):
        from .atom import AliphaticOrganic

        return AliphaticOrganic(children)

    def aromatic_organic(self, children):
        from .atom import AromaticOrganic

        return AromaticOrganic(children)

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

    # def big_smiles_fragment_definition(self, children


#    "{[][<]N=Cc(cc1)ccc1C=N[13C@OH1H2+1:3]CC[Si](C)(C)O{[<][>][Si](C)(C)O[<][>]}[Si](C)(C)CCC[>][]}"

_GLOBAL_TRANSFORMER = GBigSMILESTransformer(visit_tokens=True)
