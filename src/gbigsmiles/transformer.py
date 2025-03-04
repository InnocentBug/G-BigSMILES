# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details
import lark
from lark.visitors import Discard

from .exception import UnsupportedBigSMILES


class GBigSMILESTransformer(lark.Transformer):
    def stochastic_generation(self, children):
        return children[1]

    def NUMBER(self, children):
        return float(children)

    def DIGIT(self, children):
        return str(children)

    def INT(self, children):
        return int(children)

    def WS_INLINE(self, children):
        return Discard

    def big_smiles_fragment_definition(self, children):
        raise UnsupportedBigSMILES("big_smiles_fragment_definition", children)

    def big_smiles_fragment_declaration(self, children):
        raise UnsupportedBigSMILES("big_smiles_fragment_declaration", children)

    def ladder_bond_descriptor(self, children):
        raise UnsupportedBigSMILES("ladder_bond_descriptor", children)

    def inner_non_covalent_descriptor(self, children):
        raise UnsupportedBigSMILES("inner_non_covalent_descriptor", children)

    def inner_ambi_covalent_descriptor(self, children):
        raise UnsupportedBigSMILES("inner_ambi_covalent_descriptor", children)

    def non_covalent_bond_descriptor(self, children):
        raise UnsupportedBigSMILES("non_covalent_bond_descriptor", children)


_GLOBAL_TRANSFORMER: None | GBigSMILESTransformer = None


def get_global_transformer():
    global _GLOBAL_TRANSFORMER
    if _GLOBAL_TRANSFORMER is None:
        import gbigsmiles

        transformer = lark.ast_utils.create_transformer(ast_module=gbigsmiles, transformer=GBigSMILESTransformer(visit_tokens=True))
        _GLOBAL_TRANSFORMER = transformer

    return _GLOBAL_TRANSFORMER
