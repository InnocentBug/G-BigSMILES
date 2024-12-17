# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details
import lark
from lark.visitors import Discard


class GBigSMILESTransformer(lark.Transformer):
    def NUMBER(self, children):
        return float(children)

    def INT(self, children):
        return int(children)

    def WS_INLINE(self, children):
        return Discard


_GLOBAL_TRANSFORMER: None | GBigSMILESTransformer = None


def get_global_transformer():
    global _GLOBAL_TRANSFORMER
    if _GLOBAL_TRANSFORMER is None:
        import gbigsmiles

        transformer = lark.ast_utils.create_transformer(
            ast_module=gbigsmiles, transformer=GBigSMILESTransformer(visit_tokens=True)
        )
        _GLOBAL_TRANSFORMER = transformer

    return _GLOBAL_TRANSFORMER
