# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import lark

from .core import BigSMILESbase


class Atom(BigSMILESbase):
    def __init__(
        self, text: str | None = None, children: list[lark.Tree | lark.Token] | None = None
    ):
        super().__init__(text, children)
        self._raw_text = ""

        for token in self._children:
            print(token)
            self._raw_text += str(token.value)

        self._raw_text = text

    def generate_string(self, extension):
        return self._raw_text

    @property
    def value(self):
        return self.generate_string(True)

    @property
    def generable(self):
        return True


class Isotope(BigSMILESbase):
    def __init__(
        self, text: str | None = None, children: list[lark.Tree | lark.Token] | None = None
    ):
        super().__init__(text, children)
        self._value = int(self._children[0])

    def generate_string(self, extension):
        return str(self.value)

    @property
    def value(self):
        return self._value

    @property
    def generable(self):
        return True


class AtomSymbol(BigSMILESbase):
    def __init__(
        self, text: str | None = None, children: list[lark.Tree | lark.Token] | None = None
    ):
        super().__init__(text, children)
        self._value = str(children[0])

    def generate_string(self, extension):
        return str(self.value)

    @property
    def value(self):
        return self._value

    @property
    def generable(self):
        return True

    @property
    def aromatic(self):
        return False


class AtomSymbol(BigSMILESbase):
    def __init__(
        self, text: str | None = None, children: list[lark.Tree | lark.Token] | None = None
    ):
        super().__init__(text, children)
        print("atom_symbol", children)
        self._value = str(children[0])

    def generate_string(self, extension):
        return str(self.value)

    @property
    def value(self):
        return self._value

    @property
    def generable(self):
        return True

    @property
    def aromatic(self):
        return False


class AromaticSymbol(AtomSymbol):
    @property
    def aromatic(self):
        return True
