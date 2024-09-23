# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import lark

from .core import BigSMILESbase
from .exception import GBigSMILESParsingError, GBigSMILESTooManyTokens


class Atom(BigSMILESbase):
    _symbol = None

    @property
    def symbol(self):
        return self._symbol

    def __init__(self, children: list):
        super().__init__(children)

        for child in self._children:
            if isinstance(child, AtomSymbol):
                if self._symbol is not None:
                    raise GBigSMILESTooManyTokens(self.__class__, self._symbol, child)
                self._symbol = child

    def generate_string(self, extension):
        return str(self.symbol)

    @property
    def aromatic(self):
        return self.symbol.aromatic

    @property
    def generable(self):
        return True

    @property
    def charge(self):
        return 0


class BracketAtom(Atom):
    _isotope = None

    @property
    def isotope(self):
        return self._isotope

    _chiral = None

    @property
    def chiral(self):
        return self._chiral

    _h_count = None

    @property
    def h_count(self):
        return self._h_count

    _atom_charge = None

    @property
    def atom_charge(self):
        return self._atom_charge

    _atom_class = None

    @property
    def atom_class(self):
        return self._atom_class

    def __init__(self, children: list):
        super().__init__(children)

        for child in self._children:
            if isinstance(child, AtomSymbol):
                pass
            elif isinstance(child, Isotope):
                self._isotope = child
            elif isinstance(child, Chiral):
                self._chiral = child
            elif isinstance(child, HCount):
                self._h_count = child
            elif isinstance(child, AtomCharge):
                self._atom_charge = child
            elif isinstance(child, AtomClass):
                self._atom_class = child

    def generate_string(self, extension):
        string = "["
        if self.isotope:
            string += self.isotope.generate_string(extension)
        string += super().generate_string(extension)
        if self.chiral:
            string += self.chiral.generate_string(extension)
        if self.h_count:
            string += self.h_count.generate_string(extension)
        if self.atom_charge:
            string += self.atom_charge.generate_string(extension)
        if self.atom_class:
            string += self.atom_class.generate_string(extension)
        string += "]"
        return string

    @property
    def charge(self):
        if self.atom_charge:
            return self.atom_charge.charge
        return super().charge


class Isotope(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)
        self._value = int(self._children[0])

    def generate_string(self, extension: bool):
        return str(self._value)

    def generable(self):
        return True

    @property
    def num_nuclei(self):
        return int(self._value)


class AtomSymbol(BigSMILESbase):
    def __init__(self, children: list[lark.Token]):
        self._value = str(children[0])

    def generate_string(self, extension):
        return self._value

    @property
    def aromatic(self):
        return False

    def generable(self):
        return True


class Chiral(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)

        self._symbol = str(self._children[0])
        if len(self._children) > 1:
            self._symbol += str(self._children[1])
        if len(self._children) > 2:
            self._symbol += str(self._children[2])

    @property
    def generable(self):
        return True

    def generate_string(self, extension):
        return self._symbol


class HCount(BigSMILESbase):
    def __init__(self, children: list):
        self._count = None
        super().__init__(children)

        if str(self._children[0]) != "H":
            raise GBigSMILESParsingError(self._children[0])

        if len(self._children) > 1:
            self._count = int(self._children[1])

    @property
    def num(self) -> int:
        if self._count is None:
            return 1
        return self._count

    @property
    def generable(self):
        return True

    def generate_string(self, extension):
        if self._count:
            return f"H{self.num}"
        return "H"


class AtomCharge(BigSMILESbase):

    def __init__(self, children):
        super().__init__(children)
        self._number = False
        if str(self._children[0]) == "++":
            self._sign = +1
            self._value = 2
        elif str(self._children[0]) == "--":
            self._sign = -1
            self._value = 2
        else:
            self._sign = int(str(self._children[0] + "1"))
            self._value = 1
            if len(self._children) > 1:
                self._value = int(self._children[1])
                self._number = True

    @property
    def generable(self):
        return True

    def generate_string(self, extension):
        if not self._number:
            if self._sign > 0:
                if self._value > 1:
                    return "++"
                else:
                    return "+"
            else:
                if self._value > 1:
                    return "--"
                else:
                    return "-"
        if self._sign > 0:
            return "+" + str(self._value)
        else:
            return "-" + str(self._value)

    @property
    def charge(self):
        return self._sign * self._value


class AtomClass(BigSMILESbase):

    def __init__(self, children):
        super().__init__(children)
        self._class_num = int(self._children[1])

    @property
    def generable(self):
        return True

    def generate_string(self, extension):
        return ":" + str(self._class_num)

    @property
    def num(self):
        return self._class_num


class AromaticSymbol(AtomSymbol):
    @property
    def aromatic(self):
        return True


class AliphaticOrganic(AtomSymbol):
    @property
    def aromatic(self):
        return False


class AromaticOrganic(AtomSymbol):
    @property
    def aromatic(self):
        return True
