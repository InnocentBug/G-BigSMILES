# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import lark

from .core import BigSMILESbase
from .exception import ParsingError, TooManyTokens


class Atom(BigSMILESbase):
    """
    Represents an atom in a chemical structure.
    """

    _symbol = None

    @property
    def symbol(self):
        """Returns the atom symbol."""
        return self._symbol

    def __init__(self, children: list):
        """
        Initializes an Atom object.

        Args:
            children (list): List of child elements.

        """
        super().__init__(children)

        for child in self._children:
            if isinstance(child, AtomSymbol):
                if self._symbol is not None:
                    raise TooManyTokens(self.__class__, self._symbol, child)
                self._symbol = child

    def generate_string(self, extension):
        """
        Generates a string representation of the atom.

        Args:
            extension: Extension parameter (unused in this method).

        Returns:
            str: String representation of the atom.

        """
        return str(self.symbol)

    @property
    def aromatic(self):
        """Returns whether the atom is aromatic."""
        return self.symbol.aromatic

    @property
    def generable(self):
        """Returns whether the atom can be generated."""
        return True

    @property
    def charge(self):
        """Returns the charge of the atom."""
        return 0


class BracketAtom(Atom):
    """
    Represents a bracketed atom in a chemical structure.
    """

    _isotope = None
    _chiral = None
    _h_count = None
    _atom_charge = None
    _atom_class = None

    @property
    def isotope(self):
        """Returns the isotope of the atom."""
        return self._isotope

    @property
    def chiral(self):
        """Returns the chirality of the atom."""
        return self._chiral

    @property
    def h_count(self):
        """Returns the hydrogen count of the atom."""
        return self._h_count

    @property
    def atom_charge(self):
        """Returns the charge of the atom."""
        return self._atom_charge

    @property
    def atom_class(self):
        """Returns the class of the atom."""
        return self._atom_class

    def __init__(self, children: list):
        """
        Initializes a BracketAtom object.

        Args:
            children (list): List of child elements.

        """
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
        """
        Generates a string representation of the bracketed atom.

        Args:
            extension: Extension parameter.

        Returns:
            str: String representation of the bracketed atom.

        """

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
        """Returns the charge of the atom."""
        if self.atom_charge:
            return self.atom_charge.charge
        return super().charge


class Isotope(BigSMILESbase):
    """
    Represents an isotope of an atom.
    """

    def __init__(self, children: list):
        """
        Initializes an Isotope object.

        Args:
            children (list): List of child elements.

        """
        super().__init__(children)
        self._value = int(self._children[0])

    def generate_string(self, extension: bool):
        """
        Generates a string representation of the isotope.

        Args:
            extension (bool): Extension parameter (unused in this method).

        Returns:
            str: String representation of the isotope.

        """
        return str(self._value)

    def generable(self):
        """Returns whether the isotope can be generated."""
        return True

    @property
    def num_nuclei(self):
        """Returns the number of nuclei in the isotope."""
        return int(self._value)


class AtomSymbol(BigSMILESbase):
    """
    Represents the symbol of an atom.
    """

    def __init__(self, children: list[lark.Token]):
        """
        Initializes an AtomSymbol object.

        Args:
            children (list[lark.Token]): List of child elements.

        """
        self._value = str(children[0])

    def generate_string(self, extension):
        """
        Generates a string representation of the atom symbol.

        Args:
            extension: Extension parameter (unused in this method).

        Returns:
            str: String representation of the atom symbol.

        """
        return self._value

    @property
    def aromatic(self):
        """Returns whether the atom is aromatic."""
        return False

    def generable(self):
        """Returns whether the atom symbol can be generated."""
        return True


class Chiral(BigSMILESbase):
    """
    Represents the chirality of an atom.
    """

    def __init__(self, children: list):
        """
        Initializes a Chiral object.

        Args:
            children (list): List of child elements.

        """
        super().__init__(children)

        self._symbol = str(self._children[0])
        if len(self._children) > 1:
            self._symbol += str(self._children[1])
        if len(self._children) > 2:
            self._symbol += str(self._children[2])

    @property
    def generable(self):
        """Returns whether the chirality can be generated."""
        return True

    def generate_string(self, extension):
        """
        Generates a string representation of the chirality.

        Args:
            extension: Extension parameter (unused in this method).

        Returns:
            str: String representation of the chirality.

        """
        return self._symbol


class HCount(BigSMILESbase):
    """
    Represents the hydrogen count of an atom.
    """

    def __init__(self, children: list):
        """
        Initializes an HCount object.

        Args:
            children (list): List of child elements.

        """
        self._count = None
        super().__init__(children)

        if str(self._children[0]) != "H":
            raise ParsingError(self._children[0])

        if len(self._children) > 1:
            self._count = int(self._children[1])

    @property
    def num(self) -> int:
        """Returns the number of hydrogen atoms."""
        if self._count is None:
            return 1
        return self._count

    @property
    def generable(self):
        """Returns whether the hydrogen count can be generated."""
        return True

    def generate_string(self, extension):
        """
        Generates a string representation of the hydrogen count.

        Args:
            extension: Extension parameter (unused in this method).

        Returns:
            str: String representation of the hydrogen count.

        """
        if self._count:
            return f"H{self.num}"
        return "H"


class AtomCharge(BigSMILESbase):
    """
    Represents the charge of an atom.
    """

    def __init__(self, children):
        """
        Initializes an AtomCharge object.

        Args:
            children: List of child elements.

        """
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
        """Returns whether the atom charge can be generated."""
        return True

    def generate_string(self, extension):
        """
        Generates a string representation of the atom charge.

        Args:
            extension: Extension parameter (unused in this method).

        Returns:
            str: String representation of the atom charge.

        """
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
        """Returns the charge value."""
        return self._sign * self._value


class AtomClass(BigSMILESbase):
    """
    Represents the class of an atom.
    """

    def __init__(self, children):
        """
        Initializes an AtomClass object.

        Args:
            children: List of child elements.

        """
        super().__init__(children)
        self._class_num = int(self._children[1])

    @property
    def generable(self):
        """Returns whether the atom class can be generated."""
        return True

    def generate_string(self, extension):
        """
        Generates a string representation of the atom class.

        Args:
            extension: Extension parameter (unused in this method).

        Returns:
            str: String representation of the atom class.

        """
        return ":" + str(self._class_num)

    @property
    def num(self):
        """Returns the class number."""
        return self._class_num


class AromaticSymbol(AtomSymbol):
    """
    Represents an aromatic atom symbol.
    """

    @property
    def aromatic(self):
        """Returns whether the atom is aromatic."""
        return True


class AliphaticOrganic(AtomSymbol):
    """
    Represents an aliphatic organic atom symbol.
    """

    @property
    def aromatic(self):
        """Returns whether the atom is aromatic."""
        return False


class AromaticOrganic(AtomSymbol):
    """
    Represents an aromatic organic atom symbol.
    """

    @property
    def aromatic(self):
        """Returns whether the atom is aromatic."""
        return True
