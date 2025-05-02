# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

"""
This module defines classes representing atoms and their properties
as parsed from a BigSMILES string. It includes handling for regular atoms,
bracketed atoms with isotopes, chirality, hydrogen counts, charges, and atom classes.
The module also provides functionality to generate string representations
of these atom components and to contribute to the generation of a molecular graph.
"""

import uuid
from typing import TYPE_CHECKING, List, Optional, Union

import networkx as nx

from .core import BigSMILESbase, GenerationBase
from .exception import ParsingError, TooManyTokens
from .generating_graph import _HalfBond, _PartialGeneratingGraph

if TYPE_CHECKING:
    from lark import Token


class Atom(BigSMILESbase, GenerationBase):
    """
    Represents an atom in a chemical structure.
    """

    _symbol: Optional["AtomSymbol"] = None

    @property
    def symbol(self) -> Optional["AtomSymbol"]:
        """Returns the atom symbol."""
        return self._symbol

    def __init__(self, children: List[Union["AtomSymbol", "Isotope", "Chiral", "HCount", "AtomCharge", "AtomClass"]]):
        """
        Initializes an Atom object.

        Args:
            children (List[Union[AtomSymbol, Isotope, Chiral, HCount, AtomCharge, AtomClass]]):
                List of child elements parsed by Lark.
        """
        super().__init__(children)

        for child in self._children:
            if isinstance(child, AtomSymbol):
                if self._symbol is not None:
                    raise TooManyTokens(self.__class__, self._symbol, child)
                self._symbol = child

    def generate_string(self, extension: bool) -> str:
        """
        Generates a string representation of the atom.

        Args:
            extension (bool): Extension parameter (unused in this method).

        Returns:
            str: String representation of the atom.
        """
        return str(self.symbol)

    def _generate_partial_graph(self) -> _PartialGeneratingGraph:
        """
        Generates a partial NetworkX graph representing this atom.

        Returns:
            _PartialGeneratingGraph: A partial graph with the atom as a node
                                     and initialized left/right half-bonds.
        """
        g: nx.MultiDiGraph = nx.MultiDiGraph()
        node_id: str = str(uuid.uuid4())
        g.add_node(node_id, smi_text=str(self), obj=self)
        partial_graph: _PartialGeneratingGraph = _PartialGeneratingGraph(g)
        partial_graph.left_half_bonds.append(_HalfBond(self, node_id, {}))
        partial_graph.right_half_bonds.append(_HalfBond(self, node_id, {}))

        return partial_graph

    @property
    def aromatic(self) -> bool:
        """Returns whether the atom is aromatic."""
        return self.symbol.aromatic if self.symbol else False

    @property
    def generable(self) -> bool:
        """Returns whether the atom can be generated."""
        return True

    @property
    def charge(self) -> int:
        """Returns the charge of the atom (default is 0)."""
        return 0


class BracketAtom(Atom):
    """
    Represents a bracketed atom in a chemical structure, allowing for
    specification of isotopes, chirality, hydrogen count, charge, and class.
    """

    _isotope: Optional["Isotope"] = None
    _chiral: Optional["Chiral"] = None
    _h_count: Optional["HCount"] = None
    _atom_charge: Optional["AtomCharge"] = None
    _atom_class: Optional["AtomClass"] = None

    @property
    def isotope(self) -> Optional["Isotope"]:
        """Returns the isotope of the atom, if specified."""
        return self._isotope

    @property
    def chiral(self) -> Optional["Chiral"]:
        """Returns the chirality of the atom, if specified."""
        return self._chiral

    @property
    def h_count(self) -> Optional["HCount"]:
        """Returns the hydrogen count of the atom, if specified."""
        return self._h_count

    @property
    def atom_charge(self) -> Optional["AtomCharge"]:
        """Returns the charge of the atom, if specified."""
        return self._atom_charge

    @property
    def atom_class(self) -> Optional["AtomClass"]:
        """Returns the class of the atom, if specified."""
        return self._atom_class

    def __init__(self, children: List[Union["AtomSymbol", "Isotope", "Chiral", "HCount", "AtomCharge", "AtomClass"]]):
        """
        Initializes a BracketAtom object.

        Args:
            children (List[Union[AtomSymbol, Isotope, Chiral, HCount, AtomCharge, AtomClass]]):
                List of child elements parsed by Lark.
        """
        super().__init__(children)
        for child in self._children:
            if isinstance(child, AtomSymbol):
                pass  # Symbol is handled by the parent class
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

    def generate_string(self, extension: bool) -> str:
        """
        Generates a string representation of the bracketed atom.

        Args:
            extension (bool): Extension parameter (unused in this method).

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
    def charge(self) -> int:
        """Returns the charge of the atom, considering the AtomCharge if present."""
        if self.atom_charge:
            return self.atom_charge.charge
        return super().charge


class Isotope(BigSMILESbase):
    """
    Represents an isotope specification for an atom within brackets.
    """

    _value: int

    def __init__(self, children: List["Token"]):
        """
        Initializes an Isotope object.

        Args:
            children (List[Token]): List containing the Lark token for the isotope number.
        """
        super().__init__(children)
        self._value = int(self._children[0])

    def generate_string(self, extension: bool) -> str:
        """
        Generates a string representation of the isotope number.

        Args:
            extension (bool): Extension parameter (unused in this method).

        Returns:
            str: String representation of the isotope number.
        """
        return str(self._value)

    def generable(self) -> bool:
        """Returns whether the isotope specification can be generated."""
        return True

    @property
    def num_nuclei(self) -> int:
        """Returns the integer value of the isotope number."""
        return self._value


class AtomSymbol(BigSMILESbase):
    """
    Represents the symbol of an atom (e.g., 'C', 'O', 'n', 's').
    """

    _value: str

    def __init__(self, children: List["Token"]):
        """
        Initializes an AtomSymbol object.

        Args:
            children (List[Token]): List containing the Lark token for the atom symbol.
        """
        self._value = str(children[0])

    def generate_string(self, extension: bool) -> str:
        """
        Generates a string representation of the atom symbol.

        Args:
            extension (bool): Extension parameter (unused in this method).

        Returns:
            str: String representation of the atom symbol.
        """
        return self._value

    @property
    def aromatic(self) -> bool:
        """Returns whether the atom symbol represents an aromatic atom (default is False)."""
        return False

    def generable(self) -> bool:
        """Returns whether the atom symbol can be generated."""
        return True


class Chiral(BigSMILESbase):
    """
    Represents the chirality specification of an atom within brackets
    (e.g., '@', '@@', '@TH1', etc.).
    """

    _symbol: str

    def __init__(self, children: List["Token"]):
        """
        Initializes a Chiral object.

        Args:
            children (List[Token]): List of Lark tokens representing the chirality symbol.
        """
        super().__init__(children)

        self._symbol = "".join(str(child) for child in self._children)

    @property
    def generable(self) -> bool:
        """Returns whether the chirality specification can be generated."""
        return True

    def generate_string(self, extension: bool) -> str:
        """
        Generates a string representation of the chirality symbol.

        Args:
            extension (bool): Extension parameter (unused in this method).

        Returns:
            str: String representation of the chirality symbol.
        """
        return self._symbol


class HCount(BigSMILESbase):
    """
    Represents the hydrogen count specification for an atom within brackets
    (e.g., 'H', 'H2').
    """

    _count: Optional[int] = None

    def __init__(self, children: List["Token"]):
        """
        Initializes an HCount object.

        Args:
            children (List[Token]): List of Lark tokens representing the hydrogen count.
        """
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
    def generable(self) -> bool:
        """Returns whether the hydrogen count specification can be generated."""
        return True

    def generate_string(self, extension: bool) -> str:
        """
        Generates a string representation of the hydrogen count.

        Args:
            extension (bool): Extension parameter (unused in this method).

        Returns:
            str: String representation of the hydrogen count.
        """
        if self._count:
            return f"H{self.num}"
        return "H"


class AtomCharge(BigSMILESbase):
    """
    Represents the charge specification for an atom within brackets
    (e.g., '+', '++', '-', '--', '+2', '-3').
    """

    _sign: int
    _value: int
    _number: bool

    def __init__(self, children: List["Token"]):
        """
        Initializes an AtomCharge object.

        Args:
            children (List[Token]): List of Lark tokens representing the charge.
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
    def generable(self) -> bool:
        """Returns whether the atom charge specification can be generated."""
        return True

    def generate_string(self, extension: bool) -> str:
        """
        Generates a string representation of the atom charge.

        Args:
            extension (bool): Extension parameter (unused in this method).

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
    def charge(self) -> int:
        """Returns the integer value of the charge."""
        return self._sign * self._value


class AtomClass(BigSMILESbase):
    """
    Represents the atom class specification for an atom within brackets
    (e.g., ':1', ':10').
    """

    _class_num: int

    def __init__(self, children: List["Token"]):
        """
        Initializes an AtomClass object.

        Args:
            children (List[Token]): List of Lark tokens representing the atom class.
        """
        super().__init__(children)
        self._class_num = int(self._children[1])

    @property
    def generable(self) -> bool:
        """Returns whether the atom class specification can be generated."""
        return True

    def generate_string(self, extension: bool) -> str:
        """
        Generates a string representation of the atom class.

        Args:
            extension (bool): Extension parameter (unused in this method).

        Returns:
            str: String representation of the atom class.
        """
        return ":" + str(self._class_num)

    @property
    def num(self) -> int:
        """Returns the integer value of the atom class number."""
        return self._class_num


class AromaticSymbol(AtomSymbol):
    """
    Represents an aromatic atom symbol (lowercase, e.g., 'c', 'o', 'n').
    """

    @property
    def aromatic(self) -> bool:
        """Returns whether the atom is aromatic (True for AromaticSymbol)."""
        return True


class AliphaticOrganic(AtomSymbol):
    """
    Represents an aliphatic organic atom symbol (uppercase, e.g., 'C', 'O', 'N').
    """

    @property
    def aromatic(self) -> bool:
        """Returns whether the atom is aromatic (False for AliphaticOrganic)."""
        return False


class AromaticOrganic(AtomSymbol):
    """
    Represents an aromatic organic atom symbol (lowercase, e.g., 'c', 'o', 'n').
    This class is redundant as AromaticSymbol already covers this.
    It is kept for potential distinction in parsing or future extensions.
    """

    @property
    def aromatic(self) -> bool:
        """Returns whether the atom is aromatic (True for AromaticOrganic)."""
        return True
