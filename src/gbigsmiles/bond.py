# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import uuid

import networkx as nx

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .core import BigSMILESbase, GenerationBase
from .generating_graph import _HalfBond, _PartialGeneratingGraph


def _create_compatible_bond_text(bond):
    compatible_symbol = "$"
    if "<" in str(bond):
        compatible_symbol = "<"
    if ">" in str(bond):
        compatible_symbol = ">"
    bond_string = f"{bond.preceding_characters}[{compatible_symbol}{bond.descriptor_id}]"
    return bond_string


class BondSymbol(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)
        self._symbol = str(self._children[0])

    @property
    def generable(self):
        return True

    def generate_string(self, extension):
        return self._symbol


class RingBond(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)

        self._bond_symbol: None | BondSymbol = None
        self._has_dollar: bool = False

        num_text = ""
        for child in self._children:
            if isinstance(child, BondSymbol):
                self._bond_symbol = child
            elif str(child) == "%":
                self._has_dollar = True
            elif str(child).isdigit():
                num_text += str(child)
        self._num: int = int(num_text)

    def generate_string(self, extension):
        string = ""
        if self._bond_symbol is not None:
            string += self._bond_symbol.generate_string(extension)
        if self._has_dollar:
            string += "%"
        string += str(self._num)
        return string

    @property
    def generable():
        return True

    @property
    def idx(self) -> int:
        return self._num


class BondDescriptorSymbol(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)

    def generate_string(self, extension):
        return str(self._children[0])

    def generable(self):
        return True


class BondDescriptorSymbolIdx(BondDescriptorSymbol):
    def __init__(self, children):
        super().__init__(children)
        self._idx = 0
        if len(self._children) > 1:
            self._idx = int(self._children[1])

    @property
    def idx(self):
        return self._idx

    def generate_string(self, extension):
        string = super().generate_string(extension)
        if self.idx != 0:
            string += str(self.idx)
        return string

    def generable(self):
        return True

    def is_compatible(self, other):
        if other is None:
            return False
        if not isinstance(other, BondDescriptorSymbolIdx):
            raise RuntimeError(f"Only BondDescriptorSymbolIdx can be compared for compatibility. But 'other' is of type {type(other)}.")

        if self.idx != other.idx:
            return False

        self_str = str(self._children[0])
        other_str = str(other._children[0])

        if self_str == "$" and other_str == "$":
            return True

        if self_str in ("<", ">") and other_str in ("<", ">") and self_str != other_str:
            return True

        return False


class BondDescriptorGeneration(BigSMILESbase):
    def __init__(self, children):
        super().__init__(children)
        self._transition = None
        self._weight = 1.0

        if len(self._children) > 0:
            # Strip out the "|"
            parse = self._children[1:-1]
            self._weight = float(parse[0])
            if len(parse) > 1:
                self._transition = [float(number) for number in parse]
                self._weight = sum(self.transition)

    @property
    def transition(self):
        return self._transition

    @property
    def weight(self):
        return self._weight

    def generate_string(self, extension):
        if extension and (self.weight != 1.0 or self.transition is not None):
            string = "|"
            if self.transition:
                for trans in self.transition:
                    string += str(trans) + " "
            else:
                string += str(self.weight)
            string = string.strip() + "|"
            return string
        return ""

    def generable(self):
        return True


class InnerBondDescriptor(BigSMILESbase):
    def __init__(self, children):
        super().__init__(children)

        self._generation = BondDescriptorGeneration([])
        for child in self._children:
            if isinstance(child, BondDescriptorSymbolIdx):
                self._symbol = child
            if isinstance(child, BondDescriptorGeneration):
                self._generation = child

    def generate_string(self, extension):
        string = self._symbol.generate_string(extension)
        string += self._generation.generate_string(extension)
        return string

    def generable(self):
        return True

    @property
    def idx(self):
        return self._symbol.idx

    @property
    def weight(self):
        return self._generation.weight

    @property
    def transition(self):
        return self._generation.transition


class BondDescriptor(BigSMILESbase, GenerationBase):
    @classmethod
    def make(cls, text: str) -> Self:
        if "$" in text or "<" in text or ">" in text:
            return SimpleBondDescriptor.make(text)
        return TerminalBondDescriptor.make(text)

    @property
    def symbol(self):
        return None

    def is_compatible(self, other):
        if other is None:
            return False
        if not isinstance(other, BondDescriptor):
            raise RuntimeError(f"Only BondDescriptors can be compared for compatibility. But 'other' is of type {type(other)}.")

        if self.symbol is None or other.symbol is None:
            return False
        return self.symbol.is_compatible(other.symbol)

    @property
    def bond_descriptors(self):
        return [self]

    def _generate_partial_graph(self):
        g = nx.MultiDiGraph()
        node_idx = str(uuid.uuid4())
        g.add_node(node_idx, smi_text=str(self), obj=self)
        partial_graph = _PartialGeneratingGraph(g)
        partial_graph.left_half_bonds.append(_HalfBond(self, node_idx, {}))
        partial_graph.right_half_bonds.append(_HalfBond(self, node_idx, {}))

        return partial_graph


class SimpleBondDescriptor(BondDescriptor):
    def __init__(self, children):
        super().__init__(children)

        for child in self._children:
            if isinstance(child, InnerBondDescriptor):
                self._inner_bond_descriptor = child

    @classmethod
    def make(cls, text: str) -> Self:
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    def generate_string(self, extension):
        return "[" + self._inner_bond_descriptor.generate_string(extension) + "]"

    def generable(self):
        return self._inner_bond_descriptor.generable

    @property
    def idx(self):
        return self._inner_bond_descriptor.idx

    @property
    def weight(self):
        return self._inner_bond_descriptor.weight

    @property
    def transition(self):
        return self._inner_bond_descriptor.transition

    @property
    def symbol(self):
        return self._inner_bond_descriptor._symbol


class TerminalBondDescriptor(BondDescriptor):
    def __init__(self, children):
        super().__init__(children)
        self._symbol = None
        self._generation = BondDescriptorGeneration([])

        for child in self._children:
            if isinstance(child, BondDescriptorSymbolIdx):
                self._symbol = child
            if isinstance(child, BondDescriptorGeneration):
                self._generation = child

    @classmethod
    def make(cls, text: str) -> Self:
        # We use BigSMILESbase.make.__func__ to get the underlying function of the class method,
        # then call it with cls as the first argument to ensure child typing.
        # We do not want to call StochasticDistribution's make function, because it directs here.
        return BigSMILESbase.make.__func__(cls, text)

    @property
    def weight(self):
        return self._generation.weight

    @property
    def transition(self):
        return self._generation.transition

    def generate_string(self, extension):
        string = "["
        if self._symbol is not None:
            string += self._symbol.generate_string(extension)
        string += self._generation.generate_string(extension)
        string += "]"
        return string

    def generable(self):
        return True

    @property
    def symbol(self):
        return self._symbol
