# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .core import BigSMILESbase
from .parser import get_global_parser


def _create_compatible_bond_text(bond):
    compatible_symbol = "$"
    if "<" in str(bond):
        compatible_symbol = "<"
    if ">" in str(bond):
        compatible_symbol = ">"
    bond_string = f"{bond.preceding_characters}[{compatible_symbol}{bond.descriptor_id}]"
    return bond_string


# class BondDescriptor(BigSMILESbase):
#     """
#     Bond descriptor of the bigSMILES notation.
#     """

#     def __init__(self, big_smiles_ext, descr_num, preceding_characters, atom_bonding_to):
#         """
#         Construction of a bond descriptor.

#         Arguments:
#         ---------
#         big_smiles_ext: str
#            text representation of a bond descriptor. Example: `[$0]`

#         descr_num: int
#            Position of bond description in the line notation of the stochastic object.
#            Ensure that it starts with `0` and is strictly monotonic increasing during a stochastic object.

#         preceding_characters: str
#            Characters preceding the bond descriptor, that is not an atom.
#            These characters are used to identify the type of bond, any stereochemistry etc.
#            If no further characters are provided, it is assumed to be a single bond without stereochemistry specification.

#         atom_bonding_to:
#            Index of the atom this bond descriptor is bonding to

#         """
#         self._raw_text = big_smiles_ext

#         self.descriptor = ""
#         self.descriptor_id = ""
#         self.descriptor_num = int(descr_num)
#         self.weight = 1.0
#         self.transitions = None
#         self.preceding_characters = preceding_characters
#         self.bond_type = rc.BondType.UNSPECIFIED
#         self.bond_stereo = rc.BondStereo.STEREOANY
#         if self._raw_text == "[]":
#             return

#         if len(preceding_characters) == 0:
#             self.preceding_characters = self._raw_text[: self._raw_text.find("[")]
#             self._raw_text = self._raw_text[self._raw_text.find("[") :]

#         self.atom_bonding_to = atom_bonding_to
#         if self.atom_bonding_to is not None:
#             self.atom_bonding_to = int(self.atom_bonding_to)

#         if self._raw_text[0] != "[" or self._raw_text[-1] != "]":
#             raise RuntimeError(f"Bond descriptor {self._raw_text} does not start and end with []")
#         if self._raw_text[1] not in ("$", "<", ">"):
#             raise RuntimeError(
#                 f"Bond descriptor {self._raw_text} does not have '$<>' as its second character"
#             )
#         self.descriptor = self._raw_text[1]

#         id_end = -1
#         if "|" in self._raw_text:
#             id_end = self._raw_text.find("|")
#         id_str = self._raw_text[2:id_end]
#         if "[" in id_str or "]" in id_str:
#             raise RuntimeError("Nested bond descriptors not supported.")
#         self.descriptor_id = ""
#         if len(id_str) > 0:
#             self.descriptor_id = int(id_str.strip())

#         self.weight = 1.0
#         self.transitions = None
#         if "|" in self._raw_text:
#             if self._raw_text.count("|") != 2:
#                 raise RuntimeError(f"Invalid number of '|' in bond descriptor {self._raw_text}")
#             weight_string = self._raw_text[self._raw_text.find("|") : self._raw_text.rfind("|")]
#             weight_string = weight_string.strip("|")
#             weight_list = [float(w) for w in weight_string.split()]
#             if len(weight_list) == 1:
#                 self.weight = weight_list[0]
#             else:
#                 self.transitions = np.asarray(weight_list)
#                 self.weight = self.transitions.sum()

#         self.preceding_characters = preceding_characters
#         self.bond_type = rc.BondType.SINGLE
#         if "=" in self.preceding_characters:
#             self.bond_type = rc.BondType.DOUBLE
#         if "#" in self.preceding_characters:
#             self.bond_type = rc.BondType.TRIPLE
#         if "$" in self.preceding_characters:
#             self.bond_type = rc.BondType.QUADRUPLE
#         if ":" in self.preceding_characters:
#             self.bond_type = rc.BondType.ONEANDAHALF

#         self.bond_stereo = rc.BondStereo.STEREOANY
#         if (
#             "@" in self.preceding_characters
#             or "/" in self.preceding_characters
#             or "\\" in self.preceding_characters
#         ):
#             raise RuntimeError("Stereochemistry not implemented yet.")

#     def is_compatible(self, other):
#         if self.bond_type != other.bond_type:
#             return False
#         if self.descriptor_id != other.descriptor_id:
#             return False
#         if self.descriptor == "" or other.descriptor == "":
#             return False
#         if self.descriptor == "$" and other.descriptor == "$":
#             return True
#         if self.descriptor == "<" and other.descriptor == ">":
#             return True
#         if self.descriptor == ">" and other.descriptor == "<":
#             return True
#         return False

#     def generate_string(self, extension):
#         string = ""
#         string += f"[{self.descriptor}{self.descriptor_id}"
#         if extension and self.weight != 1.0:
#             string += "|"
#             if self.transitions is None:
#                 string += f"{self.weight}"
#             else:
#                 for t in self.transitions:
#                     string += f"{t} "
#                 string = string[:-1]
#             string += "|"
#         string += "]"
#         return string.strip()

#     @property
#     def generable(self):
#         return self.weight >= 0


class BondSymbol(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)
        self._symbol = str(self._children[0])

    @property
    def generable():
        return True

    def generate_string(self, extension):
        return self._symbol


class RingBond(BondSymbol):
    def __init__(self, children: list):
        super().__init__(children)

        num_text = "".join((str(c) for c in self._children[1:]))
        self._has_dollar = "%" in num_text
        num_text.strip("%")
        self._num = int(num_text)

    def generate_string(self, extension):
        string = ""
        if self._has_dollar:
            string += "%"
        string += str(self._num)
        return super().generate_string(extension) + string

    @property
    def generable():
        return True


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
            raise RuntimeError(
                f"Only BondDescriptorSymbolIdx can be compared for compatibility. But 'other' is of type {type(other)}."
            )

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


class BondDescriptor(BigSMILESbase):
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
            raise RuntimeError(
                f"Only BondDescriptors can be compared for compatibility. But 'other' is of type {type(other)}."
            )

        if self.symbol is None or other.symbol is None:
            return False
        return self.symbol.is_compatible(other.symbol)


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
