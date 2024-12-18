from .bond import BondSymbol, RingBond
from .core import BigSMILESbase


class Branch(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)

        self._bond_symbol: None | BondSymbol = None
        self._elements: list = []

        for child in self._children:
            if self._bond_symbol is None and isinstance(child, BondSymbol):
                self._bond_symbol = child
            elif isinstance(child, BigSMILESbase):
                self._elements.append(child)

    @property
    def bond_symbol(self):
        return self._bond_symbol

    @property
    def generable(self):
        gen = True
        for element in self._elements:
            gen = gen and element.generable
        if self.bond_symbol is not None:
            gen = gen and self.bond_symbol.generable
        return gen

    def generate_string(self, extension: bool) -> str:
        string = "("
        if self.bond_symbol is not None:
            string += self.bond_symbol.generate_string(extension)
        for element in self._elements:
            string += element.generate_string(extension)
        return string + ")"


class BranchedAtom(BigSMILESbase):
    def __init__(self, children):
        super().__init__(children)

        self._atom_stand_in: BigSMILESbase | None = None
        self._branches: list[Branch] = []
        self._ring_bonds: list[RingBond] = []

        for child in self._children:
            if self._atom_stand_in is None and isinstance(child, BigSMILESbase):
                self._atom_stand_in = child
            if isinstance(child, RingBond):
                self._ring_bonds.append(child)
            if isinstance(child, Branch):
                self._branches.append(child)

    @property
    def generable(self):
        gen = self._atom_stand_in.generable
        for ring_bond in self._ring_bonds:
            gen = gen and ring_bond.generable
        for branch in self._branches:
            gen = gen and branch.generable

        return gen

    def generate_string(self, extension: bool) -> str:
        string = self._atom_stand_in.generate_string(extension)
        for ring_bond in self._ring_bonds:
            string += ring_bond.generate_string(extension)
        for branch in self._branches:
            string += branch.generate_string(extension)
        return string


class AtomAssembly(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)
        self._symbol: None | BondSymbol = None
        self._branched_atom: None | BranchedAtom = None

        for child in self._children:
            if isinstance(child, BondSymbol):
                self._symbol = child
            elif isinstance(child, BranchedAtom):
                self._branched_atom = child

    @property
    def bond_symbol(self) -> None | BondSymbol:
        return self._symbol

    @property
    def generable(self) -> bool:
        gen = self._branched_atom.generable
        if self.bond_symbol is not None:
            gen = gen and self.bond_symbol.generable
        return gen

    def generate_string(self, extension: bool) -> str:
        string = ""
        if self.bond_symbol is not None:
            string += self.bond_symbol.generate_string(extension)
        string += self._branched_atom.generate_string(extension)
        return string


class Dot(BigSMILESbase):
    @property
    def generable(self):
        return True

    def generate_string(self, extension: bool) -> str:
        return "."


class Smiles(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)

        self._elements = self._children

    @property
    def generable(self):
        gen = True
        for element in self._elements:
            gen = gen and element.generable
        return gen

    def generate_string(self, extension: bool) -> str:
        string: str = ""
        for element in self._elements:
            string += element.generate_string(extension)
        return string
