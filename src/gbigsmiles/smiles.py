from itertools import product

from .big_smiles import _AbstractIterativeGenerativeClass
from .bond import BondSymbol, RingBond
from .core import BigSMILESbase, GenerationBase
from .exception import DoubleBondSymbolDefinition
from .generating_graph import _BOND_TYPE_NAME, _PartialGeneratingGraph


class Branch(BigSMILESbase, GenerationBase):
    def __init__(self, children: list):
        super().__init__(children)

        self._bond_symbol: None | BondSymbol = None
        self._elements: list = []

        for child in self._children:
            if self._bond_symbol is None and isinstance(child, BondSymbol):
                self._bond_symbol = child
            elif isinstance(child, BigSMILESbase):
                self._elements.append(child)

    def _set_stochastic_parent(self, parent):
        for child in self._children:
            try:
                child._set_stochastic_parent(parent)
            except AttributeError:
                pass

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

    def _generate_partial_graph(self) -> _PartialGeneratingGraph:
        partial_graph = self._elements[0]._generate_partial_graph()
        if self._bond_symbol is not None:
            for lhb in partial_graph.left_half_bonds:
                if _BOND_TYPE_NAME in lhb.bond_attributes:
                    raise DoubleBondSymbolDefinition(partial_graph, self._bond_symbol, lhb.bond_attributes)
                lhb.bond_attributes[_BOND_TYPE_NAME] = self._bond_symbol

        for element in self._elements[1:]:
            element_partial_graph = element._generate_partial_graph()
            bonds_to_add = product(partial_graph.right_half_bonds, element_partial_graph.left_half_bonds)
            # Transfer right_half_bonds to new partial graph
            partial_graph.right_half_bonds = element_partial_graph.right_half_bonds
            element_partial_graph.right_half_bonds = []
            element_partial_graph.left_half_bonds = []

            partial_graph.merge(element_partial_graph, bonds_to_add)

        # A branch cannot connect to anything on the right
        partial_graph.right_half_bonds = []
        return partial_graph

    @property
    def bond_descriptors(self):
        bond_descriptors = []
        for element in self._elements:
            bond_descriptors += element.bond_descriptors
        return bond_descriptors


class BranchedAtom(BigSMILESbase, GenerationBase):
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

    def _set_stochastic_parent(self, parent):
        for child in self._children:
            try:
                child._set_stochastic_parent(parent)
            except AttributeError:
                pass

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

    def _generate_partial_graph(self) -> _PartialGeneratingGraph:
        partial_graph = self._atom_stand_in._generate_partial_graph()
        # Adding ring bonds
        for ring_idx, half_bond in product(self._ring_bonds, partial_graph.right_half_bonds):
            partial_graph.add_ring_bond(ring_idx, half_bond)

        # Adding branches
        for branch in self._branches:
            branch_partial_graph = branch._generate_partial_graph()
            bonds_to_add = product(partial_graph.right_half_bonds, branch_partial_graph.left_half_bonds)
            # Branches have empty right hand half bonds, so only resetting left ones.
            branch_partial_graph.left_half_bonds = []

            partial_graph.merge(branch_partial_graph, bonds_to_add)

        # Not resetting right bonds, because this can bond to more on the right (not a branch)
        return partial_graph

    @property
    def bond_descriptors(self):
        bond_descriptors = []

        if self._atom_stand_in:
            bond_descriptors += self._atom_stand_in.bond_descriptors
        for branch in self._branches:
            bond_descriptors += branch.bond_descriptors

        return bond_descriptors


class AtomAssembly(BigSMILESbase, GenerationBase):
    def __init__(self, children: list):
        super().__init__(children)
        self._symbol: None | BondSymbol = None
        self._branched_atom: None | BranchedAtom = None

        for child in self._children:
            if isinstance(child, BondSymbol):
                self._symbol = child
            elif isinstance(child, BranchedAtom):
                self._branched_atom = child

    def _set_stochastic_parent(self, parent):
        self._branched_atom._set_stochastic_parent(parent)

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

    def _generate_partial_graph(self) -> _PartialGeneratingGraph:
        partial_graph = self._branched_atom._generate_partial_graph()
        if self.bond_symbol:
            for half_bond in partial_graph.left_half_bonds:
                if _BOND_TYPE_NAME in half_bond.bond_attributes:
                    raise DoubleBondSymbolDefinition(partial_graph, self.bond_symbol, half_bond.bond_attributes)
                half_bond.bond_attributes[_BOND_TYPE_NAME] = self.bond_symbol

        return partial_graph

    @property
    def bond_descriptors(self) -> list:
        return self._branched_atom.bond_descriptors


class Dot(BigSMILESbase, GenerationBase):
    @property
    def generable(self):
        return True

    def generate_string(self, extension: bool) -> str:
        return "."

    def _generate_partial_graph(self) -> _PartialGeneratingGraph:
        return _PartialGeneratingGraph()


class Smiles(_AbstractIterativeGenerativeClass):
    pass
