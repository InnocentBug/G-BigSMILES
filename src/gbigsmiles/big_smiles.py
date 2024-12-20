from itertools import product

import networkx as nx

from .core import BigSMILESbase, GenerationBase
from .generating_graph import _PartialGeneratingGraph


class _AbstractIterativeClass(BigSMILESbase):
    @property
    def generable(self):
        gen = True
        for child in self._children:
            gen = gen and child.generable
        return gen

    def generate_string(self, extension: bool) -> str:
        string = ""
        for child in self._children:
            string += child.generate_string(extension)
        return string


class _AbstractIterativeGenerativeClass(_AbstractIterativeClass, GenerationBase):
    def _generate_partial_graph(self) -> _PartialGeneratingGraph:
        partial_graph = _PartialGeneratingGraph()
        if len(self._children) > 0:
            partial_graph = self._children[0]._generate_partial_graph()

            for child in self._children[1:]:
                child_partial_graph = child._generate_partial_graph()
                bonds_to_add = product(
                    partial_graph.right_half_bonds, child_partial_graph.left_half_bonds
                )
                # Transfer the child right bond to the partial graph, and reset partial graph
                partial_graph.right_half_bonds = child_partial_graph.right_half_bonds
                child_partial_graph.right_half_bonds = []
                child_partial_graph.left_half_bonds = []

                partial_graph.merge(child_partial_graph, bonds_to_add)
        return partial_graph


class BigSmilesMolecule(_AbstractIterativeGenerativeClass):
    def __init__(self, children: list):
        super().__init__(children)

        self._dot_generation: None | DotGeneration = None
        for child in self._children:
            if isinstance(child, DotGeneration):
                assert self._dot_generation is None
                self._dot_generation = child

    @property
    def system_molecular_weight(self) -> float | None:
        if self._dot_generation:
            return self._dot_generation.molecular_weight

    def _generate_partial_graph(self) -> _PartialGeneratingGraph:
        partial_graph = super()._generate_partial_graph()
        if self.system_molecular_weight:
            nx.set_node_attributes(
                partial_graph.g, values=self.system_molecular_weight, name="system_molecular_weight"
            )
        return partial_graph


class BigSmiles(_AbstractIterativeGenerativeClass):

    @property
    def mol_molecular_weight_map(self) -> dict[BigSmilesMolecule, float | None]:
        return {mol: mol.system_molecular_weight for mol in self._children}

    @property
    def total_system_molecular_weight(self) -> None | float:
        total_mol_weight: float = 0.0
        for molw in self.mol_molecular_weight_map.values():
            if molw is None:
                return None
            total_mol_weight += molw
        return total_mol_weight


class DotGeneration(_AbstractIterativeGenerativeClass):
    def __init__(self, children):
        from .smiles import Dot

        super().__init__(children)

        self._dot: None | Dot = None
        self._dot_system_size: None | DotSystemSize = None
        for child in self._children:
            if isinstance(child, Dot):
                self._dot = child
            if isinstance(child, DotSystemSize):
                self._dot_system_size = child

    @property
    def molecular_weight(self):
        return self._dot_system_size.molecular_weight


class DotSystemSize(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)

        self._molecular_weight: float = -1.0
        for child in self._children:
            if isinstance(child, float):
                if self._molecular_weight >= 0:
                    raise ValueError("Internal Error, please report on github.com")
                self._molecular_weight = child

    @property
    def molecular_weight(self) -> float:
        return self._molecular_weight

    def generate_string(self, extension: bool) -> str:
        string = ""
        if extension:
            string += f"|{self.molecular_weight}|"
        return string
