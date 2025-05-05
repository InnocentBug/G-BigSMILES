from itertools import product

import networkx as nx

from .core import BigSMILESbase, GenerationBase
from .exception import ParsingError, SmilesHasNonZeroBondDescriptors
from .generating_graph import _PartialGeneratingGraph


class _AbstractIterativeClass(BigSMILESbase):
    @property
    def generable(self):
        gen = True
        for child in self._children:
            gen = gen and child.generable
        return gen

    def _set_stochastic_parent(self, parent):
        for child in self._children:
            try:
                child._set_stochastic_parent(parent)
            except AttributeError:
                pass

    def generate_string(self, extension: bool) -> str:
        string = ""
        for child in self._children:
            string += child.generate_string(extension)
        return string

    @property
    def bond_descriptors(self) -> list:
        bond_descriptors = []
        for child in self._children:
            bond_descriptors += child.bond_descriptors
        return bond_descriptors


class _AbstractIterativeGenerativeClass(_AbstractIterativeClass, GenerationBase):
    def _generate_partial_graph(self) -> _PartialGeneratingGraph:
        partial_graph = _PartialGeneratingGraph()
        if len(self._children) > 0:
            partial_graph = self._children[0]._generate_partial_graph()

            for child in self._children[1:]:
                child_partial_graph = child._generate_partial_graph()
                bonds_to_add = product(partial_graph.right_half_bonds, child_partial_graph.left_half_bonds)
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
                if self._dot_generation is not None:
                    raise ParsingError(self)
                self._dot_generation = child

        self._post_parse_validation()

    def _post_parse_validation(self):
        for child in self._children:
            if len(child.bond_descriptors) != 0:
                raise SmilesHasNonZeroBondDescriptors(child)

    @property
    def mol_molecular_weight(self) -> float | None:
        if self._dot_generation:
            return self._dot_generation.molecular_weight

    def _generate_partial_graph(self) -> _PartialGeneratingGraph:
        partial_graph = super()._generate_partial_graph()
        if self.mol_molecular_weight is not None:
            nx.set_node_attributes(partial_graph.g, values=self.mol_molecular_weight, name="mol_molecular_weight")

        # Remove previous assignments of init weights
        for node_idx in list(partial_graph.g.nodes()):
            try:
                del partial_graph.g.nodes[node_idx]["init_weight"]
            except KeyError:
                pass

        # set weights, if no weight is provided we use 1 as a positive fill
        init_weight = 1
        if self.mol_molecular_weight is not None:
            if len(partial_graph.left_half_bonds) > 0:
                # Divide by length of possible entry points
                init_weight = self.mol_molecular_weight / len(partial_graph.left_half_bonds)

        # Open left half bonds are entry points
        for half_bond in partial_graph.left_half_bonds:
            partial_graph.g.nodes[half_bond.node_id]["init_weight"] = init_weight

        return partial_graph


class BigSmiles(_AbstractIterativeGenerativeClass):

    @property
    def mol_molecular_weight_map(self) -> dict[BigSmilesMolecule, float | None]:
        return {mol: mol.mol_molecular_weight for mol in self._children}

    @property
    def total_molecular_weight(self) -> None | float:
        total_mol_weight: float = 0.0
        for molw in self.mol_molecular_weight_map.values():
            if molw is not None:
                total_mol_weight += molw
        if total_mol_weight > 0:
            return total_mol_weight
        return None

    def _generate_partial_graph(self) -> _PartialGeneratingGraph:
        partial_graph = super()._generate_partial_graph()
        if self.total_molecular_weight is not None:
            for node_idx in list(partial_graph.g.nodes()):
                partial_graph.g.nodes[node_idx]["total_molecular_weight"] = self.total_molecular_weight
        return partial_graph

    @property
    def num_mol_species(self):
        return len(self._children)


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
        if self._dot_system_size is not None:
            return self._dot_system_size.molecular_weight
        return 0.0


class DotSystemSize(BigSMILESbase, GenerationBase):
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

    def _generate_partial_graph(self) -> _PartialGeneratingGraph:
        return _PartialGeneratingGraph()
