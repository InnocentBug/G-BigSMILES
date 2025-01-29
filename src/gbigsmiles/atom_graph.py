import warnings
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from .exception import InvalidGenerationSource, UnvalidatedGenerationSource
from .generating_graph import (
    _AROMATIC_NAME,
    _BOND_TYPE_NAME,
    _NON_STATIC_ATTR,
    _STATIC_NAME,
)
from .util import get_global_rng


@dataclass(frozen=True)
class _HalfAtomBond:
    atom_idx: int
    gen_bond_attr: str
    target_gen_idx: str
    bond_type: str


class _PartialAtomGraph:
    _ATOM_ATTRS = {"atomic_num", _AROMATIC_NAME, "charge"}
    _BOND_ATTRS = {_BOND_TYPE_NAME, _AROMATIC_NAME}

    def __init__(self, generating_graph, static_graph, source_node):
        self._atom_id = 0
        self.generating_graph = generating_graph
        self.static_graph = static_graph
        self.atom_graph = nx.Graph()

        self.add_static_sub_graph(source_node)

    def add_static_sub_graph(self, source):
        atom_key_to_gen_key = {}
        gen_key_to_atom_key = {}

        def add_node(node_idx):
            transistion_edges = []
            termination_edges = []
            stochastic_edges = []

            data = self.gen_node_attr_to_atom_attr(self.generating_graph.nodes[source])
            self.atom_graph.add_node(self._atom_id, **data)
            atom_key_to_gen_key[self._atom_id] = source
            gen_key_to_atom_key[source] = self._atom_id

            for u, v, d in self.generating_graph.out_edges(node_idx, data=True):
                if not d["static"]:
                    for k in _NON_STATIC_ATTR:
                        if d[k] > 0:
                            stochastic_edges.append(
                                _HalfAtomBond(
                                    self._atom_id,
                                    self.gen_edge_attr_to_bond_attr(d),
                                    v,
                                    _STOCHASTIC_NAME,
                                )
                            )

            self._atom_id += 1

        edges_data_map = {}

        for u, v, k in nx.edge_dfs(self.static_graph, source=source):
            if (u, v) not in edges_data_map and (v, u) not in edges_data_map:

                edges_data_map[(u, v)] = self.gen_edge_attr_to_bond_attr(
                    self.static_graph.get_edge_data(u, v, k)
                )
        print(edges_data_map)

    def gen_node_attr_to_atom_attr(
        self, attr: dict[str, bool | float | int], keys_to_copy: None | set[str] = None
    ) -> dict[str, bool | float | int]:
        if keys_to_copy is None:
            keys_to_copy = self._ATOM_ATTRS
        return self._copy_some_dict_attr(attr, keys_to_copy)

    def gen_edge_attr_to_bond_attr(
        self, attr: dict[str, bool | int], keys_to_copy: None | set[str] = None
    ) -> dict[str, bool | int]:
        if keys_to_copy is None:
            keys_to_copy = self._BOND_ATTRS
        return self._copy_some_dict_attr(attr, keys_to_copy)

    @staticmethod
    def _copy_some_dict_attr(dictionary: dict[str, Any], keys_to_copy: set[str]) -> dict[str, Any]:
        new_dict = {}
        for k in keys_to_copy:
            new_dict[k] = dictionary[k]
        return new_dict


class AtomGraph:
    def __init__(self, ml_graph):
        self._ml_graph = ml_graph.copy()

        self._static_graph = self._create_static_graph(self.ml_graph)

        self._starting_node_idx, self._starting_node_weight = self._create_init_weights(
            self.ml_graph
        )

    @staticmethod
    def _create_init_weights(graph):
        starting_node_idx = []
        starting_node_weight = []
        for node_idx, data in graph.nodes(data=True):
            if data["init_weight"] > 0:
                starting_node_idx.append(node_idx)
                starting_node_weight.append(data["init_weight"])

        starting_node_weight = np.asarray(starting_node_weight)
        starting_node_weight /= np.sum(starting_node_weight)

        return starting_node_idx, starting_node_weight

    @staticmethod
    def _create_static_graph(ml_graph):
        static_graph = ml_graph.copy()

        edges_to_delete = set()
        for u, v, k, d in static_graph.edges(keys=True, data=True):
            if not d["static"]:
                edges_to_delete.add((u, v, k))

        static_graph.remove_edges_from(edges_to_delete)
        return static_graph

    @property
    def ml_graph(self):
        return self._ml_graph.copy()

    def _get_random_start_node(self, rng):
        return rng.choice(self._starting_node_idx, p=self._starting_node_weight)

    def sample_graph(self, source: str = None, rng=None):

        if rng is None:
            rng = get_global_rng()

        if source is None:
            source = self._get_random_start_node(rng)

        if source not in self.ml_graph.nodes():
            raise InvalidGenerationSource(source, self.ml_graph.nodes(), self.ml_graph)

        if source not in self._starting_node_idx:
            warnings.warn(
                UnvalidatedGenerationSource(source, self._starting_node_idx, self.ml_graph)
            )

        atom_graph = _PartialAtomGraph(self.ml_graph, self._static_graph, source)
