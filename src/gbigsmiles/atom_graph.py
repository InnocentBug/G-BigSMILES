import copy
import json
import warnings
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from .chem_resource import atomic_masses
from .distribution import StochasticDistribution
from .exception import InvalidGenerationSource, UnvalidatedGenerationSource
from .generating_graph import (
    _AROMATIC_NAME,
    _BOND_TYPE_NAME,
    _NON_STATIC_ATTR,
    _STATIC_NAME,
)
from .util import get_global_rng


class _HalfAtomBond:
    def __init__(self, atom_idx: int, node_idx: str, graph, bond_attr_converter):
        self.atom_idx: int = atom_idx
        self.node_idx: str = node_idx
        self.weight: float = graph.nodes[node_idx]["gen_weight"]

        self._mode_attr_map = {}
        self._mode_target_map = {}
        self._mode_weight_map = {}

        for u, v, d in graph.out_edges(node_idx, data=True):
            if not d["static"]:
                for k in _NON_STATIC_ATTR:
                    if d[k] > 0:
                        try:
                            self._mode_attr_map[k] += [bond_attr_converter(d)]
                        except KeyError:
                            self._mode_attr_map[k] = [bond_attr_converter(d)]
                        try:
                            self._mode_target_map[k] += [v]
                        except KeyError:
                            self._mode_target_map[k] = [v]
                        try:
                            self._mode_weight_map[k] += [d[k]]
                        except KeyError:
                            self._mode_weight_map[k] = [d[k]]

    def has_any_bonds(self):
        has_bonds = False
        for key in self._mode_attr_map:
            if len(self._mode_attr_map[key]) > 0:
                has_bonds = True
        return has_bonds

    def has_mode_bonds(self, mode):
        if mode not in self._mode_attr_map:
            return False
        return len(self._mode_attr_map[mode]) > 0

    def get_mode_bonds(self, mode):
        try:
            return self._mode_attr_map[mode], self._mode_target_map[mode]
        except KeyError:
            return [], []

    def __str__(self):
        return f"HalfAtomBond({self.atom_idx}, {self.node_idx}, {self.weight}, {self._mode_attr_map}, {self._mode_target_map}, {self._mode_weight_map})"


class _PartialAtomGraph:
    _ATOM_ATTRS = {"atomic_num", _AROMATIC_NAME, "charge"}
    _BOND_ATTRS = {_BOND_TYPE_NAME, _AROMATIC_NAME}

    def __init__(self, generating_graph, static_graph, source_node):
        self._atom_id = 0
        self.generating_graph = generating_graph
        self.static_graph = static_graph
        self._stochastic_vector: list[float] = generating_graph.nodes[source_node][
            "stochastic_generation"
        ]
        self._stochastic_generation: None | StochasticDistribution = (
            StochasticDistribution.from_serial_vector(self._stochastic_vector)
        )

        self.atom_graph = nx.Graph()
        self._open_half_bonds = []
        self._mol_weight: float = 0.0

        self.add_static_sub_graph(source_node)

    def merge(self, other, self_idx, other_idx, bond_attr):
        # relabel other idx
        remapping_dict = {idx: idx + self._atom_id for idx in other.atom_graph.nodes}
        other_graph = nx.relabel_nodes(other.atom_graph, remapping_dict, copy=True)
        other_open_half_bonds = []
        for half_bond in other._open_half_bonds:
            new_half_bond = copy.copy(half_bond)
            new_half_bond.atom_idx += self.atom_id
            other_open_half_bonds += [new_half_bond]
        other_idx += self._atom_id

        # Now we can do the actual merging
        self._atom_id += other._atom_id
        self._stochastic_vector = other.stochastic_vector
        self._stochastic_generation = copy.copy(other._stochastic_generation)

        self.atom_graph = nx.union(self.atom_graph, other_graph)
        self.atom_graph.add_edge(self_idx, other_idx, **bond_attr)
        self._open_half_bonds += other_open_half_bonds
        self._mol_weight += other._mol_weight

    def add_static_sub_graph(self, source):
        atom_key_to_gen_key = {}
        gen_key_to_atom_key = {}

        def add_node(node_idx):

            data = self.gen_node_attr_to_atom_attr(self.generating_graph.nodes[node_idx])
            self.atom_graph.add_node(self._atom_id, **(data | {"origin_idx": node_idx}))
            atom_key_to_gen_key[self._atom_id] = node_idx
            gen_key_to_atom_key[node_idx] = self._atom_id
            half_bond = _HalfAtomBond(
                self._atom_id, node_idx, self.generating_graph, self.gen_edge_attr_to_bond_attr
            )

            self._mol_weight += atomic_masses[data["atomic_num"]]
            self._atom_id += 1

            return half_bond

        half_bond = add_node(source)
        if half_bond.weight > 0 and half_bond.has_any_bonds():
            self._open_half_bonds += [half_bond]

        edges_data_map = {}

        for u, v, k in nx.edge_dfs(self.static_graph, source=source):
            for gen_atom_idx in (u, v):
                if gen_atom_idx not in gen_key_to_atom_key:
                    half_bond = add_node(gen_atom_idx)
                    if half_bond.weight > 0 and half_bond.has_any_bonds():
                        self._open_half_bonds += [half_bond]

            u_atom_idx = gen_key_to_atom_key[u]
            v_atom_idx = gen_key_to_atom_key[v]

            if (u_atom_idx, v_atom_idx) not in edges_data_map and (
                v_atom_idx,
                u_atom_idx,
            ) not in edges_data_map:
                edges_data_map[(u_atom_idx, v_atom_idx)] = self.gen_edge_attr_to_bond_attr(
                    self.static_graph.get_edge_data(u, v, k)
                )

        for u_atom_idx, v_atom_idx in edges_data_map:
            self.atom_graph.add_edge(
                u_atom_idx, v_atom_idx, **edges_data_map[(u_atom_idx, v_atom_idx)]
            )

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

    @property
    def stochastic_vector(self):
        return self._stochastic_vector.copy()

    def draw_mw(self, rng=None) -> None | float:
        if self._stochastic_generation is not None:
            return self._stochastic_generation.draw_mw(rng)
        return -1.0

    @property
    def molw(self):
        return self._mol_weight


class _MolWeightTracker:
    pass


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

    def sample_mol_graph(self, source: str = None, rng=None):

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

        partial_atom_graph = _PartialAtomGraph(self.ml_graph, self._static_graph, source)
        while len(partial_atom_graph._open_half_bonds) > 0:
            print(
                partial_atom_graph.atom_graph,
                partial_atom_graph.molw,
                [str(ob) for ob in partial_atom_graph._open_half_bonds],
            )

        return partial_atom_graph.atom_graph
