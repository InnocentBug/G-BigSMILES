import copy
import json
import warnings
from collections.abc import Sequence
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
    _STOCHASTIC_NAME,
    _TERMINATION_NAME,
    _TRANSITION_NAME,
)
from .util import get_global_rng


class _HalfAtomBond:
    def __init__(self, atom_idx: int, node_idx: str, graph):
        self.atom_idx: int = atom_idx
        self.node_idx: str = node_idx
        self.weight: float = graph.nodes[node_idx]["gen_weight"]

        self._mode_attr_map = {}
        self._mode_target_map = {}

        for u, v, d in graph.out_edges(node_idx, data=True):
            if not d["static"]:
                for k in _NON_STATIC_ATTR:
                    if d[k] > 0:
                        try:
                            self._mode_attr_map[k] += [d]
                        except KeyError:
                            self._mode_attr_map[k] = [d]
                        try:
                            self._mode_target_map[k] += [v]
                        except KeyError:
                            self._mode_target_map[k] = [v]

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
        return f"HalfAtomBond({self.atom_idx}, {self.node_idx}, {self.weight}, {self._mode_attr_map}, {self._mode_target_map})"


class _PartialAtomGraph:
    _ATOM_ATTRS = {"atomic_num", _AROMATIC_NAME, "charge"}
    _BOND_ATTRS = {_BOND_TYPE_NAME, _AROMATIC_NAME}

    def __init__(self, generating_graph, static_graph, source_node):
        self._atom_id = 0
        self.generating_graph = generating_graph
        self.static_graph = static_graph

        self.atom_graph = nx.Graph()
        self._open_half_bond_map: dict[int : list[_HalfAtomBond]] = {}
        self._stochastic_distribution_map: dict[int:StochasticDistribution] = {}
        self._mol_weight: float = 0.0

        self.add_static_sub_graph(source_node)

    def get_open_half_bonds(stochastic_id: int | tuple[int] | None) -> list[_HalfAtomBond]:

        if stochastic_id is None:
            fetch_ids: tuple[int] = tuple(self._open_half_bond_map.keys())
        elif isinstance(stochastic_id, Sequence):
            fetch_ids: tuple[int] = tuple(stochastic_id)
        else:
            fetch_ids: tuple[int] = [int(stochastic_id)]

        open_half_bonds: list[_HalfAtomBond] = []
        for idx in fetch_ids:
            open_half_bonds += self._open_half_bond_map[idx]
        return open_half_bonds

    def merge(self, other, self_idx, other_idx, bond_attr):
        # relabel other idx
        remapping_dict = {idx: idx + self._atom_id for idx in other.atom_graph.nodes}
        other_graph = nx.relabel_nodes(other.atom_graph, remapping_dict, copy=True)
        other_open_half_bond_map = {}
        for stochstic_id in other._open_half_bond_map:
            for half_bond in other._open_half_bond_map[stochstic_id]:
                new_half_bond = copy.copy(half_bond)
                new_half_bond.atom_idx += self._atom_id
                try:
                    other_open_half_bond_map[stochstic_id] += [new_half_bond]
                except KeyError:
                    other_open_half_bond_map[stochstic_id] = [new_half_bond]

        other_idx += self._atom_id

        # Now we can do the actual merging
        self._atom_id += other._atom_id

        self.atom_graph = nx.union(self.atom_graph, other_graph)
        self.atom_graph.add_edge(self_idx, other_idx, **bond_attr)
        for stochastic_id in other_open_half_bond_map:
            try:
                self._open_half_bond_map[stochastic_id] += other_open_half_bond_map[stochastic_id]
            except KeyError:
                self._open_half_bond_map[stochastic_id] = other_open_half_bond_map[stochastic_id]

        self._stochastic_distribution_map |= other._stochastic_distribution_map
        self._mol_weight += other._mol_weight

    def add_static_sub_graph(self, source):
        atom_key_to_gen_key = {}
        gen_key_to_atom_key = {}

        def add_node(node_idx):
            data = self.gen_node_attr_to_atom_attr(self.generating_graph.nodes[node_idx])
            self.atom_graph.add_node(self._atom_id, **(data | {"origin_idx": node_idx}))
            atom_key_to_gen_key[self._atom_id] = node_idx
            gen_key_to_atom_key[node_idx] = self._atom_id
            half_bond = _HalfAtomBond(self._atom_id, node_idx, self.generating_graph)

            self._mol_weight += atomic_masses[data["atomic_num"]]
            self._atom_id += 1

            if half_bond.weight > 0 and half_bond.has_any_bonds():
                try:
                    stochastic_id = self.generating_graph[source]["stochastic_id"]
                except KeyError:
                    stochastic_id = -1
                try:
                    self._open_half_bond_map[stochastic_id] += [half_bond]
                except KeyError:
                    self._open_half_bond_map[stochastic_id] = [half_bond]
                stochastic_vector = self.generating_graph.nodes[node_idx]["stochastic_generation"]
                self._stochastic_distribution_map[stochastic_id] = (
                    StochasticDistribution.from_serial_vector(stochastic_vector)
                )

        # Initiate with first node
        add_node(source)

        edges_data_map = {}

        for u, v, k in nx.edge_dfs(self.static_graph, source=source):
            for gen_atom_idx in (u, v):
                if gen_atom_idx not in gen_key_to_atom_key:
                    add_node(gen_atom_idx)

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

    def draw_mw(self, stochastic_id, rng=None) -> None | float:
        try:
            self._stochastic_distribution_map[stochastic_id].draw_mw(rng)
        except KeyError:
            return -1

    @property
    def molw(self):
        return self._mol_weight

    def terminate_graph(self, stochastic_id, rng):
        terminated_graph = copy.deepcopy(self)
        # Find a transition bond
        transition_idx = []
        transition_weight = []
        for i, half_bond in enumerate(terminated_graph._open_half_bonds):
            if half_bond.has_mode_bonds(_TRANSITION_NAME):
                # TODO carefully check if transition bonds have the right weight here!
                transition_weight += [half_bond.weight]
                transition_idx += [i]
        transition_weight = np.asarray(transition_weight)
        transition_prob = transition_weight / np.sum(transition_weight)

        transition_half_bond = None
        # Select one of them
        if len(transition_idx) > 0:
            selected_transition_idx = rng.choice(transition_idx, p=transition_prob)
            transition_half_bond = terminated_graph._open_half_bonds.pop(selected_transition_idx)

        while len(terminated_graph._open_half_bonds) > 0:
            termination_idx = []
            termination_weight = []
            for i, half_bond in enumerate(terminated_graph._open_half_bonds):
                assert half_bond.has_mode_bonds[_TERMINATION_NAME]
                termination_idx += [i]
                termination_weight += [half_bond.weight]
            termination_weight = np.asarray(termination_weight)
            termination_prob = termination_weight / np.sum(termination_weight)

            selected_termination_idx = rng.choice(termination_idx, p=termination_weight)
            termination_bond = terminated_graph._open_half_bonds.pop(selected_termination_idx)

            target_attributes, target_ids = termination_bond.get_mode_bonds(_TERMINATION_NAME)
            target_weight = np.asarray([attr[_TERMINATION_NAME] for attr in target_attributes])
            target_prob = target_weight / np.sum(target_weight)

            selected_target_idx = np.choice(len(target_weight), p=target_prob)
            selected_target = target_ids[selected_target_idx]
            selected_attr = self.gen_edge_attr_to_bond_attr(target_attributes[selected_target_idx])

            other_partial_graph = _PartialAtomGraph(
                terminated_graph.generating_graph, terminated_graph.static_graph, selected_target
            )
            other_half_bond_atom_idx = other_partial_graph.pop_target_open_half_bond(
                selected_target
            )

            terminated_graph.merge(
                other_partial_graph,
                termination_bond.atom_idx,
                other_half_bond_atom_idx,
                selected_attr,
            )

        if transition_half_bond is not None:
            terminated_graph._open_half_bonds = [transition_half_bond]

        return terminated_graph

    def transition_graph(self, rng):
        # Early exit if no transition necessary
        if len(self._open_half_bonds) == 0:
            return self

        transitioned_graph = copy.deepcopy(self)
        assert len(transitioned_graph._open_half_bonds) == 1
        transition_bond = transitioned_graph._open_half_bonds.pop()
        assert transition_bond.has_mode_bonds(_TRANSITION_NAME)

        target_attr, target_idx = transition_bond.get_mode_bonds(_TRANSITION_NAME)
        target_weights = np.asarray([attr[_TRANSITION_NAME] for attr in target_attr])
        target_prob = target_weights / np.sum(target_weights)

        target_id = rng.choice(len(target_idx), p=target_prob)
        selected_target_idx = target_idx[target_id]
        selected_attr = self.gen_edge_attr_to_bond_attr(target_attr[target_id])

        other_graph = _PartialAtomGraph(
            self.generating_graph, self.static_graph, selected_target_idx
        )
        other_half_bond_atom_idx = other_graph.pop_target_open_half_bond(selected_target_idx)

        transitioned_graph.merge(
            other_graph, transition_bond.atom_idx, other_half_bond_atom_idx, selected_attr
        )

        return transitioned_graph

    def pop_target_open_half_bond(self, target_idx) -> int:

        # If we terminate into a regular smiles, there is no open half bonds, but the connection point is guaranteed to be 0
        if len(self._open_half_bonds) == 0:
            return 0

        found_target_index = None
        for target_index, half_bond in enumerate(self._open_half_bonds):
            if half_bond.node_idx == target_idx:
                assert found_target_index is None
                found_target_index = target_index
        assert found_target_index is not None

        target_half_bond = self._open_half_bonds.pop(found_target_index)
        return target_half_bond.atom_idx

    def has_stochastic_bonds(self):
        for half_bond in self._open_half_bonds:
            if half_bond.has_mode_bonds(_STOCHASTIC_NAME):
                return True
        return False


class _MolWeightTracker:
    def __init__(self):
        self._target_mw = {}
        self._starting_mw = {}

    # def stop_stochastic(self, partial_graph):
    #     if partial_graph.


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
            terminated_graph = partial_atom_graph.terminate_graph(rng)

            if not partial_atom_graph.has_stochastic_bonds():
                partial_atom_graph = terminated_graph.transition_graph(rng)
            else:
                break

        print(
            partial_atom_graph.atom_graph,
            partial_atom_graph.molw,
            [str(hb) for hb in partial_atom_graph._open_half_bonds],
        )

        return partial_atom_graph.atom_graph
