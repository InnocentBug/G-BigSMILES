# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022-2025: Ludwig Schneider
# See LICENSE for details

import copy
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Optional

import networkx as nx
import numpy as np

from .chem_resource import atom_color_mapping, atom_name_mapping, atomic_masses
from .distribution import StochasticDistribution
from .exception import (
    IncompleteStochasticGeneration,
    InvalidGenerationSource,
    UnvalidatedGenerationSource,
)
from .generating_graph import (
    _AROMATIC_NAME,
    _BOND_TYPE_NAME,
    _NON_STATIC_ATTR,
    _STOCHASTIC_NAME,
    _TERMINATION_NAME,
    _TRANSITION_NAME,
)
from .util import _determine_darkness_from_hex, get_global_rng


class _HalfAtomBond:
    def __init__(self, atom_idx: int, node_idx: str, graph, rng):
        self.atom_idx: int = atom_idx
        self.node_idx: str = node_idx
        self.weight: float = graph.nodes[node_idx]["gen_weight"]
        self.parent: int = graph.nodes[node_idx]["parent_stochastic_id"]
        self._graph = graph

        self._mode_attr_map = {}
        self._mode_target_map = {}

        self._special_target = None
        special_target_list = []
        special_target_weight = []

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

                if d[_TRANSITION_NAME] > 0:
                    current_stochastic_id = graph.nodes[u]["stochastic_id"]
                    target_stochastic_id = graph.nodes[v]["stochastic_id"]
                    parent_target_stochastic_id = graph.nodes[v]["parent_stochastic_id"]
                    if current_stochastic_id != target_stochastic_id and parent_target_stochastic_id == current_stochastic_id:
                        special_target_list += [(v, d)]
                        special_target_weight += [d[_TRANSITION_NAME]]
                        # if self._special_target is not None:
                        #     raise RuntimeError("There should only be one nested special target per bond.")

        if len(special_target_weight) > 0:
            special_target_prob = special_target_weight / np.sum(special_target_weight)
            self._special_target = rng.choice(special_target_list, p=special_target_prob)

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

    @property
    def stochastic_growth_suitable(self):
        return self.has_mode_bonds(_STOCHASTIC_NAME)

    def get_mode_bonds(self, mode):
        try:
            return self._mode_attr_map[mode], self._mode_target_map[mode]
        except KeyError:
            return [], []

    def __str__(self):
        return f"HalfAtomBond({self.atom_idx}, {self.node_idx}, {self.weight}, {self._mode_attr_map}, {self._mode_target_map})"


class _StochasticObjectTracker:
    def __init__(self, generating_graph, rng=None):
        self._rng = rng
        # Stochastic **sto_gen_id** is the id of the stochastic object as found in the generative graph.
        # Stochastic **sto_atom_id** is the id of an instance of that particular stochastic gen id.
        # In most cases they are the same as we have exactly one instance for each stochastic object.
        # However, with nested stochastic objects that is not the case.
        # Consider a linear polymer, where each back-bone monomer has a stochastic side arm like {[] [<]CC({[<] [<]NN[>] [>]}[H])CC[>] []}
        # From the outer stochastic object we only have one instance. And every "C" has the same `sto_gen_id` and `sto_atom_id` of 0.
        # But each monomer spawns a new instance of the inner stochastic object. So every "N" has the sto_gen_id of 1, but every monomer has a different stochastic atom id and counting
        self._stochastic_gen_id_to_atom_id = {}
        self._stochastic_atom_id_to_gen_id = OrderedDict()
        self._sto_gen_id_distribution = {}
        self._sto_atom_id_actual_molw = OrderedDict()
        self._sto_atom_id_expected_molw = OrderedDict()
        self._terminated_sto_atom_ids = set()
        self._parent_map = {}

        for _node_idx, data in generating_graph.nodes(data=True):
            if data["stochastic_id"] >= 0:
                stochastic_vector = data["stochastic_generation"].copy()
                distribution = StochasticDistribution.from_serial_vector(stochastic_vector)
                self._register_sto_gen_id(data["stochastic_id"], distribution)

    def has_sto_gen_id_unterminated_sto_ids(self, sto_gen_id: int):
        if sto_gen_id not in self._stochastic_gen_id_to_atom_id:
            return False
        found = False
        for sto_atom_id in self._stochastic_gen_id_to_atom_id[sto_gen_id]:
            if not self.is_terminated(sto_atom_id):
                found = True
                break
        return found

    def _register_sto_gen_id(self, sto_gen_id, distribution):
        self._sto_gen_id_distribution[sto_gen_id] = distribution

    def register_new_atom_instance(self, sto_gen_id, old_atom_id, is_nested_parent):
        if sto_gen_id > 0:
            if not self._is_sto_gen_id_known(sto_gen_id):
                raise RuntimeError("You cannot register the an already known atomic instance as new. Please report on github.")

        try:
            new_sto_atom_id = max(self._stochastic_atom_id_to_gen_id) + 1
        except ValueError:
            new_sto_atom_id = 0

        self._stochastic_atom_id_to_gen_id[new_sto_atom_id] = sto_gen_id
        try:
            self._stochastic_gen_id_to_atom_id[sto_gen_id].add(new_sto_atom_id)
        except KeyError:
            self._stochastic_gen_id_to_atom_id[sto_gen_id] = {new_sto_atom_id}

        if sto_gen_id >= 0:
            new_molw = self._sto_gen_id_distribution[sto_gen_id].draw_mw(self._rng)
            self._sto_atom_id_expected_molw[new_sto_atom_id] = new_molw
        else:
            self._sto_atom_id_expected_molw[new_sto_atom_id] = -1

        self._sto_atom_id_actual_molw[new_sto_atom_id] = 0

        if is_nested_parent:
            self._parent_map[new_sto_atom_id] = old_atom_id

        return new_sto_atom_id

    def add_molw(self, sto_atom_id, molw):
        self._sto_atom_id_actual_molw[sto_atom_id] += molw

        tmp_id = sto_atom_id
        while tmp_id in self._parent_map:
            tmp_id = self._parent_map[tmp_id]
            self._sto_atom_id_actual_molw[tmp_id] += molw
        return self._sto_atom_id_actual_molw[sto_atom_id] >= self._sto_atom_id_expected_molw[sto_atom_id]

    def should_terminate(self, sto_atom_id):
        return self.add_molw(sto_atom_id, 0)

    def _is_sto_gen_id_known(self, sto_gen_id):
        return sto_gen_id in self._sto_gen_id_distribution

    def is_terminated(self, sto_atom_id):
        if sto_atom_id not in self._stochastic_atom_id_to_gen_id:
            raise ValueError("Unknown atom id. it cannot be terminated")
        return sto_atom_id in self._terminated_sto_atom_ids

    def terminate(self, sto_atom_id):
        if self.is_terminated(sto_atom_id):
            raise RuntimeError("You cannot terminate an already terminated stochastic ID. This is a bug, please report on github.")

        self._terminated_sto_atom_ids.add(sto_atom_id)

    def draw_mw(self, sto_gen_id, sto_atom_id=None, rng=None) -> None | float:
        if sto_gen_id is None:
            sto_gen_id = self._stochastic_atom_id_to_sto_gen_id[sto_atom_id]

        return self._sto_gen_id_distribution[sto_gen_id].draw_mw(rng)

    def get_unterminated_sto_atom_ids(self):
        unterminated_sto_atom_ids = []
        for sto_atom_id in reversed(self._stochastic_atom_id_to_gen_id):
            if sto_atom_id not in self._terminated_sto_atom_ids:
                unterminated_sto_atom_ids += [sto_atom_id]
        return unterminated_sto_atom_ids


class _PartialAtomGraph:
    _ATOM_ATTRS = {"atomic_num", _AROMATIC_NAME, "charge"}
    _BOND_ATTRS = {_BOND_TYPE_NAME, _AROMATIC_NAME}

    def __init__(self, generating_graph, static_graph, source_node, stochastic_tracker, sto_atom_id, rng):
        self._atom_id = 0
        self.generating_graph = generating_graph
        self.static_graph = static_graph
        self.stochastic_tracker = stochastic_tracker

        self.atom_graph = nx.Graph()
        self._open_half_bond_map: dict[int : list[_HalfAtomBond]] = {}
        self.add_static_sub_graph(source_node, sto_atom_id, rng)

    def merge(self, other, self_idx, other_idx, bond_attr):
        # relabel other idx
        remapping_dict = {idx: idx + self._atom_id for idx in other.atom_graph.nodes}
        other_graph = nx.relabel_nodes(other.atom_graph, remapping_dict, copy=True)
        other_open_half_bond_map = {}
        for stochastic_id in other._open_half_bond_map:
            for half_bond in other._open_half_bond_map[stochastic_id]:
                new_half_bond = copy.copy(half_bond)
                new_half_bond.atom_idx += self._atom_id
                try:
                    other_open_half_bond_map[stochastic_id] += [new_half_bond]
                except KeyError:
                    other_open_half_bond_map[stochastic_id] = [new_half_bond]

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

    def get_open_half_bonds(self, sto_atom_id: int | tuple[int] | None, prefer_parent: bool = False) -> list[_HalfAtomBond]:

        if sto_atom_id is None:
            fetch_ids: tuple[int] = tuple(self._open_half_bond_map.keys())
        elif isinstance(sto_atom_id, Sequence):
            fetch_ids: tuple[int] = tuple(sto_atom_id)
        else:
            fetch_ids: tuple[int] = tuple([int(sto_atom_id)])

        open_half_bonds: list[_HalfAtomBond] = []
        for idx in fetch_ids:
            try:
                open_half_bonds += self._open_half_bond_map[idx]
            except KeyError:
                pass

        idx = range(len(open_half_bonds))
        if prefer_parent:
            parent_idx = []
            parent_bonds = []
            for i, bond in enumerate(open_half_bonds):
                if bond.parent >= 0:
                    parent_idx += [i]
                    parent_bonds += [bond]
            if len(parent_bonds) > 0:
                open_half_bonds = parent_bonds
                idx = parent_idx
        return idx, open_half_bonds

    def add_static_sub_graph(self, source, sto_atom_id, rng):
        atom_key_to_gen_key = {}
        gen_key_to_atom_key = {}

        def add_node(node_idx):
            data = self.gen_node_attr_to_atom_attr(self.generating_graph.nodes[node_idx])
            self.atom_graph.add_node(self._atom_id, **(data | {"origin_idx": str(node_idx)}))
            atom_key_to_gen_key[self._atom_id] = node_idx
            gen_key_to_atom_key[node_idx] = self._atom_id
            half_bond = _HalfAtomBond(self._atom_id, node_idx, self.generating_graph, rng)

            self.stochastic_tracker.add_molw(sto_atom_id, atomic_masses[data["atomic_num"]])
            self._atom_id += 1

            if half_bond.weight > 0 and half_bond.has_any_bonds():
                try:
                    self._open_half_bond_map[sto_atom_id] += [half_bond]
                except KeyError:
                    self._open_half_bond_map[sto_atom_id] = [half_bond]

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
                edges_data_map[(u_atom_idx, v_atom_idx)] = self.gen_edge_attr_to_bond_attr(self.static_graph.get_edge_data(u, v, k))

        for u_atom_idx, v_atom_idx in edges_data_map:
            self.atom_graph.add_edge(u_atom_idx, v_atom_idx, **edges_data_map[(u_atom_idx, v_atom_idx)])

    def gen_node_attr_to_atom_attr(self, attr: dict[str, bool | float | int], keys_to_copy: None | set[str] = None) -> dict[str, bool | float | int]:
        if keys_to_copy is None:
            keys_to_copy = self._ATOM_ATTRS
        return self._copy_some_dict_attr(attr, keys_to_copy)

    def gen_edge_attr_to_bond_attr(self, attr: dict[str, bool | int], keys_to_copy: None | set[str] = None) -> dict[str, bool | int]:
        if keys_to_copy is None:
            keys_to_copy = self._BOND_ATTRS
        return self._copy_some_dict_attr(attr, keys_to_copy)

    @staticmethod
    def _copy_some_dict_attr(dictionary: dict[str, Any], keys_to_copy: set[str]) -> dict[str, Any]:
        new_dict = {}
        for k in keys_to_copy:
            new_dict[k] = dictionary[k]
        return new_dict

    def pop_target_open_half_bond(self, sto_atom_idx, target_idx) -> _HalfAtomBond:
        found_target_index = None
        try:
            for target_index, half_bond in enumerate(self._open_half_bond_map[sto_atom_idx]):
                if half_bond.node_idx == target_idx:
                    if found_target_index is not None:
                        raise RuntimeError("A matching target index was found twice, that is a bug. Please report on github.")

                    found_target_index = target_index
        except KeyError:
            pass

        if found_target_index is None:
            possible_connections = self._find_origin_to_atom(target_idx)
            if len(possible_connections) != 1:
                raise RuntimeError("There should only be one possible connection left. Please report this bug on github.")
            return possible_connections[0]

        target_half_bond = self._open_half_bond_map[sto_atom_idx].pop(found_target_index)
        return target_half_bond.atom_idx

    def terminate_graph(self, sto_atom_id, rng):
        terminated_graph = copy.deepcopy(self)

        def pop_random_transition_bond():
            # Find a transition bond
            transition_idx = []
            transition_weight = []

            for i, half_bond in zip(*terminated_graph.get_open_half_bonds(sto_atom_id)):
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
                transition_half_bond = terminated_graph._open_half_bond_map[sto_atom_id].pop(selected_transition_idx)
            return transition_half_bond

        def get_termination_bonds(graph, sto_atom_id):
            termination_bonds = []
            for _, half_bond in zip(*graph.get_open_half_bonds(sto_atom_id)):
                if half_bond.has_mode_bonds(_TERMINATION_NAME):
                    termination_bonds += [half_bond]
            return termination_bonds

        def pop_next_termination_bond():
            termination_idx = []
            termination_weight = []
            for i, half_bond in zip(*terminated_graph.get_open_half_bonds(sto_atom_id)):
                if half_bond.has_mode_bonds(_TERMINATION_NAME):
                    termination_idx += [i]
                    termination_weight += [half_bond.weight]
            termination_weight = np.asarray(termination_weight)
            termination_prob = termination_weight / np.sum(termination_weight)

            selected_termination_idx = rng.choice(termination_idx, p=termination_prob)
            termination_bond = terminated_graph._open_half_bond_map[sto_atom_id].pop(selected_termination_idx)
            return termination_bond

        transition_half_bond = pop_random_transition_bond()

        while len(get_termination_bonds(terminated_graph, sto_atom_id)) > 0:
            termination_bond = pop_next_termination_bond()

            target_attributes, target_ids = termination_bond.get_mode_bonds(_TERMINATION_NAME)
            target_weight = np.asarray([attr[_TERMINATION_NAME] for attr in target_attributes])
            target_prob = target_weight / np.sum(target_weight)

            selected_target_idx = rng.choice(len(target_weight), p=target_prob)
            selected_target = target_ids[selected_target_idx]
            selected_attr = self.gen_edge_attr_to_bond_attr(target_attributes[selected_target_idx])

            other_partial_graph = _PartialAtomGraph(terminated_graph.generating_graph, terminated_graph.static_graph, selected_target, self.stochastic_tracker, sto_atom_id, rng)
            other_half_bond_atom_idx = other_partial_graph.pop_target_open_half_bond(sto_atom_id, selected_target)

            terminated_graph.merge(
                other_partial_graph,
                termination_bond.atom_idx,
                other_half_bond_atom_idx,
                selected_attr,
            )

        if transition_half_bond is not None:
            terminated_graph._open_half_bond_map[sto_atom_id] = [transition_half_bond]
        else:
            terminated_graph._open_half_bond_map[sto_atom_id] = []

        terminated_graph.stochastic_tracker.terminate(sto_atom_id)

        return terminated_graph

    def _find_origin_to_atom(self, origin_idx):
        atom_id_list = []
        for node_id, data in self.atom_graph.nodes(data=True):
            if data["origin_idx"] == origin_idx:
                atom_id_list += [node_id]
        return atom_id_list

    def transition_graph(self, sto_atom_id, rng):
        # Early exit if no transition necessary
        if len(self.get_open_half_bonds(sto_atom_id)[0]) < 1:
            return sto_atom_id, False

        transition_bond = self._open_half_bond_map[sto_atom_id].pop(0)
        if not transition_bond.has_mode_bonds(_TRANSITION_NAME):
            self._open_half_bond_map[sto_atom_id] += [transition_bond]
            return sto_atom_id, False

        target_attr, target_idx = transition_bond.get_mode_bonds(_TRANSITION_NAME)
        target_weights = np.asarray([attr[_TRANSITION_NAME] for attr in target_attr])

        target_prob = target_weights / np.sum(target_weights)

        target_id = rng.choice(len(target_idx), p=target_prob)
        selected_target_idx = target_idx[target_id]
        selected_attr = self.gen_edge_attr_to_bond_attr(target_attr[target_id])
        selected_target_sto_gen_id = self.generating_graph.nodes[selected_target_idx]["stochastic_id"]
        # if selected_target_sto_gen_id > 0:
        #     if self.stochastic_tracker._stochastic_atom_id_to_gen_id[sto_atom_id] == selected_target_sto_gen_id:
        #         raise RuntimeError("New stochastic IDs need to be new.")

        new_sto_atom_id = None
        if selected_target_sto_gen_id in self.stochastic_tracker._stochastic_gen_id_to_atom_id:
            for a in self.stochastic_tracker._stochastic_gen_id_to_atom_id[selected_target_sto_gen_id]:
                if not self.stochastic_tracker.is_terminated(a):
                    new_sto_atom_id = a
                    break
        if new_sto_atom_id is None:
            new_sto_atom_id = self.stochastic_tracker.register_new_atom_instance(selected_target_sto_gen_id, sto_atom_id, False)

        other_graph = _PartialAtomGraph(self.generating_graph, self.static_graph, selected_target_idx, self.stochastic_tracker, new_sto_atom_id, rng)

        other_target_idx = other_graph.pop_target_open_half_bond(new_sto_atom_id, selected_target_idx)

        self.merge(other_graph, transition_bond.atom_idx, other_target_idx, selected_attr)

        return new_sto_atom_id, True

    def stochastic_growth(self, sto_atom_id, rng, prefer_parent_bonds):
        def pop_random_stochastic_bond():
            # Find a transition bond
            stochastic_idx = []
            stochastic_weight = []
            for i, half_bond in zip(*self.get_open_half_bonds(sto_atom_id, prefer_parent=prefer_parent_bonds)):
                if half_bond.stochastic_growth_suitable:
                    # TODO carefully check if stochastic bonds have the right weight here!
                    stochastic_weight += [half_bond.weight]
                    stochastic_idx += [i]

            stochastic_weight = np.asarray(stochastic_weight)
            stochastic_prob = stochastic_weight / np.sum(stochastic_weight)
            stochastic_half_bond = None

            # Select one of them
            if len(stochastic_idx) > 0:
                selected_stochastic_idx = rng.choice(stochastic_idx, p=stochastic_prob)
                stochastic_half_bond = self._open_half_bond_map[sto_atom_id].pop(selected_stochastic_idx)

            return stochastic_half_bond

        stochastic_bond = pop_random_stochastic_bond()

        if stochastic_bond is None:
            raise IncompleteStochasticGeneration(self)

        target_attr, target_idx = stochastic_bond.get_mode_bonds(_STOCHASTIC_NAME)
        target_weights = np.asarray([attr[_STOCHASTIC_NAME] for attr in target_attr])
        target_prob = target_weights / np.sum(target_weights)

        target_id = rng.choice(len(target_idx), p=target_prob)
        selected_target_idx = target_idx[target_id]
        selected_attr = self.gen_edge_attr_to_bond_attr(target_attr[target_id])
        selected_target_sto_gen_id = self.generating_graph.nodes[selected_target_idx]["stochastic_id"]

        new_sto_atom_id = sto_atom_id
        if self.stochastic_tracker._stochastic_atom_id_to_gen_id[sto_atom_id] != selected_target_sto_gen_id:

            for existing_atom_id in reversed(self.stochastic_tracker.get_unterminated_sto_atom_ids()):
                if self.stochastic_tracker._stochastic_atom_id_to_gen_id[existing_atom_id] == selected_target_sto_gen_id:
                    new_sto_atom_id = existing_atom_id
                    break

            if new_sto_atom_id == sto_atom_id:
                new_sto_atom_id = self.stochastic_tracker.register_new_atom_instance(selected_target_sto_gen_id, sto_atom_id, False)

            self.stochastic_tracker.terminate(sto_atom_id)

        other_graph = _PartialAtomGraph(self.generating_graph, self.static_graph, selected_target_idx, self.stochastic_tracker, new_sto_atom_id, rng)

        other_half_bond_atom_idx = other_graph.pop_target_open_half_bond(new_sto_atom_id, selected_target_idx)

        self.merge(other_graph, stochastic_bond.atom_idx, other_half_bond_atom_idx, selected_attr)

        return new_sto_atom_id

    def nested_transition(self, sto_atom_id, rng):
        def pop_nested_bonds():
            if sto_atom_id not in self._open_half_bond_map:
                return []

            # Find a transition bond
            normal_bonds = []
            special_bonds = []

            for half_bond in self._open_half_bond_map[sto_atom_id]:
                if half_bond._special_target is not None:
                    special_bonds += [half_bond]
                else:
                    normal_bonds += [half_bond]
            self._open_half_bond_map[sto_atom_id] = normal_bonds

            return special_bonds

        for nested_transition_bond in pop_nested_bonds():
            selected_target_idx, selected_attr = nested_transition_bond._special_target
            selected_target_sto_gen_id = self.generating_graph.nodes[selected_target_idx]["stochastic_id"]

            new_sto_atom_id = sto_atom_id
            if self.stochastic_tracker._stochastic_atom_id_to_gen_id[sto_atom_id] != selected_target_sto_gen_id:
                for existing_atom_id in reversed(self.stochastic_tracker.get_unterminated_sto_atom_ids()):
                    if self.stochastic_tracker._stochastic_atom_id_to_gen_id[existing_atom_id] == selected_target_sto_gen_id:
                        new_sto_atom_id = existing_atom_id
                        break

                if new_sto_atom_id == sto_atom_id:
                    new_sto_atom_id = self.stochastic_tracker.register_new_atom_instance(selected_target_sto_gen_id, sto_atom_id, True)

            other_graph = _PartialAtomGraph(self.generating_graph, self.static_graph, selected_target_idx, self.stochastic_tracker, new_sto_atom_id, rng)

            other_half_bond_atom_idx = other_graph.pop_target_open_half_bond(new_sto_atom_id, selected_target_idx)

            self.merge(other_graph, nested_transition_bond.atom_idx, other_half_bond_atom_idx, selected_attr)


class AtomGraph:
    def __init__(self, ml_graph):
        self._ml_graph = ml_graph.copy()

        self._static_graph = self._create_static_graph(self.ml_graph)

        self._starting_node_idx, self._starting_node_weight = self._create_init_weights(self.ml_graph)

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

    @staticmethod
    def get_dot_string(atom_graph, bond_type_colors=None, prefix="") -> str:
        if bond_type_colors is None:
            bond_type_colors = {1: "black", 2: "red", 3: "green", 4: "blue"}
        dot_str = "graph{\n"
        for node, data in atom_graph.nodes(data=True):
            label = atom_name_mapping[data["atomic_num"]]
            color = "#" + atom_color_mapping[data["atomic_num"]]

            extra_attr = f'style="filled", fillcolor="{color}", '
            if _determine_darkness_from_hex(color):
                extra_attr += "fontcolor=white, "
            dot_str += f'"{prefix}{node}" [{extra_attr} label="{label}"];\n'

        for u, v, d in atom_graph.edges(data=True):
            bond_type = d["bond_type"]
            color = bond_type_colors[bond_type]
            style = "solid"
            if d["aromatic"]:
                style = "dashed"
            dot_str += f'"{prefix}{u}" -- "{prefix}{v}" [color="{color}", style="{style}"];\n'
        dot_str += "}\n"
        return dot_str

    def sample_mol_graph(self, source: Optional[str] = None, rng=None, tolerate_incomplete_stochastic_generation_with_no_more_than_X_open_bonds=0):
        if rng is None:
            rng = get_global_rng()

        if source is None:
            source = self._get_random_start_node(rng)

        if source not in self.ml_graph.nodes():
            raise InvalidGenerationSource(source, self.ml_graph.nodes(), self.ml_graph)

        if source not in self._starting_node_idx:
            warnings.warn(
                UnvalidatedGenerationSource(source, self._starting_node_idx, self.ml_graph),
                stacklevel=2,
            )

        stochastic_object_tracker = _StochasticObjectTracker(self.ml_graph, rng)

        source_sto_gen_id = self.ml_graph.nodes[source]["stochastic_id"]
        sto_atom_id = stochastic_object_tracker.register_new_atom_instance(source_sto_gen_id, self.ml_graph.nodes[source]["parent_stochastic_id"], False)
        partial_atom_graph = _PartialAtomGraph(self.ml_graph, self._static_graph, source, stochastic_object_tracker, sto_atom_id, rng)
        del stochastic_object_tracker

        partial_atom_graph.transition_graph(sto_atom_id, rng)
        while len(partial_atom_graph.stochastic_tracker.get_unterminated_sto_atom_ids()) > 0:
            active_sto_atom_id = partial_atom_graph.stochastic_tracker.get_unterminated_sto_atom_ids()[0]
            partial_atom_graph.nested_transition(active_sto_atom_id, rng)
            active_sto_atom_id = partial_atom_graph.stochastic_tracker.get_unterminated_sto_atom_ids()[0]

            terminated_graph = partial_atom_graph.terminate_graph(active_sto_atom_id, rng)
            if terminated_graph.stochastic_tracker.should_terminate(active_sto_atom_id):

                partial_atom_graph = terminated_graph
                # After termination, there is only one transition bond left
                active_sto_atom_id, transition_success = partial_atom_graph.transition_graph(active_sto_atom_id, rng)
            else:
                try:
                    partial_atom_graph.stochastic_growth(active_sto_atom_id, rng, True)
                except IncompleteStochasticGeneration as exc:
                    sto_atom_id, transition_success = partial_atom_graph.transition_graph(active_sto_atom_id, rng)
                    if not transition_success:
                        if exc.num_open_bonds <= tolerate_incomplete_stochastic_generation_with_no_more_than_X_open_bonds:
                            return exc.atom_graph
                        raise exc from exc

        return partial_atom_graph.atom_graph
