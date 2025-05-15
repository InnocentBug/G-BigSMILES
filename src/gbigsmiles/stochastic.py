# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details
import warnings

import networkx as nx
import numpy as np

from .big_smiles import BigSmilesMolecule
from .bond import BondDescriptor, TerminalBondDescriptor
from .core import BigSMILESbase, GenerationBase
from .distribution import StochasticGeneration
from .exception import (
    ConcatenatedBondDescriptors,
    EmptyTerminalBondDescriptorWithoutEndGroups,
    EndGroupHasOneBondDescriptors,
    IncorrectNumberOfTransitionWeights,
    MonomerHasTwoOrMoreBondDescriptors,
    NoInitiationForStochasticObject,
    NoLeftTransitions,
    StochasticMissingPath,
    UndefinedDistribution,
)
from .generating_graph import (
    _STOCHASTIC_NAME,
    _TERMINATION_NAME,
    _TRANSITION_NAME,
    _HalfBond,
    _PartialGeneratingGraph,
)
from .smiles import Smiles


class StochasticObject(BigSMILESbase, GenerationBase):
    def __init__(self, children: list):
        super().__init__(children)

        self._repeat_residues: list = []
        self._termination_residues: list = []
        self._left_terminal_bond_d: None | BondDescriptor = None
        self._right_terminal_bond_d: None | BondDescriptor = None

        self._generation: None | StochasticGeneration = None
        self._stochastic_parent: None | StochasticObject = None

        # Parse info
        termination_separator_found = False
        for child in self._children:
            if isinstance(child, TerminalBondDescriptor):
                if self._left_terminal_bond_d is None:
                    self._left_terminal_bond_d = child
                else:
                    if self._right_terminal_bond_d is not None:
                        raise ValueError(f"{self}, {self._children}, {self._right_terminal_bond_d}")
                    self._right_terminal_bond_d = child

            if str(child) == ";":
                termination_separator_found = True

            if isinstance(child, BigSmilesMolecule) or isinstance(child, Smiles):
                if not termination_separator_found:
                    self._repeat_residues.append(child)
                else:
                    self._termination_residues.append(child)

            if isinstance(child, StochasticGeneration):
                self._generation = child

        self._post_parse_validation()

        # Set the parent of child stochastic objects
        for child in self._children:
            try:
                child._set_stochastic_parent(self)
            except AttributeError:
                pass

    @property
    def stochastic_parent(self):
        return self._stochastic_parent

    def _set_stochastic_parent(self, parent):
        if self._stochastic_parent is not None:
            raise RuntimeError("A stochastic parent can only be set once. If this most likely a bug, please report it on github.")
        self._stochastic_parent = parent

    def _post_parse_validation(self):
        try:
            for element in [self] + self._repeat_residues + self._termination_residues:
                gen_graph = element.get_generating_graph()
                MLgraph = gen_graph.get_ml_graph(include_bond_descriptors=True)
                for node, data in MLgraph.nodes(data=True):
                    if node in gen_graph._bd_idx_set:
                        if data["stochastic_id"] == data["parent_stochastic_id"]:
                            for _u, v in MLgraph.out_edges(node):
                                stochastic_id = MLgraph.nodes(data=True)[v]["stochastic_id"]
                                if (v in gen_graph._bd_idx_set) and (data["stochastic_id"] == stochastic_id):
                                    raise ConcatenatedBondDescriptors(element, self)
        except UndefinedDistribution:
            pass

        for smi in self._repeat_residues:
            if len(smi.bond_descriptors) < 2:
                raise MonomerHasTwoOrMoreBondDescriptors(smi, self)

        for smi in self._termination_residues:
            if len(smi.bond_descriptors) != 1:
                raise EndGroupHasOneBondDescriptors(smi, self)

        # Empty left bond descriptors need end-groups to start initiation
        if self._left_terminal_bond_d.symbol is None:
            if len(self._termination_residues) < 1:
                warnings.warn(EmptyTerminalBondDescriptorWithoutEndGroups(self), stacklevel=1)

        inner_bond_descriptors = []
        for element in self._repeat_residues + self._termination_residues:
            inner_bond_descriptors += element.bond_descriptors

        for bd in [
            self._left_terminal_bond_d,
            self._right_terminal_bond_d,
        ] + inner_bond_descriptors:
            if bd.transition is not None and len(bd.transition) != len(inner_bond_descriptors):
                raise IncorrectNumberOfTransitionWeights(self, bd, len(inner_bond_descriptors))

    def generate_string(self, extension: bool):
        string = "{" + self._left_terminal_bond_d.generate_string(extension) + " "
        if len(self._repeat_residues) > 0:
            string += self._repeat_residues[0].generate_string(extension)
            for residue in self._repeat_residues[1:]:
                string += ", " + residue.generate_string(extension)

        if len(self._termination_residues) > 0:
            string += "; " + self._termination_residues[0].generate_string(extension)
            for residue in self._termination_residues[1:]:
                string += ", " + residue.generate_string(extension)

        string += " " + self._right_terminal_bond_d.generate_string(extension) + "}"

        if self._generation:
            string += self._generation.generate_string(extension)

        return string

    @property
    def bond_descriptors(self):
        return []

    @property
    def stochastic_generation(self):
        return self._generation

    def _generate_partial_graph(self) -> _PartialGeneratingGraph:

        if self.stochastic_generation is None:
            raise UndefinedDistribution(self)

        def build_idx(residues, graph):
            """
            Build a list that maps uuid of all bond_descriptors to their position in the string.
            The position is important for the transition weights.

            Example:
            -------
            build_idx(..)[3] gives you the uuid graph index of the 3rd bond descriptor in the stochastic element.

            """
            bond_descriptors = []
            for res in residues:
                bond_descriptors += res.bond_descriptors

            bd_idx = [None] * len(bond_descriptors)
            for bd in graph.nodes(data=True):
                bd_obj = bd[1]["obj"]
                if isinstance(bd_obj, BondDescriptor):
                    if bd_obj in bond_descriptors:
                        bd_idx[bond_descriptors.index(bd[1]["obj"])] = bd[0]
            return bd_idx

        def _connect(
            graph,
            first_idx,
            second_idx,
            full_idx,
            attr_name=_STOCHASTIC_NAME,
            ignore_transitions=False,
        ):

            for bd_idx_a in first_idx:
                obj_a = graph.nodes[bd_idx_a]["obj"]
                if obj_a.transition is not None and not ignore_transitions:
                    # Note that this spans the end groups
                    probabilities = obj_a.transition
                else:
                    # This does not span the end groups
                    probabilities = [graph.nodes[bd_idx_b]["obj"].weight for bd_idx_b in second_idx]

                probabilities = np.asarray(probabilities)
                # Set weights to zero if bond are incompatible, note different lengths from above.
                for i in range(len(probabilities)):
                    bd_idx_b = full_idx[i]
                    obj_b = graph.nodes[bd_idx_b]["obj"]
                    if not obj_a.is_compatible(obj_b):
                        probabilities[i] = 0
                # Normalizing probabilities
                if probabilities.sum() > 0:
                    probabilities /= probabilities.sum()
                # Bond attributes are _STOCHASTIC_NAME
                for i, prob in enumerate(probabilities):
                    if prob > 0:
                        bd_idx_b = full_idx[i]
                        graph.add_edge(bd_idx_a, bd_idx_b, **dict([(attr_name, prob)]))
            return graph

        def connect_monomers_to_monomers(graph, mono_idx_pos, end_idx_pos):
            return _connect(
                graph,
                mono_idx_pos,
                mono_idx_pos,
                mono_idx_pos + end_idx_pos,
                ignore_transitions=False,
                attr_name=_STOCHASTIC_NAME,
            )

        def connect_monomers_to_end(graph, mono_idx_pos, end_idx_pos):
            result = _connect(
                graph,
                mono_idx_pos,
                end_idx_pos,
                end_idx_pos,
                ignore_transitions=True,
                attr_name=_TERMINATION_NAME,
            )
            return result

        def connect_end_to_monomers(graph, mono_idx_pos, end_idx_pos):
            return _connect(
                graph,
                end_idx_pos,
                mono_idx_pos,
                mono_idx_pos + end_idx_pos,
                ignore_transitions=False,
                attr_name=_TRANSITION_NAME,
            )

        # Build graph without any connections between bond descriptors.
        repeat_subgraphs = [monomer.get_generating_graph()._partial_graph for monomer in self._repeat_residues]
        terminal_subgraphs = [end.get_generating_graph()._partial_graph for end in self._termination_residues]
        partial_graph = _PartialGeneratingGraph(None)
        for gen_graph in repeat_subgraphs + terminal_subgraphs:
            gen_graph.left_half_bonds = []
            gen_graph.right_half_bonds = []
            partial_graph.merge(gen_graph, [])
        graph = partial_graph.g

        # List of monomer repeat unit bond descriptor IDX
        mono_idx_pos = build_idx(self._repeat_residues, graph)
        # Same for end units
        end_idx_pos = build_idx(self._termination_residues, graph)

        graph = connect_monomers_to_monomers(graph, mono_idx_pos, end_idx_pos)
        graph = connect_monomers_to_end(graph, mono_idx_pos, end_idx_pos)

        # If we have an empty left terminal bond descriptor, allow end groups to be initial groups.
        if self._left_terminal_bond_d.symbol is None:
            graph = connect_end_to_monomers(graph, mono_idx_pos, end_idx_pos)

        # Add initiating bonds
        if self._left_terminal_bond_d.symbol is None:
            if self._left_terminal_bond_d.transition is not None:
                weights = self._left_terminal_bond_d.transition[len(mono_idx_pos) :]
            else:
                weights = [graph.nodes[bd_idx]["obj"].weight for bd_idx in end_idx_pos]
            weights = np.asarray(weights)
            if len(weights) != len(end_idx_pos):
                raise RuntimeError(f"Implementation error, please report on GitHub https://github.com/InnocentBug/G-BigSMILES/issues . {weights} {end_idx_pos}")

            if weights.sum() == 0:
                weights += 1

            probabilities = weights / weights.sum()

            for i, prob in enumerate(probabilities):
                if prob > 0:
                    node_idx = end_idx_pos[i]
                    node = graph.nodes[node_idx]["obj"]
                    partial_graph.left_half_bonds.append(_HalfBond(node, node_idx, dict([(_TRANSITION_NAME, prob)])))
        else:
            left_partial_graph = self._left_terminal_bond_d._generate_partial_graph()
            left_partial_graph.left_half_bonds = []
            left_partial_graph.right_half_bonds = []
            left_idx = list(left_partial_graph.g.nodes)[0]
            partial_graph.merge(left_partial_graph, [])
            partial_graph.left_half_bonds.append(_HalfBond(self._left_terminal_bond_d, left_idx, {}))
            graph = partial_graph.g

            # With non-empty left bond descriptors we connect first to one of the monomers inside.
            left_bd = self._left_terminal_bond_d
            if left_bd.transition is not None:
                weights = left_bd.transition[: len(mono_idx_pos)]
            else:
                weights = [graph.nodes[bd_idx]["obj"].weight for bd_idx in mono_idx_pos]
            weights = np.asarray(weights)

            for i, bd_idx in enumerate(mono_idx_pos):
                if not left_bd.is_compatible(graph.nodes[bd_idx]["obj"]):
                    weights[i] = 0

            probabilities = []
            if weights.sum() > 0:
                probabilities = weights / weights.sum()

            for i, prob in enumerate(probabilities):
                if prob > 0:
                    node_idx = mono_idx_pos[i]
                    graph.add_edge(left_idx, node_idx, **dict([(_TRANSITION_NAME, prob)]))

        # Add out-going bonds
        if self._right_terminal_bond_d.symbol is not None:
            right_partial_graph = self._right_terminal_bond_d._generate_partial_graph()
            right_partial_graph.left_half_bonds = []
            right_partial_graph.right_half_bonds = []
            right_idx = list(right_partial_graph.g.nodes)[0]
            partial_graph.merge(right_partial_graph, [])
            partial_graph.right_half_bonds.append(_HalfBond(self._right_terminal_bond_d, right_idx, {}))

            graph = partial_graph.g

            weights = []
            for i, bd_idx in enumerate(mono_idx_pos):
                node = graph.nodes[bd_idx]["obj"]
                weight = node.weight
                if self._right_terminal_bond_d.transition is not None:
                    weight = self._right_terminal_bond_d.transition[i]
                if not self._right_terminal_bond_d.is_compatible(node):
                    weight = 0

                weights.append(weight)
            weights = np.asarray(weights)
            probabilities = []
            if weights.sum() > 0:
                probabilities = weights / weights.sum()

            for i, prob in enumerate(probabilities):
                if prob > 0:
                    bd_idx = mono_idx_pos[i]
                    graph.add_edge(bd_idx, right_idx, **dict([(_TRANSITION_NAME, prob)]))

        # Add mol weight distribution to all nodes
        for node_idx in partial_graph.g:
            if "stochastic_obj" not in graph.nodes[node_idx]:  # Nested objects have that already
                graph.nodes[node_idx]["stochastic_obj"] = self

        self._post_validate_partial_graph(partial_graph, mono_idx_pos + end_idx_pos)

        return partial_graph

    def _post_validate_partial_graph(self, partial_graph, bd_idx):
        if len(partial_graph.left_half_bonds) == 0:
            if self._left_terminal_bond_d.symbol is None:
                warnings.warn(NoInitiationForStochasticObject(self, partial_graph), stacklevel=1)
            else:
                warnings.warn(NoLeftTransitions(self, partial_graph), stacklevel=1)

        # To successfully generate molecules, there needs to be path from each entry point (left half bonds)
        # to at least one end, right half bonds
        if len(partial_graph.right_half_bonds) > 0:
            target_idx = set([rhb.node_id for rhb in partial_graph.right_half_bonds])
            source_idx = set([lhb.node_id for lhb in partial_graph.left_half_bonds])

            for source in source_idx:
                tree = nx.dfs_tree(partial_graph.g, source=source)
                reachable_nodes = set(tree.nodes)

                if target_idx.isdisjoint(reachable_nodes):
                    warnings.warn(
                        StochasticMissingPath(self, partial_graph.g.nodes[source]["obj"]),
                        stacklevel=1,
                    )


"""Deprecated with the grammar based G-BigSMILES, use StochasticObject instead."""
Stochastic = StochasticObject
