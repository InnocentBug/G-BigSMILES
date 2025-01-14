# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details
import networkx as nx
import numpy as np

from .big_smiles import BigSmilesMolecule
from .bond import BondDescriptor, TerminalBondDescriptor
from .core import BigSMILESbase, GenerationBase
from .distribution import StochasticGeneration
from .exception import EndGroupHasOneBondDescriptors, MonomerHasTwoOrMoreBondDescriptors
from .generating_graph import (
    _STOCHASTIC_NAME,
    _TERMINATION_NAME,
    _PartialGeneratingGraph,
    is_static_edge,
)
from .smiles import Smiles


class StochasticObject(BigSMILESbase, GenerationBase):
    def __init__(self, children: list):
        super().__init__(children)

        self._repeat_residues: list = []
        self._termination_residues: list = []
        self._left_terminal_bond_d: None | BondDescriptor = None
        self._right_terminal_bond_d: None | BondDescriptor = None

        self._generation: StochasticGeneration | None = None

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

    def _post_parse_validation(self):
        for smi in self._repeat_residues:
            if len(smi.bond_descriptors) < 2:
                raise MonomerHasTwoOrMoreBondDescriptors(smi, self)

        for smi in self._termination_residues:
            if len(smi.bond_descriptors) != 1:
                raise EndGroupHasOneBondDescriptors(smi, self)

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

    def _generate_partial_graph(self) -> _PartialGeneratingGraph:
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

        def add_reverse_static_edges(graph):
            # TODO may require extra thinking for nested stochastic objects
            for u, v, _k, data in list(graph.edges(keys=True, data=True)):
                if is_static_edge(data):
                    graph.add_edge(v, u, **data)

            bd_idx = set()
            for node in graph.nodes(data=True):
                if isinstance(node[1]["obj"], BondDescriptor):
                    bd_idx.add(node[0])

            # Remove edges that are not necessary for traversal from every bond descriptor
            # This makes this function idem potent
            necessary_edges = set()
            for source in bd_idx:
                edges_this_bd = nx.dfs_edges(G=graph, source=source)
                necessary_edges |= set(edges_this_bd)

            for u, v, k, data in list(graph.edges(keys=True, data=True)):
                if is_static_edge(data):
                    if (u, v) not in necessary_edges:
                        graph.remove_edge(u, v, k)
            return graph

        # Build graph without any connections between bond descriptors.
        repeat_subgraphs = [
            add_reverse_static_edges(monomer._generate_partial_graph().g)
            for monomer in self._repeat_residues
        ]
        terminal_subgraphs = [
            add_reverse_static_edges(end._generate_partial_graph().g)
            for end in self._termination_residues
        ]
        graph = nx.union_all(repeat_subgraphs + terminal_subgraphs)

        # List of monomer repeat unit bond descriptor IDX
        mono_idx_pos = build_idx(self._repeat_residues, graph)
        # Same for end units
        end_idx_pos = build_idx(self._termination_residues, graph)
        # We can combine for all bond descriptors
        bd_idx_pos = mono_idx_pos + end_idx_pos

        # Add bonds between compatible pairs monomer bond descriptors
        for bd_idx_a in mono_idx_pos:
            obj_a = graph.nodes[bd_idx_a]["obj"]
            if obj_a.transition is not None:
                # Note that this spans the end groups
                probabilities = obj_a.transition
            else:
                # This does not span the end groups
                probabilities = [graph.nodes[bd_idx_b]["obj"].weight for bd_idx_b in mono_idx_pos]

            probabilities = np.asarray(probabilities)

            # Set weights to zero if bond are incompatible, note different lengths from above.
            for i in range(len(probabilities)):
                bd_idx_b = bd_idx_pos[i]
                obj_b = graph.nodes[bd_idx_b]["obj"]
                if not obj_a.is_compatible(obj_b):
                    probabilities[i] = 0
            # Normalizing probabilities
            if probabilities.sum() > 0:
                probabilities /= probabilities.sum()

            # Bond attributes are _STOCHASTIC_NAME
            for i, prob in enumerate(probabilities):
                if prob > 0:
                    bd_idx_b = bd_idx_pos[i]
                    graph.add_edge(bd_idx_a, bd_idx_b, **dict([(_STOCHASTIC_NAME, prob)]))

            # Add bonds between monomer/end group bond_descriptors
            # Termination probs
            end_probabilities = []
            for bd_idx_b in end_idx_pos:
                obj_b = graph.nodes[bd_idx_b]["obj"]
                if obj_a.is_compatible(obj_b):
                    end_probabilities.append(obj_b.weight)
            end_probabilities = np.asarray(end_probabilities)
            if end_probabilities.sum() > 0:
                end_probabilities /= end_probabilities.sum()

            for i, prob in enumerate(end_probabilities):
                if prob > 0:
                    bd_idx_b = end_idx_pos[i]
                    graph.add_edge(bd_idx_a, bd_idx_b, **dict([(_TERMINATION_NAME, prob)]))

        # Add edges jumping over the pairs of bond descriptors with correct weights.
        for bd_idx in bd_idx_pos:
            for in_edge in list(graph.in_edges(bd_idx, data=True)):
                in_idx = in_edge[0]
                in_data = in_edge[2]
                # Ignore bonds between bond descriptors, chaining them is not allowed. I think
                if in_idx not in bd_idx_pos:
                    # Find connected out going bond descriptors
                    for bd_out_edge in list(graph.out_edges(bd_idx, data=True)):
                        bd_out_idx = bd_out_edge[1]
                        bd_out_data = bd_out_edge[2]
                        if bd_out_idx in bd_idx_pos:  # Only bond descriptors
                            # Iterate the out-going non-bond-descriptor bonds
                            for out_edge in list(graph.out_edges(bd_out_idx, data=True)):
                                out_idx = out_edge[1]
                                out_data = out_edge[2]
                                if out_idx not in bd_idx_pos:  # Only non Bond Descriptors
                                    # Build edge that jumps over both bond descriptors
                                    # TODO think about raises problem if bd attr don't match
                                    graph.add_edge(
                                        in_idx, out_idx, **dict(in_data | bd_out_data | out_data)
                                    )

        partial_graph = _PartialGeneratingGraph(graph)
        # Add left half bonds.

        # Add right half bonds

        # Remove all bond descriptors from the graph...
        graph.remove_nodes_from(bd_idx_pos)

        return partial_graph


"""Deprecated with the grammar based G-BigSMILES, use StochasticObject instead."""
Stochastic = StochasticObject
