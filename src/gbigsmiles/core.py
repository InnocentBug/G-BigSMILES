# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from abc import ABC, abstractmethod
from warnings import warn

import numpy as np

from .chem_resource import atom_color_mapping, atom_name_mapping

_GLOBAL_RNG = np.random.default_rng()


def _determine_darkness_from_hex(color):
    """
    Determine the darkness of a color from its hex string.

    Arguments:
    ---------
    color: str
       7 character string with prefix `#` followed by RGB hex code.

    Returns: bool
       if the darkness is below half

    """
    # If hex --> Convert it to RGB: http://gist.github.com/983661
    if color[0] != "#":
        raise ValueError(f"{color} is missing '#'")
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    # HSP (Highly Sensitive Poo) equation from http://alienryderflex.com/hsp.html
    hsp = np.sqrt(0.299 * red**2 + 0.587 * green**2 + 0.114 * blue**2)
    return hsp < 127.5


class BigSMILESbase(ABC):

    bond_descriptors = []

    def __str__(self):
        return self.generate_string(True)

    @abstractmethod
    def generate_string(self, extension: bool):
        pass

    @property
    @abstractmethod
    def generable(self):
        pass

    @property
    def residues(self):
        return []

    def generate(self, prefix=None, rng=_GLOBAL_RNG):
        if not self.generable:
            raise RuntimeError("Attempt to generate a non-generable molecule.")
        if prefix:
            if len(prefix.bond_descriptors) != 1:
                raise RuntimeError(
                    f"Prefixes for generating Mols must have exactly one open bond descriptor found {len(prefix.bond_descriptors)}."
                )


def get_compatible_bond_descriptor_ids(bond_descriptors, bond):
    compatible_idx = []
    for i, other in enumerate(bond_descriptors):
        if bond is None or bond.is_compatible(other):
            compatible_idx.append(i)
    return np.asarray(compatible_idx, dtype=int)


def choose_compatible_weight(bond_descriptors, bond, rng):
    weights = []
    compatible_idx = get_compatible_bond_descriptor_ids(bond_descriptors, bond)
    for i in compatible_idx:
        weights.append(bond_descriptors[i].weight)
    weights = np.asarray(weights)
    # Trick to allow 0 weights for special situations.
    if len(compatible_idx) > 0 and np.all(weights == weights[0]):
        weights += 1
    weights /= np.sum(weights)

    try:
        idx = rng.choice(compatible_idx, p=weights)
    except ValueError as exc:
        warn(
            f"Cannot choose compatible bonds, available bonds {len(compatible_idx)}, sum of weights {np.sum(weights)}.",
            stacklevel=1,
        )
        raise exc

    return idx


def reaction_graph_to_dot_string(graph, bigsmiles=None):
    from .bond import BondDescriptor

    dot_str = "strict digraph { \n"
    if bigsmiles:
        dot_str += f'label="{str(bigsmiles)}"\n'
    for node in graph.nodes():

        if isinstance(node, BondDescriptor):
            dot_str += f"\"{hash(node)}\" [label=\"{node.generate_string(False)} w={graph.nodes[node]['weight']}\", fillcolor=lightgreen, style=filled];\n"
        else:
            dot_str += f'"{hash(node)}" [label="{node.generate_string(False)}", fillcolor=lightblue, style=filled];\n'

    name_map = {
        "term_prob": "t(stochastic)",
        "trans_prob": "t(suffix)",
        "atom": "atom",
        "prob": "r",
    }
    for edge in graph.edges():
        edge_data = graph.get_edge_data(*edge)
        for name in name_map:
            if name in edge_data:
                value = edge_data[name]
                edge_label = f"{name_map[name]} = {np.round(value ,2)}"
        if "atom" in edge_label:
            dot_str += (
                f'{hash(edge[0])} -> {hash(edge[1])} [label="{edge_label}", arrowhead=none];\n'
            )
        else:
            dot_str += f'{hash(edge[0])} -> {hash(edge[1])} [label="{edge_label}"];\n'

    dot_str += "}\n"

    return dot_str


def stochastic_atom_graph_to_dot_string(stochastic_atom_graph):
    graph = stochastic_atom_graph.graph
    dot_str = "strict digraph { \n"
    for node in graph.nodes(data=True):
        label = f"{atom_name_mapping[node[1]['atomic_num']]}"

        color = "#" + atom_color_mapping[node[1]["atomic_num"]]
        extra_attr = f'style=filled, fillcolor="{color}", '
        if _determine_darkness_from_hex(color):
            extra_attr += "fontcolor=white,"

        dot_str += f'"{node[0]}" [{extra_attr} label="{label}"];\n'

    for edge in graph.edges():
        edge_data = graph.get_edge_data(*edge)
        bond_type = []
        try:
            bond_type += [edge_data["bond_type"]]
        except KeyError:
            for key in edge_data:
                bond_type += [edge_data[key]["bond_type"]]

        dot_str += f'"{int(edge[0])}" -> "{int(edge[1])}" [label="{bond_type}"];\n'
    dot_str += "}\n"
    return dot_str
