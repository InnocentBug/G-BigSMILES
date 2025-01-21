import uuid

import networkx as nx

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .chem_resource import (
    atom_color_mapping,
    atom_name_mapping,
    atom_name_num,
    smi_bond_mapping,
)
from .util import _determine_darkness_from_hex

_STOCHASTIC_NAME = "stochastic_weight"
_TERMINATION_NAME = "termination_weight"
_TRANSITION_NAME = "transition_weight"
_NON_STATIC_ATTR = (_STOCHASTIC_NAME, _TERMINATION_NAME, _TRANSITION_NAME)


def is_static_edge(edge_data):
    return all(attr not in edge_data for attr in _NON_STATIC_ATTR)


class _HalfBond:
    def __init__(self, node, node_id: uuid.UUID, bond_attributes: dict):
        self.node = node
        self.node_id = node_id
        self.bond_attributes = bond_attributes

    def __str__(self):
        return f"HalfBond({str(self.node)}, {self.node_id}, {self.bond_attributes})"


class _PartialGeneratingGraph:

    def __init__(self, g: None | nx.MultiDiGraph = None):
        if g is None:
            g = nx.MultiDiGraph()
        self.g = g

        self.left_half_bonds: list[_HalfBond] = []
        self.right_half_bonds: list[_HalfBond] = []
        self.ring_bond_map: dict[int, _HalfBond] = {}

    def merge(self, other: Self, half_bond_tuples: list[tuple[_HalfBond, _HalfBond]]) -> Self:
        """
        Strictly only merges the graphs, handling of left/right bond halves has to be performed before hand.
        It does handle the merging of ring bonds though.

        """

        if len(other.left_half_bonds) != 0:
            raise ValueError(other.left_half_bonds)
        if len(other.right_half_bonds) != 0:
            raise ValueError(other.right_half_bonds)

        half_bond_tuples = list(half_bond_tuples)

        new_ring_bond_map = self.ring_bond_map
        for ring_bond_idx in other.ring_bond_map:
            if ring_bond_idx in new_ring_bond_map:
                half_bond_tuples.append(
                    (new_ring_bond_map[ring_bond_idx], other.ring_bond_map[ring_bond_idx])
                )
                del new_ring_bond_map[ring_bond_idx]
            else:
                new_ring_bond_map[ring_bond_idx] = other.ring_bond_map[ring_bond_idx]

        self.ring_bond_map = new_ring_bond_map

        self.g = nx.union(self.g, other.g)

        for self_bond, other_bond in half_bond_tuples:
            self.add_half_bond_edge(self_bond, other_bond)

    def add_half_bond_edge(
        self, self_half_bond_edge: _HalfBond, other_half_bond_edge: _HalfBond
    ) -> None:
        overlapping_keys = (
            self_half_bond_edge.bond_attributes.keys() & other_half_bond_edge.bond_attributes.keys()
        )
        if len(overlapping_keys) > 0:
            raise ValueError(overlapping_keys)

        new_bond_attributes = (
            self_half_bond_edge.bond_attributes | other_half_bond_edge.bond_attributes
        )
        self.g.add_edge(
            self_half_bond_edge.node_id, other_half_bond_edge.node_id, **new_bond_attributes
        )

    def add_ring_bond(self, ring_bond, half_bond: _HalfBond) -> bool:

        if ring_bond.idx in self.ring_bond_map:
            self.add_half_bond_edge(half_bond, self.ring_bond_map[ring_bond.idx])
            del self.ring_bond_map[ring_bond.idx]
            return False

        self.ring_bond_map[ring_bond.idx] = half_bond
        return True

    def __str__(self):
        return f"PartialGraph({self.g}, {self.left_half_bonds}, {self.right_half_bonds}, {self.ring_bond_map})"

    def __getitem__(self, idx):
        return self.g.nodes[idx]


def _docstring_format(*args, **kwargs):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return dec


class GeneratingGraph:
    def __init__(self, final_partial_graph: _PartialGeneratingGraph):
        self._partial_graph = final_partial_graph
        self._ml_graph = None
        self._graph_without_bond_descriptors = None
        self._g = self._partial_graph.g

        GeneratingGraph._mark_aromatic_bonds(self.g)
        self._duplicate_static_edges()

    @staticmethod
    def _mark_aromatic_bonds(graph):
        from .atom import Atom

        # Post-process, marking aromatic bonds
        for edge in graph.edges(data=True):
            node_a = graph.nodes()[edge[0]]["obj"]
            node_b = graph.nodes()[edge[1]]["obj"]
            if isinstance(node_a, Atom) and isinstance(node_b, Atom):
                if node_a.aromatic and node_b.aromatic:
                    edge[2]["aromatic"] = True

    def _duplicate_static_edges(self):
        for u, v, _k, d in list(self.g.edges(keys=True, data=True)):
            if is_static_edge(d):
                alternate_direction_data = self.g.get_edge_data(v, u)
                edge_found = False
                if alternate_direction_data is not None:
                    for key in alternate_direction_data:
                        if d == alternate_direction_data[key]:
                            edge_found = True
                if not edge_found:
                    self.g.add_edge(v, u, **d)

    def __str__(self):
        return f"GeneratingGraph({self.g})"

    @property
    def g(self):
        return self._g

    def get_graph_without_bond_descriptors(self):
        from .bond import BondDescriptor

        def conditional_traversal(graph, source, stop_condition):
            """
            Perform a graph traversal starting from the source node.
            Stop traversal if a node fulfills the stop_condition and return all nodes where traversal stopped.

            Args:
                graph (nx.DiGraph): The directed graph.
                source: The starting node for the traversal.
                stop_condition (function): A function that takes a node and returns True if traversal should stop.

            Returns:
                set: A set of nodes where the traversal stopped.
            """
            stopped_nodes = set()
            visited = set()

            def dfs(node):
                if node in visited:
                    return
                visited.add(node)

                if stop_condition(node):
                    stopped_nodes.add(node)
                    return  # Stop traversal from this node

                # Continue traversal to neighbors
                for neighbor in graph.neighbors(node):
                    dfs(neighbor)

            dfs(source)
            return stopped_nodes

        graph = self.g.copy()

        bd_idx_set = set()
        for node_idx, data in graph.nodes(data=True):
            if isinstance(data["obj"], BondDescriptor):
                bd_idx_set.add(node_idx)

        edges_to_add = []
        # Add edges jumping over the pairs of bond descriptors with correct weights.
        for bd_idx in bd_idx_set:
            for in_edge in graph.in_edges(bd_idx, data=True):
                in_idx = in_edge[0]
                # Only do it for sources of the bond descriptors that are not bond descriptors themselves. I.e. at the start of a chain.
                if in_idx not in bd_idx_set:

                    in_data = in_edge[2]
                    non_bond_descriptor_successor = conditional_traversal(
                        graph, in_idx, lambda x: ((x not in bd_idx_set) and (x != in_idx))
                    )
                    for target in non_bond_descriptor_successor:
                        for path in nx.all_simple_edge_paths(graph, in_idx, target):
                            data = in_data.copy()
                            stochastic_keys = set()
                            bd_idx_found = False
                            stochastic_weight_found = False
                            for edge in path:
                                if bd_idx in edge:
                                    bd_idx_found = True
                                for node_idx in edge[:2]:
                                    stochastic_generation = graph.nodes[node_idx].get(
                                        "stochastic_generation", None
                                    )
                                    if stochastic_generation is not None:
                                        stochastic_keys.add(stochastic_generation.key)
                                    else:
                                        stochastic_keys.add(-1)
                                # TODO secure union
                                data |= graph.get_edge_data(*edge)
                                if _STOCHASTIC_NAME in data:
                                    stochastic_weight_found = True
                            if bd_idx_found:  # Only consider paths that contain the bd of interest
                                if (
                                    len(stochastic_keys) <= 2
                                ):  # We allow at most one stochastic object in the path
                                    if (
                                        len(
                                            set.intersection(
                                                set(
                                                    (
                                                        _TRANSITION_NAME,
                                                        _TERMINATION_NAME,
                                                        _STOCHASTIC_NAME,
                                                    )
                                                ),
                                                set(data.keys()),
                                            )
                                        )
                                        < 2
                                    ):
                                        if not stochastic_weight_found:
                                            print(
                                                "A", in_idx, bd_idx, data, len(stochastic_keys) <= 2
                                            )
                                            edges_to_add.append((in_idx, target, data))

        for edge in edges_to_add:
            graph.add_edge(edge[0], edge[1], **edge[2])
        # Remove all bond descriptors from the graph.
        graph.remove_nodes_from(bd_idx_set)

        GeneratingGraph._mark_aromatic_bonds(graph)

        return graph

    @_docstring_format(
        stochastic_name=_STOCHASTIC_NAME,
        termination_name=_TERMINATION_NAME,
        transition_name=_TRANSITION_NAME,
        smi_bond_mapping=smi_bond_mapping,
    )
    def get_ml_graph(self, include_bond_descriptors=False, return_extra_graph_info=False):
        r"""
        The ML Graph has well defined properties that do not rely on the specifics of this library.

        Node(Atoms) have the following properties:

        - **atomic_num**: int Atomic number, can be converted to Chemical Symbol Name or one-hot encoding.
        - **aromatic**: bool Indicating the aromaticity of the atom.
        - **charge**: float Nominal charge (not partial charge in Force-Fields) in elementary unit *e*.

        Edges(Bonds) have the following properties:

        - **static**: bool indicating static edges, that are always present.
        - **{stochastic_name}**: float Stochastic probability. If bond descriptors are connecting between monomer repeat units inside stochastic objects, this indicates the probability \in [0, 1].
        - **{termination_name}**: float Termination Probabilities. If bond descriptors terminate with end-groups after the molecular weight is reached, this is the probability \in [0, 1].
        - **{transition_name}**: float Transition Probabilities. If transitioning between stochastic objects this is the probability to take.
        - **bond_type**: int Integer category that maps to different bond_types as follows{smi_bond_mapping}.
        - **aromatic**: bool Indicates aromatic bonds.
        """

        extra_graph_info = {0: "None"}
        extra_graph_info_reverse = {"None": 0}

        if include_bond_descriptors:
            graph = self.g
        else:
            graph = self.get_graph_without_bond_descriptors()

        ml_graph = nx.MultiDiGraph()
        for node, data in graph.nodes(data=True):
            obj = data["obj"]
            try:
                aromatic = obj.aromatic
            except AttributeError:
                aromatic = False
            atomic_symbol = str(obj.symbol)
            if aromatic:
                atomic_symbol = atomic_symbol.upper()
            try:
                atomic_num = int(atom_name_num[atomic_symbol])
            except KeyError:
                string = str(obj)
                if string in extra_graph_info_reverse:
                    atomic_num = extra_graph_info_reverse[string]
                else:
                    idx = min(extra_graph_info.keys()) - 1
                    atomic_num = idx
                    extra_graph_info[idx] = string
                    extra_graph_info_reverse[string] = idx

            try:
                charge = obj.charge
            except AttributeError:
                charge = float("nan")

            ml_graph.add_node(
                node,
                **{
                    "atomic_num": atomic_num,
                    "aromatic": aromatic,
                    "charge": charge,
                },
            )

        for u, v, _k, d in graph.edges(keys=True, data=True):
            d.setdefault("static", is_static_edge(d))
            d.setdefault(_STOCHASTIC_NAME, 0)
            d.setdefault(_TERMINATION_NAME, 0)
            d.setdefault(_TRANSITION_NAME, 0)
            if "bond_type" in d:
                d["bond_type"] = smi_bond_mapping.get(str(d["bond_type"]), 1)
            else:
                d["bond_type"] = 1
            d.setdefault("aromatic", False)

            ml_graph.add_edge(u, v, **d)

        if return_extra_graph_info:
            return ml_graph, extra_graph_info
        return ml_graph

    _DEFAULT_EDGE_COLOR = {
        "static": "#000000",
        _STOCHASTIC_NAME: "#ff0000",
        _TRANSITION_NAME: "#00ff00",
        _TERMINATION_NAME: "#0000ff",
    }
    _DEFAULT_BOND_TO_ARROW = {
        1: "normal",
        2: "diamond",
        3: "dot",
        4: "box",
    }

    def get_dot_string(self, include_bond_descriptors=False, edge_colors=None, bond_to_arrow=None):

        if edge_colors is None:
            edge_colors = self._DEFAULT_EDGE_COLOR
        if bond_to_arrow is None:
            bond_to_arrow = self._DEFAULT_BOND_TO_ARROW

        graph, extra_graph_info = self.get_ml_graph(
            include_bond_descriptors=include_bond_descriptors, return_extra_graph_info=True
        )

        dot_str = "digraph{\n"
        for node in graph.nodes(data=True):
            if node[1]["atomic_num"] > 0:
                label = f"{atom_name_mapping[node[1]['atomic_num']]}"
                color = "#" + atom_color_mapping[node[1]["atomic_num"]]
            else:
                label = extra_graph_info[node[1]["atomic_num"]]
                color = "#FFFFFF"

            extra_attr = f'style=filled, fillcolor="{color}", '
            if _determine_darkness_from_hex(color):
                extra_attr += "fontcolor=white,"

            dot_str += f'"{node[0]}" [{extra_attr} label="{label}"];\n'

        for u, v, _k, d in graph.edges(keys=True, data=True):
            bond_type = d["bond_type"]
            color = "black"
            value = 1.0
            for key in edge_colors:
                if d[key] > 0:
                    color = edge_colors[key]
                    value = d[key]
            style = "solid"
            if d["aromatic"]:
                style = "dashed"

            dot_str += f'"{u}" -> "{v}" [arrowhead="{bond_to_arrow[bond_type]}", label="{float(value)}", color="{color}", style="{style}"];\n'

        dot_str += "}\n"
        return dot_str
