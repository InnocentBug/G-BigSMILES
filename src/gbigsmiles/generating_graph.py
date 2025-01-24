import uuid
import warnings

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
from .exception import IncompatibleBondTypeBondDescriptor
from .util import _determine_darkness_from_hex

_STOCHASTIC_NAME = "stochastic_weight"
_TERMINATION_NAME = "termination_weight"
_TRANSITION_NAME = "transition_weight"
_STATIC_NAME = "static"
_AROMATIC_NAME = "aromatic"
_BOND_TYPE_NAME = "bond_type"
_NON_STATIC_ATTR = (_STOCHASTIC_NAME, _TERMINATION_NAME, _TRANSITION_NAME)


def is_static_edge(edge_data):
    if _STATIC_NAME in edge_data:
        return edge_data[_STATIC_NAME]
    weight = 0
    for attr in _NON_STATIC_ATTR:
        if attr in edge_data:
            weight += edge_data[attr]
    return not weight > 0


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
    def __init__(self, final_partial_graph: _PartialGeneratingGraph, text: str = ""):
        self._partial_graph = final_partial_graph
        self._ml_graph = None
        self._graph_without_bond_descriptors = None
        self._g = self._partial_graph.g
        self.text: str = text

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
                    edge[2][_AROMATIC_NAME] = True

    def _duplicate_static_edges(self):
        for u, v, _k, d in list(self._g.edges(keys=True, data=True)):
            if is_static_edge(d):
                alternate_direction_data = self._g.get_edge_data(v, u)
                edge_found = False
                if alternate_direction_data is not None:
                    for key in alternate_direction_data:
                        if d == alternate_direction_data[key]:
                            edge_found = True
                if not edge_found:
                    self._g.add_edge(v, u, **d)

    def __str__(self):
        return f"GeneratingGraph({self.g})"

    @property
    def g(self):
        return self._g.copy()

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

        class BondDescriptorPath:
            def __init__(self, edge_path: list[tuple]):
                self.edge_path = edge_path
                node_path: list = []
                data_path: list = []
                for edge in self.edge_path:
                    data = graph.get_edge_data(*edge)
                    data_path.append(data)
                    for node in edge[:2]:
                        if len(node_path) < 1:
                            node_path.append(node)
                        else:
                            if node_path[-1] != node:
                                node_path.append(node)
                self.node_path = node_path
                self.data_path = data_path
                self._weight, self._combined_attr = self.create_combined_attr()

            def create_combined_attr(self) -> dict | None:

                data = {}
                weight = 1.0
                weight_type_list = []
                non_static_attribute_list = list(_NON_STATIC_ATTR)
                for d in self.data_path:
                    if _STATIC_NAME in d:
                        if _STATIC_NAME in data:
                            data[_STATIC_NAME] &= d[_STATIC_NAME]
                        else:
                            data[_STATIC_NAME] = d[_STATIC_NAME]
                    if _AROMATIC_NAME in d:
                        if _AROMATIC_NAME in data:
                            data[_AROMATIC_NAME] |= d[_AROMATIC_NAME]
                        else:
                            data[_AROMATIC_NAME] = d[_AROMATIC_NAME]
                    if _BOND_TYPE_NAME in d:
                        if _BOND_TYPE_NAME in data:
                            if str(data[_BOND_TYPE_NAME]) != str(d[_BOND_TYPE_NAME]):
                                warnings.warn(
                                    IncompatibleBondTypeBondDescriptor(
                                        str(data[_BOND_TYPE_NAME]), str(d[_BOND_TYPE_NAME])
                                    ),
                                    stacklevel=2,
                                )
                            return 0.0, None
                        else:
                            data[_BOND_TYPE_NAME] = d[_BOND_TYPE_NAME]

                    non_static_weights = [
                        d[attr] if attr in d else 0 for attr in non_static_attribute_list
                    ]
                    if max(non_static_weights) > 0:
                        for attr, w in zip(non_static_attribute_list, non_static_weights):
                            if w > 0:
                                current_type = attr
                                weight *= w
                    else:
                        current_type = _STATIC_NAME

                    weight_type_list.append(current_type)

                last_weight_type = _STATIC_NAME
                for weight_type in weight_type_list:
                    if last_weight_type != _STATIC_NAME and weight_type != _STATIC_NAME:
                        return 0.0, None
                    last_weight_type = weight_type

                weight_type = _STATIC_NAME
                for attr in [_TRANSITION_NAME, _TERMINATION_NAME, _STOCHASTIC_NAME]:
                    if attr in weight_type_list:
                        weight_type = attr

                if weight > 0:
                    data[weight_type] = weight

                return weight, data

            def __len__(self):
                return len(self.node_path)

            @property
            def only_bond_descriptors(self):
                return len(self.node_path) > 2 and set(self.node_path[1:-1]).issubset(bd_idx_set)

            def contains_bd(self, bd_idx):
                return bd_idx in self.node_path

            @property
            def weight(self):
                return self._weight

            @property
            def combined_attr(self):
                return self._combined_attr

            def non_zero_weight(self):
                return self.weight > 0

            def valid(self, bd_idx):
                return (
                    self.only_bond_descriptors
                    and self.contains_bd(bd_idx)
                    and (self.combined_attr is not None)
                    and (self.weight > 0)
                )

            def __str__(self):
                string = str(graph.nodes[self.node_path[0]]["obj"]) + " "
                for edge, data in zip(self.edge_path, self.data_path):
                    string += str(data) + "\n"
                    string += str(graph.nodes[edge[1]]["obj"]) + " "
                return string

        class GraphDecider:
            def __init__(self, bd_idx_set, in_idx):
                self.bd_idx_set = bd_idx_set
                self.in_idx = in_idx

            def __call__(self, x):
                return (x not in self.bd_idx_set) and (x != self.in_idx)

        edges_to_add = []
        # Add edges jumping over the pairs of bond descriptors with correct weights.
        for bd_idx in bd_idx_set:
            for in_edge in graph.in_edges(bd_idx, data=True):
                in_idx = in_edge[0]
                # Only do it for sources of the bond descriptors that are not bond descriptors themselves. I.e. at the start of a chain.
                if in_idx not in bd_idx_set:
                    traversal_condition = GraphDecider(bd_idx_set, in_idx)
                    non_bond_descriptor_successor = conditional_traversal(
                        graph, in_idx, traversal_condition
                    )
                    for target in non_bond_descriptor_successor:
                        all_paths = list(nx.all_simple_edge_paths(graph, in_idx, target))
                        for path in all_paths:
                            bond_descriptor_path = BondDescriptorPath(path)
                            if bond_descriptor_path.valid(bd_idx):
                                data = bond_descriptor_path.combined_attr
                                edges_to_add.append((in_idx, target, data))

        # The previous approach does not handle self loops on bond descriptors, since they are cycles.
        # However, these are important and easy manually address
        for bd_idx in bd_idx_set:
            in_edges = set(graph.in_edges(bd_idx, keys=True))
            out_edges = set(graph.out_edges(bd_idx, keys=True))
            # A loop to itself is the intersection
            loop_edges = in_edges.intersection(out_edges)
            for loop_edge in loop_edges:
                for in_u, in_v, in_k, in_data in graph.in_edges(bd_idx, keys=True, data=True):
                    if is_static_edge(in_data):
                        for out_u, out_v, out_k, out_data in graph.out_edges(
                            bd_idx, keys=True, data=True
                        ):
                            if is_static_edge(out_data):
                                path = [(in_u, in_v, in_k), loop_edge, (out_u, out_v, out_k)]
                                bond_descriptor_path = BondDescriptorPath(path)
                                if bond_descriptor_path.valid(bd_idx):
                                    data = bond_descriptor_path.combined_attr
                                    edges_to_add.append((in_u, out_v, data))

        for edge in edges_to_add:
            graph.add_edge(edge[0], edge[1], **edge[2])
        # Remove all bond descriptors from the graph.
        graph.remove_nodes_from(bd_idx_set)

        # Normalize transition weights
        for node in graph.nodes():
            total_weight = 0
            for _u, _v, _k, d in graph.out_edges(node, keys=True, data=True):
                if _TRANSITION_NAME in d:
                    total_weight += d[_TRANSITION_NAME]

            if total_weight > 0:
                for u, v, k, d in graph.out_edges(node, keys=True, data=True):
                    if _TRANSITION_NAME in d:
                        graph.edges[u, v, k][_TRANSITION_NAME] /= total_weight

        GeneratingGraph._mark_aromatic_bonds(graph)

        return graph

    @_docstring_format(
        stochastic_name=_STOCHASTIC_NAME,
        termination_name=_TERMINATION_NAME,
        transition_name=_TRANSITION_NAME,
        static_name=_STOCHASTIC_NAME,
        aromatic_name=_AROMATIC_NAME,
        bond_type_name=_BOND_TYPE_NAME,
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

        - **{static_name}**: bool indicating static edges, that are always present.
        - **{stochastic_name}**: float Stochastic probability. If bond descriptors are connecting between monomer repeat units inside stochastic objects, this indicates the probability \in [0, 1].
        - **{termination_name}**: float Termination Probabilities. If bond descriptors terminate with end-groups after the molecular weight is reached, this is the probability \in [0, 1].
        - **{transition_name}**: float Transition Probabilities. If transitioning between stochastic objects this is the probability to take.
        - **{bond_type_name}**: int Integer category that maps to different bond_types as follows{smi_bond_mapping}.
        - **{aromatic_name}**: bool Indicates aromatic bonds.
        """

        extra_graph_info = {0: "None"}
        extra_graph_info_reverse = {"None": 0}

        if include_bond_descriptors:
            graph = self.g.copy()
        else:
            graph = self.get_graph_without_bond_descriptors().copy()

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
                    _AROMATIC_NAME: aromatic,
                    "charge": charge,
                },
            )

        for u, v, _k, d in graph.edges(keys=True, data=True):
            d.setdefault(_STATIC_NAME, is_static_edge(d))
            d.setdefault(_STOCHASTIC_NAME, 0)
            d.setdefault(_TERMINATION_NAME, 0)
            d.setdefault(_TRANSITION_NAME, 0)
            if _BOND_TYPE_NAME in d:
                d[_BOND_TYPE_NAME] = smi_bond_mapping.get(str(d[_BOND_TYPE_NAME]), 1)
            else:
                d[_BOND_TYPE_NAME] = 1
            d.setdefault(_AROMATIC_NAME, False)

            ml_graph.add_edge(u, v, **d)

        if return_extra_graph_info:
            return ml_graph, extra_graph_info
        return ml_graph

    _DEFAULT_EDGE_COLOR = {
        _STATIC_NAME: "#000000",
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

    def get_dot_string(
        self, include_bond_descriptors=False, edge_colors=None, bond_to_arrow=None, node_prefix=""
    ):

        if edge_colors is None:
            edge_colors = self._DEFAULT_EDGE_COLOR
        if bond_to_arrow is None:
            bond_to_arrow = self._DEFAULT_BOND_TO_ARROW

        graph, extra_graph_info = self.get_ml_graph(
            include_bond_descriptors=include_bond_descriptors, return_extra_graph_info=True
        )

        dot_str = "digraph{\n"
        dot_str += f'label="{self.text}";\n'
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

            dot_str += f'"{node_prefix}{node[0]}" [{extra_attr} label="{label}"];\n'

        for u, v, _k, d in graph.edges(keys=True, data=True):
            bond_type = d[_BOND_TYPE_NAME]
            color = "black"
            value = 1.0
            for key in edge_colors:
                if d[key] > 0:
                    color = edge_colors[key]
                    value = d[key]
            style = "solid"
            if d[_AROMATIC_NAME]:
                style = "dashed"

            dot_str += f'"{node_prefix}{u}" -> "{node_prefix}{v}" [arrowhead="{bond_to_arrow[bond_type]}", label="{float(value)}", color="{color}", style="{style}"];\n'

        dot_str += "}\n"
        return dot_str
