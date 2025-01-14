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


def _docstring_parameter(*args, **kwargs):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return dec


class GeneratingGraph:
    def __init__(self, final_partial_graph: _PartialGeneratingGraph):
        self._partial_graph = final_partial_graph
        self._ml_graph = None
        self._g = self._partial_graph.g

        self.mark_aromatic_bonds()

    def mark_aromatic_bonds(self):
        from .atom import Atom

        # Post-process, marking aromatic bonds
        for edge in self.g.edges(data=True):
            node_a = self.g.nodes()[edge[0]]["obj"]
            node_b = self.g.nodes()[edge[0]]["obj"]
            if isinstance(node_a, Atom) and isinstance(node_b, Atom):
                if node_a.aromatic and node_b.aromatic:
                    edge[2]["aromatic"] = True

    def __str__(self):
        return f"GeneratingGraph({self.g}"

    @property
    def g(self):
        return self._g

    @property
    @_docstring_parameter(
        stochastic_name=_STOCHASTIC_NAME,
        termination_name=_TERMINATION_NAME,
        transition_name=_TRANSITION_NAME,
        smi_bond_mapping=smi_bond_mapping,
    )
    def ml_graph(self):
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

        if self._ml_graph is None:
            self._ml_graph = self._create_ml_graph()
        return self._ml_graph

    def _create_ml_graph(self):
        ml_graph = nx.MultiDiGraph()
        for node, data in self.g.nodes(data=True):
            obj = data["obj"]
            ml_graph.add_node(
                node,
                **{
                    "atomic_num": int(atom_name_num[str(obj.symbol)]),
                    "aromatic": obj.aromatic,
                    "charge": obj.charge,
                },
            )

        for u, v, _k, d in self.g.edges(keys=True, data=True):
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

    def get_dot_string(self, edge_colors=None, bond_to_arrow=None):

        if edge_colors is None:
            edge_colors = self._DEFAULT_EDGE_COLOR
        if bond_to_arrow is None:
            bond_to_arrow = self._DEFAULT_BOND_TO_ARROW

        graph = self.ml_graph

        dot_str = "digraph{\n"
        for node in graph.nodes(data=True):
            label = f"{atom_name_mapping[node[1]['atomic_num']]}"

            color = "#" + atom_color_mapping[node[1]["atomic_num"]]
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
