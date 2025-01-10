import uuid

import networkx as nx

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class _HalfBond:
    def __init__(self, node_id: uuid.UUID, bond_attributes: dict):
        self.node_id = node_id
        self.bond_attributes = bond_attributes

    def __str__(self):
        return f"HalfBond({self.node_id}, {self.bond_attributes})"


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
