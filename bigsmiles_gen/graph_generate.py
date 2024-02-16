import copy

import networkx as nx
import numpy as np
from rdkit import Chem

from .chem_resource import atomic_masses
from .distribution import SchulzZimm


def _is_stochastic_edge(edge):
    if edge["stochastic_weight"] != 0:
        return True
    return False


def _is_termination_edge(edge):
    if edge["termination_weight"] != 0:
        return True
    return False


def _is_transition_edge(edge):
    if edge["transition_weight"] != 0:
        return True
    return False


def _is_static_edge(edge):
    if edge["static_weight"] != 0:
        return True
    return False


class AtomGraph:
    def __init__(self, stochastic_graph, rng_seed=None):
        self.stochastic_graph = stochastic_graph.graph
        self.atom_graph = None
        self.mw = [0]
        self.fully_generated = False
        self.rng = np.random.default_rng(seed=rng_seed)
        self._mw_draw_map = {}

        start = self._find_start_source()
        if start is None:
            raise RuntimeError(
                "The graph does not contain a single source node. There is at least one start node, which reach all other nodes."
            )

        node_id = self._add_node(start)

    def to_mol(self):
        mol = Chem.EditableMol(Chem.MolFromSmiles(""))
        for node in self.atom_graph.nodes(data=True):
            atom = Chem.Atom(node[1]["atomic_num"])
            mol.AddAtom(atom)
        for edge in self.atom_graph.edges(data=True):
            bond_type = Chem.BondType(edge[2]["bond_type"])
            mol.AddBond(edge[0], edge[1], bond_type)
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol

    def generate(self):
        self.atom_graph = nx.Graph()

        transition_edge_options = True
        while transition_edge_options:
            self._fill_stochastic_edges()

            # do transition
            transition_edge_options = self._next_transition_edges()
            if transition_edge_options is None:
                break

            weights = [
                data[1]["transition_weight"] for data in transition_edge_options["transition_edges"]
            ]
            weights = np.asarray(weights)
            weights /= np.sum(weights)
            idx = self.rng.choice(
                range(len(transition_edge_options["transition_edges"])), p=weights
            )

            # Selected transition edge
            transition_edge = transition_edge_options["transition_edges"][idx]
            stochastic_graph_node_idx = transition_edge[0][1]
            new_bond_type = transition_edge[1]["bond_type"]

            # We chose one transition, all other are mood
            for node in self.atom_graph:
                self.atom_graph[node]["transition_edges"].clear()

            old_node_idx = transition_edge_options["node"]
            new_node_idx = self._add_node(
                stochastic_graph_node_idx,
                None,
                transition_allowed=False,
                termination_allowed=False,
                stochastic_allowed=False,
            )
            self.atom_graph.add_edge(old_node_idx, new_node_idx, bond_type=new_bond_type)

    def _next_static_edge(self):
        for node in self.atom_graph.nodes():
            node_data = self.atom_graph.nodes[node]
            try:
                weight_edges = node_data["static_edges"]
            except IndexError:
                continue
            else:
                for i, edge_tuple in enumerate(weight_edges):
                    weight = edge_tuple[1]["weight"]
                    edge = edge_tuple[0]
                    bond_type = edge_tuple[1]["bond_type"]
                    return node, edge, bond_type
        return None

    def _fill_static_edges(self):
        node_id = None
        edge_tuple = self._next_static_edge()
        while edge_tuple is not None:
            node, edge, bond_type = edge_tuple
            node_id = self._add_node(
                edge[1],
                static_edge=edge,
                termination_allowed=True,
                stochastic_allowed=True,
                transition_allowed=True,
            )
            self.atom_graph.add_edge(node, node_id, bond_type=bond_type)

            edge_tuple = self._next_static_edge()
        return node_id

    def _next_stochastic_edge(self):
        stochastic_edges = []
        node_weights = []
        for node in self.atom_graph.nodes():
            node_data = self.atom_graph.nodes[node]
            node_edges = node_data["stochastic_edges"]
            weights = [edge_tuple[1]["weight"] for edge_tuple in node_edges]
            weight = np.sum(weights)

        if len(node_weights) == 0:
            return None

        node_weights = np.asarray(node_weights)
        node_weights /= np.sum(node_weights)
        idx = self.rng.choice(range(len(node_weights)), p=node_weights)

        return stochastic_edges[idx]

    def _fill_stochastic_edges(self):
        def get_target_mw(self, last_node_id, swap_self):
            last_node = self.atom_graph.nodes[last_node_id]
            try:
                drawn_mw = self._mw_draw_map[(last_node["mw"], last_node["mn"])]
            except KeyError:
                distr = SchulzZimm(f"schulz_zimm({last_node['mw']}, {last_node['mn']})")
                drawn_mw = distr.draw_mw(self.rng)
                self._mw_draw_map[(mw, mn)] = drawn_mw
                swap_self._mw_draw_map[(mw, mn)] = drawn_mw
            return drawn_mw

        # do-while init
        last_node_id = self._fill_static_edges()
        next_stochastic = self._next_stochastic_edge()

        while next_stochastic and last_node_id:

            excempt_node = next_stochastic["node"]
            swap_self, last_node_id = self._terminate_graph(last_node_id, excempt_node)

            if self.mw[-1] < get_target_mw(self, last_node_id, swap_self):
                # Reverse the termination of the graph
                self = swap_atom_graph
                self._add_stochastic_connection(next_stochastic)
            else:
                # Target MW reached, so we leave the termination in place
                break

            # do-while advance
            new_last_node_id = self._fill_static_edges()
            if new_last_node_id:
                last_node_id = new_last_node_id
            next_stochastic = self._next_stochastic_edge()

        # We finished generating this stochastic element of the graph, on to the next one, which counts its mw on its own
        self.mw += [0]

    def _add_stochastic_connection(self, next_stochastic):
        node = next_stochastic["node"]
        stochastic_edges = next_stochastic["stochastic_edges"]
        weights = [edge[1]["stochastic_weight"] for edge in stochastic_edges]
        weights = np.asarray(weights)
        weights /= np.sum(weights)
        idx = self.rng.choice(len(stochastic_edges), p=weights)
        edge = stochastic_edges[idx]

        # Since we will add a stochastic edge to the node, we remove all others.
        self.atom_graph.nodes[node]["stochastic_edges"].clear()
        self.atom_graph.nodes[node]["termination_edges"].clear()
        self.atom_graph.nodes[node]["transition_edges"].clear()

        new_stochastic_node = edge[0][1]
        new_bond_type = edge[1]["bond_type"]
        new_node_idx = self._add_node(
            new_stochastic_node,
            None,
            termination_allowed=False,
            transition_allowed=False,
            stochastic_allowed=False,
        )
        self.atom_graph.add_edge(node, new_node_idx, bond_type=new_bond_type)

    def _next_termination_edge(self, exempt_node):
        for node in self.atom_graph.nodes():
            if exempt_node is None or exempt_node != node:
                node_data = self.atom_graph.nodes[node]
                edge_list = node_data["termination_edges"]
                if len(edge_list) > 0:
                    weights = [edge_tuple[1]["termination_weight"] for edge_tuple in edge_list]
                    weights = np.asarray(weights)
                    weights /= np.sum(weights)
                    idx = self.rng.choice(len(edge_list), p=weights)

                    edge = edge_list[idx][0]
                    bond_type = edge_list[idx][1]

                    return {"node": node, "edge": edge, "bond_type": bond_type}

    def _terminate_graph(self, last_node_id, exempt_node):

        swap_atom_graph = copy.deepcopy(self)

        edge_info = self._next_termination_edge(exempt_node)
        while edge_info:
            last_node_id = self._add_node(
                None, transition_allowed=False, termination_allowed=False, stochastic_allowed=False
            )
            self.atom_graph.add_edge(node, node_id, bond_type=bond_type)

            node_data = self.atom_graph.nodes[node]
            # Since we are fullfilling this termination, we clear the node
            node_data["termination_edges"].clear()
            node_data["stochastic_edges"].clear()
            node_data["transition_edges"].clear()

            edge_tuple = self._next_termination_edges(exempt_node)

        return swap_atom_graph, last_node_id

    def _next_transition_edge(self):
        transition_edges = []
        node_weights = []
        for node in self.atom_graph.nodes():
            node_data = self.atom_graph.nodes[node]
            node_edges = node_data["transition_edges"]
            weights = [edge_tuple[1]["transition_weight"] for edge_tuple in node_edges]
            weight = np.sum(weights)
            node_weights.append(weight)
            transition_edges.append({"node": node, "transition_edges": node_edges})
        if len(node_weights) == 0:
            return None
        node_weights = np.asarray(node_weights)
        node_weights /= np.sum(node_weights)
        idx = self.rng.choice(range(len(node_weights)), p=node_weights)

        return transition_edges[idx]

    def _find_start_source(self):
        for node in self.stochastic_graph:
            tree = nx.dfs_tree(self.stochastic_graph, source=node)
            # If we can reach the entire graph from with following the edges, it is suitable to span the entire molecule
            if len(tree) == len(self.stochastic_graph):
                return node

    def _add_node(
        self, node, static_edge, transition_allowed, termination_allowed, stochastic_allowed
    ):
        node_id = len(self.atom_graph)
        stochastic_edges = self.stochastic_graph.out_edges(node, data=True)
        static_edges = []
        stochastic_edges = []
        transition_edges = []
        termination_edges = []
        for full_edge in stochastic_edges:
            edge = (full_edge[0], full_edge[1])
            edge_data = full_edge[2]

            if _is_static_edge(edge_data):
                if static_edge is None or (static_edge[1], static_edge[0]) != edge:
                    static_edges += [(edge, edge_data)]
            if _is_transition_edge(edge_data) and transition_allowed:
                transition_edges += [(edge, edge_data)]
            if _is_termination_edge(edge_data) and termination_allowed:
                termination_edges += [(edge, edge_data)]
            if _is_stochastic_edge(edge_data) and stochastic_allowed:
                stochastic_edges += [(edge, edge_data)]

        node_data = self.stochastic_graph.nodes[node]
        self.atom_graph.add_node(
            node_id,
            atomic_num=node_data["atomic_num"],
            mn=node_data["mn"],
            mw=node_data["mw"],
            static_edges=static_edges,
            transition_edges=transition_edges,
            termination_edges=termination_edges,
            stochastic_edges=stochastic_edges,
        )
        mw = atomic_masses[node_data["atomic_num"]]
        self.mw[-1] += mw

        return node_id
