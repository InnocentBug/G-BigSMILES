import copy

import networkx as nx
import numpy as np
from rdkit import Chem

from .chem_resource import atomic_masses
from .distribution import SchulzZimm

EPSILON = 1e-300


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
    def __init__(self, stochastic_graph, rng=None):
        self.stochastic_graph = stochastic_graph.graph
        self.static_graph = self._build_static_graph()
        self.graph = None
        self.mw = [0]
        self.fully_generated = False
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()
        self._mw_draw_map = {}

        self._build_static_graph()

    def to_mol(self):
        mol = Chem.EditableMol(Chem.MolFromSmiles(""))
        for node in self.graph.nodes(data=True):
            atom = Chem.Atom(node[1]["atomic_num"])
            mol.AddAtom(atom)
        for edge in self.graph.edges(data=True):
            bond_type = Chem.BondType(edge[2]["bond_type"])
            mol.AddBond(edge[0], edge[1], bond_type)
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol

    def generate(self):
        self.graph = nx.Graph()
        start = self._find_start_source()
        if start is None:
            raise RuntimeError(
                "The graph does not contain a single source node. There is at least one start node, which reach all other nodes."
            )

        node_id = self._add_node(
            start, stochastic_allowed=True, termination_allowed=True, transition_allowed=True
        )

        transition_edge_options = True
        while transition_edge_options:
            self._fill_stochastic_edges(node_id)
            # do transition
            transition_edge_options = self._next_transition_edge()

            weights = [
                data[1]["transition_weight"] for data in transition_edge_options["transition_edges"]
            ]
            weights = np.asarray(weights) + EPSILON
            weights /= np.sum(weights)

            # No options for transitions means we finishing the generation
            if len(weights) == 0:
                break
            idx = self.rng.choice(
                range(len(transition_edge_options["transition_edges"])), p=weights
            )

            # Selected transition edge
            transition_edge = transition_edge_options["transition_edges"][idx]
            stochastic_graph_node_idx = transition_edge[0][1]
            new_bond_type = transition_edge[1]["bond_type"]

            # We chose one transition, all other are mood
            for node in self.graph:
                self.graph.nodes[node]["transition_edges"].clear()

            old_node_idx = transition_edge_options["node"]
            new_node_idx = self._add_node(
                stochastic_graph_node_idx,
                transition_allowed=False,
                termination_allowed=False,
                stochastic_allowed=False,
            )
            node_id = new_node_idx
            self.graph.add_edge(old_node_idx, new_node_idx, bond_type=new_bond_type)

    def _build_static_graph(self):
        static_graph = nx.Graph()
        for node in self.stochastic_graph.nodes(data=True):
            static_graph.add_node(node[0], **node[1])

        for edge in self.stochastic_graph.edges():
            edge_data_list = self.stochastic_graph.get_edge_data(*edge)
            valid_edge = None
            for edge_data in edge_data_list.values():
                if _is_static_edge(edge_data):
                    valid_edge = edge_data
                    break
            if valid_edge and (edge[1], edge[0]) not in static_graph.edges:
                static_graph.add_edge(*edge, **edge_data)
        return static_graph

    def _fill_static_edges(self, current_atom):
        stochastic_node = self.graph.nodes[current_atom]["stochastic_node"]
        static_tree = nx.dfs_tree(self.static_graph, source=stochastic_node)
        static_map = {self.graph.nodes[current_atom]["stochastic_node"]: current_atom}

        for node in static_tree:
            if self.graph.nodes[current_atom]["stochastic_node"] != node:
                local_node = self._add_node(
                    node, transition_allowed=True, termination_allowed=True, stochastic_allowed=True
                )
                static_map[node] = local_node

        for edge in self.static_graph.edges(static_map.keys(), data=True):
            atom_a = static_map[edge[0]]
            atom_b = static_map[edge[1]]
            bond_type = edge[2]["bond_type"]
            self.graph.add_edge(atom_a, atom_b, bond_type=bond_type)

    def _next_stochastic_edge(self):
        stochastic_edges = []
        node_weights = []
        for node in self.graph:
            node_data = self.graph.nodes[node]
            node_edges = node_data["stochastic_edges"]
            weights = [edge_tuple[1]["stochastic_weight"] for edge_tuple in node_edges]
            weight = np.sum(weights)
            if weight > 0:
                node_weights.append(weight)
                stochastic_edges.append({"node": node, "stochastic_edges": node_edges})

        if len(node_weights) == 0:
            return None

        node_weights = np.asarray(node_weights) + EPSILON
        node_weights /= np.sum(node_weights)
        idx = self.rng.choice(range(len(node_weights)), p=node_weights)

        return stochastic_edges[idx]

    def _fill_stochastic_edges(self, last_node_id):
        def get_target_mw(self, node_id, swap_self):
            last_node = self.graph.nodes[node_id]
            try:
                drawn_mw = self._mw_draw_map[(last_node["mw"], last_node["mn"])]
            except KeyError:
                distr = SchulzZimm(f"schulz_zimm({last_node['mw']}, {last_node['mn']})")
                drawn_mw = distr.draw_mw(self.rng)
                self._mw_draw_map[(last_node["mw"], last_node["mn"])] = drawn_mw
                swap_self._mw_draw_map[(last_node["mw"], last_node["mn"])] = drawn_mw
            return drawn_mw

        # do-while init

        self._fill_static_edges(last_node_id)
        next_stochastic = self._next_stochastic_edge()

        while next_stochastic:

            exempt_node = next_stochastic["node"]
            swap_self = self._terminate_graph(exempt_node)

            if self.mw[-1] < get_target_mw(self, exempt_node, swap_self):
                # Reverse the termination of the graph
                self.graph = swap_self.graph
                self.mw = swap_self.mw

                new_node = self._add_stochastic_connection(next_stochastic)
            else:
                # Target MW reached, so we leave the termination in place
                # But this also means, not more terminations and stochastic options are open
                for node in self.graph:
                    self.graph.nodes[node]["stochastic_edges"].clear()
                    self.graph.nodes[node]["termination_edges"].clear()
                break

            # do-while advance
            self._fill_static_edges(new_node)
            next_stochastic = self._next_stochastic_edge()

        # We finished generating this stochastic element of the graph, on to the next one, which counts its mw on its own
        self.mw += [0]

    def _add_stochastic_connection(self, next_stochastic):
        node = next_stochastic["node"]
        stochastic_edges = next_stochastic["stochastic_edges"]
        weights = [edge[1]["stochastic_weight"] for edge in stochastic_edges]
        weights = np.asarray(weights) + EPSILON
        weights /= np.sum(weights)
        idx = self.rng.choice(len(stochastic_edges), p=weights)
        edge = stochastic_edges[idx]

        # Since we will add a stochastic edge to the node, we remove all others.
        self.graph.nodes[node]["stochastic_edges"].clear()
        self.graph.nodes[node]["termination_edges"].clear()
        self.graph.nodes[node]["transition_edges"].clear()

        new_stochastic_node = edge[0][1]
        new_bond_type = edge[1]["bond_type"]
        new_node_idx = self._add_node(
            new_stochastic_node,
            termination_allowed=False,
            transition_allowed=False,
            stochastic_allowed=False,
        )
        self.graph.add_edge(node, new_node_idx, bond_type=new_bond_type)
        return new_node_idx

    def _next_termination_edge(self, exempt_node):
        for node in self.graph.nodes():
            if exempt_node is None or exempt_node != node:
                node_data = self.graph.nodes[node]
                edge_list = node_data["termination_edges"]
                if len(edge_list) > 0:
                    weights = [edge_tuple[1]["termination_weight"] for edge_tuple in edge_list]
                    weights = np.asarray(weights) + EPSILON
                    weights /= np.sum(weights)
                    idx = self.rng.choice(len(edge_list), p=weights)

                    edge = edge_list[idx][0]
                    bond_type = edge_list[idx][1]["bond_type"]

                    return {"node": node, "edge": edge, "bond_type": bond_type}

    def _terminate_graph(self, exempt_node):
        swap_graph = copy.deepcopy(self)

        edge_info = self._next_termination_edge(exempt_node)
        while edge_info:
            node = edge_info["node"]
            new_node = edge_info["edge"][1]

            last_node_id = self._add_node(
                new_node,
                transition_allowed=False,
                termination_allowed=False,
                stochastic_allowed=False,
            )
            self.graph.add_edge(node, last_node_id, bond_type=edge_info["bond_type"])

            node_data = self.graph.nodes[node]
            # Since we are fulfilling this termination, we clear the node
            node_data["termination_edges"].clear()
            node_data["stochastic_edges"].clear()
            node_data["transition_edges"].clear()

            edge_info = self._next_termination_edge(exempt_node)

        return swap_graph

    def _next_transition_edge(self):
        transition_edges = []
        node_weights = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_edges = node_data["transition_edges"]
            weights = [edge_tuple[1]["transition_weight"] for edge_tuple in node_edges]
            weight = np.sum(weights)
            node_weights.append(weight)
            transition_edges.append({"node": node, "transition_edges": node_edges})
        if len(node_weights) == 0:
            return None

        node_weights = np.asarray(node_weights) + EPSILON
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
        self,
        node,
        transition_allowed,
        termination_allowed,
        stochastic_allowed,
    ):
        node_id = len(self.graph)
        edges = self.stochastic_graph.out_edges(node, data=True)
        stochastic_edges = []
        transition_edges = []
        termination_edges = []
        for full_edge in edges:
            edge = (full_edge[0], full_edge[1])
            edge_data = full_edge[2]
            if _is_transition_edge(edge_data) and transition_allowed:
                transition_edges += [(edge, edge_data)]
            if _is_termination_edge(edge_data) and termination_allowed:
                termination_edges += [(edge, edge_data)]
            if _is_stochastic_edge(edge_data) and stochastic_allowed:
                stochastic_edges += [(edge, edge_data)]

        node_data = self.stochastic_graph.nodes[node]
        self.graph.add_node(
            node_id,
            atomic_num=node_data["atomic_num"],
            stochastic_node=node,
            mn=node_data["mn"],
            mw=node_data["mw"],
            transition_edges=transition_edges,
            termination_edges=termination_edges,
            stochastic_edges=stochastic_edges,
        )
        mw = atomic_masses[node_data["atomic_num"]]
        self.mw[-1] += mw

        return node_id
