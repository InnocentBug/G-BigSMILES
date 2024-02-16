import copy

import networkx as nx
import numpy as np
from rdkit import Chem

from .chem_resource import atomic_masses
from .distribution import SchulzZimm


def _is_weight_edge(edge):
    if edge["weight"] != 0:
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


class AtomGraph:
    def __init__(self, stochastic_graph, rng_seed=None):
        self.stochastic_graph = stochastic_graph
        self.atom_graph = nx.Graph()
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

    def generate(self):

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
            transition_edge = transition_edge_options["transition_edges"][idx]
            stochastic_graph_node_idx = transition_edge[0][1]
            new_bond_type = transition_edge[1]["bond_type"]

            self._clear_atom_graph_open_bond_descriptors()
            old_node_idx = transition_edge_options["node"]
            new_node_idx = self._add_node(
                stochastic_graph_node_idx,
                transition_edge_allowed=False,
                termination_edge_allowed=False,
            )
            self.atom_graph.add_edge(old_node_idx, new_node_idx, bond_type=new_bond_type)

    def _fill_static_edges(self):
        node_id = None
        edge_tuple = self._next_static_edges()
        while edge_tuple is not None:
            node, edge, bond_type = edge_tuple
            node_id = self._add_node(edge[1], weight_edge=edge)
            self.atom_graph.add_edge(node, node_id, bond_type=bond_type)

            edge_tuple = self._next_static_edges()
        return node_id

    def _fill_stochastic_edges(self):
        next_stochastic = True
        while next_stochastic:
            last_node_id = self._fill_static_edges()

            next_stochastic = self._next_stochastic_edges()
            if next_stochastic is None:
                excempt_node = None
            else:
                excempt_node = next_stochastic["node"]
            swap_atom_graph, last_node_id = self._terminate_graph(last_node_id, excempt_node)
            if last_node_id is None:
                break
            if next_stochastic:
                # Draw distribution etc.
                last_node = self.atom_graph.nodes[last_node_id]

                mw = last_node["mw"]
                mn = last_node["mn"]
                try:

                    drawn_mw = self._mw_draw_map[(mw, mn)]
                except KeyError:
                    distr = SchulzZimm(f"schulz_zimm({mw}, {mn})")
                    drawn_mw = distr.draw_mw(self.rng)
                    self._mw_draw_map[(mw, mn)] = drawn_mw
                    swap_atom_graph._mw_draw_map = self._mw_draw_map

                if not self.mw[-1] >= drawn_mw:
                    # Reverse the termination of the graph
                    self = swap_atom_graph
                    self._add_stochastic_connection(next_stochastic)
                else:
                    break
            else:
                break
                # We finished one element of the graph, on to the next one, which counts its mw on its own
        self.mw += [0]

    def _add_stochastic_connection(self, next_stochastic):
        node = next_stochastic["node"]
        stochastic_edges = next_stochastic["stochastic_edges"]
        weights = [edge[1]["weight"] for edge in stochastic_edges]
        weights = np.asarray(weights)
        weights /= np.sum(weights)
        idx = self.rng.choice(len(stochastic_edges), p=weights)
        edge = stochastic_edges[idx]

        # Since we will add a stochastic edge to the node, we remove all others.
        self.atom_graph.nodes[node]["weight_edges"].clear()

        new_stochastic_node = edge[0][1]
        new_bond_type = edge[1]["bond_type"]
        new_node_idx = self._add_node(
            new_stochastic_node,
            edge[0],
            termination_edge_allowed=False,
            transition_edge_allowed=False,
        )
        self.atom_graph.add_edge(node, new_node_idx, bond_type=new_bond_type)

    def _terminate_graph(self, last_node_id, exempt_node):

        swap_atom_graph = copy.deepcopy(self)

        edge_tuple = self._next_termination_edges(exempt_node)
        while edge_tuple is not None:
            node, edge_options = edge_tuple
            weights = []
            for edge_tuple in edge_options:
                (edge, bond_type, weight) = edge_tuple
                weights.append(weight)

            weights = np.asarray(weights)
            weights /= np.sum(weights)
            idx = self.rng.choice(range(len(weights)), p=weights)

            edge, bond_type, weight = edge_options[idx]

            node_id = self._add_node(
                edge[1], transition_edge_allowed=False, termination_edge_allowed=False
            )
            last_node_id = node_id
            self.atom_graph.add_edge(node, node_id, bond_type=bond_type)

            edge_tuple = self._next_termination_edges(exempt_node)

        return swap_atom_graph, last_node_id

    def _next_static_edges(self):
        for node in self.atom_graph.nodes():
            node_data = self.atom_graph.nodes[node]
            try:
                weight_edges = node_data["weight_edges"]
            except IndexError:
                continue
            else:
                for i, edge_tuple in enumerate(weight_edges):
                    weight = edge_tuple[1]["weight"]
                    if weight < 0:
                        edge = edge_tuple[0]
                        bond_type = edge_tuple[1]["bond_type"]
                        del node_data["weight_edges"][i]
                        return node, edge, bond_type
        return None

    def _next_stochastic_edges(self):
        stochastic_edges = []
        node_weights = []
        for node in self.atom_graph.nodes():
            node_data = self.atom_graph.nodes[node]
            node_edges = node_data["weight_edges"]
            weights = [edge_tuple[1]["weight"] for edge_tuple in node_edges]
            weight = np.sum(weights)
            if weight > 0:
                node_weights.append(weight)
                stochastic_edges.append({"node": node, "stochastic_edges": node_edges})
        if len(node_weights) == 0:
            return None
        node_weights = np.asarray(node_weights)
        node_weights /= np.sum(node_weights)
        idx = self.rng.choice(range(len(node_weights)), p=node_weights)

        return stochastic_edges[idx]

    def _next_termination_edges(self, exempt_node):
        for node in self.atom_graph.nodes():
            if exempt_node is None or exempt_node != node:
                node_data = self.atom_graph.nodes[node]
                edge_list = []
                while True:
                    try:
                        edge_tuple = node_data["termination_edges"].pop()
                    except IndexError:
                        break
                    edge = edge_tuple[0]
                    bond_type = edge_tuple[1]["bond_type"]
                    weight = edge_tuple[1]["termination_weight"]
                    edge_list.append((edge, bond_type, weight))
            if len(edge_list) > 0:
                return node, edge_list

        return None

    def _next_transition_edges(self):
        transition_edges = []
        node_weights = []
        for node in self.atom_graph.nodes():
            node_data = self.atom_graph.nodes[node]
            node_edges = node_data["transition_edges"]
            weights = [edge_tuple[1]["transition_weight"] for edge_tuple in node_edges]
            weight = np.sum(weights)
            if weight > 0:
                node_weights.append(weight)
                transition_edges.append({"node": node, "transition_edges": node_edges})
        if len(node_weights) == 0:
            return None
        node_weights = np.asarray(node_weights)
        node_weights /= np.sum(node_weights)
        idx = self.rng.choice(range(len(node_weights)), p=node_weights)

        return transition_edges[idx]

    def _clear_atom_graph_open_bond_descriptors(self):
        for node in self.atom_graph:
            node_data = self.atom_graph.nodes[node]
            node_data["weight_edges"].clear()
            node_data["transition_edges"].clear()
            node_data["termination_edges"].clear()

    def _find_start_source(self):
        for node in self.stochastic_graph:
            tree = nx.dfs_tree(self.stochastic_graph, source=node)
            # If we can reach the entire graph from with following the edges, it is suitable to span the entire molecule
            if len(tree) == len(self.stochastic_graph):
                return node

    def _add_node(
        self, node, weight_edge=None, transition_edge_allowed=True, termination_edge_allowed=True
    ):
        node_id = len(self.atom_graph)
        stochastic_edges = self.stochastic_graph.out_edges(node, data=True)
        weight_edges = []
        transition_edges = []
        termination_edges = []
        for full_edge in stochastic_edges:
            edge = (full_edge[0], full_edge[1])
            edge_data = full_edge[2]

            data = edge_data
            if _is_weight_edge(data):
                if weight_edge is None or (weight_edge[1], weight_edge[0]) != edge:
                    weight_edges += [(edge, data)]
            if _is_transition_edge(data):
                if transition_edge_allowed:
                    transition_edges += [(edge, data)]
            if _is_termination_edge(data):
                if termination_edge_allowed:
                    termination_edges += [(edge, data)]

        node_data = self.stochastic_graph.nodes[node]
        self.atom_graph.add_node(
            node_id,
            atomic_num=node_data["atomic_num"],
            mn=node_data["mn"],
            mw=node_data["mw"],
            weight_edges=weight_edges,
            transition_edges=transition_edges,
            termination_edges=termination_edges,
        )
        mw = atomic_masses[node_data["atomic_num"]]
        self.mw[-1] += mw

        return node_id

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
