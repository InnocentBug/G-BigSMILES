import copy

import networkx as nx
import numpy as np

from .chem_resource import atomic_masses


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

        start = self._find_start_source()
        if start is None:
            raise RuntimeError(
                "The graph does not contain a single source node. There is at least one start node, which reach all other nodes."
            )

        node_id = self._add_node(start)

    def generate(self):

        next_stochastic = [True]
        while next_stochastic:
            last_node_id = self._fill_static_edges()

            new_node = self.atom_graph.nodes[last_node_id]

            next_stochastic = self._next_stochastic_edges()
            swap_atom_graph, last_node_id = self._terminate_graph(last_node_id)
            if next_stochastic:
                # Draw distribution etc.
                pass
            else:
                break

        # do transition

        print(next_stochastic)
        print(new_node)

    def _fill_static_edges(self):
        edge_tuple = self._next_static_edges()
        while edge_tuple is not None:
            node, edge, bond_type = edge_tuple

            node_id = self._add_node(edge[1], weight_edge=edge)
            self.atom_graph.add_edge(node, node_id, bond_type=bond_type)

            edge_tuple = self._next_static_edges()
        return node_id

    def _terminate_graph(self, last_node_id):

        swap_atom_graph = copy.deepcopy(self)

        edge_tuple = self._next_termination_edges()
        while edge_tuple is not None:
            node, edge_options = edge_tuple
            print("xkcd", edge_options)
            weights = []
            for edge_tuple in edge_options:
                (edge, bond_type, weight) = edge_tuple
                weights.append(weight)
            print(weights)
            weights = np.asarray(weights)
            weights /= np.sum(weights)
            idx = rng.choice(range(len(weights)), p=weights)

            edge, bond_type, weight = edge_options[idx]

            node_id = self._add_node(edge[1], weight_edge=edge)
            last_node_id = node_id
            self.atom_graph.add_edge(node, node_id, bond_type=bond_type)

            edge_tuple = self._next_termination_edges()

        return swap_atom_graph, last_node_id

    def _next_static_edges(self):
        for node in self.atom_graph.nodes():
            node_data = self.atom_graph.nodes[node]
            try:
                edge_tuple = node_data["weight_edges"].pop()
                edge = edge_tuple[0]
                bond_type = edge_tuple[1]["bond_type"]
                weight = edge_tuple[1]["weight"]
                if weight > 0:
                    node_data["weight_edges"].push(edge_tuple)
                else:
                    return node, edge, bond_type
            except IndexError:
                pass

        return None

    def _next_stochastic_edges(self):
        stochastic_edges = []
        for node in self.atom_graph.nodes():
            node_data = self.atom_graph.nodes[node]
            node_edges = node_data["weight_edges"]
            weights = [edge_tuple[1]["weight"] for edge_tuple in node_edges]
            weight = np.sum(weights)
            stochastic_edges.append(
                {"node": node, "stochastic_edges": node_edges, "weight": weight}
            )
        return stochastic_edges

    def _next_termination_edges(self):
        for node in self.atom_graph.nodes():
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

    def _find_start_source(self):
        for node in self.stochastic_graph:
            tree = nx.dfs_tree(self.stochastic_graph, source=node)
            # If we can reach the entire graph from with following the edges, it is suitable to span the entire molecule
            if len(tree) == len(self.stochastic_graph):
                return node

    def _add_node(self, node, weight_edge=None, transition_edge=None, termination_edge=None):
        node_id = len(self.atom_graph)
        stochastic_edges = self.stochastic_graph.out_edges([node])
        weight_edges = []
        transition_edges = []
        termination_edges = []
        for edge in stochastic_edges:
            edge_data = self.stochastic_graph.get_edge_data(*edge)
            for idx in edge_data:
                data = edge_data[idx]
                if _is_weight_edge(data):
                    if weight_edge is None or (weight_edge[1], weight_edge[0]) != edge:
                        weight_edges += [(edge, data)]
                if _is_transition_edge(data):
                    if transition_edge is None or (transition_edge[1], transition_edge[0]) != edge:
                        transition_edges += [(edge, data)]
                if _is_termination_edge(data):
                    if (
                        termination_edge is None
                        or (termination_edge[1], termination_edge[0]) != edge
                    ):

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
