import networkx as nx


def find_start_source(stochastic_graph):
    for node in stochastic_graph:
        tree = nx.dfs_tree(stochastic_graph, source=node)
        if len(tree) == len(stochastic_graph):
            return node


def add_node(node, atom_graph, stochastic_graph):
    atom_graph.add_node(node["atomic_num"], stochastic_edges=stochastic_graph.out_edges(node))


def generate_full_mol_graph(stochastic_graph):
    start = find_start_source(stochastic_graph)
    atom_graph = nx.Graph()
