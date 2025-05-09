import networkx as nx

import gbigsmiles


def node_match(node1_attrs, node2_attrs):
    return node1_attrs == node2_attrs


# Define a custom edge matcher
def edge_match(edge1_attrs, edge2_attrs):
    return edge1_attrs == edge2_attrs


def test_generating_ml_graph_generation(graph_validation_dict):
    for big_smi in graph_validation_dict:
        print(big_smi)
        big = gbigsmiles.BigSmiles.make(big_smi)
        gen_graph = big.get_generating_graph()
        ml_graph = gen_graph.get_ml_graph(include_bond_descriptors=False)
        assert nx.is_isomorphic(ml_graph, graph_validation_dict[big_smi], node_match=node_match, edge_match=edge_match)

        dot_string_A = gen_graph.get_dot_string(include_bond_descriptors=True)
        dot_string_B = gen_graph.get_dot_string(include_bond_descriptors=False, node_prefix="bd-")
        dot_string_A = dot_string_A[: dot_string_A.rfind("}")]
        dot_string_B = dot_string_B[len("digraph{") :]
        dot_string = dot_string_A + dot_string_B
        assert len(dot_string) > 0
