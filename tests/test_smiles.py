import networkx as nx
import numpy as np
import pysmiles
import pytest

import gbigsmiles


def test_smiles_parsing(chembl_smi_list):
    for smi in chembl_smi_list:
        if len(smi) > 0:
            smiles_instance = gbigsmiles.Smiles.make(smi)
            assert smi == smiles_instance.generate_string(True)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_smiles_weight(n, chembl_smi_list):
    rng = np.random.default_rng()
    no_dot_smi = []
    for smi in chembl_smi_list:
        if "." not in smi and len(smi) > 0:
            no_dot_smi.append(smi)

    for i in range(len(no_dot_smi) // n - 1):
        smis = no_dot_smi[i * n : (i + 1) * n]
        system_string = ""
        total_mw = 0.0
        for smi in smis:
            molw = np.round(rng.uniform(1.0, 1e5), 1)
            system_string += f"{smi}.|{molw}|"
            total_mw += molw
        print(system_string)
        big_smiles = gbigsmiles.BigSmiles.make(system_string)
        for mol in big_smiles.mol_molecular_weight_map:
            print("x", mol, big_smiles.mol_molecular_weight_map[mol])
        assert abs(total_mw - big_smiles.total_system_molecular_weight) < 1e-6


def test_smiles_graph(chembl_smi_list):

    def node_match(gb_node, pysmi_node):
        return_value = True
        if str(gb_node["obj"].symbol).upper() != pysmi_node["element"].upper():
            return_value = False
        if gb_node["obj"].charge != pysmi_node["charge"]:
            return_value = False
        if gb_node["obj"].aromatic != pysmi_node["aromatic"]:
            return_value = False

        # print(
        #     "node_match",
        #     gb_node,
        #     str(gb_node["obj"]),
        #     gb_node["obj"].aromatic,
        #     pysmi_node,
        #     return_value,
        #     gb_node["obj"].aromatic != pysmi_node["aromatic"],
        # )
        return return_value

    def edge_match(gb_edge, pysmi_edge):
        return_value = True

        bond_type = gb_edge.get("bond_type", None)
        if bond_type is None and pysmi_edge["order"] != 1:
            if not (gb_edge.get("aromatic", False) and pysmi_edge["order"] == 1.5):
                return_value = False
        if str(bond_type) == "=" and pysmi_edge["order"] != 2:
            return_value = False
        if str(bond_type) == "#" and pysmi_edge["order"] != 3:
            return_value = False

        # print("edge match", gb_edge, pysmi_edge, return_value)
        return return_value

    for smi in chembl_smi_list:
        big_smiles = gbigsmiles.BigSmiles.make(smi)

        pysmiles_graph = pysmiles.read_smiles(smi, reinterpret_aromatic=False)

        # PYSMILES and us treat hydrogen differently
        for node in list(pysmiles_graph.nodes(data=True)):
            if "H" in node[1]["element"]:
                pysmiles_graph.remove_node(node[0])

        # Pysmiles adds 0 order bonds to the graph, but we do not.
        for edge in list(pysmiles_graph.edges(data=True)):
            if edge[2]["order"] == 0:
                pysmiles_graph.remove_edge(edge[0], edge[1])

        graph = big_smiles.generating_graph
        graph = nx.Graph(nx.to_undirected(graph))

        # Remove hydrogens from bigsmiles graph for comparison:
        for node in list(graph.nodes(data=True)):
            if "H" == str(node[1]["obj"].symbol):
                graph.remove_node(node[0])

        print("\n", smi, pysmiles_graph, graph, "\n")
        assert nx.is_isomorphic(graph, pysmiles_graph, node_match=node_match, edge_match=edge_match)
