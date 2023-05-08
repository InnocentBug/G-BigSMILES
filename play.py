#!/usr/bin/env python

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydot
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D

import bigsmiles_gen


def test_mirror(bigA):
    molA = bigsmiles_gen.Molecule(bigA)
    print(bigA)
    print(molA)
    print(molA.generate().smiles)
    molB = molA.gen_mirror()
    print(molB)
    print(molB.generate().smiles)


def test_prob(bigA):
    mol = bigsmiles_gen.Molecule(bigA)
    smi = mol.generate().smiles
    print(bigA)
    print(smi)
    prob, matches = bigsmiles_gen.mol_prob.get_ensemble_prob(smi, mol)
    print(prob)
    print(matches)


bigA = "{[][<]C(N)C[>]; [<][H][>]}|uniform(500, 600)|{[<][<]C(=O)C[>]; [>][H][]}|uniform(500, 600)|"
bigA = "CCO{[<][<]C(N)C[>][>]}|uniform(500, 600)|{[<][<]C(=O)C[>][>]}|uniform(500, 600)|CCN"
# test_mirror(bigA)
# bigA = "CCO"
# test_prob(bigA)
bigA = "{[][<]C(N)C[>]; [<][H], [>]CO []}|uniform(560, 600)|"
# test_prob(bigA)

bigA = "{[] CC([<])=NCC[>], [$2]CC(=O)C([<])CC[$2]; [F][>], CCO[<], [H][$2][]}|uniform(100, 150)|"
bigA = "OCC{[$] [$]C(N)C[$], [$]CC(C(=O)C[$2])C[$], [$2]CCC[$2] ;[H][$], [Si][$2] [$]}|gauss(500, 10)|CCN"
bigA = "OC{[>] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0 0|])C[>]; [<][H], [<]C [<]}|flory_schulz(5e-3)|COOC{[<] [<]COC[>], [<]C(ON)C[>] [>]}|schulz_zimm(200, 150)|{[<] [<]COCOC[>], [<]CONOC[>] [>]}|schulz_zimm(170, 150)|F"
bigA = (
    "OC{[<] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0|])C[>]; [<][H] [>]}|flory_schulz(5e-3)|C=O.|5000|"
)

mol = bigsmiles_gen.Molecule(bigA)
mol_gen = mol.generate()
print(mol_gen.smiles)
molSize = (450, 150)
mc = Chem.Mol(mol_gen.mol.ToBinary())
drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
drawer.DrawMolecule(mc)
drawer.FinishDrawing()
svg = drawer.GetDrawingText()
with open(f"molPlay.svg", "w") as filehandle:
    filehandle.write(svg)

print(mol.generate_string(True))
graph = mol.gen_reaction_graph()


def graph_dot(graph):
    dot_str = "strict digraph { \n"
    for node in graph.nodes():
        try:
            dot_str += (
                f"\"{hash(node)}\" [label=\"{graph.nodes[node]['smiles']}\", color=orange];\n"
            )
        except:
            dot_str += f"\"{hash(node)}\" [label=\"{node.generate_string(False)} {graph.nodes[node]['atom']}\", color=green];\n"

    name_map = {"term_prob": "pt", "trans_prob": "pa", "weight": "w", "prob": "p"}
    for edge in graph.edges():
        edge_data = graph.get_edge_data(*edge)
        for name in name_map:
            if name in edge_data:
                value = edge_data[name]
                edge_label = f"{name_map[name]} = {np.round(value ,2)}"
        dot_str += f'{hash(edge[0])} -> {hash(edge[1])} [label="{edge_label}"];\n'

    dot_str += "}\n"

    return dot_str


with open("graph.dot", "w") as filehandle:
    filehandle.write(graph_dot(graph))


# for i in range(100):
#     smi = mol.generate().smiles
#     print(smi)
#     # print(bigsmiles_gen.mol_prob.get_ensemble_prob(smi, mol)[0])
