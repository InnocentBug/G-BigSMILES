#!/usr/bin/env python

# import matplotlib.pyplot as plt

# import pydot

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

import gbigsmiles
from gbigsmiles.graph_generate import AtomGraph


def test_mirror(bigA):
    molA = gbigsmiles.Molecule(bigA)
    print(bigA)
    print(molA)
    print(molA.generate().smiles)
    molB = molA.gen_mirror()
    print(molB)
    print(molB.generate().smiles)


def test_prob(bigA):
    mol = gbigsmiles.Molecule(bigA)
    smi = mol.generate().smiles
    print(bigA)
    print(smi)
    prob, matches = gbigsmiles.mol_prob.get_ensemble_prob(smi, mol)
    print(prob)
    print(matches)


def gen_calc_prob(big):
    mol = gbigsmiles.Molecule(big)
    mol_gen = mol.generate()
    calc_prob, matches = gbigsmiles.mol_prob.get_ensemble_prob(mol_gen.smiles, mol)
    print(mol_gen.smiles, calc_prob)


bigA = "{[][<]C(N)C[>]; [<][H][>]}|uniform(500, 600)|{[<][<]C(=O)C[>]; [>][H][]}|uniform(500, 600)|"
bigA = "CCO{[<][<]C(N)C[>][>]}|uniform(500, 600)|{[<][<]C(=O)C[>][>]}|uniform(500, 600)|CCN"
# test_mirror(bigA)
# bigA = "CCO"
# test_prob(bigA)
bigA = "{[][<]C(N)C[>]; [<][H], [>]CO []}|uniform(560, 600)|"
# test_prob(bigA)

bigA = "{[] CC([<])=NCC[>], [$2]CC(=O)C([<])CC[$2]; [F][>], CCO[<], [H][$2][]}|uniform(100, 150)|"
bigA = "OCC{[$] [$]C(N)C[$], [$]CC(C(=O)C[$2])C[$], [$2]CCC[$2] ;[H][$], [Si][$2] [$]}|gauss(500, 10)|CCN"
bigA = "OC{[>] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0 0|])C[>]; [<][H], [<]C [<]}|schulz_zimm(750, 600)|COOC{[<] [<]COC[>], [<]C(ON)C[>] [>]}|schulz_zimm(200, 150)|{[<] [<]COCOC[>], [<]CONOC[>] [>]}|schulz_zimm(170, 150)|F"
bigA = "OC{[<] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0|])C[>]; [<][H] [>]}|schulz_zimm(750, 600)|C=O.|5000|"

bigA = "CCC(C){[>][<]CC([>])c1ccccc1[<]}|schulz_zimm(2000, 1800)|{[>][<]CC([>])C(=O)OC[<]}|schulz_zimm(1000, 900)|[H]"
bigA = "CCC(C){[>][<]CC([>])c1ccccc1[<]}|schulz_zimm(1000, 900)|{[>][<]CC([>])C(=O)OC [<]}|schulz_zimm(500, 450)|"
# bigA = "CCC(C){[>][<]CC([>])c1ccccc1, [<]CC([>])C(=O)OC [<]}|schulz_zimm(1500, 925)|"

# bigA = "F{[<] [>]CC[<] [>]}|uniform(0, 20)|[H]"
# gen_calc_prob(bigA)
# print("ASD")
# bigA = "F{[<] [<]CC[>] [>]}|uniform(0, 20)|[H]"
# gen_calc_prob(bigA)

bigA = "CCOC{[$] O([<|3|])(C([$])C[$]), [>]CCO[<|0 0 0 1 0 2|] ; [>][H] [$]}|poisson(900)|CCCC"
bigA = "CCOC{[$] C([<|0.3|])(C([$])C[$]), [>|0.2|]C=CCc1ccccc1[<|0 0 0 0.1 0 0.2|] ; [>][H] [$]}|schulz_zimm(900, 800)|N"
bigA = "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(5000, 4100)|[<]CC.|60000|"
# bigA = "OC{[>] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0 0|])C[>]; [<][H], [<]C [<]}|schulz_zimm(5000, 4500)|COOC{[<] [<]COC[>], [<]C(ON)C[>] [>]}|schulz_zimm(5000, 4500)|{[<] [<]COCOC[>], [<]CONOC[>] [>]}|schulz_zimm(1700, 1500)|F"
bigA = "OC{[>] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0 0|])C[>]; [<][H], [<]C [<]}|schulz_zimm(5000, 4500)|COOC{[<] [<]COC[>], [<]C(ON)C[>] [>]}|schulz_zimm(5000, 4500)|{[<] [<]COCOC[>], [<]CONOC[>] [>]}|schulz_zimm(1700, 1500)|F"


mol = gbigsmiles.Molecule(bigA)

stochastic_atom_graph = mol.gen_stochastic_atom_graph(expect_schulz_zimm_distribution=True)
graph_dot = gbigsmiles.core.stochastic_atom_graph_to_dot_string(stochastic_atom_graph)
print(stochastic_atom_graph.graph)
with open("stochastic_atom_graph.dot", "w") as filehandle:
    filehandle.write(graph_dot)

rng = np.random.default_rng(45)
full_graph = AtomGraph(stochastic_atom_graph, rng=rng)
full_graph.generate()
print(full_graph.mw, full_graph._mw_draw_map, full_graph.graph)
atom_dot = gbigsmiles.core.molecule_atom_graph_to_dot_string(full_graph)
with open("atom_graph.dot", "w") as filehandle:
    filehandle.write(atom_dot)

rd_mol = full_graph.to_mol()
print(Chem.MolToSmiles(rd_mol))
print(Chem.Descriptors.HeavyAtomMolWt(rd_mol), np.sum(full_graph.mw))

molSize = (2000, 1000)
my_mol = rd_mol
AllChem.EmbedMolecule(my_mol)
mc = Chem.Mol(my_mol.ToBinary())
rdDepictor.Compute2DCoords(mc)
drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
drawer.DrawMolecule(mc)
drawer.FinishDrawing()
svg = drawer.GetDrawingText()
with open("rd_mol.svg", "w") as filehandle:
    filehandle.write(svg)


# mol_gen = mol.generate()
# print(mol_gen.smiles)

# molSize = (2000, 1000)
# my_mol = rd_mol
# AllChem.EmbedMolecule(my_mol)
# mc = Chem.Mol(my_mol.ToBinary())
# rdDepictor.Compute2DCoords(mc)
# drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
# drawer.DrawMolecule(mc)
# drawer.FinishDrawing()
# svg = drawer.GetDrawingText()
# with open("molPlay.svg", "w") as filehandle:
#     filehandle.write(svg)
# # calc_prob, matches = gbigsmiles.mol_prob.get_ensemble_prob(mol_gen.smiles, mol)
# # print(calc_prob)

# print(mol.generate_string(True))
# graph = mol.gen_reaction_graph()
# graph_dot = gbigsmiles.reaction_graph_to_dot_string(graph, mol)

# with open("graph.dot", "w") as filehandle:
#     filehandle.write(graph_dot)
