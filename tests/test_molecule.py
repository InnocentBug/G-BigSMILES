# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy

import numpy as np
import pytest
from rdkit import Chem

import gbigsmiles

test_args = [
    (
        "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}CC{[>][<]CC([>])c1ccccc1[<]}C(C)CC(c1ccccc1)c1ccccc1",
        "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1",
        "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1",
    ),
    (
        "{[] CC([$])=NCC[$]; [H][$][]}|schulz_zimm(1000, 900)|",
        "{[]CC([$])=NCC[$]; [H][$][]}",
        "{[]CC([$])=NCC[$]; [H][$][]}|schulz_zimm(1000.0, 900.0)|",
    ),
    (
        "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(1500, 1400)|[<]CC.|60000|",
        "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC.",
        "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(1500.0, 1400.0)|[<]CC.|60000.0|",
    ),
    (
        "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(1500, 1000)|CC{[>][>]CC([<])c1ccccc1[<]}|schulz_zimm(1500, 1000)|C(C)CC(c1ccccc1)c1ccccc1.|60000|",
        "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][>]CC([<])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
        "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(1500.0, 1000.0)|[<]CC[>|0.0|]{[>][>]CC([<])c1ccccc1[<]}|schulz_zimm(1500.0, 1000.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|60000.0|",
    ),
    (
        "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}CC{[>][<]CC([>])c1ccccc1[<]}|gauss(500, 10)|C(C)CC(c1ccccc1)c1ccccc1.|60000|",
        "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
        "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|gauss(500.0, 10.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|60000.0|",
    ),
    (
        "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(1500, 1000)|CC{[>][<]CC([>])c1ccccc1[<]}|schulz_zimm(1500, 1000)|C(C)CC(c1ccccc1)c1ccccc1.|50%|",
        "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
        "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(1500.0, 1000.0)|[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|schulz_zimm(1500.0, 1000.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|50.0%|",
    ),
    (
        "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}|schulz_zimm(1000, 950)|CC{[>][<]CC([>])c1ccccc1[<]}|schulz_zimm(500, 400)|C(C)CC(c1ccccc1)c1ccccc1.|5e7|",
        "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
        "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}|schulz_zimm(1000.0, 950.0)|[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|schulz_zimm(500.0, 400.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|50000000.0|",
    ),
    (
        "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N][<]}|schulz_zimm(1000, 450)|{[>][<]CC([>])c1ccccc1; [>]N, [<][H][]}|schulz_zimm(400, 300)|.|5e7|",
        "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N][<]}{[>][<]CC([>])c1ccccc1; [>]N, [<][H][]}.",
        "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N][<]}|schulz_zimm(1000.0, 450.0)|{[>][<]CC([>])c1ccccc1; [>]N, [<][H][]}|schulz_zimm(400.0, 300.0)|.|50000000.0|",
    ),
    (
        "OC{[>] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0 0|])C[>]; [<][H], [<]C [<]}|schulz_zimm(5000, 4500)|COOC{[<] [<]COC[>], [<]C(ON)C[>] [>]}|schulz_zimm(5000, 4500)|{[<] [<]COCOC[>], [<]CONOC[>] [>]}|schulz_zimm(1700, 1500)|F",
        "OC[>]{[>][<]CC[>], [<]C(N[>])C[>]; [<][H], [<]C[<]}[<]COOC[<]{[<][<]COC[>], [<]C(ON)C[>][>]}{[<][<]COCOC[>], [<]CONOC[>][>]}[>]F",
        "OC[>|0.0|]{[>][<]CC[>], [<|0.5|]C(N[>|0.1 0.0 0.0 0.0 0.0 0.0 0.0|])C[>]; [<][H], [<]C[<]}|schulz_zimm(5000.0, 4500.0)|[<]COOC[<|0.0|]{[<][<]COC[>], [<]C(ON)C[>][>]}|schulz_zimm(5000.0, 4500.0)|{[<][<]COCOC[>], [<]CONOC[>][>]}|schulz_zimm(1700.0, 1500.0)|[>]F",
    ),
    (
        "OC{[>] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0 0|])C[>]; [<][H], [<]C [<]}|schulz_zimm(5000, 4000)|COOC{[<] [<]COC[>], [<]C(ON)C[>] [>]}|schulz_zimm(5000, 4500)|{[<] [<]COCOC[>], [<]CONOC[>] [>]}|schulz_zimm(1700, 1600)|F",
        "OC[>]{[>][<]CC[>], [<]C(N[>])C[>]; [<][H], [<]C[<]}[<]COOC[<]{[<][<]COC[>], [<]C(ON)C[>][>]}{[<][<]COCOC[>], [<]CONOC[>][>]}[>]F",
        "OC[>|0.0|]{[>][<]CC[>], [<|0.5|]C(N[>|0.1 0.0 0.0 0.0 0.0 0.0 0.0|])C[>]; [<][H], [<]C[<]}|schulz_zimm(5000.0, 4000.0)|[<]COOC[<|0.0|]{[<][<]COC[>], [<]C(ON)C[>][>]}|schulz_zimm(5000.0, 4500.0)|{[<][<]COCOC[>], [<]CONOC[>][>]}|schulz_zimm(1700.0, 1600.0)|[>]F",
    ),
]


@pytest.mark.parametrize(("text", "big", "ref"), test_args)
def test_molecule(text, big, ref):

    global_rng = np.random.Generator(np.random.MT19937(42))
    test = global_rng.uniform()
    assert test == 0.5419938930062744

    rng = copy.deepcopy(global_rng)
    mol = gbigsmiles.Molecule(text)
    assert str(mol) == ref
    assert mol.generate_string(False) == big

    mol.gen_mirror()

    schulz_zimm_distribution = True
    # Check distribution possibility
    for element in mol.elements:
        if isinstance(element, gbigsmiles.stochastic.Stochastic):
            if not isinstance(element.distribution, gbigsmiles.distribution.SchulzZimm):
                schulz_zimm_distribution = False
                break

    stochastic_graph = mol.gen_stochastic_atom_graph(schulz_zimm_distribution)
    if schulz_zimm_distribution:
        full_atom_graph = gbigsmiles.AtomGraph(stochastic_graph, rng=rng)
        full_atom_graph.generate()
        Chem.MolToSmiles(full_atom_graph.to_mol())
        # assert graph_smi == graph_gen

    if mol.generable:
        mol.generate(rng=copy.deepcopy(rng))
        mol.gen_reaction_graph()


if __name__ == "__main__":
    test_molecule()
