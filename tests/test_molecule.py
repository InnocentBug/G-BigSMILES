# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy

import numpy as np

import bigsmiles_gen


def test_molecule():
    test_args = [
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}CC{[>][<]CC([>])c1ccccc1[<]}C(C)CC(c1ccccc1)c1ccccc1",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1",
            None,
            None,
        ),
        (
            "{[] CC([$])=NCC[$]; [H][$][]}|schulz_zimm(1000, 900)|",
            "{[]CC([$])=NCC[$]; [H][$][]}",
            "{[]CC([$])=NCC[$]; [H][$][]}|schulz_zimm(1000.0, 900.0)|",
            "[H]CCN=C(C)C(C)=NCCCCN=C(C)CCN=C(C)C(C)=NCCCCN=C(C)CCN=C(C)C(C)=NCC[H]",
            None,
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(500, 10)|[<]CC.|60000|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC.",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(500.0, 10.0)|[<]CC.|60000.0|",
            "[H]CC(C)(CC(C)(CC)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
            "[H]CC(C)(CC(C)(CC)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(500, 10)|CC{[>][>]CC([<])c1ccccc1[<]}|gauss(500, 10)|C(C)CC(c1ccccc1)c1ccccc1.|60000|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][>]CC([<])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(500.0, 10.0)|[<]CC[>|0.0|]{[>][>]CC([<])c1ccccc1[<]}|gauss(500.0, 10.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|60000.0|",
            "[H]CC(C)(CC(C)(CCC(CC(CC(CC(CC(CC(CC(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
            "[H]CC(C)(CC(C)(CCC(CC(CC(CC(CC(CC(CC(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}CC{[>][<]CC([>])c1ccccc1[<]}|gauss(500, 10)|C(C)CC(c1ccccc1)c1ccccc1.|60000|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|gauss(500.0, 10.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|60000.0|",
            None,
            None,
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(500, 10)|CC{[>][<]CC([>])c1ccccc1[<]}|gauss(500, 10)|C(C)CC(c1ccccc1)c1ccccc1.|50%|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(500.0, 10.0)|[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|gauss(500.0, 10.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|50.0%|",
            "[H]CC(C)(CC(C)(CCCC(CC(CC(CC(CC(CC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
            "[H]CC(C)(CC(C)(CCCC(CC(CC(CC(CC(CC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}|gauss(1000, 45)|CC{[>][<]CC([>])c1ccccc1[<]}|flory_schulz(0.11)|C(C)CC(c1ccccc1)c1ccccc1.|5e7|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}|gauss(1000.0, 45.0)|[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|flory_schulz(0.11)|[<]C(C)CC(c1ccccc1)c1ccccc1.|50000000.0|",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CCCC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSC(F)(F)F",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CCCC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1ccc(F)c(F)c1",
        ),
        (
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N][<]}|gauss(1000, 45)|{[>][<]CC([>])c1ccccc1; [>]N, [<][H][]}|flory_schulz(0.011)|.|5e7|",
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N][<]}{[>][<]CC([>])c1ccccc1; [>]N, [<][H][]}.",
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N][<]}|gauss(1000.0, 45.0)|{[>][<]CC([>])c1ccccc1; [>]N, [<][H][]}|flory_schulz(0.011)|.|50000000.0|",
            "[H]C(CC(C)(CC(C)(CC(C)(CC(C)(C[N])C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSC(F)(F)F)c1ccccc1",
            "[H]C(CC(C)(CC(C)(CC(C)(CC(C)(C[N])C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSc1ccc(F)c(F)c1)c1ccccc1",
        ),
        (
            "OC{[>] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0 0|])C[>]; [<][H], [<]C [<]}|flory_schulz(5e-3)|COOC{[<] [<]COC[>], [<]C(ON)C[>] [>]}|schulz_zimm(500, 450)|{[<] [<]COCOC[>], [<]CONOC[>] [>]}|schulz_zimm(1700, 1500)|F",
            "OC[>]{[>][<]CC[>], [<]C(N[>])C[>]; [<][H], [<]C[<]}[<]COOC[<]{[<][<]COC[>], [<]C(ON)C[>][>]}{[<][<]COCOC[>], [<]CONOC[>][>]}[>]F",
            "OC[>|0.0|]{[>][<]CC[>], [<|0.5|]C(N[>|0.1 0.0 0.0 0.0 0.0 0.0 0.0|])C[>]; [<][H], [<]C[<]}|flory_schulz(0.005)|[<]COOC[<|0.0|]{[<][<]COC[>], [<]C(ON)C[>][>]}|schulz_zimm(500.0, 450.0)|{[<][<]COCOC[>], [<]CONOC[>][>]}|schulz_zimm(1700.0, 1500.0)|[>]F",
            "[H]NC(CCCCCCOOCCC(COCCOCCOCCOCCOCCOCCONOCCOCOCCONOCCONOCCOCOCCONOCCOCOCCONOCCONOCCONOCCONOCCOCOCCOCOCCONOCCONOCCONOCCONOCCONOCCOCOCCONOCCOCOCCOCOCCONOCCONOCCOCOCCONOCCONOCCOCOCCOCOCCOCOCCONOCCONOCCOCOCCONOCCOCOCCONOCCONOCCOCOCCOCOCCONOCF)ON)CCCC(CCCC(CO)NC)NCCC(CC(CCCCCC)NCCC(CC)NC)N[H]",
            "CNC(CCCCCCCO)CCCCOOCCOCCC(CC(COCCC(COCCC(COCCOCCOCCOCCC(COCCOCCONOCCONOCCONOCCONOCCONOCCOCOCCOCOCCONOCCOCOCCONOCCONOCCONOCCONOCCOCOCCONOCCOCOCCOCOCCONOCCONOCCOCOCCOCOCCONOCCONOCCONOCF)ON)ON)ON)ON)ON",
        ),
        (
            "OC{[>] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0 0|])C[>]; [<][H], [<]C [<]}|flory_schulz(5e-3)|COOC{[<] [<]COC[>], [<]C(ON)C[>] [>]}|schulz_zimm(500, 450)|{[<] [<]COCOC[>], [<]CONOC[>] [>]}|log_normal(1700, 1.2)|F",
            "OC[>]{[>][<]CC[>], [<]C(N[>])C[>]; [<][H], [<]C[<]}[<]COOC[<]{[<][<]COC[>], [<]C(ON)C[>][>]}{[<][<]COCOC[>], [<]CONOC[>][>]}[>]F",
            "OC[>|0.0|]{[>][<]CC[>], [<|0.5|]C(N[>|0.1 0.0 0.0 0.0 0.0 0.0 0.0|])C[>]; [<][H], [<]C[<]}|flory_schulz(0.005)|[<]COOC[<|0.0|]{[<][<]COC[>], [<]C(ON)C[>][>]}|schulz_zimm(500.0, 450.0)|{[<][<]COCOC[>], [<]CONOC[>][>]}|log_normal(1700.0, 1.2)|[>]F",
            "[H]NC(CCCCCCOOCCC(COCCOCCOCCOCCOCCOCCONOCCOCOCCONOCCONOCCOCOCCONOCCOCOCCONOCCONOCCONOCCONOCCOCOCCOCOCCONOCCONOCCONOCCONOCCONOCCOCOCCONOCCOCOCCOCOCCONOCCONOCCOCOCCONOCCONOCCOCOCCOCOCCOCOCCONOCCONOCCOCOCCONOCCOCOCCONOCCONOCCOCOCCOCOCCONOCCOCOCCONOCCONOCCOCOCCOCOCCONOCCOCOCCOCOCCOCOCCONOCCOCOCCONOCF)ON)CCCC(CCCC(CO)NC)NCCC(CC(CCCCCC)NCCC(CC)NC)N[H]",
            "[H]CCCCCCCCCC(CCCCNC(CCOOCCOCCC(COCCC(COCCOCCOCCOCCC(CONOCCOCOCCONOCCONOCCONOCCONOCCONOCCOCOCCOCOCCONOCCOCOCCONOCCONOCCONOCCONOCCOCOCCONOCCOCOCCOCOCCONOCCONOCCOCOCCOCOCCONOCCONOCCONOCF)ON)ON)ON)CC(CCCCCC(CC(CCCCCC(CO)NCCCCC(C[H])NC)N[H])NCCC(CC(CC(CCCC(CCC[H])NCCCCCCCCCCC(C[H])NC)N[H])NC)NCCC)NC)N[H]",
        ),
    ]

    rng = np.random.Generator(np.random.MT19937(42))
    test = rng.uniform()
    assert test == 0.5419938930062744

    for text, big, ref, gen, mir_gen in test_args:
        mol = bigsmiles_gen.Molecule(text)
        assert str(mol) == ref
        assert mol.generate_string(False) == big
        assert mol.generable == (gen is not None)

        mol_mirror = mol.gen_mirror()

        schulz_zimm_distribution = True
        # Check distribution possibility
        for element in mol.elements:
            if isinstance(element, bigsmiles_gen.stochastic.Stochastic):
                if not isinstance(element.distribution, bigsmiles_gen.distribution.SchulzZimm):
                    schulz_zimm_distribution = False
                    break

        stochastic_graph = mol.gen_stochastic_atom_graph(distribution=schulz_zimm_distribution)
        if schulz_zimm_distribution:
            full_atom_graph = bigsmiles_gen.AtomGraph(stochastic_graph)
            full_atom_graph.generate()

        if mol.generable:
            gen_mol = mol.generate(rng=copy.deepcopy(rng))
            assert gen == gen_mol.smiles

            if mir_gen:
                mirror_gen = mol_mirror.generate(rng=copy.deepcopy(rng)).smiles
                assert mirror_gen == mir_gen
            mol.gen_reaction_graph()


if __name__ == "__main__":
    test_molecule()
