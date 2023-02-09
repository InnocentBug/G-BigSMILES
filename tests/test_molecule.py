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
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1",
            None,
            None,
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(500, 10)|[<]CC.|60000|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC.",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(500.0, 10.0)|[<]CC.|60000.0|",
            "[H]CC(C)(CC(C)(CC)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
            "[H]CC(C)(CC(C)(CC)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(500, 10)|CC{[>][>]CC([<])c1ccccc1[<]}|gauss(500, 10)|C(C)CC(c1ccccc1)c1ccccc1.|60000|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][>]CC([<])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(500.0, 10.0)|[<]CC[>]{[>][>]CC([<])c1ccccc1[<]}|gauss(500.0, 10.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|60000.0|",
            "[H]CC(C)(CC(C)(CCC(CC(CC(CC(CC(CC(CC(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
            "[H]CC(C)(CC(C)(CCC(CC(CC(CC(CC(CC(CC(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}CC{[>][<]CC([>])c1ccccc1[<]}|gauss(500, 10)|C(C)CC(c1ccccc1)c1ccccc1.|60000|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}|gauss(500.0, 10.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|60000.0|",
            None,
            None,
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(500, 10)|CC{[>][<]CC([>])c1ccccc1[<]}|gauss(500, 10)|C(C)CC(c1ccccc1)c1ccccc1.|50%|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(500.0, 10.0)|[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}|gauss(500.0, 10.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|50.0%|",
            "[H]CC(C)(CC(C)(CCCC(CC(CC(CC(CC(CC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
            "[H]CC(C)(CC(C)(CCCC(CC(CC(CC(CC(CC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}|gauss(1000, 45)|CC{[>][<]CC([>])c1ccccc1[<]}|flory_schulz(0.11)|C(C)CC(c1ccccc1)c1ccccc1.|5e7|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}|gauss(1000.0, 45.0)|[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}|flory_schulz(0.11)|[<]C(C)CC(c1ccccc1)c1ccccc1.|50000000.0|",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CCCC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSC(F)(F)F",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CCCC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSC(F)(F)F",
        ),
        (
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N][<]}|gauss(1000, 45)|{[>][<]CC([>])c1ccccc1; [>]N, [<][H][]}|flory_schulz(0.011)|.|5e7|",
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N][<]}{[>][<]CC([>])c1ccccc1; [>]N, [<][H][]}.",
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N][<]}|gauss(1000.0, 45.0)|{[>][<]CC([>])c1ccccc1; [>]N, [<][H][]}|flory_schulz(0.011)|.|50000000.0|",
            "[H]C(CC(C)(CC(C)(CC(C)(CC(C)(C[N])C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSc1ccc(F)c(F)c1)c1ccccc1",
            "[H]C(CC(C)(CC(C)(CC(C)(CC(C)(C[N])C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSc1ccc(F)c(F)c1)c1ccccc1",
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

        if mol.generable:
            gen_mol = mol.generate(rng=copy.deepcopy(rng))
            assert gen == gen_mol.smiles
            mirror_gen = mol_mirror.generate(rng=copy.deepcopy(rng)).smiles
            assert mirror_gen == mir_gen


if __name__ == "__main__":
    test_molecule()
