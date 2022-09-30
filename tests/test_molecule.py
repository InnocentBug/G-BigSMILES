# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import numpy as np

import bigsmiles_gen


def test_molecule():
    test_args = [
        (
            "[H]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}CC{[<][<]CC([>])c1ccccc1[>]}C(C)CC(c1ccccc1)c1ccccc1",
            "[H][>]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}[<]CC[>]{[<][<]CC([>])c1ccccc1[>]}[<]C(C)CC(c1ccccc1)c1ccccc1",
            "[H][>]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}[<]CC[>]{[<][<]CC([>])c1ccccc1[>]}[<]C(C)CC(c1ccccc1)c1ccccc1",
            False,
        ),
        (
            "[H]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}|gauss(500, 10)|CC{[<][<]CC([>])c1ccccc1[>]}|gauss(500, 10)|C(C)CC(c1ccccc1)c1ccccc1.|60000|",
            "[H][>]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}[<]CC[>]{[<][<]CC([>])c1ccccc1[>]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}|gauss(500.0, 10.0)|[<]CC[>]{[<][<]CC([>])c1ccccc1[>]}|gauss(500.0, 10.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|60000.0|",
            True,
        ),
        (
            "[H]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}CC{[<][<]CC([>])c1ccccc1[>]}|gauss(500, 10)|C(C)CC(c1ccccc1)c1ccccc1.|60000|",
            "[H][>]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}[<]CC[>]{[<][<]CC([>])c1ccccc1[>]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}[<]CC[>]{[<][<]CC([>])c1ccccc1[>]}|gauss(500.0, 10.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|60000.0|",
            False,
        ),
        (
            "[H]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}|gauss(500, 10)|CC{[<][<]CC([>])c1ccccc1[>]}|gauss(500, 10)|C(C)CC(c1ccccc1)c1ccccc1.|50%|",
            "[H][>]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}[<]CC[>]{[<][<]CC([>])c1ccccc1[>]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}|gauss(500.0, 10.0)|[<]CC[>]{[<][<]CC([>])c1ccccc1[>]}|gauss(500.0, 10.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|50.0%|",
            False,
        ),
        (
            "[H]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1|0.3|, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[>]}|gauss(1000, 45)|CC{[<][<]CC([>])c1ccccc1[>]}|flory_schulz(0.11)|C(C)CC(c1ccccc1)c1ccccc1.|5e7|",
            "[H][>]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[>]}[<]CC[>]{[<][<]CC([>])c1ccccc1[>]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1|0.3|, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F|0.7|[>]}|gauss(1000.0, 45.0)|[<]CC[>]{[<][<]CC([>])c1ccccc1[>]}|flory_schulz(0.11)|[<]C(C)CC(c1ccccc1)c1ccccc1.|50000000.0|",
            True,
        ),
    ]

    rng = np.random.Generator(np.random.MT19937(42))
    test = rng.uniform()
    assert test == 0.5419938930062744

    for text, big, ref, gen in test_args:
        print(text)
        mol = bigsmiles_gen.Molecule(text)
        assert str(mol) == ref
        assert mol.generate_string(False) == big
        assert mol.generable == gen

        if mol.generable:
            gen_mol = mol.generate(rng=rng)
            print(gen_mol.smiles)


if __name__ == "__main__":
    test_molecule()
