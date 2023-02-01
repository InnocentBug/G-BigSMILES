# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy

import numpy as np

import bigsmiles_gen


def test_mol_prob():
    test_objects = [
        ("CCO", 1.0, 1),
        ("{[][<]C(N)C[>]; [<][H], [>]CO []}|uniform(500, 600)|", 0.01, 2),
        ("[H]{[<][<]C(N)C[>]; [>]CO []}|uniform(500, 600)|", 0.02, 2),
        ("[H]{[<][<]C(N)C[>][>]}|uniform(500, 600)|CO", 0.02, 2),
        (
            "{[][<]C(N)C[>]; [<][H][>]}|uniform(100, 200)|{[<][<]C(=O)C[>]; [>][H][]}|uniform(100, 200)|",
            0.0002,
            2,
        ),
        (
            "[H]{[<][<]C(N)C[>] [>]}|gauss(100, 20)|{[<][<]C(=O)C[>]; [>][H][]}|gauss(100, 20)|",
            0.00037778155245353777,
            2,
        ),
        (
            "OCC{[<][<]C(N)C[>] [>]}|gauss(100, 20)|{[<][<]C(=O)C[>]; [>]}|gauss(100, 20)|[Si]",
            0.00037778155245353777,
            2,
        ),
        (
            "OCC{[<][<]C(N)C[>] [>]}|flory_schulz(0.1)|CC[Si]CC{[<][<]C(=O)C[>]; [>]}|flory_schulz(0.1)|[Si]",
            2.992503549677072e-06,
            2,
        ),
        (
            "OCC{[<][<]C(N)C[>], [<]CC(C(=O)C[<])[>] ;[H][>] [>]}|gauss(50, 10)|CC[Si]",
            0.022459119845361735,
            2,
        ),
        (
            "OCC{[<][<]C(N)C[>], [<]CC(C(=O)C[<|0.01|])[>] ;[H][>], CCF[$] [>]}|gauss(250, 10)|CC[Si]",
            4.393301174734012e-07,
            72,
        ),
    ]

    rng = np.random.default_rng(42)
    for big_smi, prob, num_matches in test_objects:
        mol = bigsmiles_gen.Molecule(big_smi)
        smi = mol.generate(rng=copy.deepcopy(rng)).smiles
        print(smi)
        calc_prob, matches = bigsmiles_gen.mol_prob.get_ensemble_prob(smi, mol)
        print(big_smi, prob, calc_prob, len(matches))
        assert abs(calc_prob - prob) < 1e-10
        assert len(matches) == num_matches

        tmp_rng = copy.deepcopy(rng)
        for i in range(6):
            smi = mol.generate(rng=tmp_rng).smiles
            print(i, smi)
            calc_prob, matches = bigsmiles_gen.mol_prob.get_ensemble_prob(smi, mol)
            print(i, calc_prob, len(matches))


if __name__ == "__main__":
    test_mol_prob()
