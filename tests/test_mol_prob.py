# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy

import numpy as np

import gbigsmiles


def test_mol_prob():
    test_objects = [
        ("CCO", 1.0, 1),
        ("{[][<]C(N)C[>]; [<][H], [>]CO []}|uniform(500, 600)|", 0.38029, 2),
        ("[H]{[<][<]C(N)C[>]; [>]CO []}|uniform(500, 600)|", 0.29565000000000047, 1),
        ("[H]{[<][<]C(N)C[>][>]}|uniform(500, 600)|CO", 0.29565000000000047, 1),
        (
            "{[][<]C(N)C[>]; [<][H][>]}|uniform(100, 200)|{[<][<]C(=O)C[>]; [>][H][]}|uniform(100, 200)|",
            0.1521958609,
            1,
        ),
        (
            "[H]{[<][<]C(N)C[>] [>]}|gauss(100, 20)|{[<][<]C(=O)C[>]; [>][H][]}|gauss(100, 20)|",
            0.4396498059364593,
            1,
        ),
        (
            "OCC{[<][<]C(N)C[>] [>]}|gauss(100, 20)|{[<][<]C(=O)C[>]; [>]}|gauss(100, 20)|[Si]",
            0.4396498059364593,
            1,
        ),
        (
            "OCC{[<][<]C(N)C[>] [>]}|flory_schulz(0.1)|CC[Si]CC{[<][<]C(=O)C[>]; [>]}|flory_schulz(0.1)|[Si]",
            0.06563705212253623,
            1,
        ),
        (
            "OCC{[<][<]C(N)C[>], [<]CC(C(=O)C[<])[>] ;[H][>] [>]}|gauss(50, 10)|CC[Si]",
            0.4599424731010503,
            1,
        ),
        (
            "OCC{[<][<]C(N)C[>], [<]CC(C(=O)C[<])[>] ;[H][>] [>]}|log_normal(50, 1.1)|CC[Si]",
            0.4152512429696172,
            1,
        ),
        (
            "OCC{[<][<]C(N)C[>], [<]CC(C(=O)C[<])[>] ;[H][>] [>]}|poisson(65)|CC[Si]",
            0.129121815076718,
            1,
        ),
        (
            "OCC{[<][<]C(N)C[>], [<]CC(C(=O)C[<|0.01|])[>] ;[H][>], CCF[$] [>]}|gauss(250, 10)|CC[Si]",
            0.0025427279427714814,
            35,
        ),
    ]

    rng = np.random.default_rng(42)
    for big_smi, prob, num_matches in test_objects:
        mol = gbigsmiles.Molecule(big_smi)
        smi = mol.generate(rng=copy.deepcopy(rng)).smiles
        print(smi)
        calc_prob, matches = gbigsmiles.mol_prob.get_ensemble_prob(smi, mol)
        print(big_smi, prob, calc_prob, len(matches))
        assert abs(calc_prob - prob) < 1e-10
        assert len(matches) == num_matches

        tmp_rng = copy.deepcopy(rng)
        for i in range(6):
            smi = mol.generate(rng=tmp_rng).smiles
            calc_prob, matches = gbigsmiles.mol_prob.get_ensemble_prob(smi, mol)
            assert 0 < calc_prob <= 1.0


if __name__ == "__main__":
    test_mol_prob()
