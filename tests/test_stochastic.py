# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import numpy as np

import gbigsmiles

# trunk-disable-all(cspell/error)


def test_stochastic():
    test_args = [
        (
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H], [<][H][]}|gauss(1500, 50)|",
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H], [<][H][]}",
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H], [<][H][]}|gauss(1500.0, 50.0)|",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CC([H])(C)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1c(F)cccc1F)C(=O)OCC(O)CSc1c(F)cccc1F)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1c(F)cccc1F)C(=O)OCC(O)CSc1c(F)cccc1F",
        ),
        (
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H], [<][H][]}|schulz_zimm(4500, 3500)|",
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H], [<][H][]}",
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H], [<][H][]}|schulz_zimm(4500.0, 3500.0)|",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC([H])(C)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1c(F)cccc1F)C(=O)OCC(O)CSc1c(F)cccc1F)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1c(F)cccc1F",
        ),
        (
            "{[$][$1]C([$1])C=O,[$1]CC([$1])CO;[$1][H], [$1]O[$]}",
            "{[$][$1]C([$1])C=O, [$1]CC([$1])CO; [$1][H], [$1]O[$]}",
            "{[$][$1]C([$1])C=O, [$1]CC([$1])CO; [$1][H], [$1]O[$]}",
            None,
        ),
        (
            "{[][$]C([$])C=O,[$]CC([$])CO;[$][H], [$1]O[]}",
            "{[][$]C([$])C=O, [$]CC([$])CO; [$][H], [$1]O[]}",
            "{[][$]C([$])C=O, [$]CC([$])CO; [$][H], [$1]O[]}",
            None,
        ),
        (
            "{[][$]C([$])C=O,[$]CC([$])CO;[$][H], [$]O[]}|flory_schulz(0.0011)|",
            "{[][$]C([$])C=O, [$]CC([$])CO; [$][H], [$]O[]}",
            "{[][$]C([$])C=O, [$]CC([$])CO; [$][H], [$]O[]}|flory_schulz(0.0011)|",
            "O=CC(O)CC(CO)C(C=O)C(C=O)C(CO)CC(C=O)C(C=O)CC(O)CO",
        ),
        (
            "{[][$|3 4 5 6 0 8|]C([$|4.|])C=O,[$|6.|]CC([$|10.1|])CO;[$][H], [$]O[]}|flory_schulz(9e-4)|",
            "{[][$]C([$])C=O, [$]CC([$])CO; [$][H], [$]O[]}",
            "{[][$|3.0 4.0 5.0 6.0 0.0 8.0|]C([$|4.0|])C=O, [$|6.0|]CC([$|10.1|])CO; [$][H], [$]O[]}|flory_schulz(0.0009)|",
            "O=CC(O)C(CO)CC(C=O)C(CO)CC(C=O)C(C=O)C(CO)CC(C=O)C(C=O)CC(O)CO",
        ),
    ]

    for text, big, ref, gen in test_args:
        stochastic = gbigsmiles.Stochastic(text, 0)
        assert str(stochastic) == ref
        assert stochastic.generate_string(False) == big
        assert stochastic.generable == (gen is not None)

        rng = np.random.Generator(np.random.MT19937(42))
        test = rng.uniform()
        assert test == 0.5419938930062744
        if stochastic.generable:
            mol = stochastic.generate(rng=rng)
            assert gen == mol.smiles


if __name__ == "__main__":
    test_stochastic()
