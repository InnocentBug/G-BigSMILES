# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import pytest

import gbigsmiles

# trunk-disable-all(cspell/error)
# trunk-ignore-all(bandit/B101)

test_args = [
    (
        "{[]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H], [<][H] []}|gauss(1500, 50)|",
        "{[] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H], [<][H] []}",
        "{[] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H], [<][H] []}|gauss(1500.0, 50.0)|",
    ),
    (
        "{[]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H], [<][H][]}|schulz_zimm(4500, 3500)|",
        "{[] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H], [<][H] []}",
        "{[] CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][H], [<][H] []}|schulz_zimm(4500.0, 3500.0)|",
    ),
    (
        "{[$1][$1]C([$1])C=O,[$1]CC([$1])CO;[$1][H], [$1]O[$1]}|gauss(1500.0, 50.0)|",
        "{[$1] [$1]C([$1])C=O, [$1]CC([$1])CO; [$1][H], [$1]O [$1]}",
        "{[$1] [$1]C([$1])C=O, [$1]CC([$1])CO; [$1][H], [$1]O [$1]}|gauss(1500.0, 50.0)|",
    ),
    (
        "{[][$]C([$])C=O,[$]CC([$])CO;[$][H], [$1]O[]}|poisson(10.0)|",
        "{[] [$]C([$])C=O, [$]CC([$])CO; [$][H], [$1]O []}",
        "{[] [$]C([$])C=O, [$]CC([$])CO; [$][H], [$1]O []}|poisson(10.0)|",
    ),
    (
        "{[][$]C([$])C=O,[$]CC([$])CO;[$][H], [$]O[]}|flory_schulz(0.0011)|",
        "{[] [$]C([$])C=O, [$]CC([$])CO; [$][H], [$]O []}",
        "{[] [$]C([$])C=O, [$]CC([$])CO; [$][H], [$]O []}|flory_schulz(0.0011)|",
    ),
    (
        "{[][$|3 4 5 6 0 8|]C([$|4.|])C=O,[$|6.|]CC([$|10.1|])CO;[$][H], [$]O[]}|flory_schulz(9e-4)|",
        "{[] [$]C([$])C=O, [$]CC([$])CO; [$][H], [$]O []}",
        "{[] [$|3.0 4.0 5.0 6.0 0.0 8.0|]C([$|4.0|])C=O, [$|6.0|]CC([$|10.1|])CO; [$][H], [$]O []}|flory_schulz(0.0009)|",
    ),
]


@pytest.mark.parametrize(("text", "big", "ref"), test_args)
def test_stochastic(text, big, ref):
    stochastic = gbigsmiles.StochasticObject.make(text)

    assert str(stochastic) == ref
    assert stochastic.generate_string(False) == big

    stochastic.get_generating_graph()

    # rng = np.random.Generator(np.random.MT19937(42))
    # test = rng.uniform()
    # assert test == 0.5419938930062744
    # if stochastic.generable:
    #     mol = stochastic.generate(rng=rng)
    #     smi = mol.smiles
    #     assert smi


if __name__ == "__main__":
    test_stochastic()
