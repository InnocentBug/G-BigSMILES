# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import bigsmiles_gen


def test_stochastic():
    test_args = [
        (
            "{[<]CC([<])([>])C(=O)OCC(O)CSc1c(F)cccc1F|0.3|, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F, [<]C([>])c1ccccc1|0.23|[>]}|gauss(100, 50)|",
            "{[<]CC([<])([>])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F, [<]C([>])c1ccccc1[>]}",
            "{[<]CC([<])([>])C(=O)OCC(O)CSc1c(F)cccc1F|0.3|, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F|0.47|, [<]C([>])c1ccccc1|0.23|[>]}|gauss(100.0, 50.0)|",
            True,
        ),
        (
            "{[$][$1]C([$1])C=O,[$1]CC[$1]CO;[$1][H], [$1]O[$]}",
            "{[$][$1]C([$1])C=O, [$1]CC[$1]CO; [$1][H], [$1]O[$]}",
            "{[$][$1]C([$1])C=O, [$1]CC[$1]CO; [$1][H], [$1]O[$]}",
            False,
        ),
        (
            "{[][$]C([$])C=O|0.1|,[$]CC[$]CO;[$][H]|0.2|, [$1]O[]}",
            "{[][$]C([$])C=O, [$]CC[$]CO; [$][H], [$1]O[]}",
            "{[][$]C([$])C=O|0.1|, [$]CC[$]CO|0.9|; [$][H]|0.2|, [$1]O|0.8|[]}",
            False,
        ),
        (
            "{[][$]C([$])C=O|0.1|,[$]CC[$]CO;[$][H]|0.2|, [$1]O[]}|flory_schulz(0.011)|",
            "{[][$]C([$])C=O, [$]CC[$]CO; [$][H], [$1]O[]}",
            "{[][$]C([$])C=O|0.1|, [$]CC[$]CO|0.9|; [$][H]|0.2|, [$1]O|0.8|[]}|flory_schulz(0.011)|",
            True,
        ),
        (
            "{[][$|0 3 4 5 6 0 8 9|]C([$|4.|])C=O|0.5|,[$|6.|]CC[$|0.1|]CO;[$][H], [$1]O|.1|[]}|flory_schulz(1e-3)|",
            "{[][$]C([$])C=O, [$]CC[$]CO; [$][H], [$1]O[]}",
            "{[][$|0.0 3.0 4.0 5.0 6.0 0.0 8.0 9.0|]C([$|4.0|])C=O|0.5|, [$|6.0|]CC[$|0.1|]CO|0.5|; [$][H]|0.9|, [$1]O|0.1|[]}|flory_schulz(0.001)|",
            True,
        ),
    ]

    for text, big, ref, gen in test_args:
        stochastic = bigsmiles_gen.Stochastic(text)
        assert str(stochastic) == ref
        assert stochastic.generate_string(False) == big
        assert stochastic.generable == gen


if __name__ == "__main__":
    test_stochastic()
