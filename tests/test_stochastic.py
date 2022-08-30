# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import bigsmiles_gen

def test_stochastic():
    test_args = [
        ("{[<]CC([<])([>])C(=O)OCC(O)CSc1c(F)cccc1F|0.3|, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F, [<]C([>])c1ccccc1|0.23|[>]}|gauss(100, 50)|.|50%|",
         "{[<]CC([<])([>])C(=O)OCC(O)CSc1c(F)cccc1F, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F, [<]C([>])c1ccccc1[>]}.",
         "{[<]CC([<])([>])C(=O)OCC(O)CSc1c(F)cccc1F|0.3|, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F|0.47|, [<]C([>])c1ccccc1|0.23|[>]}|gauss(100.0, 50.0)|.|50.0%|", ),

        ("{[$][$1]C([$1])C=O,[$1]CC[$1]CO;[$1][H], [$1]O[$]}",
         "{[$][$1]C([$1])C=O, [$1]CC[$1]CO; [$1][H],  [$1]O[$]}",
         "{[$][$1]C([$1])C=O, [$1]CC[$1]CO; [$1][H],  [$1]O[$]}", ),

    ]

    for text, big, ref in test_args:
        stochastic = bigsmiles_gen.Stochastic(text)
        assert str(stochastic) == ref
        assert stochastic.generate_string(True) == ref


if __name__ == "__main__":
    test_stochastic()
