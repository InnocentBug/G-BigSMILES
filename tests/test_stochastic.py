# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import bigsmiles_gen

def test_stochastic():
    test_args = [ ("{[<]CC([<])([>])C(=O)OCC(O)CSc1c(F)cccc1F|0.3|, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F, [<]C([>])c1ccccc1|0.23|[>]}", )]

    for text, in test_args:
        stochastic = bigsmiles_gen.Stochastic(text)
        print(str(stochastic))
        print(stochastic.generate_string(False))
        print(text)


if __name__ == "__main__":
    test_stochastic()
