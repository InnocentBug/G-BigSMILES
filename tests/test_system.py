# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import bigsmiles_gen


def test_molecule():
    test_args = [
        (
            "[H]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}C(C{[<][<]CC([>])c1ccccc1[>]}C(C)CC)(c1ccccc1)c1ccccc1.|90%|[H]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1|0.3|, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[>]}|gauss(1000, 45)|C(C{[<][<]CC([>])c1ccccc1[>]}|flory_schulz(0.11)|C(C)CC)(c1ccccc1)c1ccccc1.|5e7|",
            "[H]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}C(C{[<][<]CC([>])c1ccccc1[>]}C(C)CC)(c1ccccc1)c1ccccc1.[H]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[>]}C(C{[<][<]CC([>])c1ccccc1[>]}C(C)CC)(c1ccccc1)c1ccccc1.",
            "[H]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[>]}C(C{[<][<]CC([>])c1ccccc1[>]}C(C)CC)(c1ccccc1)c1ccccc1.|450000000.0|[H]{[<]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1|0.3|, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F|0.7|[>]}|gauss(1000.0, 45.0)|C(C{[<][<]CC([>])c1ccccc1[>]}|flory_schulz(0.11)|C(C)CC)(c1ccccc1)c1ccccc1.|50000000.0|",
        ),
        (
            "CCCCC.[H]{[<][<]CC([>])c1ccccc1[>]}[H]",
            "CCCCC.[H]{[<][<]CC([>])c1ccccc1[>]}[H].",
            "CCCCC.[H]{[<][<]CC([>])c1ccccc1[>]}[H].|100.0%|",
        ),
        (
            "CCCCC.|90%|[H]{[<][<]CC([>])c1ccccc1[>]}|gauss(5000, 50)|[H].|50000|",
            "CCCCC.[H]{[<][<]CC([>])c1ccccc1[>]}[H].",
            "CCCCC.|450000.0|[H]{[<][<]CC([>])c1ccccc1[>]}|gauss(5000.0, 50.0)|[H].|50000.0|",
        ),
        (
            "CCCCC.|80%|[H]{[<][<]CC([>])c1ccccc1, [<]CC[>]|0.1|; [>]O|0.2|, [<][H][>]}|gauss(15000, 150)|[H].|150000|",
            "CCCCC.[H]{[<][<]CC([>])c1ccccc1, [<]CC[>]; [>]O, [<][H][>]}[H].",
            "CCCCC.|600000.0|[H]{[<][<]CC([>])c1ccccc1|0.9|, [<]CC[>]|0.1|; [>]O|0.2|, [<][H]|0.8|[>]}|gauss(15000.0, 150.0)|[H].|150000.0|",
        ),
    ]

    for text, big, ref in test_args:
        mol = bigsmiles_gen.System(text)
        print(ref)
        print(str(mol))
        assert str(mol) == ref
        print(mol.generate_string(False))
        print(big)
        assert mol.generate_string(False) == big


if __name__ == "__main__":
    test_molecule()
