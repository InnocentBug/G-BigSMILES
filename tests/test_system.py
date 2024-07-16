# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import gbigsmiles


def test_molecule():
    test_args = [
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(5000.0, 150.)|CC{[>][<]CC([>])c1ccccc1[<]}|gauss(5000.0, 150.)|C(C)CC(c1ccccc1)c1ccccc1.|90%|[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}|gauss(1000, 45)|CC{[>][<]CC([>])c1ccccc1[<]}|flory_schulz(0.11)|C(C)CC(c1ccccc1)c1ccccc1.|5e4|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|gauss(5000.0, 150.0)|[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|gauss(5000.0, 150.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|450000.0|[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}|gauss(1000.0, 45.0)|[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|flory_schulz(0.11)|[<]C(C)CC(c1ccccc1)c1ccccc1.|50000.0|",
            True,
        ),
        (
            "CCCCC.[H]{[>][<]CC([>])c1ccccc1[<]}|gauss(5000, 50)|[H]",
            "CCCCC.[H][>]{[>][<]CC([>])c1ccccc1[<]}[<][H].",
            "CCCCC.[H][>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|gauss(5000.0, 50.0)|[<][H].|100.0%|",
            False,
        ),
        (
            "CCCCC.|10.0%|[H]{[>][<]CC([>])c1ccccc1[<]}|gauss(5000, 50)|[H]",
            "CCCCC.[H][>]{[>][<]CC([>])c1ccccc1[<]}[<][H].",
            "CCCCC.|10.0%|[H][>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|gauss(5000.0, 50.0)|[<][H].|90.0%|",
            False,
        ),
        (
            "CCCCC.|90%|[H]{[>][<]CC([>])c1ccccc1[<]}|gauss(5000, 50)|[H].|50000|",
            "CCCCC.[H][>]{[>][<]CC([>])c1ccccc1[<]}[<][H].",
            "CCCCC.|450000.0|[H][>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|gauss(5000.0, 50.0)|[<][H].|50000.0|",
            True,
        ),
        (
            "CCCCC.|80%|[H]{[>][<]CC([>])c1ccccc1, [<]CC[>]; [>|1|]O, [<|8.0|][H][<|8.0|]}|gauss(15000, 150)|[H].|150000|",
            "CCCCC.[H][>]{[>][<]CC([>])c1ccccc1, [<]CC[>]; [>]O, [<][H][<]}[<][H].",
            "CCCCC.|600000.0|[H][>|0.0|]{[>][<]CC([>])c1ccccc1, [<]CC[>]; [>]O, [<|8.0|][H][<|8.0|]}|gauss(15000.0, 150.0)|[<][H].|150000.0|",
            True,
        ),
        (
            "NC{[$][$]C[$][$]}|uniform(12, 72)|COOC{[$][$]C[$][$]}|uniform(12, 72)|CO.|1000|",
            "NC[$]{[$][$]C[$][$]}[$]COOC[$]{[$][$]C[$][$]}[$]CO.",
            "NC[$|0.0|]{[$][$]C[$][$]}|uniform(12, 72)|[$]COOC[$|0.0|]{[$][$]C[$][$]}|uniform(12, 72)|[$]CO.|1000.0|",
            True,
        ),
    ]

    for text, big, ref, gen in test_args:
        system = gbigsmiles.System(text)
        assert str(system) == ref
        assert system.generate_string(False) == big
        assert system.generable == gen

        if system.generable:
            for mol in system.generator:
                mol.smiles


if __name__ == "__main__":
    test_molecule()
