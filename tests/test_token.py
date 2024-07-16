# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import gbigsmiles


def test_token_str():

    test_args = [
        ("[$]CC([$])C#N", 0, "[$]CC([$])C#N", "[$]CC([$])C#N", "CCC#N"),
        ("[$]C([H])(C#N)[$]", 0, "[$]C([H])(C#N)[$]", "[$]C([H])(C#N)[$]", "CC#N"),
        (
            "[$]CC(C[$])(c1ccccc1)",
            1,
            "[$]CC(C[$])(c1ccccc1)",
            "[$]CC(C[$])(c1ccccc1)",
            "CC(C)c1ccccc1",
        ),
        (
            "[$][Si]CC(c1ccccc1)[$]",
            1,
            "[$][Si]CC(c1ccccc1)[$]",
            "[$][Si]CC(c1ccccc1)[$]",
            "[Si]CCc1ccccc1",
        ),
        (
            "[<|2.3|]C(=O)c1ccc(cc1)C(=O)[<|1.3|]",
            5,
            "[<]C(=O)c1ccc(cc1)C(=O)[<]",
            "[<|2.3|]C(=O)c1ccc(cc1)C(=O)[<|1.3|]",
            "O=Cc1ccc(C=O)cc1",
        ),
        ("[$]CC([$|0.5|])[$]", 3, "[$]CC([$])[$]", "[$]CC([$|0.5|])[$]", "CC"),
        (
            "CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F",
            4,
            "CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F",
            "CC([>])(C[<])C(=O)OCC(O)CSc1c(F)cccc1F",
            "CC(C)C(=O)OCC(O)CSc1c(F)cccc1F",
        ),
        (
            "CC([>])([<])C(=O)OCC(O)CSc1c(F)cccc1F",
            4,
            "CC([>])([<])C(=O)OCC(O)CSc1c(F)cccc1F",
            "CC([>])([<])C(=O)OCC(O)CSc1c(F)cccc1F",
            "CCC(=O)OCC(O)CSc1c(F)cccc1F",
        ),
    ]

    for text, offset, big, ref, smi in test_args:
        token = gbigsmiles.SmilesToken(text, offset, 0)
        assert ref == str(token)
        assert big == token.generate_string(False)
        assert token.bond_descriptors[0].descriptor_num == offset
        assert token.generable
        assert token.res_id == 0

        if token.generable:
            mol = token.generate()
            assert smi == mol.smiles


if __name__ == "__main__":
    test_token_str()
