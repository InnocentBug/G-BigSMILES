# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import bigsmiles_gen


def test_token_str():

    test_args = [
        ("[$]CC(C#N)[$]", 0, "[$]CC(C#N)[$]", "[$|1.0|]CC(C#N)[$|1.0|]", "CC(C#N)"),
        (
            "[$]CC(c1ccccc1)[$]",
            1,
            "[$]CC(c1ccccc1)[$]",
            "[$|1.0|]CC(c1ccccc1)[$|1.0|]",
            "CC(c1ccccc1)",
        ),
        (
            "[<|2.3|]C(=O)c1ccc(cc1)C(=O)[<|1.3|]|0.25|",
            5,
            "[<]C(=O)c1ccc(cc1)C(=O)[<]",
            "[<|2.3|]C(=O)c1ccc(cc1)C(=O)[<|1.3|]|0.25|",
            "C(=O)c1ccc(cc1)C(=O)",
        ),
        ("[$]CC([$|0.5|])[$]", 3, "[$]CC([$])[$]", "[$|1.0|]CC([$|0.5|])[$|1.0|]", "CC()"),
        (
            "[<|234|]OCC{[<][>]OCC[<][>|123|]}O[<]|0.4|",
            2,
            "[<]OCC{[<][>]OCC[<][>]}O[<]",
            "[<|234.0|]OCC{[<|1.0|][>|1.0|]OCC[<|1.0|][>|123.0|]}O[<|1.0|]|0.4|",
            "OCC{OCC}O",
        ),
    ]

    for text, offset, big, ref, smi in test_args:
        token = bigsmiles_gen.SmilesToken(text, offset)
        assert ref == str(token)
        assert big == token.pure_big_smiles()
        assert token.bond_descriptors[0].descriptor_num == offset
        assert token.strip_smiles == smi


if __name__ == "__main__":
    test_token_str()
