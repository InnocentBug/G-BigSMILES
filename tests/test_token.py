# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from rdkit import Chem

import bigsmiles_gen


def test_token_str():

    test_args = [
        ("[$]CC([$])C#N", 0, "[$]CC([$])C#N", "[$]CC([$])C#N", "CCC#N"),
        ("[$]C([H])(C#N)[$]", 0, "[$]C([H])(C#N)[$]", "[$]C([H])(C#N)[$]", "C([H])(C#N)"),
        (
            "[$]CC(C[$])(c1ccccc1)",
            1,
            "[$]CC(C[$])(c1ccccc1)",
            "[$]CC(C[$])(c1ccccc1)",
            "CC(C)(c1ccccc1)",
        ),
        (
            "[$][Si]CC(c1ccccc1)[$]",
            1,
            "[$][Si]CC(c1ccccc1)[$]",
            "[$][Si]CC(c1ccccc1)[$]",
            "[Si]CC(c1ccccc1)",
        ),
        (
            "[<|2.3|]C(=O)c1ccc(cc1)C(=O)[<|1.3|]|0.25|",
            5,
            "[<]C(=O)c1ccc(cc1)C(=O)[<]",
            "[<|2.3|]C(=O)c1ccc(cc1)C(=O)[<|1.3|]|0.25|",
            "C(=O)c1ccc(cc1)C(=O)",
        ),
        ("[$]CC([$|0.5|])[$]", 3, "[$]CC([$])[$]", "[$]CC([$|0.5|])[$]", "CC()"),
        # (
        #     "[<|234|]OCC{[<][>]OCC[<][>|123|]}O[<]|0.4|",
        #     2,
        #     "[<]OCC{[<][>]OCC[<][>]}O[<]",
        #     "[<|234.0|]OCC{[<][>]OCC[<][>|123.0|]}O[<]|0.4|",
        #     "OCC{OCC}O",
        # ),
    ]

    for text, offset, big, ref, smi in test_args:
        token = bigsmiles_gen.SmilesToken(text, offset)
        assert ref == str(token)
        assert big == token.generate_string(False)
        assert token.bond_descriptors[0].descriptor_num == offset
        assert token.generable

        if token.generable:
            mol = token.generate()
            smi = Chem.MolToSmiles(mol.mol)
            print(smi, big)


if __name__ == "__main__":
    test_token_str()
