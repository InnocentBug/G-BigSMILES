# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import bigsmiles_gen


def test_descriptors_str():

    test_args = [
        ("[$0]", 0, "", "[$0|1.0|]"),
        ("[<]", 0, "", "[<|1.0|]"),
        ("[>]", 1, "", "[>|1.0|]"),
        ("[$]", 0, "-", "[$|1.0|]"),
        ("[<]", 0, "#", "[<|1.0|]"),
        ("[>]", 1, "=", "[>|1.0|]"),
        ("[$5| 2.|]", 34, "", "[$5|2.0|]"),
        ("[<| 21. 234. 2134. 64. 657.|]", 0, "#", "[<|21.0 234.0 2134.0 64.0 657.0|]"),
    ]

    for text, idx, char, ref in test_args:
        bond = bigsmiles_gen.BondDescriptor(text, idx, char)
        assert str(bond) == char + ref


def test_descriptors_compatible():

    bondA = bigsmiles_gen.BondDescriptor("[$]", 0, "")
    bondB = bigsmiles_gen.BondDescriptor("[$]", 0, "")

    assert bondA.is_compatible(bondB)
    assert bondB.is_compatible(bondA)

    bondA = bigsmiles_gen.BondDescriptor("[$0]", 0, "")
    bondB = bigsmiles_gen.BondDescriptor("[$0]", 0, "")

    assert bondA.is_compatible(bondB)
    assert bondB.is_compatible(bondA)

    bondA = bigsmiles_gen.BondDescriptor("[$0]", 0, "")
    bondB = bigsmiles_gen.BondDescriptor("[$1]", 0, "")

    assert not bondA.is_compatible(bondB)
    assert not bondB.is_compatible(bondA)

    bondA = bigsmiles_gen.BondDescriptor("[<]", 0, "")
    bondB = bigsmiles_gen.BondDescriptor("[<]", 0, "")

    assert not bondA.is_compatible(bondB)
    assert not bondB.is_compatible(bondA)

    bondA = bigsmiles_gen.BondDescriptor("[>]", 0, "")
    bondB = bigsmiles_gen.BondDescriptor("[>]", 0, "")

    assert not bondA.is_compatible(bondB)
    assert not bondB.is_compatible(bondA)

    bondA = bigsmiles_gen.BondDescriptor("[<]", 0, "")
    bondB = bigsmiles_gen.BondDescriptor("[>]", 0, "")

    assert bondA.is_compatible(bondB)
    assert bondB.is_compatible(bondA)


if __name__ == "__main__":
    test_descriptors_str()
    test_descriptors_compatible()
