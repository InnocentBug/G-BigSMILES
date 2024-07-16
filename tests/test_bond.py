# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import gbigsmiles


def test_descriptors_str():

    test_args = [
        ("[$0]", 0, "", 5, "[$0]", "[$0]"),
        ("[<]", 0, "", 5, "[<]", "[<]"),
        ("[]", 12, "", 5, "[]", "[]"),
        ("[>|2|]", 1, "", 5, "[>|2.0|]", "[>]"),
        ("[$|3|]", 0, "-", 5, "[$|3.0|]", "[$]"),
        ("[<]", 0, "#", 5, "[<]", "[<]"),
        ("[>]", 1, "=", 5, "[>]", "[>]"),
        ("[$5| 2.|]", 34, "", 5, "[$5|2.0|]", "[$5]"),
        ("[<| 21. 234. 2134. 64. 657.|]", 0, "#", 5, "[<|21.0 234.0 2134.0 64.0 657.0|]", "[<]"),
    ]

    for text, idx, char, bl, ref, big in test_args:
        bond = gbigsmiles.BondDescriptor(text, idx, char, bl)
        assert str(bond) == ref
        assert bond.generate_string(False) == big
        assert bond.generable


def test_descriptors_compatible():

    bondA = gbigsmiles.BondDescriptor("[$]", 0, "", 5)
    bondB = gbigsmiles.BondDescriptor("[$]", 0, "", 5)

    assert bondA.is_compatible(bondB)
    assert bondB.is_compatible(bondA)

    bondA = gbigsmiles.BondDescriptor("[$0]", 0, "", 5)
    bondB = gbigsmiles.BondDescriptor("[$0]", 0, "", 5)

    assert bondA.is_compatible(bondB)
    assert bondB.is_compatible(bondA)

    bondA = gbigsmiles.BondDescriptor("[$0]", 0, "", 5)
    bondB = gbigsmiles.BondDescriptor("[$1]", 0, "", 5)

    assert not bondA.is_compatible(bondB)
    assert not bondB.is_compatible(bondA)

    bondA = gbigsmiles.BondDescriptor("[<]", 0, "", 5)
    bondB = gbigsmiles.BondDescriptor("[<]", 0, "", 5)

    assert not bondA.is_compatible(bondB)
    assert not bondB.is_compatible(bondA)

    bondA = gbigsmiles.BondDescriptor("[>]", 0, "", 5)
    bondB = gbigsmiles.BondDescriptor("[>]", 0, "", 5)

    assert not bondA.is_compatible(bondB)
    assert not bondB.is_compatible(bondA)

    bondA = gbigsmiles.BondDescriptor("[<]", 0, "", 5)
    bondB = gbigsmiles.BondDescriptor("[>]", 0, "", 5)

    assert bondA.is_compatible(bondB)
    assert bondB.is_compatible(bondA)

    bondA = gbigsmiles.BondDescriptor("[<]", 0, "=", 5)
    bondB = gbigsmiles.BondDescriptor("[>]", 0, "", 5)

    assert not bondA.is_compatible(bondB)
    assert not bondB.is_compatible(bondA)

    bondA = gbigsmiles.BondDescriptor("[<]", 0, "=", 5)
    bondB = gbigsmiles.BondDescriptor("[>]", 0, "=", 5)

    assert bondA.is_compatible(bondB)
    assert bondB.is_compatible(bondA)


if __name__ == "__main__":
    test_descriptors_str()
    test_descriptors_compatible()
