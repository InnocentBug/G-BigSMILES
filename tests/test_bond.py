# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import bigsmiles_gen


def test_descriptors():

    test_args = [
        ("[$]", 0, "", "[$|1.0|]"),
        ("[<]", 0, "", "[<|1.0|]"),
        ("[>]", 1, "", "[>|1.0|]"),
        ("[$]", 0, "-", "[$|1.0|]"),
        ("[<]", 0, "#", "[<|1.0|]"),
        ("[>]", 1, "@=", "[>|1.0|]"),
        ("[$| 2.|]", 34, "", "[$|2.0|]"),
        ("[<| 21. 234. 2134. 64. 657.|]", 0, "#", "[<|21.0 234.0 2134.0 64.0 657.0|]"),
    ]

    for text, idx, char, ref in test_args:
        bond = bigsmiles_gen.BondDescriptor(text, idx, char)
        assert str(bond) == char + ref
