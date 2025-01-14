# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import pytest

import gbigsmiles

test_simple_bond_descriptors = [
    ("[$0]", 0, "[$]", "[$]"),
    ("[<]", 0, "[<]", "[<]"),
    ("[>|2|]", 0, "[>|2.0|]", "[>]"),
    ("[$|3|]", 0, "[$|3.0|]", "[$]"),
    ("[<]", 0, "[<]", "[<]"),
    ("[>]", 0, "[>]", "[>]"),
    ("[$5| 2.|]", 5, "[$5|2.0|]", "[$5]"),
    ("[<4| 21. 234. 2134. 64. 657.|]", 4, "[<4|21.0 234.0 2134.0 64.0 657.0|]", "[<4]"),
    ("[<7| 0.0 1.0 0.0|]", 7, "[<7|0.0 1.0 0.0|]", "[<7]"),
]


@pytest.mark.parametrize(("text", "idx", "ref", "big"), test_simple_bond_descriptors)
def test_simple_bond_descriptors(text, idx, ref, big):
    bond = gbigsmiles.BondDescriptor.make(text)
    if idx is not None:
        assert bond.idx == idx
    assert str(bond) == ref
    assert bond.generate_string(False) == big
    assert bond.generable


test_terminal_bond_descriptors = [
    ("[]", None, "[]", "[]"),
    ("[|6|]", None, "[|6.0|]", "[]"),
    ("[|6. 3. 1 0|]", None, "[|6.0 3.0 1.0 0.0|]", "[]"),
]


@pytest.mark.parametrize(("text", "idx", "ref", "big"), test_terminal_bond_descriptors)
def test_terminal_bond_descriptors(text, idx, ref, big):
    bond = gbigsmiles.bond.TerminalBondDescriptor.make(text)
    if idx is not None:
        assert bond.idx == idx
    assert str(bond) == ref
    assert bond.generate_string(False) == big
    assert bond.generable


compatibility_list = [
    ("[$]", "[$]", True),
    ("[$1]", "[$1]", True),
    ("[$0]", "[$1]", False),
    ("[<]", "[<]", False),
    ("[>]", "[>]", False),
    ("[<]", "[>]", True),
    ("[>]", "[<]", True),
]


@pytest.mark.parametrize(("textA", "textB", "compatible"), compatibility_list)
def test_descriptors_compatible(textA, textB, compatible):
    bondA = gbigsmiles.BondDescriptor.make(textA)
    bondB = gbigsmiles.BondDescriptor.make(textB)

    assert bondA.is_compatible(bondB) == compatible
    assert bondB.is_compatible(bondA) == compatible


bd_smiles_list = [
    ("[C@@H]N(=O)c1ccncc1", []),
    ("[$][C@@H]N(=O)c1ccncc1[$]", ["[$]", "[$]"]),
    ("[$|3|][C@@H]N(=O)c1ccncc1[$|1 2|]", ["[$|3.0|]", "[$|1.0 2.0|]"]),
    ("[$|3|][C@@H]N(=[<]O)c1ccncc1[$|1 2|]", ["[$|3.0|]", "[<]", "[$|1.0 2.0|]"]),
    (
        "[$|3|][C@@H][>]N(=[<]O)c1cc[<|3|]ncc1[$|1 2|]",
        ["[$|3.0|]", "[>]", "[<]", "[<|3.0|]", "[$|1.0 2.0|]"],
    ),
]


@pytest.mark.parametrize(("smi", "expected_bd_list"), bd_smiles_list)
def test_bond_descriptor_recognition(smi, expected_bd_list):
    big_smi = gbigsmiles.Smiles.make(smi)

    assert len(big_smi.bond_descriptors) == len(expected_bd_list)

    for actual, expected in zip(big_smi.bond_descriptors, expected_bd_list):
        assert str(actual) == expected
