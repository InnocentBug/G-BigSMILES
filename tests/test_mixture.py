# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import pytest

import gbigsmiles

test_args = [
    (".|25%|", True, ".|25.0%|", 2500, 25.0),
    (".|1%|", True, ".|1.0%|", 100, 1.0),
    (".|5000|", True, ".|5000.0|", 5000, 50.0),
    (".|5004|", True, ".|5004.0|", 5004, 50.04),
]


@pytest.mark.parametrize(("text", "gen", "ref", "absm", "relm"), test_args)
def test_descriptors_str(text, gen, ref, absm, relm):
    TOTAL_MASS = 10000.0
    mix = gbigsmiles.Mixture(text)
    assert str(mix) == ref
    assert mix.generate_string(False) == "."
    assert mix.generable == gen
    mix.system_mass = TOTAL_MASS
    assert abs(mix.absolute_mass - absm) < 1e-3
    assert abs(mix.relative_mass - relm) < 1e-3


if __name__ == "__main__":
    test_descriptors_str()
