# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

from importlib.resources import files

from lark import Lark


def _make_parser(filename=None, start_tokens=None):
    if filename is None:
        filename = files("gbigsmiles").joinpath("data", "g-bigsmiles.lark")
    with open(filename, "r") as file_handle:
        grammar_text = file_handle.read()

    if start_tokens is None:
        start_tokens = [
            "big_smiles",
            "big_smiles_molecule",
            "stochastic_object",
            "atom",
            "big_smiles_fragment_declaration",
            "big_smiles_fragment_definition",
            "bond",
            "ringbond",
            "branch",
            "smiles",
            "bond_descriptor_symbol",
            "bond_descriptor_generation",
            "ladder_bond_descriptor",
            "non_covalent_bond_descriptor",
            "bond_descriptor",
            "terminal_bond_descriptor",
            "stochastic_generation",
            "stochastic_distribution",
            "dot_system_size",
            "dot_generation",
            "dot",
            "end_group",
            "isotope",
            "atom_symbol",
            "aromatic_symbol",
            "bracket_atom",
        ]
    parser = Lark(rf"{grammar_text}", start=start_tokens, keep_all_tokens=True)
    return parser


_GLOBAL_PARSER = _make_parser()
