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
            "bond_symbol",
            "ring_bond",
            "aliphatic_organic",
            "aromatic_organic",
            "branch",
            "smiles",
            "bond_descriptor_symbol",
            "bond_descriptor_generation",
            "ladder_bond_descriptor",
            "non_covalent_bond_descriptor",
            "bond_descriptor",
            "simple_bond_descriptor",
            "terminal_bond_descriptor",
            "stochastic_generation",
            "stochastic_distribution",
            "dot_system_size",
            "dot_generation",
            "dot",
            "isotope",
            "atom_symbol",
            "aromatic_symbol",
            "bracket_atom",
            "flory_schulz",
            "uniform",
            "schulz_zimm",
            "log_normal",
            "gauss",
            "poisson",
        ]
    parser = Lark(rf"{grammar_text}", start=start_tokens, keep_all_tokens=True)
    return parser


_GLOBAL_PARSER: None | Lark = None


def get_global_parser():
    global _GLOBAL_PARSER
    if _GLOBAL_PARSER is None:
        _GLOBAL_PARSER = _make_parser()

    return _GLOBAL_PARSER
