import lark
import pytest

import gbigsmiles


def test_chembl_smi_grammar(grammar_parser, chembl_smi_list):
    for smi in chembl_smi_list:
        assert grammar_parser.parse(smi)


def test_big_smi_grammar(grammar_parser, big_smi_list, big_smi_valid_unsupported):
    for smi in big_smi_list:
        assert grammar_parser.parse(smi)
    for smi in big_smi_valid_unsupported:
        assert grammar_parser.parse(smi)


def test_invalid_big_smi_grammar(grammar_parser, invalid_big_smi_list):
    for smi in invalid_big_smi_list:
        print(smi)
        with pytest.raises(lark.UnexpectedInput):
            tree = grammar_parser.parse(smi)
            print(tree.pretty())


def test_big_smi_graph(grammar_parser, big_smi_list):
    for smi in big_smi_list:
        print(smi)
        big_smi = gbigsmiles.BigSmiles.make(smi)
        gen_graph = big_smi.get_generating_graph()
        gen_graph.get_ml_graph(include_bond_descriptors=True)
        gen_graph.get_ml_graph(include_bond_descriptors=False)
