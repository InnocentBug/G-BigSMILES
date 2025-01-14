import lark
import pytest


def test_chembl_smi_grammar(grammar_parser, chembl_smi_list):
    for smi in chembl_smi_list:
        assert grammar_parser.parse(smi)


def test_big_smi_grammar(grammar_parser, big_smi_list):
    for smi in big_smi_list:
        assert grammar_parser.parse(smi)


def test_invalid_big_smi_grammar(grammar_parser, invalid_big_smi_list):
    for smi in invalid_big_smi_list:
        with pytest.raises(lark.UnexpectedInput):
            tree = grammar_parser.parse(smi)
            print(tree.pretty())
