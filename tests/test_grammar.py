import os
import pytest
import json
from importlib.resources import files
import lark
import gbigsmiles


@pytest.fixture(scope="session")
def smi_dict():
    path = os.path.dirname(__file__)
    with open(os.path.join(path, "smi.json"), "r") as file_handle:
        data = json.load(file_handle)
    return data

@pytest.fixture(scope="session")
def chembl_smi_list(smi_dict):
    return smi_dict["chembl_smiles"]

@pytest.fixture(scope="session")
def big_smi_list(smi_dict):
    return smi_dict["big_smiles"]

@pytest.fixture(scope="session")
def invalid_big_smi_list(smi_dict):
    return smi_dict["invalid_big_smiles"]
@pytest.fixture(scope="session")
def grammar_text():
    grammar_file = files("gbigsmiles").joinpath("data", "g-bigsmiles.lark")
    with open(grammar_file, "r") as file_handle:
        grammar_text = file_handle.read()
    return grammar_text

@pytest.fixture(scope="session")
def grammar_parser(grammar_text):
    parser = lark.Lark(rf"{grammar_text}", start="big_smiles")
    return parser


def test_chembl_smi_grammar(grammar_parser, chembl_smi_list):
    for smi in chembl_smi_list:
        assert grammar_parser.parse(smi)

def test_big_smi_grammar(grammar_parser, big_smi_list):
    for smi in big_smi_list:
        assert grammar_parser.parse(smi)

def test_invalid_big_smi_grammar(grammar_parser, invalid_big_smi_list):
    for smi in invalid_big_smi_list:
        print(smi)
        with pytest.raises(lark.UnexpectedInput):
            grammar_parser.parse(smi)
