import json
import os
from importlib.resources import files

import lark
import networkx as nx
import pytest


@pytest.fixture(scope="session")
def smi_dict():
    path = os.path.dirname(__file__)
    with open(os.path.join(path, "smi.json"), "r") as file_handle:
        data = json.load(file_handle)
    return data


@pytest.fixture(scope="session")
def graph_validation_dict():
    path = os.path.dirname(__file__)
    with open(os.path.join(path, "graph_validation.json"), "r") as file_handle:
        raw_data = json.load(file_handle)
    data = {}
    for string in raw_data:
        data[string] = nx.adjacency_graph(raw_data[string])
    return data


@pytest.fixture(scope="session")
def chembl_smi_list(smi_dict):
    return smi_dict["chembl_smiles"]


@pytest.fixture(scope="session")
def big_smi_list(smi_dict):
    return smi_dict["big_smiles"]


@pytest.fixture(scope="session")
def big_smi_valid_unsupported(smi_dict):
    return smi_dict["valid_but_unsupported_big_smiles"]


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
