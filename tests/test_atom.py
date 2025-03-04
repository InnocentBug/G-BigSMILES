import pytest

from gbigsmiles.atom import Atom, BracketAtom
from gbigsmiles.parser import get_global_parser
from gbigsmiles.transformer import get_global_transformer


@pytest.mark.parametrize("string", ["B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"])
def test_simple_aliphatic_atom(string):
    tree = get_global_parser().parse(string, start="atom")
    a = get_global_transformer().transform(tree)
    assert a.aromatic is False
    assert str(a) == string

    b = Atom.make(string)
    assert b.aromatic is False
    assert str(b) == string
    assert b.charge == 0


@pytest.mark.parametrize(
    "string",
    [
        "b",
        "c",
        "n",
        "o",
        "p",
        "s",
    ],
)
def test_simple_aromatic_atom(string):
    a = Atom.make(string)
    assert a.aromatic is True
    assert str(a) == string
    assert a.charge == 0


@pytest.mark.parametrize("string", ["[se]", "[H]", "[He]", "[Li]", "[Be]", "[B]", "[C]"])
def test_simple_bracket_atom(string):
    a = BracketAtom.make(string)
    assert str(a) == string
    assert isinstance(a, Atom)
    assert a.charge == 0


@pytest.mark.parametrize("string", ["[se@]", "[H@@]", "[He@SP2]", "[Li@TB4]", "[Be@TB14]", "[B@OH3]", "[C@OH45]"])
def test_chiral_bracket_atom(string):
    a = BracketAtom.make(string)
    assert str(a) == string
    assert isinstance(a, Atom)


@pytest.mark.parametrize("string", ["[seH]", "[H]", "[BkH]", "[UH2]"])
def test_h_count_bracket_atom(string):
    a = BracketAtom.make(string)
    assert str(a) == string
    assert isinstance(a, Atom)


@pytest.mark.parametrize(
    ("string", "charge"),
    [
        ("[se-]", -1),
        ("[H+]", +1),
        ("[Bk--]", -2),
        ("[UH++]", 2),
        ("[C-2]", -2),
        ("[Pa+2]", +2),
        ("[Cf]", 0),
    ],
)
def test_charge_bracket_atom(string, charge):
    a = BracketAtom.make(string)
    assert str(a) == string
    assert isinstance(a, Atom)
    assert a.charge == charge


@pytest.mark.parametrize("string", ["[N:2]", "[p:1]", "[I:4]", "[s:14]"])
def test_class_bracket_atom(string):
    a = BracketAtom.make(string)
    assert str(a) == string
    assert isinstance(a, Atom)


def test_everything_atom():
    string = "[13C@OH1H2+1:3]"
    a = BracketAtom.make(string)
    assert str(a) == string
    assert a.isotope.num_nuclei == 13
    assert str(a.chiral) == "@OH1"
    assert a.h_count.num == 2
    assert str(a.h_count) == "H2"
    assert a.charge == 1
    assert a.atom_class.num == 3
    assert str(a.atom_class) == ":3"
