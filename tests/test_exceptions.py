import lark
import pytest

import gbigsmiles


@pytest.mark.parametrize("invalid_name", ("asdf", "xkcd"))
def test_unknown_distribution(invalid_name):
    with pytest.raises(lark.exceptions.UnexpectedCharacters):
        gbigsmiles.BigSmiles.make("C{[$] [$]CC[$] [$]}" + f"|{invalid_name}(4, 3)|C")

    with pytest.raises(gbigsmiles.exception.UnknownDistribution):
        gbigsmiles.StochasticDistribution.make(f"{invalid_name}(4, 3)")


def test_unsupported_bigsmiles(big_smi_valid_unsupported):
    for smi in big_smi_valid_unsupported:
        with pytest.raises(gbigsmiles.exception.UnsupportedBigSMILES):
            try:
                gbigsmiles.BigSmiles.make(smi)
            except lark.exceptions.VisitError as exc:
                raise exc.__context__  # trunk-ignore(ruff/B904)


@pytest.mark.parametrize("invalid_smiles", [])
def test_double_bond_symbol_smiles(invalid_smiles):
    with pytest.raises(gbigsmiles.exception.DoubleBondSymbolDefinition):
        gbigsmiles.Smiles.make(invalid_smiles)


invalid_monomer_stochastic = [
    "{[] [$]CC []}",
    "{[] CC []}",
    "{[] [$]CC[$], CC; [$]Br []}",
    "{[$] [$]CC[$], [$]CC; [$]Br [$]}",
    "{[<] [>]CC, [<]C([$])C[>]; [$]Br [>]}",
]


@pytest.mark.parametrize("stochastic_smi", invalid_monomer_stochastic)
def test_invalid_monomer_stochastic(stochastic_smi):
    with pytest.raises(gbigsmiles.exception.MonomerHasTwoOrMoreBondDescriptors):
        try:
            gbigsmiles.StochasticObject.make(stochastic_smi)
        except lark.exceptions.VisitError as exc:
            raise exc.__context__  # trunk-ignore(ruff/B904)
        r
    with pytest.raises(gbigsmiles.exception.IncorrectNumberOfBondDescriptors):
        try:
            gbigsmiles.StochasticObject.make(stochastic_smi)
        except lark.exceptions.VisitError as exc:
            raise exc.__context__  # trunk-ignore(ruff/B904)


invalid_end_stochastic = [
    "{[] [$]CC[$]; [$]C[$] []}",
    "{[] [$]CC[$]; C []}",
    "{[] [$]CC[$]; [$]Br, N []}",
    "{[$] [$]CC[$], [$]CC[$]; [$]Br, [$]CC[$] [$]}",
]


@pytest.mark.parametrize("stochastic_smi", invalid_end_stochastic)
def test_invalid_end_stochastic(stochastic_smi):
    with pytest.raises(gbigsmiles.exception.EndGroupHasOneBondDescriptors):
        try:
            gbigsmiles.StochasticObject.make(stochastic_smi)
        except lark.exceptions.VisitError as exc:
            raise exc.__context__  # trunk-ignore(ruff/B904)

    with pytest.raises(gbigsmiles.exception.IncorrectNumberOfBondDescriptors):
        try:
            gbigsmiles.StochasticObject.make(stochastic_smi)
        except lark.exceptions.VisitError as exc:
            raise exc.__context__  # trunk-ignore(ruff/B904)


@pytest.mark.parametrize(
    "smi",
    [
        "{[] [$]CCC[$] []}",
        "{[] [$][C-][$] []}",
    ],
)
def test_warn_empty_terminal_bond_descriptor_without_end_groups(smi):
    with pytest.warns(gbigsmiles.exception.EmptyTerminalBondDescriptorWithoutEndGroups):
        gbigsmiles.BigSmiles.make(smi)
