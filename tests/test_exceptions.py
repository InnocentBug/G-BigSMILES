import lark
import pytest

import gbigsmiles


@pytest.mark.parametrize("invalid_name", ("Alfred", "Hitch"))
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

invalid_bond_descriptor_sequence = [
    "{[] [>]CC[>][<] []}",
    "{[] [<]N(CC[>][<])N[>], [>]CC[<]; [$][H], [<]Br []}",
    "{[<] [<]NN[>], [$]CC[$][$]; [$][H], [<]Br [>]}",
    "{[<] [<]NN[>], [>]CC[<]; [$][$][H], [<]Br [>]}",
]

# @Gervasio reactivate, when we fix the problem catching
# @pytest.mark.parametrize("stochastic_smi", invalid_bond_descriptor_sequence)
# def test_bond_descriptor_sequence(stochastic_smi):
#     with pytest.raises(gbigsmiles.exception.TwoConsecutiveBondDescriptors):
#         try:
#             gbigsmiles.StochasticObject.make(stochastic_smi)
#         except lark.exceptions.VisitError as exc:
#             raise exc.__context__  # trunk-ignore(ruff/B904)


@pytest.mark.parametrize("stochastic_smi", invalid_monomer_stochastic)
def test_invalid_monomer_stochastic(stochastic_smi):
    with pytest.raises(gbigsmiles.exception.MonomerHasTwoOrMoreBondDescriptors):
        try:
            gbigsmiles.StochasticObject.make(stochastic_smi)
        except lark.exceptions.VisitError as exc:
            raise exc.__context__  # trunk-ignore(ruff/B904)

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


@pytest.mark.parametrize("smi", [])
def test_warn_no_initiation_for_stochastic_object(smi):
    with pytest.warns(gbigsmiles.exception.NoInitiationForStochasticObject):
        obj = gbigsmiles.BigSmiles.make(smi)
        obj.get_generating_graph()


@pytest.mark.parametrize("smi", ["{[$] [>]CC[<] [>]}"])
def test_warn_stochastic_missing_path(smi):
    with pytest.warns(gbigsmiles.exception.StochasticMissingPath):
        obj = gbigsmiles.BigSmiles.make(smi)
        obj.get_generating_graph()


@pytest.mark.parametrize("smi", [])
def test_warn_incompatible_bond_type_bond_descriptors(smi):
    with pytest.warns(gbigsmiles.exception.IncompatibleBondTypeBondDescriptor):
        obj = gbigsmiles.BigSmiles.make(smi)
        obj.get_generating_graph()


@pytest.mark.parametrize(
    "smi",
    [
        "{[] [$]C[$]; [$][H] []}",
        "{[] [<]C([<2])C[>], [<2]NN[>2]; [>][H], [<][H], [>2][H], [<2][H] []}",
        "{[] [<]C([<2])C[>], [<2]N[>2]; [>][H], [<][H], [>2][H], [<2][H] []}",
    ],
)
def test_warn_too_many_bd_per_atom(smi):
    big_smi = gbigsmiles.BigSmiles.make(smi)
    with pytest.warns(gbigsmiles.exception.TooManyBondDescriptorsPerAtomForGeneration):
        big_smi.get_generating_graph()
