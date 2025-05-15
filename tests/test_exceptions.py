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


invalid_bond_descriptor_sequence = [
    "{[] [>]CC[>][<] []}|gauss(400.,50.)|",
    "{[] [<]N(CC[>][<])N[>], [>]CC[<]; [$][H], [<]Br []}|gauss(100.,20.)|",
    "{[<] [<]NN[>], [$]CC[$][$]; [$][H], [<]Br [>]}|gauss(400.,20.)|",
    "{[<] [<]NN[>], [>]CC[<]; [$][$][H], [<]Br [>]}|gauss(100.,14.)|",
    "{[<] [>]CC(O[>2]{[<2] [>2]N(N[$]{[$] [$][$]CC[$]; [$][H] []}|gauss(100.,10.)|)NN[<2]; [>2][H] []}|gauss(100.,10.)|)[<] [>]}|gauss(100,10)|",
    "{[<] [>]CC(O[>2]{[<2] [>2][$]NN[$][<2]; [>2][H] []}|gauss(100.,10.)|)[<] [>]}|gauss(100,10)|",
]


@pytest.mark.parametrize("stochastic_smi", invalid_bond_descriptor_sequence)
def test_bond_descriptor_sequence(stochastic_smi):
    with pytest.raises(gbigsmiles.exception.ConcatenatedBondDescriptors):
        try:
            gbigsmiles.StochasticObject.make(stochastic_smi)
        except lark.exceptions.VisitError as exc:
            raise exc.__context__  # trunk-ignore(ruff/B904)


valid_concatenated_bond_descriptors = [
    "{[] [>]CC[>][<] []}",
    "{[] [<]N(CC[>][<])N[>], [>]CC[<]; [$][H], [<]Br []}",
    "{[<] [<]NN[>], [$]CC[$][$]; [$][H], [<]Br [>]}",
    "{[<] [>]CC(O[>2]{[<2] [>2]N(N[$]{[$] [$][$]CC[$]; [$][H] []})NN[<2]; [>2][H] []})[<] [>]}",
    "{[<] [>]CC(O[>2]{[<2] [>2][$]NN[$][<2]; [>2][H] []})[<] [>]}",
]


@pytest.mark.parametrize("stochastic_smi", valid_concatenated_bond_descriptors)
def test_valid_concatenated_bond_descriptors(stochastic_smi):
    gbigsmiles.StochasticObject.make(stochastic_smi)


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
        "{[] [$]CCC[$] []}|poisson(10)|",
        "{[] [$][C-][$] []}|flory_schulz(0.8)|",
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


@pytest.mark.parametrize("smi", ["{[$] [>]CC[<] [>]}|flory_schulz(0.9)|"])
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
        "{[] [$]C[$]; [$][H] []}|schulz_zimm(1000.,500.)|",
        "{[] [<]C([<2])C[>], [<2]NN[>2]; [>][H], [<][H], [>2][H], [<2][H] []}|poisson(100)|",
        "{[] [<]C([<2])C[>], [<2]N[>2]; [>][H], [<][H], [>2][H], [<2][H] []}|gauss(500.,50.)|",
    ],
)
def test_warn_too_many_bd_per_atom(smi):
    big_smi = gbigsmiles.BigSmiles.make(smi)
    with pytest.warns(gbigsmiles.exception.TooManyBondDescriptorsPerAtomForGeneration):
        big_smi.get_generating_graph()


undefined_distribution = [
    "{[] [$]CC[$]; [$]C []}",
    "{[] [$]CC([$2]{[$2] [$2]OO[$2]; [$2]Cl []})C[$]; [$]C []}|poisson(20)|",
]


@pytest.mark.parametrize("smi", undefined_distribution)
def test_undefined_distribution(smi):
    with pytest.raises(gbigsmiles.exception.UndefinedDistribution):
        obj = gbigsmiles.BigSmiles.make(smi)
        obj.get_generating_graph()
