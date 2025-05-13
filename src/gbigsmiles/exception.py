# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details


class GBigSMILESError(Exception):
    """
    Generic Exception raised by G-BigSMILES.
    """

    pass


class GBigSMILESWarning(Warning):
    """
    Generic Warning raised by G-BigSMILES.
    """

    def __init__(self, token):
        self.token = token


class ParsingError(GBigSMILESError):
    """
    Parsing the Grammar went in an unanticipated manner.
    Please report bug with input string.
    """

    def __init__(self, token):
        super().__init__()
        self.token = token

    def __str__(self):
        return f"Unanticipated error while parsing. Please report and provide the input string. Token: {self.token}"


class ParsingWarning(GBigSMILESWarning):
    """
    Parsing the your string doesn't invalidate the grammar, but there is something to consider to fix.
    """

    pass


class TooManyTokens(ParsingError):
    def __init__(self, class_name, existing_token, new_token):
        super().__init__()
        self.class_name = class_name
        self.existing_token = existing_token
        self.new_token = new_token

    def __str__(self):
        string = f"Parsing Error {self.class_name} only expected one token, but got more. "
        string += f"The existing token is {self.existing_token} which conflicts with the new "
        string += f"token {self.new_token}. Most likely in implementation error, please report."
        return string


class UnknownDistribution(GBigSMILESError):
    def __init__(self, distribution_text: str):
        super().__init__()
        self.distribution_text = distribution_text

    def __str__(self):
        string = f"GBigSMILES a distribution with the following text {self.distribution_text} is unknown."
        string += " Typo or not implemented distribution."
        return string


class UnsupportedBigSMILES(GBigSMILESError):
    def __init__(self, token_type: str, other):
        self.token_type = token_type
        self.other = other

    def __str__(self):
        string = f"The provided token {self.token_type} is supported in BigSMILES but not in G-BigSMILES. "
        string += f"This token was requested by the following parsed text {self.other}."
        return string


class GenerationError(GBigSMILESError):
    pass


class DoubleBondSymbolDefinition(GenerationError):
    def __init__(self, partial_graph, symbol, bond_attributes):
        self.partial_graph = partial_graph
        self.symbol = symbol
        self.bond_attributes = bond_attributes

    def __str__(self):
        return f"{self.partial_graph}, {self.symbol}, {self.bond_attributes}"


class UndefinedDistribution(GenerationError):
    def __init__(self, stochastic_obj):
        self.stochastic_obj = stochastic_obj

    def __str__(self):
        return f"The stochastic distribution of the stochastic object {self.stochastic_obj} is not defined. The creation of the generative graph requires that the distribution is defined for each stochastic object."


class ConcatenatedBondDescriptors(ParsingError):
    def __init__(self, obj, stochastic_obj):
        self.obj = obj
        self.stochastic_obj = stochastic_obj

    def __str__(self):
        return f"The object {self.obj} in stochastic object {self.stochastic_obj} has concatenated bond descriptors which is forbidden by G-BigSmiles grammar."


class IncorrectNumberOfBondDescriptors(ParsingError):
    def __init__(self, obj, expected_number_of_bond_descriptors):
        self.obj = obj
        self.expected_number_of_bond_descriptors = expected_number_of_bond_descriptors

    def __str__(self):
        return f"Incorrect Number of BondDescriptors we expected {self.expected_number_of_bond_descriptors} but the object {str(self.obj)} of type {type(self.obj)} has {len(self.obj.bond_descriptors)}."


class SmilesHasNonZeroBondDescriptors(IncorrectNumberOfBondDescriptors):
    def __init__(self, obj):
        super().__init__(obj, 0)

    def __str__(self):
        return f"Outside of Stochastic Objects we expect {self.expected_number_of_bond_descriptors} bond descriptors, but the object {self.obj} of type {type(self.obj)} has the following bond descriptors {[str(bd) for bd in self.obj.bond_descriptors]}."


class MonomerHasTwoOrMoreBondDescriptors(IncorrectNumberOfBondDescriptors):
    def __init__(self, monomer_obj, stochastic_obj):
        super().__init__(monomer_obj, 2)

        self.stochastic_obj = stochastic_obj

    def __str__(self) -> str:
        return f"Monomer repeat units must have at least {self.expected_number_of_bond_descriptors} bond descriptors. But this object {str(self.obj)} has {len(self.obj.bond_descriptors)} bond descriptors inside this stochastic object {str(self.stochastic_obj)}."


class EndGroupHasOneBondDescriptors(IncorrectNumberOfBondDescriptors):
    def __init__(self, end_obj, stochastic_obj):
        super().__init__(end_obj, 1)

        self.stochastic_obj = stochastic_obj

    def __str__(self) -> str:
        return f"End groups must have exactly {self.expected_number_of_bond_descriptors} bond descriptor. But this object {str(self.obj)} has {len(self.obj.bond_descriptors)} bond descriptors inside this stochastic object {str(self.stochastic_obj)}."


class EmptyTerminalBondDescriptorWithoutEndGroups(ParsingWarning):
    def __init__(self, stochastic_object):
        super().__init__(stochastic_object)

    def __str__(self) -> str:
        string = f"The stochastic object {str(self.token)} has either, a left or right terminal bond descriptor as an empty '[]' bond descriptor. "
        string += f"In this case, please use end-groups to specify how to initiate (left terminal {str(self.token._left_terminal_bond_d)}) or finalize the molecule (right terminal {str(self.token._right_terminal_bond_d)}). "
        string += "Consider adding a hydrogen end group '{[] ... ; [$/</>][H] []}' with a matching bond descriptor symbol to your BigSMILES string for clarification."
        return string


class IncorrectNumberOfTransitionWeights(ParsingError):
    def __init__(self, token, bond_descriptor, expected_length):
        super().__init__(token)
        self.bond_descriptor = bond_descriptor
        self.expected_length = expected_length
        if self.bond_descriptor.transition is not None:
            raise RuntimeError(f"Implementation error, please report on GitHub https://github.com/InnocentBug/G-BigSMILES/issues . {str(self.bond_descriptor)} {str(self.token)} ")

    def __str__(self):
        return f"The bond descriptor '{str(self.bond_descriptor)}' from the stochastic object '{str(self.token)}' specifies {len(self.bond_descriptor.transition)} transition weights, but the stochastic object has {self.expected_length} bond descriptors. Adjust the transition weights to match the bond descriptors."


class NoInitiationForStochasticObject(ParsingWarning):
    def __init__(self, stochastic_obj, partial_graph):
        super().__init__(stochastic_obj)
        self.partial_graph = partial_graph

    def __str__(self):
        return f"The stochastic object {str(self.token)} cannot generate entry points to start initiations. Check if the left terminal bond descriptor is meant to be empty or if you have correct end groups that can act as initiators."


class NoLeftTransitions(ParsingWarning):
    def __init__(self, stochastic_obj, partial_graph):
        super().__init__(stochastic_obj)
        self.partial_graph = partial_graph

    def __str__(self):
        return f"The stochastic object {str(self.token)} cannot generate left connections, check the left terminal bond descriptor and how it relates to monomeric repeat units."


class StochasticMissingPath(ParsingWarning):
    def __init__(self, stochastic_obj, source_bd_pos):
        super().__init__(stochastic_obj)
        self.source_bd_pos = source_bd_pos

    def __str__(self):
        return f"The stochastic object {str(self.token)} defines that it can be entered via the bond descriptor in position {str(self.source_bd_pos)} as defined by the left terminal descriptor. However, when entered there there is no path to reach any of the exit bond descriptors as defined by the right terminal bond descriptor."


class IncompatibleBondTypeBondDescriptor(ParsingWarning):
    def __init__(self, bond_type_lhs, bond_type_rhs):
        super().__init__(None)
        self.bond_type_lhs = bond_type_lhs
        self.bond_type_rhs = bond_type_rhs

    def __str__(self):
        return f"There is a connection between bond descriptors with different bond types, the left is {str(self.bond_type_lhs)} and the right is {str(self.bond_type_rhs)}. There will be no generation path, since the bond type is undefined. This maybe an incorrect input BigSMILES, check the bond types for compatibility around the bond descriptors: connecting bond descriptors have of same type i.e. '[<]=CC[>]' is invalid, since it connects a double bond = with a single bond implicit `-`, correct would be `[<]=CC=[>]`."


class UnvalidatedGenerationSource(GBigSMILESWarning):
    def __init__(self, source, known_source_ids, graph):
        self.source = source
        self.known_source_ids = known_source_ids
        self.graph = graph

    def __str__(self):
        return f"Attempt to create an atom graph from a generating graph with source node_idx {self.source} but this is not one of the known starting points ({self.known_source_ids}) of the generating graph."


class InvalidGenerationSource(GBigSMILESError):
    def _init__(self, source, nodes, graph):
        self.source = source
        self.nodes = nodes
        self.graph = graph

    def __str__(self):
        return str(self.__dict__)
        return f"Attempt to create and atom graph from a generating graph with a source node_idx {self.source} but this source is not a valid node idx of the graph. Valid node idx {self.nodes}."


class TooManyBondDescriptorsPerAtomForGeneration(ParsingWarning):
    def __init__(self, graph, text, atom_idx, bd_idx_set):
        self.graph = graph
        self.atom_idx = atom_idx
        self.bd_idx_set = bd_idx_set
        self.text = text

        token = self.graph.nodes[atom_idx]["obj"]
        super().__init__(token)
        self.bd_objs = [self.graph.nodes[idx]["obj"] for idx in self.bd_idx_set]

    def __str__(self):
        return f"For generation purposes any atom can only be attached to at most one Bond Descriptor. That atom {self.token} from {self.text} is however connected to {len(self.bd_idx_set)} bond descriptors {[str(obj) for obj in self.bd_objs]}. This can usual be fixed by rearranging your polymer description."


class IncompleteStochasticGeneration(GBigSMILESError):
    def __init__(self, partial_atom_graph):
        self._partial_atom_graph = partial_atom_graph
        self.atom_graph = partial_atom_graph.atom_graph

    @property
    def num_open_bonds(self):
        num_bonds = 0
        for sto_atom_id in self._partial_atom_graph._open_half_bond_map:
            for _bond in self._partial_atom_graph._open_half_bond_map[sto_atom_id]:
                num_bonds += 1
        return num_bonds

    def __str__(self):
        num_bonds = self.num_open_bonds
        if num_bonds == 0:
            return f"Incomplete Stochastic Generation: since there are {num_bonds} open bonds this may be intended. You can catch this exception and use the `atom_graph` property as a result."
        return f"Incomplete Stochastic Generation: {num_bonds} are still unaccounted for this is likely an imprecise G-BigSMILES string or a bug."
