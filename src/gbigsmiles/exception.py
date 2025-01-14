# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details


class GBigSMILESError(Exception):
    """
    Generic Exception raised by G-BigSMILES.
    """

    pass


class ParsingError(GBigSMILESError):
    """
    Parsing the Grammar went in an unanticipated manner.
    Please report bug with input string.
    """

    def __init__(self, token):
        super().__init__()
        self.token = token

    def __str__(self):
        return f"Unanticipated error while parsing. Please report and provide the input string. Token: {self.token} start: {self.token.start_pos}"


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
