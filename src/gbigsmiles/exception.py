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
        self.token = token

    def __str__(self):
        return f"Unanticipated error while parsing. Please report and provide the input string. Token: {self.token} start: {self.token.start_pos}"


class TooManyTokens(ParsingError):
    def __init__(self, class_name, existing_token, new_token):
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
