# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details


class GBigSMILESError(Exception):
    """
    Generic Exception raised by G-BigSMILES.
    """

    pass


class GBigSMILESParsingError(GBigSMILESError):
    """
    Parsing the Grammar went in an unanticipated manner.
    Please report bug with input string.
    """

    def __init__(self, token):
        self.token = token

    def __str__(self):
        return f"Unanticipated error while parsin. Please report and provide the input string. Token: {token} start: {token.start_pos}"


class GBigSMILESInitNotEnoughError(GBigSMILESError):
    """
    GBigSMILES classes usually need to be initialized either via text,
    or as part of parsing a different string.

    If this isn't followed, this exception is raise.
    Initialize the elements of G-BigSMILES with (part of) a G-BigSMILES string.
    """

    def __init__(self, class_name):
        self.class_name = class_name

    def __str__(self):
        return f"Attempt to initialize {self.class_name} without sufficient arguments. Initialize objects of {self.class_name} by passing (part of) a G-BigSMILES string."


class GBigSMILESInitTooMuchError(GBigSMILESError):
    """
    GBigSMILES classes usually need to be initialized either via text,
    or as part of parsing a different string, but not both.

    If this isn't followed, this exception is raise.
    Initialize the elements of G-BigSMILES with (part of) a G-BigSMILES string.
    """

    def __init__(self, class_name):
        self.class_name = class_name

    def __str__(self):
        return f"Attempt to initialize {self.class_name} with tree and text arguments. Initialize objects of {self.class_name} by passing (part of) a G-BigSMILES string."
