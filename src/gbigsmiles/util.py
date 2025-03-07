# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy
import re

import numpy as np

_GLOBAL_RNG: None | np.random.Generator = None


class RememberAdd:
    def __init__(self, value):
        self._value = value
        self._previous = 0.0

    @property
    def value(self):
        return self._value

    @property
    def previous(self):
        return self._previous

    def __iadd__(self, other):
        old_value = self._value
        self._value += other
        self._previous = old_value
        return self

    def __add__(self, other):
        tmp = copy(self)
        tmp += other
        return tmp

    def _radd__(self, other):
        return self + other

    def __eq__(self, other):
        return self.value == other.value and self.previous == other.previous


def snake_to_camel(snake_str):
    """
    Convert a string from snake_case to CamelCase.

    Args:
    ----
        snake_str (str): The snake_case string to convert.

    Returns:
    -------
        str: The converted CamelCase string.

    Examples:
    --------
        >>> snake_to_camel("snake_case_string")
        'SnakeCaseString'
        >>> snake_to_camel("another_example_here")
        'AnotherExampleHere'
        >>> snake_to_camel("simple_test")
        'SimpleTest'
        >>> snake_to_camel("alreadyCamelCase")
        'AlreadyCamelCase'

    """
    camel_str = ""
    capitalize_next = False

    for char in snake_str:
        if char == "_":
            capitalize_next = True
        elif capitalize_next:
            camel_str += char.upper()
            capitalize_next = False
        else:
            camel_str += char

    return camel_str[0].upper() + camel_str[1:]


def camel_to_snake(name):
    """
    Convert a string from CamelCase to snake_case, handling acronyms correctly.

    Args:
    ----
        name (str): The CamelCase string to convert.

    Returns:
    -------
        str: The converted snake_case string.

    Examples:
    --------
        >>> camel_to_snake("camelCaseString")
        'camel_case_string'
        >>> camel_to_snake("XMLHttpRequest")
        'xml_http_request'
        >>> camel_to_snake("HTTPRequest")
        'http_request'

    """
    # First, handle the case where an acronym is followed by a word
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Then, handle the case where there are consecutive uppercase letters (acronyms)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    # Finally, handle the case where an acronym is at the start of the string
    s3 = re.sub("([A-Z])([A-Z][a-z])", r"\1_\2", s2)
    return s3.lower()


def get_global_rng(seed=None):
    global _GLOBAL_RNG
    if _GLOBAL_RNG is None:
        _GLOBAL_RNG = np.random.default_rng(seed)

    return _GLOBAL_RNG


def _determine_darkness_from_hex(color):
    """
    Determine the darkness of a color from its hex string.

    Arguments:
    ---------
    color: str
       7 character string with prefix `#` followed by RGB hex code.

    Returns: bool
       if the darkness is below half

    """
    # If hex --> Convert it to RGB: http://gist.github.com/983661
    if color[0] != "#":
        raise ValueError(f"{color} is missing '#'")
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    # HSP (Highly Sensitive Poo) equation from http://alienryderflex.com/hsp.html
    hsp = np.sqrt(0.299 * red**2 + 0.587 * green**2 + 0.114 * blue**2)
    return hsp < 127.5
