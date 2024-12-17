# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import re

import numpy as np

_GLOBAL_RNG: None | np.random.Generator = None


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
