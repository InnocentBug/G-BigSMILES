#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

"""Main entry point for the generation of SMILES string ensembles.

This function serves as a convenience entry point.
"""

import argparse
import sys

import gbigsmiles


def _print_license(args):
    license_text = "\nbigSMILES extension to generate ensembles of polymer smiles strings.\n\n"
    license_text += "Copyright (C) 2022 Ludwig Schneider\n"
    license_text += "This program is free software: you can redistribute it and/or modify\n"
    license_text += "it under the terms of the GNU General Public License as published by\n"
    license_text += "the Free Software Foundation, either version 3 of the License, or\n"
    license_text += "(at your option) any later version.\n"
    license_text += "\n"
    license_text += "This program is distributed in the hope that it will be useful,\n"
    license_text += "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
    license_text += "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
    license_text += "GNU General Public License for more details.\n"
    license_text += "\n"
    license_text += "You should have received a copy of the GNU General Public License\n"
    license_text += "along with this program.  If not, see <http://www.gnu.org/licenses/>.\n"

    print(license_text)


SHORT_LICENSE_TEXT = "bigSMILES extension  Copyright (C) 2022 - 2025  Ludwig Schneider\n\n"
SHORT_LICENSE_TEXT += "This program comes with ABSOLUTELY NO WARRANTY; for details type 'license'.\n"
SHORT_LICENSE_TEXT += "This is free software, and you are welcome to redistribute it\n"
SHORT_LICENSE_TEXT += "under certain conditions; type `license' for details.\n"


def main(argv):
    """
    Main entry point function for the generation of SMILES string ensembles.
    Pass '--help' for more options.
    """

    print(SHORT_LICENSE_TEXT)
    main_parser = argparse.ArgumentParser(
        description="bigSMILES generation -- automatic generation of SMILES string a bigSMILES ensemble.",
        prog="gbigsmiles",
    )
    main_parser.add_argument("--version", "-v", action="version", version="%(prog)s " + gbigsmiles.__version__)
    main_parser.set_defaults(func=lambda x: main_parser.print_usage())

    subparsers = main_parser.add_subparsers(help="sub-command help")
    license_parser = subparsers.add_parser("license", help="Print license details: GPL-3")
    license_parser.set_defaults(func=_print_license)

    # Parsing and setup
    args = main_parser.parse_args(argv[1:])
    args.func(args)


if __name__ == "__main__":
    main(sys.argv)
