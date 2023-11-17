# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details


"""
bigSMILES extension to generation SMILES string ensembles.
"""

from . import distribution

try:
    from ._version import version as __version__  # noqa: E402, F401
    from ._version import version_tuple
except ImportError as exc:
    raise RuntimeError(
        "Please make sure to install this module correctly via setuptools with setuptools_scm activated to generate a `_version.py` file."
    ) from exc
from .bond import BondDescriptor
from .core import _GLOBAL_RNG, BigSMILESbase, reaction_graph_to_dot_string
from .distribution import Distribution, FlorySchulz, Gauss
from .mixture import Mixture
from .mol_prob import get_ensemble_prob
from .molecule import Molecule
from .stochastic import Stochastic
from .system import System
from .token import SmilesToken
