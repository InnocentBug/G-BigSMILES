# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details


"""
bigSMILES extension to generation SMILES string ensembles.
"""

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError as exc:
    raise RuntimeError(
        "Please make sure to install this module correctly via setuptools with setuptools_scm activated to generate a `_version.py` file."
    ) from exc
from .bond import BondDescriptor
from .core import _GLOBAL_RNG, BigSMILESbase, reaction_graph_to_dot_string
from .distribution import Distribution, FlorySchulz, Gauss
from .graph_generate import AtomGraph
from .mixture import Mixture
from .mol_prob import get_ensemble_prob
from .molecule import Molecule
from .stochastic import Stochastic
from .system import System
from .token import SmilesToken

__all__ = [
    "__version__",
    "version_tuple",
    "BondDescriptor",
    "_GLOBAL_RNG",
    "BigSMILESbase",
    "reaction_graph_to_dot_string",
    "Distribution",
    "FlorySchulz",
    "Gauss",
    "AtomGraph",
    "Mixture",
    "get_ensemble_prob",
    "Molecule",
    "Stochastic",
    "System",
    "SmilesToken",
]
