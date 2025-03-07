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
    raise RuntimeError("Please make sure to install this module correctly via setuptools with setuptools_scm activated to generate a `_version.py` file.") from exc

from .atom import (
    AliphaticOrganic,
    AromaticOrganic,
    AromaticSymbol,
    Atom,
    AtomCharge,
    AtomClass,
    AtomSymbol,
    BracketAtom,
    Chiral,
    HCount,
    Isotope,
)
from .big_smiles import BigSmiles, BigSmilesMolecule, DotGeneration, DotSystemSize
from .bond import (
    BondDescriptor,
    BondDescriptorGeneration,
    BondDescriptorSymbol,
    BondDescriptorSymbolIdx,
    BondSymbol,
    InnerBondDescriptor,
    RingBond,
    SimpleBondDescriptor,
    TerminalBondDescriptor,
)
from .core import BigSMILESbase
from .distribution import (
    FlorySchulz,
    Gauss,
    LogNormal,
    Poisson,
    SchulzZimm,
    StochasticDistribution,
    Uniform,
)
from .nx_rdkit_mol import mol_graph_to_rdkit_mol
from .parser import get_global_parser
from .smiles import AtomAssembly, Branch, BranchedAtom, Dot, Smiles
from .stochastic import StochasticObject
from .transformer import GBigSMILESTransformer, get_global_transformer
from .util import camel_to_snake, get_global_rng, snake_to_camel

# from .graph_generate import AtomGraph
# from .mixture import Mixture
# from .mol_prob import get_ensemble_prob
# from .molecule import Molecule
# from .system import System
# from .token import SmilesToken

__all__ = [
    "__version__",
    "version_tuple",
    "Atom",
    "BracketAtom",
    "Isotope",
    "AtomSymbol",
    "Chiral",
    "HCount",
    "AtomCharge",
    "AtomClass",
    "AromaticSymbol",
    "AliphaticOrganic",
    "AromaticOrganic",
    "BondSymbol",
    "RingBond",
    "BondDescriptorSymbol",
    "BondDescriptorSymbolIdx",
    "BondDescriptorGeneration",
    "InnerBondDescriptor",
    "BondDescriptor",
    "SimpleBondDescriptor",
    "TerminalBondDescriptor",
    "BigSMILESbase",
    "camel_to_snake",
    "snake_to_camel",
    "get_global_rng",
    "GBigSMILESTransformer",
    "get_global_transformer",
    "get_global_parser",
    "FlorySchulz",
    "SchulzZimm",
    "Gauss",
    "LogNormal",
    "Poisson",
    "StochasticDistribution",
    "StochasticObject",
    "Uniform",
    "Branch",
    "BranchedAtom",
    "AtomAssembly",
    "Dot",
    "Smiles",
    "BigSmiles",
    "BigSmilesMolecule",
    "DotGeneration",
    "DotSystemSize",
    "mol_graph_to_rdkit_mol",
]
