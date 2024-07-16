# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors as rdDescriptors
from rdkit.Chem import rdFingerprintGenerator

from .forcefield_helper import FfAssignmentError, get_assignment_class

_RDKGEN = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=512)


class MolGen:
    """
    Wrapper class to hold incompletely generated molecules.
    It mainly holds the incompletely generated rdkit.mol,
    as well as a list of not completed bond_descriptors.
    """

    def __init__(self, token):
        """
        Generate a new incompletely generated molecule from a token.

        Arguments:
        ---------
        token: SmilesToken
           Token, that starts a new molecule generation.

        """
        if not token.generable:
            raise RuntimeError(f"Attempting to generate token {str(token)}, which isn't generable.")
        self.bond_descriptors = copy.deepcopy(token.bond_descriptors)
        self.graph = nx.Graph()
        if len(token.residues) != 1:
            raise ValueError(f"{token} is not long enough")
        res_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        res_name = res_names[token.res_id]
        smiles = token.generate_smiles_fragment()
        params = Chem.SmilesParserParams()
        params.removeHs = True
        mol = Chem.MolFromSmiles(smiles, params.removeHs)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol, maxIters=200000)
        rdFP = _RDKGEN.GetFingerprint(mol)
        self.graph.add_node(0, smiles=smiles, big_smiles=str(token), rdFP=rdFP)
        for bd in self.bond_descriptors:
            # Our graph has only 1 node and all BD are associated with that.
            bd.node_idx = 0
        self._mol = mol

        atom_serial = 1
        elem_count = {}
        for atom in self._mol.GetAtoms():
            monomer_info = Chem.AtomPDBResidueInfo()
            atom.SetMonomerInfo(monomer_info)
            elem_count[atom.GetAtomicNum()] = atom_serial
            atom_serial = min(atom_serial, 99)
            monomer_info = atom.GetPDBResidueInfo()
            monomer_info.SetResidueName(res_name)
            monomer_info.SetResidueNumber(token.res_id)
            monomer_info.SetIsHeteroAtom(False)
            monomer_info.SetOccupancy(0.0)
            monomer_info.SetTempFactor(0.0)
            if monomer_info.GetSerialNumber() == 0:
                atom_name = "%2s%02d" % (atom.GetSymbol(), atom_serial)
                monomer_info.SetName(atom_name)
                monomer_info.SetSerialNumber(atom_serial)
                atom_serial += 1

            if atom_serial - 1 > 99:
                raise ValueError(
                    "Number of atoms exceeds 99. This causes issues in properly labeling the atoms."
                )

    @property
    def fully_generated(self):
        """
        Is this molecule fully generated?.
        """
        return len(self.bond_descriptors) == 0

    def attach_other(self, self_bond_idx: int, other, other_bond_idx: int):
        """
        Combine two molecules and store the result in this one.

        Arguments:
        ---------
        self_bond_idx: int
           Bond descriptor index of this molecule, that is going to bind.

        other: MolGen
           Other half generated molecules, that binds to this one.

        other_bond_idx: int
           Index of the BondDescriptor in the other molecule, that is going to bind.

        """

        if self_bond_idx >= len(self.bond_descriptors):
            raise RuntimeError(f"Invalid bond descriptor id {self_bond_idx} (self).")
        if other_bond_idx >= len(other.bond_descriptors):
            raise RuntimeError(f"Invalid bond descriptor id {other_bond_idx} (other).")

        current_atom_number = len(self._mol.GetAtoms())
        other_bond_descriptors = copy.deepcopy(other.bond_descriptors)

        if not other_bond_descriptors[other_bond_idx].is_compatible(
            self.bond_descriptors[self_bond_idx]
        ):
            print(self.bond_descriptors)
            raise RuntimeError(
                f"Unable to attach mols, because bond descriptor {str(other_bond_descriptors[other_bond_idx])}"
                f" is incompatible with {str(self.bond_descriptors[self_bond_idx])}."
            )
        self_graph_len = len(self.graph)

        # Align position in space
        self_bond_point = self._mol.GetConformer().GetAtomPosition(
            self.bond_descriptors[self_bond_idx].atom_bonding_to
        )
        other_bond_point = other._mol.GetConformer().GetAtomPosition(
            other_bond_descriptors[other_bond_idx].atom_bonding_to
        )

        rcm = np.zeros(3)
        for i in range(other._mol.GetConformer().GetNumAtoms()):
            rcm += other._mol.GetConformer().GetAtomPosition(i)
        rcm /= other._mol.GetConformer().GetNumAtoms()
        rg2 = np.zeros(3)
        for i in range(other._mol.GetConformer().GetNumAtoms()):
            rg2 += (rcm - other._mol.GetConformer().GetAtomPosition(i)) ** 2
        rg2 /= other._mol.GetConformer().GetNumAtoms()
        rg_len = np.sqrt(np.sum(rg2))
        if rg_len < 0.5:
            rg_len = 1
        offset = np.asarray((rg_len, 0, 0))

        for i in range(other._mol.GetConformer().GetNumAtoms()):
            old_pos = other._mol.GetConformer().GetAtomPosition(i)
            new_pos = old_pos - other_bond_point + self_bond_point + offset
            other._mol.GetConformer().SetAtomPosition(i, new_pos)

        for i in range(other._mol.GetConformer().GetNumAtoms()):
            rcm += other._mol.GetConformer().GetAtomPosition(i)
        rcm /= other._mol.GetConformer().GetNumAtoms()

        for bd in other_bond_descriptors:
            bd.atom_bonding_to += current_atom_number
            bd.node_idx += self_graph_len

        # print([atom.GetSymbol() for atom in self._mol.GetAtoms()],
        #       [bd.atom_bonding_to for bd in self.bond_descriptors])
        # print([atom.GetSymbol() for atom in other._mol.GetAtoms()],
        #       [bd.atom_bonding_to - current_atom_number for bd in other_bond_descriptors])

        new_mol = Chem.CombineMols(self._mol, other._mol)
        new_mol = Chem.EditableMol(new_mol)

        new_mol.AddBond(
            self.bond_descriptors[self_bond_idx].atom_bonding_to,
            other_bond_descriptors[other_bond_idx].atom_bonding_to,
            self.bond_descriptors[self_bond_idx].bond_type,
        )
        self.graph = nx.disjoint_union(self.graph, other.graph)
        self.graph.add_edge(
            self.bond_descriptors[self_bond_idx].node_idx,
            other_bond_descriptors[other_bond_idx].node_idx,
            bond_type=self.bond_descriptors[self_bond_idx].bond_type,
        )

        # Remove bond descriptors from list, as they have reacted now.
        del self.bond_descriptors[self_bond_idx]
        del other_bond_descriptors[other_bond_idx]

        self.bond_descriptors += other_bond_descriptors

        self._mol = new_mol.GetMol()
        return self

    @property
    def mol(self):
        """
        Obtain a sanitized copy of the generated (so far) generated molecule.
        """

        mol = copy.deepcopy(self._mol)
        Chem.SanitizeMol(mol)
        return mol

    @property
    def smiles(self):
        """
        Get SMILES of the (so far) generated molecule.
        """
        mol = self.mol
        return Chem.MolToSmiles(mol)

    @property
    def weight(self):
        return rdDescriptors.HeavyAtomMolWt(self._mol)

    def add_graph_res(self, residues):
        for n in self.graph:
            self.graph.nodes[n]["res"] = residues.index(self.graph.nodes[n]["smiles"])

    def get_forcefield_types(self, smarts_filename=None, nb_filename=None):
        if not self.fully_generated:
            raise RuntimeError(
                "Forcefield assignment is only possible for fully generated molecules"
            )

        assigner = get_assignment_class(smarts_filename, nb_filename)
        mol = self.mol
        mol = Chem.AddHs(mol)
        try:
            ffparam = assigner.get_type_assignments(mol)
        except FfAssignmentError as exc:
            exc.attach_mol(mol)
            raise exc
        return ffparam, mol

    @property
    def forcefield_types(self):
        return self.get_forcefield_types(smarts_filename=None, nb_filename=None)
