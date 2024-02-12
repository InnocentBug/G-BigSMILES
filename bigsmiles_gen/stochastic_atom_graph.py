import networkx as nx
import rdkit
from rdkit import Chem

from .molecule import Molecule
from .stochastic import Stochastic
from .token import SmilesToken


def _generate_stochastic_atom_graph(molecule: Molecule):
    if not molecule.generable:
        raise RuntimeError("G-BigSMILES Molecule must be generable for a stochastic atom graph.")

    # Store information about each block of the generation graph and its molecular weight in global.
    global_mw_info = []
    atom_counter = 0
    for ele in molecule.elements:
        if isinstance(ele, SmilesToken):
            smi = ele.generate_smiles_fragment()
            mol = Chem.MolFromSmiles(smi)
            mw_info = (Chem.Descriptors.HeavyAtomMolWt(mol), Chem.Descriptors.HeavyAtomMolWt(mol))
            nodes = _get_token_nodes(ele, mw_info)
            print(nodes)
        if isinstance(ele, Stochastic):
            pass


def _get_token_nodes(token: SmilesToken, mw_info):
    smi = token.generate_smiles_fragment()
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)

    atoms_to_be_deleted = _identify_extra_hydrogen_atoms(token, mol)
    nodes = []
    for atom_idx in range(mol.GetNumAtoms()):
        if atom_idx not in atoms_to_be_deleted:
            atom = mol.GetAtomWithIdx(atom_idx)
            nodes += [atom]
    return nodes


def _identify_extra_hydrogen_atoms(token, mol):
    # We need to remove hydrogens, where we have a bond descriptor connected to.
    atoms_to_be_deleted = []
    for bd in token.bond_descriptors:
        origin_atom_idx = bd.atom_bonding_to
        origin_atom = mol.GetAtomWithIdx(origin_atom_idx)
        for bond in origin_atom.GetBonds():
            if bond.GetBeginAtomIdx() == origin_atom_idx:
                other_atom_idx = bond.GetEndAtomIdx()
            if bond.GetEndAtomIdx() == origin_atom_idx:
                other_atom_idx = bond.GetBeginAtomIdx()

            if other_atom_idx not in atoms_to_be_deleted:
                # We can't remove atoms twice (shouldn't happen anyways)
                other_atom = mol.GetAtomWithIdx(other_atom_idx)

                # Make sure we remove a hydrogen
                if other_atom.GetAtomicNum() == 1:
                    atoms_to_be_deleted += [other_atom_idx]
                    # Exit from the bond loop, since we only delete one atom
                    break
    assert len(token.bond_descriptors) == len(atoms_to_be_deleted)
    return atoms_to_be_deleted
