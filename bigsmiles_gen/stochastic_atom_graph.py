import networkx as nx
import rdkit
from rdkit import Chem

from .molecule import Molecule
from .stochastic import Stochastic
from .token import SmilesToken
from .distribution import SchulzZimm


def _generate_stochastic_atom_graph(molecule: Molecule):
    if not molecule.generable:
        raise RuntimeError("G-BigSMILES Molecule must be generable for a stochastic atom graph.")

    graph = nx.Graph()
    node_counter = 0
    node_offset_list = []
    for ele in molecule.elements:
        if isinstance(ele, SmilesToken):
            smi = ele.generate_smiles_fragment()
            mol = Chem.MolFromSmiles(smi)
            mw_info = (Chem.Descriptors.HeavyAtomMolWt(mol), Chem.Descriptors.HeavyAtomMolWt(mol))
            nodes = _get_token_nodes(ele, mw_info)

            graph = _add_nodes_to_graph(graph, nodes, node_counter)

            node_counter += len(nodes) +1
            node_offset_list.append(node_counter + len(nodes))

        if isinstance(ele, Stochastic):
            distribution = ele.distribution
            if not isinstance(distribution, SchulzZimm):
                raise RuntimeError("At the moment, we only support SchulzZimm Distribution for stochastic atom graphs.")
            mw_info = (distribution._Mn, distribution._Mw)
            nested_offset = []
            for token in ele.repeat_tokens:
                print(token, node_counter)
                nodes = _get_token_nodes(token, mw_info)
                graph = _add_nodes_to_graph(graph, nodes, node_counter)
                node_counter += len(nodes) + 1
                nested_offset.append(node_counter+len(nodes))
            for token in ele.end_tokens:
                print(token, node_counter)
                nodes = _get_token_nodes(token, mw_info)
                graph = _add_nodes_to_graph(graph, nodes, node_counter)
                node_counter += len(nodes) + 1
                nested_offset.append(node_counter+len(nodes))

            node_offset_list.append(nested_offset)
    return graph

def _add_nodes_to_graph(graph, nodes, node_counter):
    for node in nodes:
        atom = node["atom"]
        mw_info = node["mw_info"]

        graph.add_node(node_counter + atom.GetIdx(),
                       atomic_num=atom.GetAtomicNum(),
                       valence = atom.GetTotalValence(),
                       formal_charge = atom.GetFormalCharge(),
                       aromatic = atom.GetIsAromatic(),
                       hybridization = int(atom.GetHybridization()),
                       mn=mw_info[0],
                       mw=mw_info[1],
                       )
    for node in nodes:
        atom = node["atom"]
        static_bonds = node["static_bonds"]
        for other_idx in static_bonds:
            bonda = atom.GetIdx() + node_counter
            bondb = other_idx + node_counter
            graph.add_edge(bonda, bondb, bond_type=int(static_bonds[other_idx].GetBondType()))

    return graph



def _get_token_nodes(token: SmilesToken, mw_info):
    smi = token.generate_smiles_fragment()
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)

    atoms_to_be_deleted = _identify_extra_hydrogen_atoms(token, mol)
    nodes = []
    for atom_idx in range(mol.GetNumAtoms()):
        if atom_idx not in atoms_to_be_deleted:
            atom = mol.GetAtomWithIdx(atom_idx)

            static_bonds = {}
            for bond in atom.GetBonds():
                if bond.GetBeginAtomIdx() == atom_idx:
                    other_atom_idx = bond.GetEndAtomIdx()
                if bond.GetEndAtomIdx() == atom_idx:
                    other_atom_idx = bond.GetBeginAtomIdx()
                if other_atom_idx not in atoms_to_be_deleted:
                    static_bonds[other_atom_idx] = bond

            nodes += [{"atom": atom, "mw_info": mw_info, "static_bonds": static_bonds}]

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
    if mol.GetNumAtoms() != 1:
        assert len(token.bond_descriptors) == len(atoms_to_be_deleted)
        return atoms_to_be_deleted
    return []
