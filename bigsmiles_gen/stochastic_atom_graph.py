import networkx as nx
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from .distribution import SchulzZimm
from .molecule import Molecule
from .stochastic import Stochastic
from .token import SmilesToken

STATIC_BOND_WEIGHT = -1


def _generate_stochastic_atom_graph(molecule: Molecule):
    if not molecule.generable:
        raise RuntimeError("G-BigSMILES Molecule must be generable for a stochastic atom graph.")

    graph = nx.Graph()
    node_counter = 0
    node_offset_list = []
    # Add all nodes and static weights
    for element in molecule.elements:
        if isinstance(element, SmilesToken):
            smi = element.generate_smiles_fragment()
            mol = Chem.MolFromSmiles(smi)
            mw_info = (Chem.Descriptors.HeavyAtomMolWt(mol), Chem.Descriptors.HeavyAtomMolWt(mol))
            nodes = _get_token_nodes(element, mw_info)

            graph = _add_nodes_to_graph(graph, nodes, node_counter)

            node_counter += len(nodes)
            node_offset_list.append(node_counter + len(nodes))

        if isinstance(element, Stochastic):
            distribution = element.distribution
            if not isinstance(distribution, SchulzZimm):
                raise RuntimeError(
                    "At the moment, we only support SchulzZimm Distribution for stochastic atom graphs."
                )
            mw_info = (distribution._Mn, distribution._Mw)
            nested_offset = [node_counter]
            for token in element.repeat_tokens:
                print(token, node_counter)
                nodes = _get_token_nodes(token, mw_info)
                graph = _add_nodes_to_graph(graph, nodes, node_counter)
                node_counter += len(nodes)
                nested_offset.append(nested_offset[-1] + len(nodes))
            for token in element.end_tokens:
                print(token, node_counter)
                nodes = _get_token_nodes(token, mw_info)
                graph = _add_nodes_to_graph(graph, nodes, node_counter)
                node_counter += len(nodes)
                nested_offset.append(nested_offset[-1] + len(nodes))

            # Add stochastic bonds inside the stochastic element
            for graph_bd in element.bond_descriptors:
                graph_bd_token_idx = _find_bd_token(element, graph_bd)
                # Add regular weights for listed bd
                if graph_bd.transitions is not None:
                    prob = graph_bd.transitions
                    for i, p in enumerate(prob):
                        other_bd = element.bond_descriptors[i]
                        other_bd_token_idx = _find_bd_token(element, other_bd)
                        if p > 0:
                            first_atom = (
                                graph_bd.atom_bonding_to + nested_offset[graph_bd_token_idx]
                            )
                            second_atom = (
                                other_bd.atom_bonding_to + nested_offset[other_bd_token_idx]
                            )
                            print(
                                nested_offset,
                                other_bd_token_idx,
                                other_bd.atom_bonding_to,
                                first_atom,
                                second_atom,
                            )
                            graph.add_edge(
                                first_atom,
                                second_atom,
                                weight=p,
                                termination_weight=graph_bd.weight,
                            )
                else:
                    for other_bd in element.bond_descriptors:
                        if graph_bd.is_compatible(other_bd) and other_bd.weight > 0:
                            other_bd_token_idx = _find_bd_token(element, other_bd)
                            first_atom = (
                                graph_bd.atom_bonding_to + nested_offset[graph_bd_token_idx]
                            )
                            second_atom = (
                                other_bd.atom_bonding_to + nested_offset[other_bd_token_idx]
                            )

                            if other_bd_token_idx < len(element.repeat_tokens):
                                graph.add_edge(
                                    first_atom,
                                    second_atom,
                                    weight=other_bd.weight,
                                    termination_weight=0,
                                )
                            else:
                                graph.add_edge(
                                    first_atom,
                                    second_atom,
                                    termination_weight=other_bd.weight,
                                    weight=0,
                                )

            node_offset_list.append(nested_offset)

    return graph


def _find_bd_token(element, bd):
    for i, token in enumerate(element.repeat_tokens):
        if bd in token.bond_descriptors:
            return i
    for i, token in enumerate(element.end_tokens):
        if bd in token.bond_descriptors:
            return i + len(element.repeat_tokens)


def _add_nodes_to_graph(graph, nodes, node_counter):
    for node in nodes:
        atom = node["atom"]
        mw_info = node["mw_info"]

        graph.add_node(
            node_counter + atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            valence=atom.GetTotalValence(),
            formal_charge=atom.GetFormalCharge(),
            aromatic=atom.GetIsAromatic(),
            hybridization=int(atom.GetHybridization()),
            mn=mw_info[0],
            mw=mw_info[1],
        )
    for node in nodes:
        atom = node["atom"]
        static_bonds = node["static_bonds"]
        for other_idx in static_bonds:
            bonda = atom.GetIdx() + node_counter
            bondb = other_idx + node_counter
            graph.add_edge(
                bonda,
                bondb,
                bond_type=int(static_bonds[other_idx].GetBondType()),
                weight=STATIC_BOND_WEIGHT,
                termination_weight=STATIC_BOND_WEIGHT,
            )

    return graph


def _get_token_nodes(token: SmilesToken, mw_info):
    smi = token.generate_smiles_fragment()
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)

    mol = _remove_extra_hydrogen_atoms(token, mol)

    nodes = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)

        static_bonds = {}
        for bond in atom.GetBonds():
            if bond.GetBeginAtomIdx() == atom_idx:
                other_atom_idx = bond.GetEndAtomIdx()
            if bond.GetEndAtomIdx() == atom_idx:
                other_atom_idx = bond.GetBeginAtomIdx()
            static_bonds[other_atom_idx] = bond

        nodes += [{"atom": atom, "mw_info": mw_info, "static_bonds": static_bonds}]

    return nodes


def _remove_extra_hydrogen_atoms(token, mol):
    # Quick exit for single atom tokens:
    if mol.GetNumAtoms() == 1:
        return mol
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

    atoms_to_be_deleted = sorted(atoms_to_be_deleted)
    edit_mol = Chem.EditableMol(mol)
    for i, atom_idx in enumerate(atoms_to_be_deleted):
        edit_mol.RemoveAtom(atom_idx - i)
    mol = edit_mol.GetMol()
    Chem.SanitizeMol(mol)

    return mol
