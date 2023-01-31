import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors as rdDescriptors

from .molecule import Molecule
from .stochastic import Stochastic
from .token import SmilesToken


def get_starting_tokens(smiles, big_mol):
    start_element = big_mol.elements[0]
    start_fragments = []
    start_probabilities = []
    if isinstance(start_element, SmilesToken):
        start_fragments.append(start_element)
        start_probabilities.append(1.0)

    if isinstance(start_element, Stochastic):
        end_weights = []
        for end_token in start_element.end_tokens:
            start_fragments.append(end_token)
            weight = 0
            for bd in end_token.bond_descriptors:
                weight += bd.weight
            end_weights.append(weight)
        end_weights = np.asarray(end_weights)
        end_weights /= np.sum(end_weights)
        start_probabilities += list(end_weights)

    assert len(start_fragments) > 0

    return start_fragments, start_probabilities


class OpenAtom:
    def __init__(self, bond, bond_descriptor):
        self.handled_atom = bond[0]
        self.new_atom = bond[1]
        self.bond_descriptor = bond_descriptor

    def __str__(self):
        return f"OpenAtom({self.handled_atom}, {self.new_atom}, {self.bond_descriptor})"


class PossibleMatch:
    def __init__(self, mol, big, substructure, token, initial_prob=1.0):
        self._big = big
        self._Nelements = len(big.elements)
        self._active_element = 0
        self._mol = mol
        self._handled_atoms = set()
        self._probability = 0

        # Verify that token and substructure match together
        params = Chem.SmilesParserParams()
        params.removeHs = False
        pattern = Chem.MolFromSmiles(token.generate_smiles_fragment(), params.removeHs)
        pattern_mw = rdDescriptors.HeavyAtomMolWt(pattern)
        self._element_weights = np.zeros(self._Nelements)
        self._open_atoms = []

        possible_substructures = mol.GetSubstructMatches(pattern)
        if substructure in possible_substructures:
            open_atoms = self._find_open_atoms(substructure, token)
            self._add_new_open_atoms(open_atoms)
            self._probability = initial_prob
            self._handled_atoms |= set(substructure)
            self._element_weights[self._active_element] += pattern_mw

        # Always pop SmilesToken, but not stochastic elements
        if isinstance(self._big.elements[self._active_element], SmilesToken):
            self._active_element += 1

    def _find_open_atoms(self, substructure, token):
        outside_bonds = []
        open_atoms = []
        for atom_idx in substructure:
            if atom_idx in self._handled_atoms:
                return []
            aoi = self._mol.GetAtomWithIdx(atom_idx)
            for bond in aoi.GetBonds():
                if (
                    bond.GetBeginAtomIdx() in substructure
                    and bond.GetEndAtomIdx() not in substructure
                ):
                    outside_bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                if (
                    bond.GetBeginAtomIdx() not in substructure
                    and bond.GetEndAtomIdx() in substructure
                ):
                    outside_bonds.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
        # Consider it only a true match if the number of outside bonds matches the number of bond descriptors
        if len(outside_bonds) != len(token.bond_descriptors):
            return []
        # Ensure that the outside bonds correspond to bond descriptors
        tmp_bd = copy.deepcopy(token.bond_descriptors)
        for bond in outside_bonds:
            token_pos = substructure.index(bond[0])
            bd_idx = None
            for i in range(len(tmp_bd)):
                if token_pos == tmp_bd[i].atom_bonding_to:
                    if not bond[1] in self._handled_atoms:
                        open_atoms.append(OpenAtom(bond, tmp_bd[i]))
                    assert bd_idx is None
                    bd_idx = i
            if bd_idx is None:
                return []
            del tmp_bd[bd_idx]
        assert len(tmp_bd) == 0
        return open_atoms

    def pop_open_atom(self, pos=-1):
        if len(self._open_atoms) == 0:
            return None
        return self._open_atoms.pop(pos)

    @property
    def fully_explored(self):
        return len(self._handled_atoms) == self._mol.GetNumAtoms()

    @property
    def probability(self):
        if self.fully_explored and len(self._open_atoms) == 0:
            mol_weight_prob = 1.0
            for i, element in enumerate(self._big.elements):
                if isinstance(element, Stochastic):
                    mol_weight_prob *= element.distribution.prob_mw(int(self._element_weights[i]))
            return self._probability * mol_weight_prob
        return 0

    @property
    def possible(self):
        return self._probability > 0

    def copy(self, adjust_probability=1.0):
        new_object = copy.deepcopy(self)
        new_object._probability *= adjust_probability
        return new_object

    @staticmethod
    def pop_stochastic_element(match):
        if match._active_element >= match._Nelements:
            return None
        active = match._big.elements[match._active_element]
        # If we couldn't add more, and we have a stochastic element, move to the next.
        if isinstance(active, Stochastic) and str(active.right_terminal) != "[]":
            # But only move on if there is no other open bond.
            if len(match._open_atoms) == 1:
                atom = match._open_atoms[0]
                # Ensure that the last open bond matches with expected terminal:
                if atom.bond_descriptor.is_compatible(active.right_terminal):
                    # Ok, now we just have to pop the stochastic element
                    match._active_element += 1
                    if match._Nelements > match._active_element:
                        return match
        return None

    def _add_new_open_atoms(self, new_open_atoms):
        for atom in new_open_atoms:
            if atom is None or not isinstance(atom, OpenAtom):
                raise RuntimeError(f"Adding invalid open atom {atom}")
        self._open_atoms += new_open_atoms

    @staticmethod
    def react_open(pm, atom=None):
        def react(match, token, atom):
            def id_open_atom(substructure, handled_atoms, open_atom):
                open_atom_idx = None
                for idx, atom_idx in enumerate(substructure):
                    if atom_idx in handled_atoms:
                        return None
                    if atom_idx == open_atom:
                        open_atom_idx = idx
                return open_atom_idx

            def id_bond_descriptor(open_atom_idx, open_bd, bond_descriptors):
                found_bd = []
                for bd in bond_descriptors:
                    if bd.atom_bonding_to == open_atom_idx:
                        if bd.is_compatible(open_bd):
                            found_bd.append(bd)
                return found_bd

            def get_reaction_prob(open_bond, bd, token):
                if open_bond.transitions:
                    reaction_prob = open_bond.transitions[bd.descriptor_num]
                    reaction_prob /= open_bond.weight
                else:
                    reaction_prob = bd.weight
                    element_weight = 0
                    for bond_descriptors_weight in token.bond_descriptors:
                        if bond_descriptors_weight.is_compatible(open_bond):
                            element_weight += bond_descriptors_weight.weight
                    reaction_prob /= element_weight
                return reaction_prob

            params = Chem.SmilesParserParams()
            params.removeHs = False
            new_open = []
            new_full = []
            pattern = Chem.MolFromSmiles(token.generate_smiles_fragment(), params.removeHs)

            for substructure in match._mol.GetSubstructMatches(pattern):
                open_atom_idx = id_open_atom(substructure, match._handled_atoms, atom.new_atom)
                if open_atom_idx is not None:
                    possible_bd = id_bond_descriptor(
                        open_atom_idx, atom.bond_descriptor, token.bond_descriptors
                    )
                    for bd in possible_bd:
                        reaction_prob = get_reaction_prob(atom.bond_descriptor, bd, token)
                        if reaction_prob > 0:
                            new_match = match.copy(reaction_prob)
                            # Find new open bond that can react
                            new_open_atoms = new_match._find_open_atoms(substructure, token)
                            new_match._add_new_open_atoms(new_open_atoms)
                            # Add the handled atoms to the match
                            new_match._handled_atoms |= set(substructure)

                            if new_match.fully_explored:
                                new_full += [new_match]
                            else:
                                new_open += [new_match]
            return new_open, new_full

        params = Chem.SmilesParserParams()
        params.removeHs = False
        new_open = []
        new_full = []
        if pm.fully_explored and atom is None:
            new_full.append(pm)
        elif pm._active_element < pm._Nelements:
            if atom is None:
                atom = pm.pop_open_atom()
                if atom is None:
                    return new_open, new_full
            token_mols = []
            # Let's find possible token that could be reactable
            if isinstance(pm._big.elements[pm._active_element], SmilesToken):
                new_mol = pm.copy()
                token = new_mol._big.elements[new_mol._active_element]
                pattern = Chem.MolFromSmiles(token.generate_smiles_fragment(), params.removeHs)
                pattern_mw = rdDescriptors.HeavyAtomMolWt(pattern)
                new_mol._element_weights[new_mol._active_element] += pattern_mw
                token_mols.append((new_mol, token))
                new_mol._active_element += 1

            if isinstance(pm._big.elements[pm._active_element], Stochastic):
                for token in pm._big.elements[pm._active_element].repeat_tokens:
                    new_mol = pm.copy()
                    pattern = Chem.MolFromSmiles(token.generate_smiles_fragment(), params.removeHs)
                    pattern_mw = rdDescriptors.HeavyAtomMolWt(pattern)
                    new_mol._element_weights[new_mol._active_element] += pattern_mw
                    token_mols.append((new_mol, token))

                for token in pm._big.elements[pm._active_element].end_tokens:
                    new_mol = pm.copy()
                    pattern = Chem.MolFromSmiles(token.generate_smiles_fragment(), params.removeHs)
                    # Note that end tokens do not increase the molecular weight (for generation purposes)
                    token_mols.append((new_mol, token))

            # Handle the reaction
            for mol, token in token_mols:
                react_new_open, react_new_full = react(mol, token, atom)
                new_open += react_new_open
                new_full += react_new_full
        return new_open, new_full


def get_prob(smiles, big_mol):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    open_matches = []
    starting_token, starting_prob = get_starting_tokens(smiles, big_mol)
    for token, prob in zip(starting_token, starting_prob):
        pattern = Chem.MolFromSmiles(token.generate_smiles_fragment(), params.removeHs)
        possible_substructures = mol.GetSubstructMatches(pattern)
        for substructure in possible_substructures:
            pm = PossibleMatch(mol, big_mol, substructure, token, prob)
            if pm.possible:
                open_matches.append(pm)

    full_matches = []
    while len(open_matches) > 0:
        match = open_matches.pop(-1)
        print(match._handled_atoms, [str(oa) for oa in match._open_atoms])
        stochastic_pop = PossibleMatch.pop_stochastic_element(match.copy())
        if stochastic_pop is not None:
            open_matches.append(stochastic_pop)
        new_open, new_full = match.react_open(match)

        open_matches += new_open
        full_matches += new_full

    probabilities = [pm.probability for pm in full_matches]
    probability = np.sum(probabilities)
    return probability, full_matches


def get_ensemble_prob(smi: str, big_mol: Molecule):
    if not big_mol.generable:
        return 0

    big_mols = [big_mol]
    big_mirror = big_mol.gen_mirror()
    if big_mirror is not None:
        big_mols.append(big_mirror)

    matches = []
    probability = 0
    for bm in big_mols:
        prob, match = get_prob(smi, bm)
        probability += prob
        matches += match
    return probability, matches
