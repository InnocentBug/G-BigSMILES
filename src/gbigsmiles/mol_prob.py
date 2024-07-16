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

    if len(start_fragments) <= 0:
        raise ValueError(f"not enough fragments {start_fragments}")

    return start_fragments, start_probabilities


class RememberAdd:
    def __init__(self, value):
        self._value = value
        self._previous = 0.0

    @property
    def value(self):
        return self._value

    @property
    def previous(self):
        return self._previous

    def __iadd__(self, other):
        old_value = self._value
        self._value += other
        self._previous = old_value
        return self

    def __add__(self, other):
        tmp = copy(self)
        tmp += other
        return tmp

    def _radd__(self, other):
        return self + other

    def __eq__(self, other):
        return self.value == other.value and self.previous == other.previous


class OpenAtom:
    def __init__(self, bond, bond_descriptor):
        self.handled_atom = bond[0]
        self.new_atom = bond[1]
        self.bond_descriptor = bond_descriptor

    def __str__(self):
        return f"OpenAtom({self.handled_atom}, {self.new_atom}, {self.bond_descriptor})"

    def __eq__(self, other):
        if (
            self.handled_atom == other.handled_atom
            and self.new_atom == other.new_atom
            and str(self.bond_descriptor) == str(other.bond_descriptor)
        ):
            return True
        return False


class PossibleMatch:
    def __init__(self, mol, big, substructure, token, initial_prob=1.0):
        self._big = big
        self._Nelements = len(big.elements)
        self._Nmol_atoms = mol.GetNumAtoms()
        self._active_element = 0
        self._mol = mol
        self._handled_atoms = []
        self._log_prob = -np.inf

        # Verify that token and substructure match together
        params = Chem.SmilesParserParams()
        params.removeHs = False
        pattern = Chem.MolFromSmiles(token.generate_smiles_fragment(), params.removeHs)
        pattern_mw = rdDescriptors.HeavyAtomMolWt(pattern)
        self._element_weights = [RememberAdd(0.0) for _ in range(self._Nelements)]
        self._open_atoms = []

        possible_substructures = mol.GetSubstructMatches(pattern)
        if substructure in possible_substructures:
            open_atoms = self._find_open_atoms(substructure, token)
            self._add_new_open_atoms(open_atoms)
            self._log_prob = np.log(initial_prob)
            self.add_handled_atoms(substructure)
            self._element_weights[self._active_element] += pattern_mw

        # Always pop SmilesToken, but not stochastic elements
        if isinstance(self._big.elements[self._active_element], SmilesToken):
            self._active_element += 1

    def __eq__(self, other):
        equal = True
        if self._big != other._big:
            return False
        if self._active_element != other._active_element:
            return False
        if self._handled_atoms != other._handled_atoms:
            return False
        if self._open_atoms != other._open_atoms:
            return False
        if self._element_weights != self._element_weights:
            return False
        return equal

    def is_atom_handled(self, atom_idx):
        atom_found = False
        for subs in self._handled_atoms:
            if atom_idx in subs:
                atom_found = True
        return atom_found

    def add_handled_atoms(self, substructure):
        for atom in substructure:
            if atom >= self._Nmol_atoms:
                raise RuntimeError(
                    f"Attempting to add invalid atom to handled: atom {atom}, substructure {substructure}, Nmol {self._Nmol_atoms}"
                )
            atom_found = self.is_atom_handled(atom)
            if atom_found:
                raise RuntimeError(
                    f"Attempting to add atom {atom} that from substructure {substructure} that is already present in handled."
                )
        self._handled_atoms.append(tuple(substructure))

    def _find_open_atoms(self, substructure, token):
        outside_bonds = []
        open_atoms = []
        for atom_idx in substructure:
            if self.is_atom_handled(atom_idx):
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
                    if not self.is_atom_handled(bond[1]):
                        open_atoms.append(OpenAtom(bond, tmp_bd[i]))
                    if bd_idx is not None:
                        raise RuntimeError(f"Invalid {bd_idx} should be None")
                    bd_idx = i
                    break
            if bd_idx is None:
                return []
            del tmp_bd[bd_idx]
        if len(tmp_bd) != 0:
            raise RuntimeError("length should be 0 here")
        return open_atoms

    def pop_open_atom(self, pos=-1):
        if len(self._open_atoms) == 0:
            return None
        return self._open_atoms.pop(pos)

    @property
    def fully_explored(self):
        N_atoms_handled = 0
        for substructure in self._handled_atoms:
            N_atoms_handled += len(substructure)
        return N_atoms_handled == self._Nmol_atoms

    @property
    def probability(self):
        return np.exp(self.log_prob)

    @property
    def log_prob(self):
        if self.fully_explored and len(self._open_atoms) == 0:
            mol_weight_log_prob = np.log(1.0)
            for i, element in enumerate(self._big.elements):
                if isinstance(element, Stochastic):
                    mol_weight_log_prob += np.log(
                        element.distribution.prob_mw(self._element_weights[i])
                    )
            log_prob = self._log_prob + mol_weight_log_prob
            return log_prob
        return -np.inf

    @property
    def possible(self):
        return not np.isinf(self._log_prob)

    def copy(self, adjust_probability=1.0):
        new_object = copy.deepcopy(self)
        new_object._log_prob += np.log(adjust_probability)
        return new_object

    @staticmethod
    def pop_stochastic_element(match):
        if match._active_element >= match._Nelements:
            return None
        active = match._big.elements[match._active_element]
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
    def react_open(match):
        def react(match, token, atom):
            def id_open_atom(substructure, match, open_atom):
                open_atom_idx = None
                for idx, atom_idx in enumerate(substructure):
                    if match.is_atom_handled(atom_idx):
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
                if open_bond.transitions is not None:
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
                open_atom_idx = id_open_atom(substructure, match, atom.new_atom)
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
                            new_match.add_handled_atoms(substructure)

                            if new_match.fully_explored:
                                new_full += [new_match]
                            else:
                                new_open += [new_match]
            return new_open, new_full

        def handle_atom(match_input, atom_idx):
            new_open = []
            new_full = []
            match = match_input.copy()
            atom = match.pop_open_atom(i)
            if atom is None:
                raise RuntimeError("Atom must be valid")
            total_atom_weight = atom.bond_descriptor.weight
            for remaining_atom in match._open_atoms:
                total_atom_weight += remaining_atom.bond_descriptor.weight
            if total_atom_weight > 0:
                atom_prob = atom.bond_descriptor.weight / total_atom_weight
            else:
                atom_prob = 1.0

            token_mols = []
            # Let's find possible token that could be reactable
            if isinstance(match._big.elements[match._active_element], SmilesToken):
                new_mol = match.copy(atom_prob)
                token = new_mol._big.elements[new_mol._active_element]
                pattern = Chem.MolFromSmiles(token.generate_smiles_fragment(), params.removeHs)
                pattern_mw = rdDescriptors.HeavyAtomMolWt(pattern)
                new_mol._element_weights[new_mol._active_element] += pattern_mw
                token_mols.append((new_mol, token))
                new_mol._active_element += 1

            if isinstance(match._big.elements[match._active_element], Stochastic):
                for token in match._big.elements[match._active_element].repeat_tokens:
                    new_mol = match.copy(atom_prob)
                    pattern = Chem.MolFromSmiles(token.generate_smiles_fragment(), params.removeHs)
                    pattern_mw = rdDescriptors.HeavyAtomMolWt(pattern)
                    new_mol._element_weights[new_mol._active_element] += pattern_mw
                    token_mols.append((new_mol, token))

                for token in match._big.elements[match._active_element].end_tokens:
                    new_mol = match.copy(atom_prob)
                    pattern = Chem.MolFromSmiles(token.generate_smiles_fragment(), params.removeHs)
                    # Note that end tokens do not increase the molecular weight (for generation purposes)
                    token_mols.append((new_mol, token))

            # Handle the reaction
            for mol, token in token_mols:
                react_new_open, react_new_full = react(mol, token, atom)
                new_open += react_new_open
                new_full += react_new_full

            return new_open, new_full

        params = Chem.SmilesParserParams()
        params.removeHs = False
        new_open = []
        new_full = []
        if match.fully_explored:
            new_full.append(match)
        elif match._active_element < match._Nelements:
            for i in range(len(match._open_atoms)):
                atom_open, atom_full = handle_atom(match, i)
                new_open += atom_open
                new_full += atom_full
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
            match = PossibleMatch(mol, big_mol, substructure, token, prob)
            if match.possible:
                open_matches.append(match)

    full_matches = []
    handled_matches = []
    while len(open_matches) > 0:
        match = open_matches.pop(-1)
        stochastic_pop = PossibleMatch.pop_stochastic_element(match.copy())
        if stochastic_pop is not None and stochastic_pop not in handled_matches:
            open_matches.append(stochastic_pop)
        new_open, new_full = match.react_open(match)

        # We consider this particular match as done, so are all fully handled matches
        handled_matches.append(match)
        handled_matches += new_full
        full_matches += new_full
        for open_match in new_open:
            if open_match.fully_explored:
                full_matches += [open_match]
            elif open_match not in handled_matches:
                open_matches += [open_match]

    probabilities = [match.probability for match in full_matches]
    probability = np.sum(probabilities)
    return probability, full_matches


def get_ensemble_prob(smi: str, big_mol: Molecule):
    if not big_mol.generable:
        return 0

    big_mols = [big_mol]

    matches = []
    probability = 0
    for bm in big_mols:
        prob, match = get_prob(smi, bm)
        probability += prob
        matches += match
    return probability, matches
