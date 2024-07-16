# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy

import networkx as nx

from .bond import _create_compatible_bond_text
from .core import _GLOBAL_RNG, BigSMILESbase
from .mixture import Mixture
from .stochastic import Stochastic
from .stochastic_atom_graph import StochasticAtomGraph
from .token import SmilesToken


class Molecule(BigSMILESbase):
    """
    Part of an extended bigSMILES description hat contains up to exactly one mixture description.
    """

    def __init__(self, big_smiles_ext, res_id_prefix=0):
        """
        Construction of a molecule to make up mixtures.

        Arguments:
        ---------
        big_smiles_ext: str
           text representation

        res_id_prefix: int = 0
           If the molecule is part of a system of molecules, specify which number it is.

        """
        self._raw_text = big_smiles_ext.strip()

        self._elements = []
        stochastic_text = copy.copy(self._raw_text)

        self.mixture = None
        # TODO: find and verify non-extension non-bonds '.'
        # without confusing them with floating point numbers
        if stochastic_text.find(".|") >= 0:
            start = stochastic_text.find(".|")
            end = stochastic_text.find("|", start + 3) + 1
            mixture_text = stochastic_text[start:end]
            end_text = stochastic_text[end:].strip()
            if len(end_text) > 0:
                raise RuntimeError(
                    f"Molecule {stochastic_text} does not end with a mixture descriptor '.|'."
                )
            stochastic_text = stochastic_text[:start]
            self.mixture = Mixture(mixture_text)

        res_id_counter = 0
        while stochastic_text.find("{") >= 0:
            pre_token = stochastic_text[: stochastic_text.find("{")].strip()
            pre_stochastic = None
            if len(pre_token) > 0:
                pre_stochastic = SmilesToken(pre_token, 0, res_id_prefix + res_id_counter)
                res_id_counter += 1
                # Find the connecting terminal bond descriptor of previous element.
                if len(self._elements) > 0:
                    # Get expected terminal bond descriptor
                    if isinstance(self._elements[-1], Stochastic):
                        other_bd = self._elements[-1].right_terminal
                    else:
                        other_bd = self._elements[-1].bond_descriptors[-1]
                    if len(pre_stochastic.bond_descriptors) > 0:
                        found_compatible = False
                        for bd in pre_stochastic.bond_descriptors[0]:
                            if bd.is_compatible(other_bd):
                                found_compatible = True
                        if not found_compatible:
                            raise RuntimeError(
                                f"Token {pre_token} only has incompatible bond descriptors with previous element {str(self._elements[-1])}."
                            )
                    # Since this isn't standard, we add a bond descriptor here.
                    else:
                        bond_string = _create_compatible_bond_text(other_bd)
                        pre_token = bond_string + pre_token
                        pre_stochastic = SmilesToken(pre_token, 0, res_id_prefix + res_id_counter)
                        res_id_counter += 1

            stochastic_text = stochastic_text[stochastic_text.find("{") :].strip()
            end_pos = stochastic_text.find("}") + 1
            if end_pos < 0:
                raise RuntimeError(
                    f"System {stochastic_text} contains an opening '{' for a stochastic object, but no closing '}'."
                )
            # Find distribution extension
            if stochastic_text[end_pos] == "|":
                end_pos = stochastic_text.find("|", end_pos + 2) + 1
            stochastic = Stochastic(stochastic_text[:end_pos], res_id_prefix + res_id_counter)
            res_id_counter += len(stochastic.residues)
            if pre_stochastic:
                min_expected_bond_descriptors = 2
                if len(self._elements) == 0:
                    min_expected_bond_descriptors = 1
                if len(pre_stochastic.bond_descriptors) < min_expected_bond_descriptors:
                    # Attach a compatible bond descriptor automatically
                    other_bd = stochastic.left_terminal
                    bond_text = _create_compatible_bond_text(other_bd)
                    bond_text = bond_text[:-1] + "|0|]"
                    pre_token += bond_text
                    pre_stochastic = SmilesToken(pre_token, 0, res_id_prefix + res_id_counter)
                    res_id_counter += 1
                self._elements.append(pre_stochastic)
            self._elements.append(stochastic)

            stochastic_text = stochastic_text[end_pos:].strip()

        if len(stochastic_text) > 0:
            token = SmilesToken(stochastic_text, 0, res_id_prefix + res_id_counter)
            if len(self._elements) > 0 and len(token.bond_descriptors) == 0:
                if isinstance(self._elements[-1], Stochastic):
                    bond_text = _create_compatible_bond_text(self._elements[-1].right_terminal)
                else:
                    bond_text = _create_compatible_bond_text(
                        self._elements[-1].bond_descriptors[-1]
                    )

                token = SmilesToken(bond_text + stochastic_text, 0, res_id_prefix + res_id_counter)
            res_id_counter += 1
            self._elements.append(token)

    @property
    def generable(self):
        if self.mixture is not None:
            if not self.mixture.generable:
                return False

        for ele in self._elements:
            if not ele.generable:
                return False

        return True

    def generate_string(self, extension):
        string = ""
        for ele in self._elements:
            string += ele.generate_string(extension)
        if self.mixture:
            string += self.mixture.generate_string(extension)
        return string

    def generate(self, prefix=None, rng=_GLOBAL_RNG):
        my_mol = prefix
        for element in self._elements:
            my_mol = element.generate(my_mol, rng)

        return my_mol

    @property
    def elements(self):
        return copy.deepcopy(self._elements)

    @property
    def residues(self):
        residues = []
        for element in self._elements:
            residues += element.residues
        return residues

    def gen_mirror(self):
        """
        Generate a bigSMILES molecule, that is identical to the original,
        but as if the the elements of the bigSMILES string would have been written in reverse.
        """
        if len(self._elements) < 2:
            return None
        mirror = copy.deepcopy(self)
        mirror._raw_text = None
        mirror._elements = list(reversed(mirror.elements))
        for ele in mirror._elements:
            ele._raw_text = None
            if isinstance(ele, Stochastic):
                # ele.bond_descriptors = list(reversed(ele.bond_descriptors))
                tmp = ele.left_terminal
                ele.left_terminal = ele.right_terminal
                ele.right_terminal = tmp

        return mirror

    def gen_reaction_graph(self):
        def validate_graph(graph):
            for node in graph:
                weight = 0
                prob = 0
                term_prob = 0
                trans_prob = 0
                edges = graph.edges(node)
                for edge in edges:
                    edge_data = graph.get_edge_data(*edge)
                    weight += edge_data.get("weight", 0)
                    prob += edge_data.get("prob", 0)
                    term_prob += edge_data.get("term_prob", 0)
                    trans_prob += edge_data.get("trans_prob", 0)
            # assert abs(weight - 1) < 1e-6 or abs(weight) < 1e-6
            if not (abs(prob - 1) < 1e-6 or abs(prob) < 1e-6):
                raise RuntimeError("invalid graph static probabilities")
            if not (abs(term_prob - 1) < 1e-6 or abs(term_prob) < 1e-6):
                raise RuntimeError("invalid graph termination probabilities")
            if not (abs(trans_prob - 1) < 1e-6 or abs(trans_prob) < 1e-6):
                raise RuntimeError("invalid graph transition probs")

        # Add primary nodes
        residues = {}
        for element in self._elements:
            if isinstance(element, SmilesToken):
                residues[element] = element
            elif isinstance(element, Stochastic):
                for res in element.repeat_tokens + element.end_tokens:
                    residues[res] = element
            else:
                raise RuntimeError("Unexpected element of type {type(element)} {element}.")

        G = nx.DiGraph(big_smiles=str(self))
        bond_descriptors = {}
        # Add all nodes and edges from residues to BondDescriptor
        for res in residues:
            try:  # Add the mol weight distribution if available
                G.add_node(res, smiles=str(res), distribution=residues[res].distribution)
            except AttributeError:
                G.add_node(res, smiles=str(res))

            for bd in res.bond_descriptors:
                bond_descriptors[bd] = res
                G.add_node(bd, weight=bd.weight)

            for bd in res.bond_descriptors:
                if bd.weight >= 0:
                    G.add_edge(res, bd, atom=bd.atom_bonding_to)

        # Add missing transition edges generation edges
        for graph_bd in bond_descriptors:
            element = residues[bond_descriptors[graph_bd]]

            # Add regular weights for listed bd
            if graph_bd.transitions is not None:
                prob = graph_bd.transitions / graph_bd.weight
                for i, p in enumerate(prob):
                    other_bd = element.bond_descriptors[i]
                    if p >= 0:
                        G.add_edge(graph_bd, other_bd, prob=p)
            elif isinstance(element, Stochastic):  # Non-transition edges
                repeat_weight = 0
                end_weight = 0
                for element_bd in element.bond_descriptors:
                    if graph_bd.is_compatible(element_bd):
                        if bond_descriptors[element_bd] in element.repeat_tokens:
                            repeat_weight += element_bd.weight
                        if bond_descriptors[element_bd] in element.end_tokens:
                            end_weight += element_bd.weight

                for element_bd in element.bond_descriptors:
                    if graph_bd.is_compatible(element_bd) and element_bd.weight > 0:
                        if bond_descriptors[element_bd] in element.repeat_tokens:
                            G.add_edge(graph_bd, element_bd, prob=element_bd.weight / repeat_weight)
                        if bond_descriptors[element_bd] in element.end_tokens:
                            G.add_edge(
                                graph_bd, element_bd, term_prob=element_bd.weight / end_weight
                            )

        # Add transitions between elements
        for graph_bd in bond_descriptors:
            res = bond_descriptors[graph_bd]
            element_id = -1
            for i, element in enumerate(self._elements):
                if res == element:
                    element_id = i
                    break
                if isinstance(element, Stochastic):
                    if res in element.repeat_tokens + element.end_tokens:
                        element_id = i
                        break
            if element_id < len(self._elements) - 1:
                element = self._elements[element_id]
                next_element = self._elements[element_id + 1]

                # Smiles token to smiles token
                if isinstance(element, SmilesToken) and isinstance(next_element, SmilesToken):
                    for other_bd in next_element.bond_descriptors:
                        if graph_bd.is_compatible(other_bd):
                            G.add_edge(graph_bd, other_bd, trans_prob=1.0)

                if isinstance(element, SmilesToken) and isinstance(next_element, Stochastic):
                    total_weight = 0
                    for other_bd in next_element.bond_descriptors:
                        if (
                            graph_bd.is_compatible(other_bd)
                            and other_bd.is_compatible(next_element.left_terminal)
                            and bond_descriptors[other_bd] in next_element.repeat_tokens
                        ):
                            total_weight += other_bd.weight
                    # total_weight = 1
                    if total_weight >= 0 and total_weight < 1e-16:
                        total_weight = 1
                    for other_bd in next_element.bond_descriptors:
                        if (
                            graph_bd.is_compatible(other_bd)
                            and other_bd.is_compatible(next_element.left_terminal)
                            and bond_descriptors[other_bd] in next_element.repeat_tokens
                        ):
                            G.add_edge(
                                graph_bd, other_bd, trans_prob=other_bd.weight / total_weight
                            )

                if isinstance(element, Stochastic) and isinstance(next_element, SmilesToken):
                    for other_bd in next_element.bond_descriptors:
                        if (
                            graph_bd.is_compatible(other_bd)
                            and graph_bd.is_compatible(element.right_terminal)
                            and bond_descriptors[graph_bd] in element.repeat_tokens
                            and other_bd.weight > 0
                        ):
                            G.add_edge(graph_bd, other_bd, trans_prob=1.0)

                if isinstance(element, Stochastic) and isinstance(next_element, Stochastic):
                    total_weight = 0
                    for other_bd in next_element.bond_descriptors:
                        if (
                            graph_bd.is_compatible(other_bd)
                            and other_bd.is_compatible(next_element.left_terminal)
                            and bond_descriptors[other_bd] in next_element.repeat_tokens
                            and graph_bd.is_compatible(element.right_terminal)
                            and bond_descriptors[graph_bd] in element.repeat_tokens
                        ):
                            total_weight += other_bd.weight
                    # total_weight = 1
                    if total_weight >= 0 and total_weight < 1e-16:
                        total_weight = 1
                    for other_bd in next_element.bond_descriptors:
                        if (
                            graph_bd.is_compatible(other_bd)
                            and other_bd.is_compatible(next_element.left_terminal)
                            and bond_descriptors[other_bd] in next_element.repeat_tokens
                            and graph_bd.is_compatible(element.right_terminal)
                            and bond_descriptors[graph_bd] in element.repeat_tokens
                        ):
                            G.add_edge(
                                graph_bd, other_bd, trans_prob=other_bd.weight / total_weight
                            )

        validate_graph(G)
        return G

    def gen_stochastic_atom_graph(self, expect_schulz_zimm_distribution: bool = True):
        stochastic_atom_graph = StochasticAtomGraph(self, expect_schulz_zimm_distribution)
        stochastic_atom_graph.generate()
        return stochastic_atom_graph
