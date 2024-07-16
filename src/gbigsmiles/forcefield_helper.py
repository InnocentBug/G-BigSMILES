import dataclasses
from importlib.resources import files

from rdkit import Chem

_global_nonbonded_itp_file = None
_global_smarts_rule_file = None
_global_assignment_class = None


class FfAssignmentError(Exception):
    def __init__(self, incomplete_ff_dict, mol=None):
        self.incomplete_ff_dict = incomplete_ff_dict
        self.mol = mol

    def attach_mol(self, mol):
        self.mol = mol

    def __str__(self):
        return "Not all atoms could be assigned"


@dataclasses.dataclass
class FFParam:
    mass: float
    charge: float
    sigma: float
    epsilon: float
    bond_type_name: str
    bond_type_id: int = -1


class SMARTS_ASSIGNMENTS:
    def _read_smarts_rules(self, filename):
        if filename is None:
            filename = files("gbigsmiles").joinpath("data", "opls.par")
        self._type_dict = {}
        self._type_dict_rev = {}
        self._rule_dict = {}

        opls_counter = 0
        with open(filename, "r") as smarts_file:
            for line in smarts_file:
                line = line.strip()
                if line and line[0] != "*":
                    ELE, SYMBOL, TYPE, RULE = line.split("|")
                    TYPE = TYPE.strip()
                    RULE = RULE.strip()
                    try:
                        type_id = self._type_dict[TYPE]
                    except KeyError:
                        type_id = opls_counter
                        opls_counter += 1
                    self._type_dict[TYPE] = type_id
                    self._type_dict_rev[type_id] = TYPE
                    self._rule_dict[RULE] = TYPE

    def _read_nb_param(self, filename):
        def assign_bond_index(type_param):
            bond_dict = {}
            bond_dict_rev = {}

            bond_counter = 0
            for name in type_param:
                bond_name = type_param[name].bond_type_name
                if bond_name not in bond_dict:
                    bond_dict[bond_name] = bond_counter
                    bond_dict_rev[bond_counter] = bond_name
                    bond_counter += 1
                type_param[name].bond_type_id = bond_dict[type_param[name].bond_type_name]
            return bond_dict, bond_dict_rev

        self._type_param = {}

        if filename is None:
            filename = files("gbigsmiles").joinpath("data", "ffnonbonded.itp")
        with open(filename, "r") as ff_file:
            for line in ff_file:
                if line and line[0] != "[" and line[0] != ";":
                    line = line.strip()
                    line = line.split()
                    if line[0] in self._type_dict:
                        bond_type_name = line[1]
                        mass = float(line[3])
                        charge = float(line[4])
                        sigma = float(line[6])
                        epsilon = float(line[7])
                        self._type_param[line[0]] = FFParam(
                            mass=mass,
                            charge=charge,
                            sigma=sigma,
                            epsilon=epsilon,
                            bond_type_name=bond_type_name,
                        )
        bond_dict, bond_dict_rev = assign_bond_index(self._type_param)

    def __init__(self, smarts_filename, nb_filename):
        self._read_smarts_rules(smarts_filename)
        self._read_nb_param(nb_filename)

    def get_type(self, type):
        try:
            return self._type_dict_rev[type]
        except KeyError:
            pass
        try:
            return self._type_dict[type]
        except KeyError:
            pass
        # Fallback, we generate a key error
        return self._type_dict[type]

    def get_ffparam(self, type):
        return self._type_param[self.get_type(type)]

    def get_type_assignments(self, mol):
        match_dict = {}
        for rule in self._rule_dict:
            rule_mol = Chem.MolFromSmarts(rule)
            matches = mol.GetSubstructMatches(rule_mol)
            for match in matches:
                if len(match) > 1:
                    RuntimeError("Match with more then atom, that doesn't make sense here.")
                match = match[0]
                try:
                    match_dict[match].append(rule)
                except KeyError:
                    match_dict[match] = [rule]

        for atom_num in match_dict:
            final_match = match_dict[atom_num][0]
            for match_rule in match_dict[atom_num]:
                # Debatable rule, that longer SMARTS strings make a better assignment
                if len(match_rule) > len(final_match):
                    final_match = match_rule
            match_dict[atom_num] = final_match

        final_dict = {}
        for atom_num in match_dict:
            final_dict[atom_num] = self.get_ffparam(
                self.get_type(self._rule_dict[match_dict[atom_num]])
            )

        if len(final_dict) != mol.GetNumAtoms():
            raise FfAssignmentError(final_dict)

        return final_dict

        # graph = nx.Graph()
        # for atom_num in match_dict:
        #     atom = mol.GetAtomWithIdx(atom_num)
        #     graph.add_node(
        #         atom_num,
        #         atomic=atom.GetAtomicNum(),
        #         param=opls_param[rule_dict[match_dict[atom_num]]],
        #     )
        # for node in graph.nodes():
        #     atom = mol.GetAtomWithIdx(node)
        #     for bond in atom.GetBonds():
        #         graph.add_edge(
        #             bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType()
        #         )

        # return graph


def get_assignment_class(smarts_filename, nb_filename):
    global _global_assignment_class, _global_nonbonded_itp_file, _global_smarts_rule_file
    if (
        _global_assignment_class is None
        or smarts_filename != _global_nonbonded_itp_file
        or nb_filename != _global_smarts_rule_file
    ):
        _global_nonbonded_itp_file = nb_filename
        _global_nonbonded_itp_file = smarts_filename
        _global_assignment_class = SMARTS_ASSIGNMENTS(
            _global_smarts_rule_file, _global_nonbonded_itp_file
        )
    return _global_assignment_class
