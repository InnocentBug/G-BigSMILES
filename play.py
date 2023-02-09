#!/usr/bin/env python

import bigsmiles_gen


def test_mirror(bigA):
    molA = bigsmiles_gen.Molecule(bigA)
    print(bigA)
    print(molA)
    print(molA.generate().smiles)
    molB = molA.gen_mirror()
    print(molB)
    print(molB.generate().smiles)


def test_prob(bigA):
    mol = bigsmiles_gen.Molecule(bigA)
    smi = mol.generate().smiles
    print(bigA)
    print(smi)
    prob, matches = bigsmiles_gen.mol_prob.get_ensemble_prob(smi, mol)
    print(prob)
    print(matches)


bigA = "{[][<]C(N)C[>]; [<][H][>]}|uniform(500, 600)|{[<][<]C(=O)C[>]; [>][H][]}|uniform(500, 600)|"
bigA = "CCO{[<][<]C(N)C[>][>]}|uniform(500, 600)|{[<][<]C(=O)C[>][>]}|uniform(500, 600)|CCN"
# test_mirror(bigA)
# bigA = "CCO"
# test_prob(bigA)
bigA = "{[][<]C(N)C[>]; [<][H], [>]CO []}|uniform(560, 600)|"
test_prob(bigA)
