import gbigsmiles


def test_forcefield_assignment():
    smi = "CCC(C){[>][<]CC([>])c1ccccc1[<]}|schulz_zimm(2000, 1800)|{[>][<]CC([>])C(=O)OC[<]}|schulz_zimm(1000, 900)|[H]"
    mol = gbigsmiles.Molecule(smi)
    mol_gen = mol.generate()
    ffparam, mol = mol_gen.forcefield_types
    assert len(ffparam) == mol.GetNumAtoms()
