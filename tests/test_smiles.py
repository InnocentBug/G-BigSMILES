import gbigsmiles


def test_smiles_parsing(chembl_smi_list):
    for smi in chembl_smi_list:
        if len(smi) > 0:
            smiles_instance = gbigsmiles.Smiles.make(smi)
            assert smi == smiles_instance.generate_string(True)
