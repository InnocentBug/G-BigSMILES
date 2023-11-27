import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

import bigsmiles_gen


def make_pdb_nice_string(pdb_string, pdbfile, write_conect):

    """Reformats PDB file nicely."""
    fmt = "{:6s}{:5d} {:>4s}{:1s}{:>3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:4s}{:>2s}{:1s}  \n"

    lins = pdb_string.strip().split("\n")
    min_x, max_x, min_y, max_y, min_z, max_z = (None, None, None, None, None, None)
    with open(pdbfile, "w") as output:
        for lin in lins:
            if lin[:4] == "ATOM" or lin.startswith("HETATM"):
                lin = lin.split(" ")
                lin = [x for x in lin if x]
                element = " ".join(re.findall("[a-zA-Z]+", lin[10]))
                pos_x, pos_y, pos_z = (float(lin[5]), float(lin[6]), float(lin[7]))
                new_line = fmt.format(
                    lin[0],
                    int(lin[1]),
                    lin[2],
                    "",
                    lin[3],
                    "",
                    int(lin[4]),
                    "",
                    pos_x,
                    pos_y,
                    pos_z,
                    float(lin[8]),
                    float(lin[9]),
                    "",
                    element,
                    "",
                )
                output.write(new_line)
                if min_x is None or pos_x < min_x:
                    min_x = pos_x
                if min_y is None or pos_y < min_y:
                    min_y = pos_y
                if min_z is None or pos_z < min_z:
                    min_z = pos_z

                if max_x is None or pos_x > max_x:
                    max_x = pos_x
                if max_y is None or pos_y > max_y:
                    max_y = pos_y
                if max_z is None or pos_z > max_z:
                    max_z = pos_z

            elif not write_conect and lin.startswith("CONECT"):
                pass
            else:
                output.write(lin + "\n")
    return max_x - min_x, max_y - min_y, max_z - min_z


bigA = "CCC(C){[>][<]CC([>])c1ccccc1[<]}|schulz_zimm(100, 90)|{[>][<]CC([>])C(=O)OC [<]}|schulz_zimm(500, 450)|"
# bigA = "CCOC(=O)C(C)(C){[>][<|0 "+str(b)+" 0 "+str(1-b)+"|]CC([>|"+str(b)+" 0 "+str(1-b)+" 0|])c1ccccc1, [<|0 "+str(1-b)+" 0 "+str(b)+"|]CC([>|"+str(1-b)+" 0 "+str(b)+" 0|])C(=O)OC [<]}|gauss(1500, 10)|[Br].|5e5|"


for b in np.linspace(0, 1.0, 30):
    print(b)
    bigA = "CCOC(=O)C(C)(C)"
    bigA += (
        "{[>][<|"
        + str(b)
        + "|]CC([>|"
        + str(b)
        + "|])c1ccccc1, [<|"
        + str(1 - b)
        + "|]CC([>|"
        + str(1 - b)
        + "|])C(=O)OC, N1(C(CCC1)=O)[C@@H](C[>|"
        + str(b / 2)
        + "|])[<|"
        + str(b / 2)
        + "|] [<]}|gauss(4000, 10)|"
    )
    bigA += (
        "{[>][<|"
        + str(1 - b)
        + "|]CC([>|"
        + str(1 - b)
        + "|])c1ccccc1, [<|"
        + str(b)
        + "|]CC([>|"
        + str(b)
        + "|])C(=O)OC, N1(C(CCC1)=O)[C@@H](C[>|"
        + str(b / 2)
        + "|])[<|"
        + str(b / 2)
        + "|] [<]}|gauss(4000, 10)|"
    )
    bigA += "[Br]"
    print(bigA)

    big_mol = bigsmiles_gen.Molecule(bigA)
    while True:
        try:
            mol_gen = big_mol.generate()
            mol = mol_gen.mol
            # AllChem.MMFFOptimizeMolecule(mol,maxIters=200000)
        except ValueError:
            print("X")
            continue
        else:
            break
    print(mol_gen.smiles)
    pdb_string = Chem.MolToPDBBlock(mol)

    make_pdb_nice_string(pdb_string, f"play{b}.pdb", False)
    # with open(f"play{b}.pdb", "w") as pdb_handle:
    #     pdb_handle.write(pdb_string)
