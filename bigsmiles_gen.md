---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.3"
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG

from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

counter = 0


def moltosvg(mol, molSize=(450, 150), kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    global counter
    # with open(f"mol{counter}.svg", "w") as filehandle:
    #     filehandle.write(svg)
    # print(counter)
    counter += 1
    return svg


def render_svg(svg):
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return SVG(svg.replace("svg:", ""))
```

```python
# import pydot
import networkx as nx
import matplotlib.pyplot as plt

graph_counter = 0


def render_graph(mol_gen, residues):
    # pydot_graph = nx.drawing.nx_pydot.to_pydot(mol_gen.graph)
    # return render_svg(pydot_graph.create_svg().decode("utf-8"))
    fig, ax = plt.subplots()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    node_list = list(mol_gen.graph.nodes)
    node_res = [mol_gen.graph.nodes[n]["res"] for n in node_list]
    node_color = [colors[res] for res in node_res]
    labels = {}
    for n in mol_gen.graph.nodes:
        labels[n] = mol_gen.graph.nodes[n]["smiles"]
    layout = nx.kamada_kawai_layout(mol_gen.graph)
    drawing = nx.draw(mol_gen.graph, pos=layout, node_color=node_color, ax=ax, labels=labels)
    global graph_counter
    # fig.savefig(f"g{graph_counter}.pdf")
    # print(f"g{graph_counter}")
    graph_counter += 1
    return plt.show()
```

#### Helper function to generate and visualize bigSMILES strings

```python
from bigsmiles_gen import System, mol_prob, Molecule

# Consistent random numbers also across calls
rng = np.random.default_rng(42)


def big_smiles_gen(string):
    global rng
    # Generate the abstract python object, parsing bigSMILES
    ext = Molecule(big)
    # Generate molecules according to extension
    mol = ext.generate(rng=rng)
    # Draw said molecule
    print(mol.smiles)
    # print(mol_prob.get_ensemble_prob(mol.smiles, ext)[0])

    return render_svg(moltosvg(mol.get_mol()))


def big_smiles_graph(string):
    global rng
    # Generate the abstract python object, parsing bigSMILES
    ext = Molecule(big)
    # Generate molecules according to extension
    mol = ext.generate(rng=rng)
    mol.add_graph_res(ext.residues)
    # Draw said molecule
    print(mol.smiles)
    # print(mol_prob.get_ensemble_prob(mol.smiles, ext)[0])

    return render_graph(mol, ext.residues)
```

# Polyesters

## Di-Carboxyl Acids with 2-5 middle C total Mw 500

BigSMILES:

```python
big = "OOC{[$][$]C(=O)C[$][$]}|uniform(120, 720)|COO"
big
```

### Explanation

- `OOC` just normal SMILES for how the first initial group looks like
- `{....}` stochastic object, here just one carbon that is repeated
- `{[$]` and `[$]}` terminal descriptors, how does the stochastic object connect with the pre- and suffix
- `[$]C[$]` one C to repeat, with a bond in front and one at the end
- `|uniform(12, 72)|` extension detailing that the stochastic object is between 12 and 72 g/mol heavy
- `COO` suffix of the molecule
- `.|500|` extension detailing that the total system component of this as a min Mw of 500 g/mol

```python
# Plot the first generated molecule
big_smiles_gen(big)
```

You can also render the graph of connected monomers.
In this case, we don't draw the atoms, but the connected monomers. This can be handy to visualize bigger molecules.

```python
big_smiles_graph(big)
```

```python
# Generate the second generated molecule
big_smiles_gen(big)
```

Notice how the repeated C are different because of the stochastic object

## Di-ols with 2-5 middle C total Mw 500

```python
big = "OC{[$][$]C[$][$]}|schulz_zimm(80, 72)|CO"
big
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

## Di-ols with 2-5 middle Cyclohexane ring total Mw 500

```python
big = "OC{[$][$]C([$])C1CCCCC1[$]}|uniform(84, 420)|CO"
big
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

# Polyamides

## Di-Carboxylic Acids

```python
big = "NC{[$][$]C[$][$]}|gauss(60, 24)|COO"
big
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

## Diamines

```python
big = "NC{[$][$]C[$][$]}|gauss(60,24)|CN"
big
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

## New: Reaction of diol with di-acid

```python
big = "NC{[$][$]C[$][$]}|uniform(12, 72)|COOC{[$][$]C[$][$]}|uniform(12, 72)|CO"
big
```

```python
big_smiles_gen(big)
```

```python
big_smiles_graph(big)
```

```python
big_smiles_gen(big)
```

## Same diol and di-acid reaction, but with phenyl ring

```python
big = "NC{[$][$]C[$][$]}|uniform(12, 72)|COOC{[$][$]C([$])C1CCCCC1[$]}|uniform(50, 250)|CO"
big
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

# New! Polyolefins, because a '=' remains

## CH2=CH-(CH2)nCH3 where n varies from 1 to 5

```python
big = "C=C{[$][$]C[$][$]}|uniform(12,60)|C"
big
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

## CH2-CH-(CH2)n-CH=CH2 where n varies from 1 to 3

```python
big = "CC{[$][$]C[$][$]}|uniform(12, 36)|C=C"
big
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

## Classic diblock PS-b-P2VP

```python
big = "[H]{[<][<]C([>])c1ccccc1[>]}|gauss(100,20)|{[<][>]C([<])c1ccncc1[>]}|gauss(100, 20)|[H]"
```

```python
big_smiles_gen(big)
```

```python
big_smiles_graph(big)
```

## Polymer in Solvent PS in THF

```python
# big = "C1CCOC1.|10%|{[][$]C([$])c1ccccc1; [$][H][]]}|gauss(400,20)|.|100000|"
```

```python
# big_smiles_gen(big)
```

```python
# big_smiles_gen(big)
```

## Poly(acrylic acid butyl - r - acrylamide) (25% acrylamide)

```python
big = "[H]{[>][>|75|]C([<|75|])CC(=O)OCCCC, [<|25|]C([>|25|])CC(=O)N[<]}|gauss(500,50)|[H]"
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_graph(big)
```

```python
big_smiles_graph(big)
```

## Vulcanized poly(1,3-butadiene)

```python
big = "{[][$1|2|]CC=CC[$1|2|], [$1]CC([<])C([<])C[$1], [>]S[$2|10|], [$2]S[$2]; [$1][H], [<][H], [>][H], [$2][H][]}|gauss(1500,500)|"
big_smiles_gen(big)
```

```python
big_smiles_graph(big)
```

```python
big_smiles_graph(big)
```

# Example stochastic polymer

```python
big = "F{[<] [<|3|]CC[>|3|],[<]C([>])c1ccccc1, [<|0.5|]CCC(C[>|0.1|])CC[>|0.5|]; [<]COC, [>][H] [>]}|uniform(500,1000)|CN{[$][$]CC[$][$]}|uniform(400,500)|[H]"
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_graph(big)
```

```python
big_smiles_graph(big)
```

```python
big_smiles_graph(big)
```

# Motifs

```python
render_svg(moltosvg(Chem.MolFromSmiles("NC")))
```

```python
render_svg(moltosvg(Chem.MolFromSmiles("CC")))
```

```python
render_svg(moltosvg(Chem.MolFromSmiles("CCC(C)CC")))
```

```python
render_svg(moltosvg(Chem.MolFromSmiles("C=O")))
```

```python
render_svg(moltosvg(Chem.MolFromSmiles("C=Cc1ccccc1")))
```

```python
big = "{[] [<]C([>])c1ccccc1 ; [H][>], [H][<] []}|uniform(100, 200)|"
big_smiles_gen(big)
```

```python
big = "{[] [<]C([>])c1ccccc1 ; [H][>], [H][<] []}|uniform(500, 1000)|"
big_smiles_gen(big)
```

```python
big = "OC{[<] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0|])C[>]; [<][H] [>]}|flory_schulz(5e-3)|C=O.|5000|"
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_graph(big)
```

```python
big_smiles_graph(big)
```

```python
big = (
    "O=C{[<]CC(C)NC(=O)C([<|10|])C[>|10|], CCCCOC(=O)C([<|90|])C[>|90|] [>]}|flory_schulz(5e-3)|CC"
)
```

```python
big_smiles_graph(big)
```

```python
big_smiles_graph(big)
```

```python
big_smiles_graph(big)
```

```python
big_smiles_graph(big)
```

```python
big_smiles_graph(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big = "N#CC(C)(C){[$] O([<])(C([$])C[$]), [>]CCO[<|0 0 0 1 0 2|] ; [>][H] [$]}|schulz_zimm(11500,6400)|[H]"
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```

```python
big_smiles_gen(big)
```
