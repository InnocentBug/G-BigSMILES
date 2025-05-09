# Generative BigSMILES: (G-BigSMILES)

[![CI](https://github.com/InnocentBug/G-BigSMILES/actions/workflows/ci.yml/badge.svg)](https://github.com/InnocentBug/G-BigSMILES/actions/workflows/ci.yml)
[![notebooks](https://github.com/InnocentBug/G-BigSMILES/actions/workflows/notebook.yml/badge.svg)](https://github.com/InnocentBug/G-BigSMILES/actions/workflows/notebook.yml)
[![trunk](https://github.com/InnocentBug/G-BigSMILES/actions/workflows/trunk.yml/badge.svg)](https://github.com/InnocentBug/G-BigSMILES/actions/workflows/trunk.yml)

This code provides a parser for an extended version of the original [bigSMILES notation](https://olsenlabmit.github.io/BigSMILES/docs/line_notation.html#the-bigsmiles-line-notation). The parsing process leverages an Extended Backus–Naur Form (EBNF) grammar implemented with LARK, followed by additional semantic validation.

The key innovation of this extension is the inclusion of specific details within the line notation. These details enable the generative modeling of molecules from a defined ensemble. Notably, the standard bigSMILES notation can be recovered by simply removing any content enclosed between the `|` symbols (inclusive).

**Publication:**

The scientific basis and details of this work are described in our peer-reviewed article published in RSC Digital Discoveries: [Generative BigSMILES: an extension for polymer informatics, computer simulations & ML/AI](https://doi.org/10.1039/D3DD00147D).

```text
@article{schneider2024generative,
  title={Generative BigSMILES: an extension for polymer informatics, computer simulations \& ML/AI},
  author={Schneider, Ludwig and Walsh, Dylan and Olsen, Bradley and de Pablo, Juan},
  journal={Digital Discovery},
  volume={3},
  number={1},
  pages={51--61},
  year={2024},
  publisher={Royal Society of Chemistry}
}
```

Please cite the above article if you utilize this code in your work. Your acknowledgment is greatly appreciated.

## Updates and Versioning

This version represents an advancement over the published work. Significant improvements include enhanced parsing through EBNF, support for nested stochastic objects, and the generation of machine learning-ready generative graphs.

Please be aware that certain features present in the older version are not yet implemented in this updated release. If your workflow currently relies on these features, you can revert to version 0.2.2 for the previous functionality.

## Installation

The installation procedure outlined below is generally applicable across operating systems but has been primarily tested on Debian-based Linux distributions. You might need to make slight adjustments for other operating systems.

This package is implemented entirely in Python. However, it has a dependency on RDKit. The recommended and easiest method for installing RDKit is via pip and PyPI.

To install this package itself, use pip:

```shell
pip install gbigsmiles
```

To verify the installation, you can try importing the package from a directory outside the installation location:

```shell
cd ~ && python -c "import gbigsmiles" && cd -
```

For a more comprehensive test, you can install and run pytest:

```shell
python -m pytest ./tests/
```

Successful execution of the automated tests located in the ./tests directory indicates a successful installation. Examining the test files can also provide valuable insights into the package's capabilities and usage.
Running the Jupyter Notebook SI.ipynb

To execute the supplementary information Jupyter notebook:

1. Follow the installation steps described above (the pytest step is optional for this).
2. Navigate to the directory containing SI.ipynb.
3. Start the Jupyter Notebook server from within your Python environment:

```shell
jupyter-notebook SI.ipynb
```

The terminal should display instructions on how to access the notebook in your web browser, or the browser might open automatically.

Important: The SI.ipynb notebook is designed to be executed sequentially from top to bottom without skipping any cells.
Notation Details and Examples

This section details the user-facing classes of the package, the specific parts of the G-BigSMILES notation they implement, and illustrative examples of their usage. For a more formal and complete definition of the syntax, please refer to the EBNF grammar for G-BigSMILES.
User Interface

The package exposes four primary classes for direct user interaction. The following subsections provide descriptions and usage examples for each. More in-depth explanations of the notation and features can be found in the subsequent "Details" section.
Stochastic Object (gbigsmiles.BigSmiles)

The gbigsmiles.BigSmiles class is initialized with a single string representing a bigSMILES stochastic object.

```python
smi = "{[][<]CC([>])c1ccccc1, [<]CC([>])C(=O)OC; CC(C)[>], CC(C)[>], [<][Br][]}|schulz_zimm(700, 600)|"
stochastic = gbigsmiles.BigSmiles.make(smi)
```

To generate a molecular representation from this stochastic object, we first obtain the generative graph:

```python
graph = stochastic.get_generating_graph()
```

Then, we derive the atom graph:

```python
atom_graph = graph.get_atom_graph()
```

From the AtomGraph object, we can sample individual molecules:

```python
mol_graph = atom_graph.sample_mol_graph()
```

At this stage, mol_graph represents a polymer molecule as a networkx graph. These graphs have not yet undergone valence checking. To obtain a chemically valid representation, we can convert the networkx graph to an RDKit Mol object:

```python
from rdkit import Chem
mol = gbigsmiles.mol_graph_to_rdkit_mol(mol_graph)
print(Chem.MolToSmiles(mol))
```

This conversion to an RDKit molecule automatically performs valence checks, ensuring the generated molecule's validity.

## Details

This section provides a detailed explanation of the G-BigSMILES notation and the corresponding Python objects within this package. For a precise definition of the syntax, please consult the EBNF grammar for G-BigSMILES.

### BondDescriptor (BondDescriptor)

The BondDescriptor class handles the parsing of bond descriptors as defined in the bigSMILES specification. A bond descriptor adheres to the following structure:

```text
[ + Symbol + ID + | + weights +| +]
```

Where:

- Symbol: Represents the type of bond connection. Possible values are $, <, or >.
- ID: An optional positive integer specifying the identifier of the bond descriptor.
- |: Optional delimiters used to enclose weight information. If weights are not specified, these delimiters must be omitted.
  - weights: Can be a single positive floating-point number or an array of positive floating-point numbers separated by spaces.
    A single float represents the general reactivity weight of this bond descriptor within a molecule.
    An array of floats is used within stochastic objects. The number of elements in the array must match the number of bond descriptors in the stochastic object. Each number corresponds to the weight of reaction between this bond descriptor and the bond descriptor at the same position in the other units. Note that Symbol and ID take precedence over these weights. The sum of the weights in the list serves a similar purpose to the single float weight, influencing the overall reactivity of the bond descriptor.

The single weight functionality can be used to bias the representation of different monomers within a stochastic object (e.g., a 90%/10% mixture).

The empty bond descriptor [] has a special meaning and is only allowed in terminal groups of a stochastic object.

Weights are an extension introduced by G-BigSMILES and are optional. If omitted, the weight is assumed to be equivalent to |1.0|.

### Distribution Object (Distribution)

A Distribution object describes the stochastic molecular weight distribution of a BigSmiles object. The syntax for specifying a distribution follows immediately after the stochastic object:

```text
| + name + ( + parameter + , + ... + ) + |
```

Where:

- name: Specifies the name of the distribution function.
- parameter: Represents the floating-point parameters of the distribution, separated by commas.

Currently, the following distribution functions are implemented (details can be found in distribution.py):

1.  Flory-Schulz
2.  Gaussian
3.  Uniform
4.  Poisson
5.  LogNormal

Distribution objects are an extension of bigSMILES introduced by G-BigSMILES and are optional. If omitted, the molecular weight is not explicitly controlled by a distribution.

### Stochastic Object Syntax

A stochastic object in G-BigSMILES is defined by the following elements:

```text
{ + terminal bond descriptor + repeat unit token + , + ... ; + end group token + ... terminal bond descriptor + } + | + distribution text + |
```

Where:

- terminal bond descriptors: These can be empty ([]) but must be non-empty if the stochastic object is preceded or followed by other components in a Molecule string.
- repeat unit tokens: These are Token objects that typically contain two or more bond descriptors (more than two for branching structures). Multiple repeat units can be listed, separated by commas (,).
- end group tokens: These are Token objects containing a single bond descriptor, usually terminating a branch. Multiple end groups can be listed, separated by commas (,).
- distribution text: As described above, this optional part specifies the molecular weight distribution.

The process of generating molecules from a stochastic object is implemented as follows:

1.  Determine the target heavy atom molecular weight for the stochastic molecule based on the specified distribution.
    If a prefix molecule exists, select a BondDescriptor from the repeat units that matches the open bond descriptor of the prefix, weighted by the reactivity weights of all repeat units.
    If no prefix exists, select a BondDescriptor from the end group tokens, weighted by their respective bond descriptor weights.
2.  Generate the molecular fragment corresponding to the selected token and add it to the growing molecule. If a prefix was involved, connect the prefix to the generated fragment using the selected bond descriptor.
3.  In the partially generated molecule, select a random open bond descriptor based on the weights of all currently open bond descriptors.
    If the selected bond descriptor has a list of weights: select the next bond descriptor from the repeat or end units according to these listed weights.
    If the selected bond descriptor has a single weight: select the next bond descriptor from the repeat units (excluding end groups) according to their bond descriptor weights.
4.  Generate the molecular fragment of the selected unit and connect it to the partially generated molecule using the two chosen bond descriptors.
5.  Repeat steps 3 and 4 until either no open bond descriptors remain or the heavy atom molecular weight of the generated molecule is greater than or equal to the target molecular weight.
6.  If the right terminal bond descriptor is not empty, select one matching open bond descriptor in the partially generated molecule to remain open.
7.  Close all other remaining open bond descriptors:
    1.  Pick a random open bond descriptor based on its weight.
    2.  Pick a matching bond descriptor from the end groups based on their weights.
    3.  Add the corresponding end group to the generated molecule and connect it using the two selected bond descriptors.
    4.  Repeat until all open bond descriptors (except the one potentially left open in step 6) are closed.

### Molecule Object Syntax

The syntax for a Molecule object is:

```text
prefix + stochastic object + connector + ... + stochastic object + suffix
```

Any of these elements can be omitted. A molecule string can contain multiple stochastic objects, optionally connected by connector tokens.
System Object Syntax

A System object defines an ensemble of molecules, rather than a single molecule. The total heavy atom molecular weight for the ensemble can be specified after a Molecule object:

```text
molecule + . + | + mol_weight + |
```

Where mol_weight represents the total heavy atom molecular weight of all molecules in the system.

A system can also describe a mixture of different molecule types by concatenating multiple molecule specifications:

```text
moleculeA + . + | + mol_weightA + ``| + moleculeB+`.`+`|`+`mol_weightB`+`|` + ...
```

In the case of mixtures, all but one of the mol_weight specifications can be relative, indicating a percentage of the total system weight rather than an absolute molecular weight. In such cases, mol_weight should be a positive floating-point number less than 100, followed by the % symbol (e.g., 10%). Ensure that the sum of specified percentages is less than 100%.
Limitations

The G-BigSMILES notation introduced here has certain inherent limitations:

1. Uniqueness: A given molecular system might not have a unique G-BigSMILES representation.
2. Crosslinking: Stochastic connections that define network structures, including rings formed through stochastic bonds, are not currently supported.
3. Compactness: Some users might find the notation less compact than desired for certain systems.
4. Reaction Kinetics: The describable reaction kinetics within this notation remain relatively simple. Complex, time- or spatially-dependent reaction scenarios cannot be represented.

Furthermore, the current implementation of this syntax also has limitations:

1. Rings: Defining large rings that incorporate stochastic objects is not yet possible.
2. Ladder Polymers: The representation and generation of ladder polymers are not currently supported.
