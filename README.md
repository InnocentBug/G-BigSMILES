# Generative BigSMILES: (G-BigSMILES)

[![CI](https://github.com/InnocentBug/G-BigSMILES/actions/workflows/ci.yml/badge.svg)](https://github.com/InnocentBug/G-BigSMILES/actions/workflows/ci.yml)
[![notebooks](https://github.com/InnocentBug/G-BigSMILES/actions/workflows/notebook.yml/badge.svg)](https://github.com/InnocentBug/G-BigSMILES/actions/workflows/notebook.yml)
[![trunk](https://github.com/InnocentBug/G-BigSMILES/actions/workflows/trunk.yml/badge.svg)](https://github.com/InnocentBug/G-BigSMILES/actions/workflows/trunk.yml)

Generator of SMILES string from bigSMILES with extension.

This code implements a parser for an extension of the original [bigSMILES notation](https://olsenlabmit.github.io/BigSMILES/docs/line_notation.html#the-bigsmiles-line-notation).
The extension is designed to add details into the line notation that enable the generation of molecules from that specific ensemble.
The syntax of the extension of bigSMILES can be removed if everything between the `|` symbols and the `|` symbols is removed from the string.

The corresponding peer-reviewed journal article can be found published in RSC Digital Discoveries [here](https://doi.org/10.1039/D3DD00147D).
Please cite this article if you are using this code. Thank you.

## Installation

The following instructions are designed to be independent of the operating system, but are tested for debian-linux systems only.
You may have to slightly adjust the procedure for a differing operating system.

The package is python-only, but it requires a rdkit installation.
The easiest way to install this dependency is via pip and pypi.

You can install this package via pip.

```shell
pip install gbigsmiles
```

Outside the installation directory, you can test importing the installed package

```shell
cd ~ && python -c "import gbigsmiles" && cd -
```

For a more detailed test, you can install and use `pytest`.

```shell
python -m pytest
```

Should execute our automated test and succeed if the installation was successful.
Examining the tests in `./test` can also help to get an overview of this package's capabilities.

### Running the Jupyter Notebook SI.ipynb

Follow the above described steps (the pytest step can be omitted).
Then start the jupyter notebook server from inside the conda environment:

```shell
jupyter-notebook SI.ipynb
```

The shell should either print out instructions of how to connect to the notebook with your browser or open it the browser automatically.

Note that the jupyter notebook is designed to be execute from the top, without skipping entries.

## Notation of details and Examples

In this section, we discuss the user-facing classes of this package, which part of the notation it implements, and how it can be used.

### User interface

Four classes are directly facing the user.
Here we describe the objects we usable examples, for more details on notation and features, check the more detailed sections later on.

#### Stochastic object

The `gbigsmiles.Stochastic` object takes as user input a single string of a bigSMILES stochastic object.

```python
stochastic = gbigsmiles.Stochastic("{[][$]C([$])C=O,[$]CC([$])CO;[$][H], [$]O[]}|flory_schulz(0.0011)|"}
```

Because this stochastic object defines its molecular weight distribution explicitly and both terminal bond descriptors are empty it can generate a full molecule.

```python
assert stochastic.generable
```

To generate this molecule we can call the `generate()` function.

```python
generated_molecule = stochastic.generate()
```

The resulting object is a wrapped `rdkit` molecule `MolGen`.

#### MolGen object

`gbigsmiles.MolGen` objects are the resulting molecules from the generation of bigSMILES strings.
It can contain partially generated molecules and fully generated molecules.
Only fully generated molecules are chemically meaningful, so we can ensure this:

```python
assert generated_molecule.fully_generated
```

For fully generated molecules we can obtain the underlying `rdkit.mol` object.

```python
mol = generated_molecule.mol
```

This enables you to do all the operations with the generated molecule that `rdkit` offers.
So calculating the SMILES string, various chemical properties, structure matching, and saving in various formats is possible.

For convenience, we offer direct access to the molecular weight of all heavy atoms and the SMILES string.

```python
print(generated_molecule.weight)
print(generated_molecule.smiles)
```

#### Molecule object

The `gbigsmiles.Stochastic` object was only generable without prefixes and suffixes, so the `gbigsmiles.Molecule` object offers more flexibility.
It allows the prefixes and suffixes to combine different stochastic objects.

```python
molecule = gbigsmiles.Molecule("NC{[$][$]C[$][$]}|uniform(12, 72)|COOC{[$][$]C[$][$]}|uniform(12, 72)|CO")
```

Similar to before we can ensure that this molecule is generable and subsequently generate the molecule.

```python
assert molecule.generable
generated_molecule = molecule.generate()
```

#### System object

If it is desired to generate not just a single molecule but a full ensemble system with one or more different types of molecules, this can be expressed with a `gbigsmiles.System` object.

This can be a simple system with just a single molecule type, where only the total molecular weight is specified like this one:

```python
system = gbigsmiles.System("NC{[$][$]C[$][$]}|uniform(12, 72)|COOC{[$][$]C[$][$]}|uniform(12, 72)|CO.|1000|")
```

Or a more complicated situation that covers for example a polymer and a solvent.

```python
system = gbigsmiles.System("C1CCOC1.|10%|{[][$]C([$])c1ccccc1; [$][H][]]}|gauss(400,20)|.|500|")
```

We can still generate these systems as before, but now it returns a random `MolGen` from the ensemble.

```python
generated_molecule = system.generate()
```

If we want to generate the entire ensemble completely and collect generated molecules in a list, we can use the generator function of the system.

```python
generated_molecule_list = []
for mol in system.generator:
   print(mol.smiles)
```

### Details

This section lists details about the notation as well as other python objects this package supports.

#### BondDescriptor

A `BondDescriptor` implements the parsing of a bond descriptor as described in bigSMILES.
In particular, a bond descriptor has the following elements

`[` + `Symbol` + `ID` + `|` + `weights` +`|` +`]`

- `Symbol` can be `$`, `<`, or `>` indicating the type of bond connection.
- `ID` is an optional positive integer indicating the ID of the bond descriptor
- `|` is optional to describe the weight of this bond descriptor.
- if not used, everything between the `|` and the `|` has to be omitted.
- `weights` can be a single positive float number of an array of positive float numbers separated by spaces.
- a single float number represents the weight of how likely this bond descriptor reacts in a molecule.
- if an array of float numbers is listed, the number of elements has to be equal to the number of bond descriptors in the stochastic it is a part of. Each of the numbers represents the weight this bond descriptor reacts with the bond descriptor it corresponds to. `Symbol` and `ID` take precedence over this weight. The sum of all weights in the list plays the equivalent role of a single float number: weighting the bond descriptor for reactions.

The weight functionality (single) can be used to weigh monomers in a stochastic object.
This allows the specification of for example a 90%/10% representation of monomers in a molecule.

The empty bond descriptor `[]` is special and only permitted in a terminal group of a stochastic object.
Weights are part of the bigSMILES extension and can be omitted. If omitted it is assumed to be equivalent to `|1.0|`.

#### Token

A token describes a short smiles string that can contain fragments of SMILES strings as well as bond descriptors.
Tokens have two functions, they serve as repeat and end units inside the stochastic object as well as prefixes, suffixes, and connectors surrounding stochastic objects.

In standard bigSMILES, the prefix, suffix, and connectors token are not supposed to have bond descriptors.
In this implementation, however, bond descriptors are supported and they are determined by the corresponding terminal bond descriptors.

#### Distribution object

A distribution object describes the stochastic molecular weight of a stochastic object.
The syntax is that it follows immediately after a stochastic object and takes the following form:

`|` + `name` + `(` + `parameter` + `,` + ... + `)` + `|`

- `name` specifies the name of the distribution
  ` and it is followed by the parameters (float) of the distribution

Currently, there are 3 distributions implemented, Flory-Schulz, Gaussian, and uniform.
For more details on the distributions, check them out in `distribution.py`.

Distribution objects are part of the bigSMILES extension and can be omitted.

#### Stochastic object syntax

A stochastic is comprised of the following elements

`{` + `terminal bond descriptor` + `repeat unit token` + `,` + ... `;` + `end group token` + ... `terminal bond descriptor` + `}` + `|` + `distribution text` + `|`

- `terminal bond descriptors` can be empty `[]` but must not be empty if there is something in front (or after) the stochastic object
- `repeat unit tokens` are tokens that usually contain 2 or more bond descriptors. (more than 2 for branching).
  - you can list as many repeat units as necessary, separated by `,`
- `end group tokens` are tokens with a single bond descriptor, usually terminating a branch.
  - you can list as many end groups as necessary separated by `,`
- the distribution text is explained above and may be omitted.

The generation of stochastic objects is implemented as follows:

1. Determine the heavy atom molecular weight of this stochastic molecule according to the specified distribution.
   1. If a there is an existing molecule (i.e. prefix) is present select a bond descriptor that matches the open bond descriptor from the prefix according to the weight of all repeat units.
   1. If there is no prefix, select a bond descriptor from the end group tokens according to the weight of bond descriptors of the end groups.
1. Generate the molecule fraction of the selected token and add it to the generating molecule. In case of a prefix, react the prefix with the selected bond descriptor.
1. In the partially generated molecule, select a random bond descriptor according to the weight of all open bond descriptors present.
   1. In case the selected bond descriptor has a list of weights: select the next bond descriptor from the repeat- or end-units according to the listed weights.
   1. In case of a single weight: select the next bond descriptor from the repeat unit (not end group) according to the weight of the bond descriptors.
1. Generate the selected unit and react with the two selected bond descriptors.
1. Repeat until no bond descriptors are open or the heavy atom molecular weight of the partially generated molecule is bigger or equal to the pre-determined heavyweight molecular weight of the stochastic object.
1. If the right terminal bond descriptor is not empty, select one matching open bond descriptor from the partially generated molecule to be left open.
1. Close all other remaining open bond descriptors.
   1. Pick a random open bond descriptor according to its bond descriptor weight.
   1. Pick a matching bond descriptor from the end groups according to the weight of their bond descriptors.
   1. Add the end group to the generated molecule and react with the two bond descriptors.
   1. Repeat until all open bond descriptors (except the one selected previously) are closed.

#### Molecule object syntax

The syntax for a molecule object is as follows:

`prefix` + `stochastic object` + `connector` + ... + `stochastic object` + `suffix`

Any of the elements can be omitted.
And the molecule can contain as many stochastic objects as necessary that can optionally be connected by `connector tokens`.

#### System object syntax

A system defines an ensemble of molecules instead of just a single molecule from the ensemble.
To determine the number of molecules the total molecular weight can be specified after a molecule.

`molecule` + `.` + `|` + `mol_weight` + `|`

where `mol_weight` is the total molecular weight of heavy atoms that of all molecules.

A system can also contain more than just one molecule type, but suffixing multiple molecules in a string:

`moleculeA` + `.` + `|` + `mol_weightA` + ``| + `moleculeB`+`.`+`|`+`mol_weightB`+`|` + ...

In this case, a mixture of molecules is generated.
In the case of mixtures, all but one of the `mol_weight` specifiers can be relatively specifying a percentage rather than molecular weight.
In that case, `mol_weight` is a positive floating point literally smaller than 100 followed by `%`.
Make sure that the number of specified percentages is below 100%.

## Limitations

The notation we introduce here has some limitations.
Here we are listing the known limitations:

- Uniqueness: there is not necessarily a unique bigSMILES for a given system.
- Crosslinking: stochastic connections that define a network (including rings) are not supported.
- Compact notation: some might find this notation not compact enough.
- Time or spatial-dependent reaction kinetics. The describable reaction kinetics of this notation remains simple. Complicated situations cannot be represented.

Further, the implementation of this syntax has limitations too. These are the known limitations of the implementation:

- Rings: Defining large rings that include stochastic objects is not possible
- Ladder polymers: not supported
- Nested stochastic objects: bigSMILES does support nested objects, but they are not supported here.
- Chemically valid tokens: Every token has to be chemically valid for molecule generation.
