# bigSMILESgen

Generator of SMILES string from bigSMILES with extension.

This code implements a parser for an extension of the original [bigSMILES notation](https://olsenlabmit.github.io/BigSMILES/docs/line_notation.html#the-bigsmiles-line-notation).
The extension is designed to add details into the line notation that enable the generation of molecules from that specific ensemble.
The syntax of the extension of bigSMILES can be removed if everything between the `|` symbols and the `|` symbols is removed from the string.

## Installation

The package is python-only, but it requires a rdkit installation.
The easiest way to install this dependency is via conda.
We provide the `environments.yml` file to install the conda packages.

With the rdkit dependency requirement fulfilled, you can install this package via pip.

```shell
pip install .
```

Outside the installation directory, you can test importing the installed package

```shell
cd ~ && python -c "import bigsmiles_gen && cd -"
```

For a more detailed test, you can install and use `pytest`.

```shell
python -m pytest
```

Should execute our automated test and succeed if the installation was successful.
Examining the tests in `./test` can also help to get an overview of this package's capabilities.

## Notation of details and Examples

In this section, we discuss the user-facing classes of this package, which part of the notation it implements, and how it can be used.

### User interface

Four classes are directly facing the user.
Here we describe the objects we usable examples, for more details on notation and features, check the more detailed sections later on.

#### Stochastic object

The `bigsmiles_gen.Stochastic` object takes as user input a single string of a bigSMILES stochastic object.

```python
stochastic = bigsmiles_gen.Stochastic("{[][$]C([$])C=O,[$]CC([$])CO;[$][H], [$]O[]}|flory_schulz(0.0011)|"}
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

`bigsmiles_gen.MolGen` objects are the resulting molecules from the generation of bigSMILES strings.
It can contain partially generated molecules and fully generated molecules.
Only fully generated molecules are chemically meaningful, so we can ensure this:

```python
assert generated_molecule.fully_generated
```

For fully generated molecules we can obtain the underlying `rdkit.mol` object.

```python
mol = generated_molecule.get_mol()
```

This enables you to do all the operations with the generated molecule that `rdkit` offers.
So calculating the SMILES string, various chemical properties, structure matching, and saving in various formats is possible.

For convenience, we offer direct access to the molecular weight of all heavy atoms and the SMILES string.

```python
print(generated_molecule.weight)
print(generated_molecule.smiles)
```

#### Molecule object

The `bigsmiles_gen.Stochstic` object was only generable without prefixes and suffixes, so the `bigsmiles_gen.Molecule` object offers more flexibility.
It allows the prefixes and suffixes to combine different stochastic objects.

```python
molecule = bigsmiles_gen.Molecule("NC{[$][$]C[$][$]}|uniform(12, 72)|COOC{[$][$]C[$][$]}|uniform(12, 72)|CO")
```

Similar to before we can ensure that this molecule is generable and subsequently generate the molecule.

```python
assert molecule.generable
generated_molecule = molecule.generate()
```

#### System object

If it is desired to generate not just a single molecule but a full ensemble system with one or more different types of molecules, this can be expressed with a `bigsmiles_gen.System` object.

This can be a simple system with just a single molecule type, where only the total molecular weight is specified like this one:

```python
system = bigsmiles_gen.System("NC{[$][$]C[$][$]}|uniform(12, 72)|COOC{[$][$]C[$][$]}|uniform(12, 72)|CO.|1000|")
```

Or a more complicated situation that covers for example a polymer and a solvent.

```python
system = bigsmiles_gen.System("C1CCOC1.|10%|{[][$]C([$])c1ccccc1; [$][H][]]}|gauss(400,20)|.|500|")
```

We can still generate these systems as before, but now it returns a list of `MolGen` objects instead of a single `MolGen` object.

```python
generated_molecule_list = system.generate()
assert isinstance(generated_molecule_list, list)
for molgen in generated_molecule_list:
   print(molgen.smiles)
```

## Limitations

The notation we introduce here has some limitations.
Here we are listing the known limitations:

- Uniqueness: there is not necessarily a unique bigSMILES for a given system.
- Crosslinking: stochastic connections that define a network (including rings) are not supported.
- Compact notation: some might find this notation not compact enough.

Further, the implementation of this syntax has limitations too. These are the known limitations of the implementation:

- Rings: Defining large rings that include stochastic objects is not possible
- Ladder polymers: not supported
- Nested stochastic objects: bigSMILES does support nested objects, but they are not supported here.
- Chemically valid tokens: Every token has to be chemically valid for molecule generation.
