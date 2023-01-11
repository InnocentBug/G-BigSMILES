# bigSMILESgen

Generator of SMILES string from bigSMILES with extension.

This code implements a parser ofr an extension of the original [bigSMILES notation](https://olsenlabmit.github.io/BigSMILES/docs/line_notation.html#the-bigsmiles-line-notation).
The extension is designed to add details into the line notation that enable the generation of molecules from that specific ensemble.
The syntax of the extension of bigSMILES can be removed if everything between `|` symbols and the `|` symbols are removed from the string.

## Installation

The package is a python only package, but it requires an rdkit installation.
The easiest way to install this dependency is via conda.
We provide the `environments.yml` file to install the conda packages.

With the rdkit dependency requirement fulfilled, you can install this package via pip.

```shell
pip install .
```

Outside the installation directory you can test to import the installed package

```shell
cd ~ && python -c "import bigsmiles_gen && cd -"
```

For a more detailed test you can install and use `pytest`.

```shell
python -m pytest
```

Should execute our automated test and succeed if the installation was successful.
Examining the tests in `./test` can also help to get an overview of this packages capabilities.

## Notation details and Examples

In this section we discuss the user facing classes of this package, which part of the notation it implements and how it can be used.

### BondDescriptor

A `BondDescriptor` is implements the parsing of a bond descriptor as described in bigSMILES.
In particular a bond descriptor has the following elements

`[` + `Symbol` + `ID` + `|` + `weights` +`|` +`]`

- `Symbol` can be `$`, `<`, or `>` indicating the type of bond connection.
- `ID` is optional positive integer indicating the ID of the bond descriptor
- `|` is optional to describe the weight of this bond descriptor.
  - if not used, everything between `|` and the `|` has to be omitted.
- `weights` can be a single positive float number of an array of positive float number separated by spaces.
  - a single float number represents the weight of how likely this bond descriptor reacts in a molecule.
  - if an array of float number is listed, the number of elements has to be equal to the number of bond descriptors in the stochastic it is a part of. Each of the numbers represents the weight this bond descriptor reacts with the bond descriptor it corresponds to. `Symbol` and `ID` take precedence of this weight.

The empty bond descriptor `[]` is special and only permitted in a terminal group of a stochastic object.

## Limitations

The notation we introduce here has some limitations.
Here we are listing the known limitations:

- Uniqueness: there is not necessarily a unique bigSMILES for a given system.
- Crosslinking: stochastic connections that define a network (including rings) is not supported.
- Compact notation: some might find this notation not compact enough.

Further, the implementation of this syntax has limitations too. These are the known limitations of the implementation:

- Rings: Defining large rings that include stochastic objects is not possible
- Ladder polymers: not supported
- Nested stochastic objects: bigSMILES does support nested objects, but they are not supported here.
- Chemically valid tokens: Every token has to be chemically valid for molecule generation.
