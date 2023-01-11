# bigSMILESgen

Generator of SMILES string from bigSMILES with extension.

This code implements a parser ofr an extension of the original [bigSMILES notation](https://olsenlabmit.github.io/BigSMILES/docs/line_notation.html#the-bigsmiles-line-notation).
The extension is designed to add details into the line notation that enable the generation of molecules from that specific ensemble.

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
