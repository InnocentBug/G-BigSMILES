# bigSMILESgen

Generator of SMILES string from bigSMILES with extension

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

- Uniqueness

Further, the implementation of this syntax has limitations too. These are the known limitations of the implementation:

-
