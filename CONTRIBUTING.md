# Contributing to IGUA

For bug fixes or new features, please file an issue before submitting a
pull request. If the change isn't trivial, it may be best to wait for
feedback.

## Setting up a local repository

Simply clone the public repository:

```console
$ git clone https://github.com/zellerlab/IGUA
```

## Building the code from source

The code can be built locally with `maturin` directly, or with `pip`. For 
instance to build the code in editable mode (if you have all the 
dependencies installed):

```console
$ cd IGUA
$ python -m pip install --no-build-isolation -e .
```

## Coding guidelines

This project targets all Python versions supported by the latest release of
PyO3. It should be able to compile with the `stable` version of the Rust
compiler.

### Docstrings

The docstring lines should not be longer than 76 characters (which allows
rendering the entire module in a 80x24 terminal window without soft-wrap).  
Docstrings should be written in Google format.

### Format

Make sure to format the code with `cargo fmt` before making a commit. This can
be done automatically with a pre-commit hook.
