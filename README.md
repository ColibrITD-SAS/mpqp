# The MPQP library

## Install

### Install MPQP (and user dependencies)

```
pip install .
```

### Install dev dependencies

```
pip install -r requirements-dev.txt
```

### Install only user dependencies

```
pip install -r requirements.txt
```

## Documentation

To generate the website documentation with sphinx

```
sphinx-build -b html docs build
```

## Tests

To run the test suite, run the following command:

```sh
python -m pytest
```

By default, long tests are disables to be more friendly to regularly run for
devs. The full suit can be run by adding the option `-l` or `--long` to the
previous command. This should still be run regularly to validate retro
compatibility.

TODO: add doctest for doc testing and tox for multiversions testing

## Contributors

Henri de Boutray - ColibrITD

Hamza Jaffali - ColibrITD

Muhammad Attallah - ColibrITD