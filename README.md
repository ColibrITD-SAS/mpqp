![license badge](https://img.shields.io/github/license/ColibrITD-SAS/mpqp)
![test status badge](https://img.shields.io/github/actions/workflow/status/ColibrITD-SAS/mpqp/test?label=tests)
![doc status badge](https://img.shields.io/github/actions/workflow/status/ColibrITD-SAS/mpqp/doc?label=doc)
![pipy deployment status badge](https://img.shields.io/github/actions/workflow/status/ColibrITD-SAS/mpqp/pipy?label=pipy)

![mpqp logo](resources/logo.svg)

# The MPQP library

MPQP stands for Multi-Platform Quantum Programming. It is a python library we at
Colibri felt the need for but couldn't find a solution. We are working on
quantum algorithms, but until now, there was no good solution to study quantum
algorithms across devices, compares the devices, etc...

MPQP is thus the solution we bring to the community to tackle this problem.

![mpqp examples](resources/mpqp-usage.gif)

On this page, you will find:

1. how to [install](#install) the library;
2. how to [start using](#usage) it;
3. how to [contribute](#contribute) to the development;
4. and the current active [contributors](#contributors).

## Install

For now, we support python versions 3.9 to 3.11, and both Windows and Linux
(specifically, if was validated on Ubuntu LTS 20.04, while Ubuntu 18.04 is not
supported). MacOS versions will come very soon!

The preferred installation method is with the `pipy` repo. In order to use this
installation method, simply run

```
pip install mpqp
```

You can also clone this repo and install from source, for instance if you need
to modify something. In that case, we advise you to see the
[Contribute](#contribute) section of this document.

## Usage

To get started with MPQP, you can create a quantum circuit with a few gates, and
run it against the backend of your choice:

```py
from mpqp import QCircuit
from mpqp.gates import *
from mpqp.execution import run, IBMDevice
circ = QCircuit([H(0), H(1), Rx(0,0), CNOT(1,2), Y(2)])
print(circ)
#      ┌───┐┌───────┐
# q_0: ┤ H ├┤ Rx(0) ├─────
#      ├───┤└───────┘
# q_1: ┤ H ├────■─────────
#      └───┘  ┌─┴─┐  ┌───┐
# q_2: ───────┤ X ├──┤ Y ├
#             └───┘  └───┘
print(run(circ, IBMDevice.AER_SIMULATOR_STATEVECTOR))
# Result: IBMDevice, AER_SIMULATOR_STATEVECTOR
#
#         State vector: [0.-0.j  0.+0.5j 0.-0.5j 0.+0.j  0.-0.j  0.+0.5j 0.-0.5j 0.+0.j ]
#         Probabilities: [0.   0.25 0.25 0.   0.   0.25 0.25 0.  ]
#         Number of qubits: 3
```

More details are available in our [documentation](https://mpqpdoc.colibri-quantum.com).

## Contribute

To contribute, once you cloned this repo, you'll need to install the development
dependencies.

```
pip install -r requirements-dev.txt
```

We advise you to get in touch with us on
[our Discord server](https://discord.gg/yyukutWbzf) so we help you on any
difficulty you may encounter along the way.

Two tools are useful when working on MPQP:

- the test suite
- the documentation generation

If you need to try out some behaviors of the library during the development, you
can install it from source using the command.

```
pip install .
```

### Tests

To run the test suite, run the following command:

```sh
python -m pytest
```

By default, long tests are disables to be more friendly to regularly run for
devs. The full suit can be run by adding the option `-l` or `--long` to the
previous command. This should still be run regularly to validate retro
compatibility.

<!-- 3M-TODO: add doctest for doc testing and tox for multiversions testing -->

### Documentation

The website documentation is generated with
[sphinx](https://www.sphinx-doc.org/en/master/index.html). You probably don't
need to generate it if you work on new features, but if you want to help us by
improving the documentation, you need to know two things:

- how our documentation is structured, _i.e._ most of it is in the docstrings in
  the code. This is done on purpose to keep code and documentation close
  together.
- you only need to run one command to generate the documentation:

```
sphinx-build -b html docs build
```

The changelog is generated from github's versions section. For this to work, you
need to get a github token with public repo read right and save is as an
environment variable with the key `SPHINX_GITHUB_CHANGELOG_TOKEN`.
Alternatively, you can create a `.env` file by duplicating the `.env.example`
one and removing the `.example` extension, and replace in this file the ellipsis
by your token. This said, you don't need the changelog to generate to work on
the documentation.

## Contributors

Henri de Boutray - ColibriTD

Hamza Jaffali - ColibriTD

Muhammad Attallah - ColibriTD
