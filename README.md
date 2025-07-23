![license badge](https://img.shields.io/github/license/ColibrITD-SAS/mpqp?logo=openaccess&logoColor=white)
![test status badge](https://img.shields.io/github/actions/workflow/status/ColibrITD-SAS/mpqp/tests.yml?branch=dev&label=tests&logo=pytest&logoColor=white)
![doc status badge](https://img.shields.io/github/actions/workflow/status/ColibrITD-SAS/mpqp/doc.yml?label=doc&logo=read-the-docs&logoColor=white)
![pipy deployment status badge](https://img.shields.io/github/actions/workflow/status/ColibrITD-SAS/mpqp/pipy.yml?label=pipy&logo=pypi&logoColor=white)
![github stars badge](https://img.shields.io/github/stars/ColibrITD-SAS/mpqp?logo=github)

![[alt:mpqp logo]](resources/dark-logo.svg)

# The MPQP library

MPQP stands for Multi-Platform Quantum Programming. It is a python library we at
Colibri felt the need for but couldn't find a solution. We are working on
quantum algorithms, but until now, there was no good solution to study quantum
algorithms across devices, compares the devices, etc...

MPQP is thus the solution we bring to the community to tackle this problem.

![[alt:mpqp examples]](resources/mpqp-usage.gif)

On this page, you will find:

1. how to [install](#install) the library;
2. how to [start using](#usage) it;
3. and the current active [contributors](#contributors).

## Install

For now, we support python versions 3.9 to 3.11, and every major OS (Windows,
Linux and MacOS). We are dependant on the SDKs we support to enable various
python versions and OS support, for instance, MPQP was validated on Ubuntu LTS
20.04, while Ubuntu 18.04 is not supported because myQLM does not support it.

The preferred installation method is with the `pipy` repo. In order to use this
installation method, simply run

```
pip install mpqp
```

You can also clone this repo and install from source, for instance if you need
to modify something. In that case, we advise you to have a look at our
[contribution guide](CONTRIBUTING.md).

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
#  State vector: [0.-0.j  0.+0.5j 0.-0.5j 0.+0.j  0.-0.j  0.+0.5j 0.-0.5j 0.+0.j ]
#  Probabilities: [0.   0.25 0.25 0.   0.   0.25 0.25 0.  ]
#  Number of qubits: 3
```

More details are available in our [documentation](https://mpqpdoc.colibri-quantum.com).

## Contributors

<table>
<tr>
<td style="text-align:center">
<img src="https://github.com/Henri-ColibrITD.png" width="60px;"/><br/><sub><a href="https://github.com/Henri-ColibrITD">Henri de Boutray</a></sub>
</td>
<td style="text-align:center">
<img src="https://github.com/hJaffaliColibritd.png" width="60px;"/><br/><sub><a href="https://github.com/hJaffaliColibritd">Hamza Jaffali</a></sub>
</td>
<td style="text-align:center">
<img src="https://github.com/MoHermes.png" width="60px;"/><br /><sub><a href="https://github.com/MoHermes">Muhammad Attallah</a></sub>
</td>
<td style="text-align:center">
<img src="https://github.com/JulienCalistoTD.png" width="60px;"/><br/><sub><a href="https://github.com/JulienCalistoTD">JulienCalisto</a></sub>
</td>
<td style="text-align:center">
<img src="https://github.com/ah4dev.png" width="60px;"/><br /><sub><a href="https://github.com/ah4dev">Ahmed Bejaoui</a></sub>
</td>
</tr>
</table>
