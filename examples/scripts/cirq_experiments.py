# %%
from mpqp import QCircuit
from mpqp.core.languages import Language
from mpqp.execution import run
from mpqp.execution.connection.key_connection import config_ionq_key
from mpqp.execution.devices import GOOGLEDevice, IBMDevice
from mpqp.gates import H, Rx, Ry, Rz
from mpqp.measures import BasisMeasure

# %%
circuit = QCircuit(3)
circuit.add(H(0))
circuit.add(H(1))
circuit.add(H(2))
circuit.add(Rx(1.76, 1))
circuit.add(Ry(1.76, 1))
circuit.add(Rz(1.987, 0))
circuit.add(BasisMeasure([0, 1, 2], shots=10000))
print(f"MPQP circuit:\n{circuit}\n")
# %%


results = run(
    circuit,
    [
        IBMDevice.AER_SIMULATOR,
        GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
        GOOGLEDevice.PROCESSOR_RAINBOW,
        GOOGLEDevice.PROCESSOR_WEBER,
    ],
)
print(results)

results.plot()

# %%
cirq_circuit = circuit.to_other_language(Language.CIRQ)
print(f"Cirq circuit:\n{cirq_circuit}\n")

# %%
# @title Choose a processor ("rainbow" or "weber")
processor_id = "rainbow"
grid_circuit = circuit.to_other_language(Language.CIRQ, cirq_proc_id=processor_id)
print(f"circuit for processor {processor_id}:\n{grid_circuit}\n")
