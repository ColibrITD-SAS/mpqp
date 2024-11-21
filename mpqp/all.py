# pyright: reportUnusedImport=false
import numpy as np
from sympy import symbols

from mpqp.execution.providers.atos import get_result_from_qlm_job_id

from . import Barrier, Breakpoint, Instruction, Language, QCircuit
from .execution import (
    Job,
    JobStatus,
    JobType,
    Result,
    Sample,
    StateVector,
    adjust_measure,
    get_remote_result,
    run,
    submit,
)
from .execution.devices import (
    ATOSDevice,
    AWSDevice,
    GOOGLEDevice,
    IBMDevice,
    AZUREDevice,
)
from .execution.simulated_devices import IBMSimulatedDevice
from .execution.remote_handler import get_all_job_ids
from .execution.vqa import Optimizer, minimize
from .gates import (
    CNOT,
    CZ,
    SWAP,
    TOF,
    ControlledGate,
    CRk,
    CRk_dagger,
    CustomGate,
    Gate,
    GateDefinition,
    H,
    Id,
    P,
    CP,
    ParametrizedGate,
    Rk,
    Rk_dagger,
    Rx,
    Ry,
    Rz,
    S,
    T,
    U,
    UnitaryMatrix,
    X,
    Y,
    Z,
)
from .measures import (
    Basis,
    BasisMeasure,
    ComputationalBasis,
    ExpectationMeasure,
    HadamardBasis,
)
from .measures import I as Iop
from .measures import Measure, Observable, VariableSizeBasis
from .measures import X as Xop
from .measures import Y as Yop
from .measures import Z as Zop
from .noise import AmplitudeDamping, BitFlip, Depolarizing, PhaseDamping
from .qasm import open_qasm_file_conversion_2_to_3, open_qasm_hard_includes
from .tools.circuit import random_circuit
from .tools.display import pprint

theta, k = symbols("θ k")
obs = Observable(np.array([[0, 1], [1, 0]]))
circ = QCircuit(
    [P(theta, 0), ExpectationMeasure(targets=[0], observable=obs, shots=1000)]
)
