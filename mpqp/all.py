# pyright: reportUnusedImport=false
import numpy as np

from mpqp.execution.providers_execution.atos_execution import get_result_from_qlm_job_id

from . import Barrier, Instruction, Language, QCircuit
from .execution import (
    Job,
    JobStatus,
    JobType,
    Result,
    Sample,
    StateVector,
    adjust_measure,
    run,
    submit,
)
from .execution.connection.qlm_connection import get_all_job_ids
from .execution.devices import ATOSDevice, AWSDevice, IBMDevice
from .execution.vqa import Optimizer, minimize
from .gates import (
    CNOT,
    CZ,
    SWAP,
    TOF,
    ControlledGate,
    CRk,
    CustomGate,
    Gate,
    GateDefinition,
    H,
    Id,
    KrausRepresentation,
    P,
    ParametrizedGate,
    PauliDecomposition,
    Rk,
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
    symbols,
)
from .measures import (
    Basis,
    BasisMeasure,
    ComputationalBasis,
    ExpectationMeasure,
    HadamardBasis,
    Measure,
    Observable,
    VariableSizeBasis,
)
from .qasm import open_qasm_file_conversion_2_to_3, open_qasm_hard_includes

theta, k = symbols("θ k")  # type: ignore
obs = Observable(np.array([[0, 1], [1, 0]]))
circ = QCircuit([P(theta, 0), ExpectationMeasure([0], observable=obs, shots=1000)])
