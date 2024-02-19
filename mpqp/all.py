# pyright: reportUnusedImport=false
import numpy as np
from . import QCircuit, Barrier, Language, Instruction
from .gates import (
    X,
    Y,
    Z,
    H,
    P,
    S,
    T,
    SWAP,
    U,
    Rx,
    Ry,
    Rz,
    Rk,
    CNOT,
    CZ,
    CRk,
    TOF,
    ControlledGate,
    Gate,
    ParametrizedGate,
    symbols,
    GateDefinition,
    KrausRepresentation,
    PauliDecomposition,
    CustomGate,
    UnitaryMatrix,
)
from .measures import (
    ComputationalBasis,
    Basis,
    HadamardBasis,
    VariableSizeBasis,
    BasisMeasure,
    ExpectationMeasure,
    Observable,
    Measure,
)
from .execution import (
    Result,
    StateVector,
    Sample,
    run,
    adjust_measure,
    submit,
    Job,
    JobStatus,
    JobType,
)
from .execution.devices import (
    ATOSDevice,
    IBMDevice,
    AWSDevice,
)
from .execution.vqa import minimize, Optimizer
from .execution.connection.qlm_connection import get_all_job_ids
from mpqp.execution.providers_execution.atos_execution import get_result_from_qlm_job_id
from .qasm import open_qasm_file_conversion_2_to_3, open_qasm_hard_includes

theta, k = symbols("θ k")  # type: ignore
obs = Observable(np.array([[0, 1], [1, 0]]))
circ = QCircuit([P(theta, 0), ExpectationMeasure([0], observable=obs, shots=1000)])
