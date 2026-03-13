# pyright: reportUnusedImport=false
from mpqp.core.circuit import QCircuit
from mpqp.core.instruction import Instruction
from mpqp.core.instruction.barrier import Barrier
from mpqp.core.instruction.breakpoint import Breakpoint
from mpqp.core.languages import Language

from ._version import __version__
from .execution import (
    BatchResult,
    Job,
    JobStatus,
    JobType,
    Result,
    Sample,
    StateVector,
    get_remote_result,
    run,
    submit,
)
from .execution.devices import (
    ATOSDevice,
    AWSDevice,
    AZUREDevice,
    GOOGLEDevice,
    IBMDevice,
)
from .execution.remote_handler import get_all_remote_job_ids
from .execution.simulated_devices import IBMSimulatedDevice
from .execution.vqa import Optimizer, minimize
from .execution.vqa.qaoa import qaoa_solver
from .execution.vqa.qubo import QuboAtom
from .gates import (
    CNOT,
    CP,
    CZ,
    SWAP,
    TOF,
    ControlledGate,
    CRk,
    CRk_dagger,
    CustomControlledGate,
    CustomGate,
    Gate,
    GateDefinition,
    H,
    Id,
    P,
    ParametrizedGate,
    Rk,
    Rk_dagger,
    Rx,
    Ry,
    Rz,
    S,
    S_dagger,
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
    Measure,
    Observable,
    VariableSizeBasis,
    pI,
    pX,
    pY,
    pZ,
)
from .noise import AmplitudeDamping, BitFlip, Depolarizing, PhaseDamping
from .tools.display import pprint
