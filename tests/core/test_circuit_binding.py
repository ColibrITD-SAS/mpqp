from __future__ import annotations


import pytest

from mpqp.core.circuit import BindingMode, CircuitBinding
from mpqp.execution.devices import (
    AvailableDevice,
    ATOSDevice,
    GOOGLEDevice,
    IBMDevice,
    AWSDevice,
)

from mpqp import (
    CNOT,
    CZ,
    SWAP,
    TOF,
    AmplitudeDamping,
    ATOSDevice,
    Barrier,
    BasisMeasure,
    BitFlip,
    CRk,
    Depolarizing,
    ExpectationMeasure,
    Gate,
    H,
    IBMDevice,
    Id,
    Instruction,
    Language,
    Measure,
    Observable,
    P,
    PhaseDamping,
    QCircuit,
    Result,
    Rx,
    Ry,
    Rz,
    S,
    T,
    U,
    X,
    Y,
    Z,
    pI,
    pZ,
    pX,
    run,
)
from mpqp.execution.result import BatchResult
from sympy import Symbol

theta, phi, psi = Symbol('θ'), Symbol('phi'), Symbol('psi')
c1 = QCircuit([U(theta, phi, psi, 0)], label="c1") 
c2 = QCircuit([H(0)], label="c2")                  
c3 = QCircuit([H(0), CNOT(0, 1)], label="c3")     
c4 = QCircuit([H(0), H(1), CNOT(0,1)], label="c4")  

v1 = {'θ': 1.0, 'phi': 1.0, 'psi': 1.0}
v2 = {'θ': 2.0, 'phi': 2.0, 'psi': 2.0}
v3 = {'θ': 3.0, 'phi': 3.0, 'psi': 3.0}
v4 = {'θ': 4.0, 'phi': 4.0, 'psi': 4.0}

m1 = ExpectationMeasure(Observable(pI), label="Exp1")   
m2 = ExpectationMeasure(Observable(pX*pZ), label="Exp2") 
m3 = BasisMeasure(label="b3")                            
m4 = None                                                 

@pytest.fixture
def list_circuit_binding_produit():
    """Test du mode PRODUCT et validation via les propriétés de Result."""
    
    cb1 = CircuitBinding(circuits=c1, values=[v1, v2], measurements=m1, mode=BindingMode.PRODUCT)
    def val_cb1(res: BatchResult):
        assert isinstance(res, BatchResult)
        for r in res.results:
            assert "Exp1" in r.expectation_values
            assert r.expectation_values["Exp1"] == pytest.approx(1.0, abs=1e-5)

    cb2 = CircuitBinding(circuits=[c2, c3], measurements=m3, mode=BindingMode.PRODUCT)
    def val_cb2(res: BatchResult):
        assert isinstance(res, BatchResult)
        
        r_c2 = res.results[0]
        assert r_c2.counts[0] > 0
        assert r_c2.counts[1] > 0
        assert sum(r_c2.counts) == r_c2.shots
        
        r_c3 = res.results[1]
        assert r_c3.counts[0] > 0
        assert r_c3.counts[3] > 0
        assert r_c3.counts[1] == 0 
        assert r_c3.counts[2] == 0

    cb3 = CircuitBinding(circuits=c3, measurements=m2, mode=BindingMode.PRODUCT)
    def val_cb3(res: Result):
        assert isinstance(res, Result)
        assert res.expectation_values["Exp2"] == pytest.approx(0.0, abs=1e-2)

    return [(cb1, val_cb1), (cb2, val_cb2), (cb3, val_cb3)]


@pytest.fixture
def list_circuit_binding_zip():
    """Test du mode ZIP et validation avec broadcast Numpy."""
    
    cb1 = CircuitBinding(circuits=c1, values=[v1, v2, v3], measurements=m1, mode=BindingMode.ZIP)
    def val_cb1(res: BatchResult):
        assert isinstance(res, BatchResult)
        assert len(res.results) == 3
        for r in res.results:
            assert r.expectation_values["Exp1"] == pytest.approx(1.0, abs=1e-5)
            
    cb2 = CircuitBinding(circuits=[c2, c4], values=m4, measurements=[m3, m3], mode=BindingMode.ZIP)
    def val_cb2(res: BatchResult):
        assert isinstance(res, BatchResult)
        assert len(res.results) == 2
        
        assert len(res.results[0].counts) == 2
        assert res.results[0].counts[0] > 0
        
        # Circuit 2 (2 Qubits)
        assert len(res.results[1].counts) == 4  
        assert res.results[1].counts[0] > 0

    return [(cb1, val_cb1), (cb2, val_cb2)]


@pytest.fixture
def list_circuit_binding_recursif():
    """Test d'imbrication avec préservation des contextes (Job/Measure)."""
    
    cb_inner = CircuitBinding(circuits=c1, measurements=m1, mode=BindingMode.ZIP)
    cb_outer = CircuitBinding(circuits=cb_inner, values=[v1, v2], mode=BindingMode.PRODUCT)
    
    def val_cb_outer(res: BatchResult):
        assert isinstance(res, BatchResult)
        assert len(res.results) == 2
        for r in res.results:
            assert "Exp1" in r.expectation_values
            assert r.expectation_values["Exp1"] == pytest.approx(1.0, abs=1e-5)

    return [(cb_outer, val_cb_outer)]

def execute_and_validate(bindings_and_validators):
    for binding, validator in bindings_and_validators:
        result = run(binding, device=IBMDevice.AER_SIMULATOR)
        validator(result)

@pytest.mark.provider("qiskit")
def test_circuit_binding_produit(list_circuit_binding_produit):
    execute_and_validate(list_circuit_binding_produit)

@pytest.mark.provider("qiskit")
def test_circuit_binding_zip(list_circuit_binding_zip):
    execute_and_validate(list_circuit_binding_zip)

@pytest.mark.provider("qiskit")
def test_circuit_binding_recursif(list_circuit_binding_recursif):
    execute_and_validate(list_circuit_binding_recursif)