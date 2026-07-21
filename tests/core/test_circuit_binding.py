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
m2 = ExpectationMeasure(Observable(pX@pZ), label="Exp2") 
m3 = BasisMeasure(label="b3", shots=1024)                            
m4 = None                                                                                     

m_I = ExpectationMeasure(Observable(pI), label="Exp_I")
m_Z = ExpectationMeasure(Observable(pZ), label="Exp_Z")

def test_broadcasting_product_shapes():
    """Valide que le mode PRODUCT génère la bonne grille 2D pour Qiskit V2."""
    binding = CircuitBinding(
        circuits=c1, 
        values=[v1, v2],
        measurements=[m_I, m_Z],
        mode=BindingMode.PRODUCT
    )
    
    pubs_with_context = binding.Broadcasting(IBMDevice.AER_SIMULATOR)

    assert len(pubs_with_context) == 1
    
    pub, context_job = pubs_with_context[0]
    assert len(pub) == 3


def test_broadcasting_zip_shapes():
    """Valide que le mode ZIP aligne les tableaux en 1D."""
    binding = CircuitBinding(
        circuits=c1, 
        values=[v1, v2],           
        measurements=[m_I, m_Z],   
        mode=BindingMode.ZIP
    )
    
    pubs_with_context = binding.Broadcasting(IBMDevice.AER_SIMULATOR)
    assert len(pubs_with_context) > 0



def test_broadcasting_zip_broadcasting_rules():
    """Valide que le mode ZIP duplique correctement un élément unique (règle de broadcast)."""
    binding = CircuitBinding(
        circuits=c1, 
        values=v1,                
        measurements=[m_I, m_Z],  
        mode=BindingMode.ZIP
    )
    
    pubs_with_context = binding.Broadcasting(IBMDevice.AER_SIMULATOR)
    _, q_obs, q_params = pubs_with_context[0][0]
    
    assert len(q_obs) == 2
    assert len(q_params) == 2
    assert q_params[0] == q_params[1] == [1.0, 1.0, 1.0]


def test_broadcasting_context_job_creation():
    """Vérifie que le Job de contexte MPQP est correctement créé et attaché au PUB."""
    from mpqp.execution.job import JobType
    
    binding = CircuitBinding(
        circuits=c1, 
        measurements=m_I,
        mode=BindingMode.PRODUCT
    )
    
    pubs_with_context = binding.Broadcasting(IBMDevice.AER_SIMULATOR)
    _, context_job = pubs_with_context[0]
    
    assert context_job.job_type == JobType.OBSERVABLE
    assert context_job.device == IBMDevice.AER_SIMULATOR


def test_broadcasting_recursive_bindings():
    """S'assure que les CircuitBindings imbriqués sont bien aplatis et parsés."""
    inner_binding = CircuitBinding(
        circuits=c2, 
        measurements=m1
    )
    outer_binding = CircuitBinding(
        circuits=inner_binding,
        values=[v1, v2]
    )
    
    pubs_with_context = outer_binding.Broadcasting(IBMDevice.AER_SIMULATOR)
    assert len(pubs_with_context) == 1

def test_broadcasting_exceptions():
    """Valide les sécurités de la méthode Broadcasting."""
    
    binding_dev = CircuitBinding(circuits=c1, measurements=m_I)
    with pytest.raises(NotImplementedError, match="only implemented for IBMDevice"):
        binding_dev.Broadcasting(ATOSDevice.MYQLM_PYLINALG)
        
    m_basis = BasisMeasure()
    binding_type = CircuitBinding(circuits=c1, measurements=m_basis)
    
    with pytest.raises(ValueError, match="only supported for circuits with expectation measurements"):
        binding_type.Broadcasting(IBMDevice.AER_SIMULATOR)

@pytest.fixture
def list_circuit_binding_produit():
    """Test du mode PRODUCT et validation via les propriétés de Result."""
    
    cb1 = CircuitBinding(circuits=c1, values=[v1, v2], measurements=m1, mode=BindingMode.PRODUCT)
    def val_cb1(res: BatchResult):
        assert isinstance(res, BatchResult)
        for r in res.results:
            if isinstance(r.expectation_values, dict):
                val = list(r.expectation_values.values())[0]
            else:
                val = r.expectation_values
            assert val == pytest.approx(1.0, abs=1e-5)

    cb2 = CircuitBinding(circuits=[c2, c3], measurements=m3, mode=BindingMode.PRODUCT)
    def val_cb2(res: BatchResult):
        assert isinstance(res, BatchResult)
        
        r_c2 = res.results[0]
        assert r_c2.counts[0] > 0
        assert r_c2.counts[1] > 0
        assert sum(r_c2.counts) == r_c2.shots
        
        r_c3 = res.results[1]
        assert len(r_c3.counts) >= 2
        assert sum(r_c3.counts) == r_c3.shots
        assert r_c3.counts[0] > 0

    cb3 = CircuitBinding(circuits=c3, measurements=m2, mode=BindingMode.PRODUCT)
    def val_cb3(res: Result):
        assert isinstance(res, Result)
        if isinstance(res.expectation_values, dict):
            val = list(res.expectation_values.values())[0]
        else:
            val = res.expectation_values
        assert val == pytest.approx(0.0, abs=1e-2)

    return [(cb1, val_cb1), (cb2, val_cb2), (cb3, val_cb3)]


@pytest.fixture
def list_circuit_binding_zip():
    """Test du mode ZIP et validation avec broadcast Numpy."""
    
    cb1 = CircuitBinding(circuits=c1, values=[v1, v2, v3], measurements=m1, mode=BindingMode.ZIP)
    def val_cb1(res: BatchResult):
        assert isinstance(res, BatchResult)
        for r in res.results:
            if isinstance(r.expectation_values, dict):
                val = list(r.expectation_values.values())[0]
            else:
                val = r.expectation_values
            assert val == pytest.approx(1.0, abs=1e-5)
            
    cb2 = CircuitBinding(circuits=[c2, c4], values=m4, measurements=[m3, m3], mode=BindingMode.ZIP)
    def val_cb2(res: BatchResult):
        assert isinstance(res, BatchResult)
        assert len(res.results) == 2
        
        assert len(res.results[0].counts) >= 2
        assert res.results[0].counts[0] > 0
        
        assert len(res.results[1].counts) >= 4  
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
            if isinstance(r.expectation_values, dict):
                val = list(r.expectation_values.values())[0]
            else:
                val = r.expectation_values
            assert val == pytest.approx(1.0, abs=1e-5)

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
