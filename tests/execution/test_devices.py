from unittest.mock import patch

import pytest

from mpqp import CZ, PRX, AWSDevice, QCircuit, Rxx, Ryy, Rzz

# TODO: test methods


def test_from_arn():
    arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    assert AWSDevice.from_arn(arn) == AWSDevice.BRAKET_SV1_SIMULATOR


def test_get_arn():
    with patch(
        "mpqp.environment.env_manager.get_env_variable",
        return_value="us-west-1",
    ):
        assert (
            AWSDevice.RIGETTI_ANKAA_3.get_arn()
            == "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"
        )


def test_iqm_native_gates():
    assert AWSDevice.IQM_GARNET.compatible_gates(native_set=True) == {CZ, PRX}
    assert AWSDevice.IQM_EMERALD.compatible_gates(native_set=True) == {CZ, PRX}


@pytest.mark.provider("braket")
def test_iqm_translation_preserves_qubit_indices():
    circuit = QCircuit([PRX(0.1, 0.2, 0), CZ(0, 2)])

    translated = circuit.to_other_device(AWSDevice.IQM_GARNET)

    assert circuit.instructions[0].targets == [0]
    controlled_gate = circuit.instructions[1]
    assert isinstance(controlled_gate, CZ)
    assert controlled_gate.controls == [0]
    assert controlled_gate.targets == [2]
    assert {int(qubit) for qubit in translated.qubits} == {0, 2}


@pytest.mark.provider("braket")
def test_iqm_translation_preserves_supported_rotation_gates():
    circuit = QCircuit(
        [
            Rxx(0.1, 0, 1),
            Ryy(0.2, 0, 1),
            Rzz(0.3, 0, 1),
            PRX(0.4, 0.5, 0),
        ]
    )

    translated = circuit.to_other_device(AWSDevice.IQM_GARNET)

    assert [instruction.operator.name for instruction in translated.instructions] == [
        "XX",
        "YY",
        "ZZ",
        "PRx",
    ]
