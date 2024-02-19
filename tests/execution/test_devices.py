from unittest.mock import patch

from mpqp.execution.devices import ATOSDevice, AWSDevice, IBMDevice


def test_ibm_device():
    assert IBMDevice.PULSE_SIMULATOR.is_remote() is False
    assert IBMDevice.PULSE_SIMULATOR.is_gate_based() is False
    assert IBMDevice.PULSE_SIMULATOR.is_simulator() is True

    assert IBMDevice.IBM_BRISBANE.is_remote() is True
    assert IBMDevice.IBM_BRISBANE.is_gate_based() is True
    assert IBMDevice.IBM_BRISBANE.is_simulator() is False


def test_atos_device():
    assert ATOSDevice.QLM_LINALG.is_remote() is True
    assert ATOSDevice.QLM_LINALG.is_gate_based() is True
    assert ATOSDevice.QLM_LINALG.is_simulator() is True

    assert ATOSDevice.MYQLM_PYLINALG.is_gate_based() is True
    assert ATOSDevice.MYQLM_PYLINALG.is_simulator() is True

    assert ATOSDevice.QLM_MPS.is_remote() is True
    assert ATOSDevice.QLM_MPS.is_gate_based() is True
    assert ATOSDevice.QLM_MPS.is_simulator() is True

    assert ATOSDevice.MYQLM_CLINALG.is_gate_based() is True
    assert ATOSDevice.MYQLM_CLINALG.is_simulator() is True


def test_aws_device():
    local_simulator = AWSDevice.BRAKET_LOCAL_SIMULATOR
    assert local_simulator.is_remote() is False
    assert local_simulator.is_gate_based() is True
    assert local_simulator.is_simulator() is True

    harmony_device = AWSDevice.BRAKET_IONQ_HARMONY
    assert harmony_device.is_remote() is True
    assert harmony_device.is_gate_based() is True
    assert harmony_device.is_simulator() is False

    with patch(
        "mpqp.execution.connection.env_manager.get_env_variable",
        return_value="us-east-1",
    ):
        assert (
            harmony_device.get_arn()
            == "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony"
        )
    arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    assert AWSDevice.from_arn(arn) == AWSDevice.BRAKET_SV1_SIMULATOR


def test_env_manager():
    with patch(
        "mpqp.execution.connection.env_manager.get_env_variable",
        return_value="us-west-1",
    ):
        assert (
            AWSDevice.BRAKET_RIGETTI_ASPEN_M_3.get_arn()
            == "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3"
        )
