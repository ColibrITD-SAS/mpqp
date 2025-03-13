from unittest.mock import patch

from mpqp.execution.devices import AWSDevice

# TODO: test methods


def test_from_arn():
    arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    assert AWSDevice.from_arn(arn) == AWSDevice.BRAKET_SV1_SIMULATOR


def test_get_arn():
    with patch(
        "mpqp.execution.connection.env_manager.get_env_variable",
        return_value="us-west-1",
    ):
        assert (
            AWSDevice.RIGETTI_ANKAA_2.get_arn()
            == "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2"
        )
