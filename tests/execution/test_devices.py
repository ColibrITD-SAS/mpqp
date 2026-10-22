import pytest

from mpqp import (
    CZ,
    PRX,
    AWSDevice,
    Gate,
    QCircuit,
    Rx,
    Rxx,
    Ryy,
    Rz,
    Rzz,
)


def test_from_arn():
    arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    assert AWSDevice.from_arn(arn) == AWSDevice.BRAKET_SV1_SIMULATOR


@pytest.mark.parametrize(
    ("device", "expected_arn"),
    [
        (
            AWSDevice.RIGETTI_ANKAA_3,
            "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
        ),
        (
            AWSDevice.AQT_IBEX_Q1,
            "arn:aws:braket:eu-north-1::device/qpu/aqt/Ibex-Q1",
        ),
        (
            AWSDevice.RIGETTI_CEPHEUS_1_108Q,
            "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
        ),
    ],
)
def test_get_arn(device: AWSDevice, expected_arn: str):
    assert device.get_arn() == expected_arn


def test_retired_aws_devices():
    retired_devices = {
        AWSDevice.BRAKET_TN1_SIMULATOR,
        AWSDevice.IONQ_ARIA_1,
        AWSDevice.IONQ_ARIA_2,
    }

    assert {device for device in AWSDevice if device.is_retired()} == retired_devices


@pytest.mark.parametrize(
    ("device", "expected_gates"),
    [
        (AWSDevice.AQT_IBEX_Q1, {PRX, Rxx, Rz}),
        (AWSDevice.IQM_GARNET, {CZ, PRX}),
        (AWSDevice.IQM_EMERALD, {CZ, PRX}),
        (AWSDevice.IONQ_FORTE_1, {Rzz}),
        (AWSDevice.IONQ_FORTE_ENTERPRISE_1, {Rzz}),
        (AWSDevice.RIGETTI_ANKAA_3, {Rx, Rz}),
    ],
)
def test_native_gates(device: AWSDevice, expected_gates: set[type[Gate]]):
    assert device.compatible_gates(native_set=True) == expected_gates


@pytest.mark.provider("braket")
def test_iqm_translation_preserves_qubit_indices():
    circuit = QCircuit([PRX(0.1, 0.2, 0), CZ(0, 2)])

    translated = circuit.to_other_device(AWSDevice.IQM_GARNET)

    assert circuit.instructions[0].targets == [0]
    controlled_gate = circuit.instructions[1]
    assert isinstance(controlled_gate, CZ)
    assert controlled_gate.controls == [0]
    assert controlled_gate.targets == [2]

    translated_cz = translated.instructions[-1]
    assert translated_cz.operator.name == "CZ"
    assert [int(qubit) for qubit in translated_cz.target] == [0, 2]


@pytest.mark.provider("braket")
def test_iqm_translation_preserves_supported_rotation_gates():
    circuit = QCircuit(
        [
            Rxx(0.1, 0, 1),
            Ryy(0.2, 0, 1),
            Rzz(0.3, 0, 1),
        ]
    )

    translated = circuit.to_other_device(AWSDevice.IQM_GARNET)

    assert [instruction.operator.name for instruction in translated.instructions] == [
        "XX",
        "YY",
        "ZZ",
    ]


@pytest.mark.provider("braket")
def test_iqm_translation_decomposes_prx():
    circuit = QCircuit([PRX(0.1, 0.2, 0)])

    translated = circuit.to_other_device(AWSDevice.IQM_GARNET)

    assert [instruction.operator.name for instruction in translated.instructions] == [
        "Rz",
        "Rx",
        "Rz",
    ]
