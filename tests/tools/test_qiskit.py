import sys
from types import ModuleType
from typing import Any

import pytest

import mpqp.environment.env_manager as env_manager
from mpqp.tools.qiskit import get_qiskit_function


def test_get_qiskit_function_uses_configured_instance(
    monkeypatch: pytest.MonkeyPatch,
):
    values = {
        "IBM_CONFIGURED": "True",
        "IBM_TOKEN": "token",
        "IBM_CHANNEL": "ibm_quantum_platform",
        "IBM_INSTANCE": "configured-crn",
    }
    monkeypatch.setattr(env_manager, "get_env_variable", values.__getitem__)

    catalog_arguments = {}
    expected_function = object()

    class FakeCatalog:
        def __init__(self, **kwargs: Any):
            catalog_arguments.update(kwargs)

        def load(self, name: str):
            assert name == "q-ctrl/performance-management"
            return expected_function

    catalog_module = ModuleType("qiskit_ibm_catalog")
    catalog_module.QiskitFunctionsCatalog = FakeCatalog  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit_ibm_catalog", catalog_module)

    function = get_qiskit_function("q-ctrl/performance-management")

    assert function is expected_function
    assert catalog_arguments == {
        "token": "token",
        "channel": "ibm_quantum_platform",
        "instance": "configured-crn",
    }


def test_get_qiskit_function_requires_an_instance(monkeypatch: pytest.MonkeyPatch):
    values = {
        "IBM_CONFIGURED": "True",
        "IBM_TOKEN": "token",
        "IBM_CHANNEL": "ibm_quantum_platform",
        "IBM_INSTANCE": "",
    }
    monkeypatch.setattr(env_manager, "get_env_variable", values.__getitem__)

    catalog_module = ModuleType("qiskit_ibm_catalog")
    catalog_module.QiskitFunctionsCatalog = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit_ibm_catalog", catalog_module)

    with pytest.raises(ValueError, match="No IBM Catalog instance"):
        get_qiskit_function("q-ctrl/performance-management")
