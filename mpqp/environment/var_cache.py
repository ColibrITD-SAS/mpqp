from enum import Flag, auto
from functools import lru_cache
from importlib.util import find_spec

from mpqp.environment.env_manager import get_env_variable


@lru_cache(maxsize=1)
def translation_warning_enabled() -> bool:
    return not get_env_variable("MPQP_TRANSLATION_WARNING").lower() == "false"


class InstalledProviders(Flag):
    NONE = 0
    QISKIT = auto()
    QISKIT_IBM_RUNTIME = auto()
    CIRQ = auto()
    BRAKET = auto()
    MY_QLM = auto()


def installed_providers() -> InstalledProviders:
    flags = InstalledProviders.NONE

    if find_spec("qiskit") is not None:
        flags |= InstalledProviders.QISKIT

    try:
        from qiskit_ibm_runtime import fake_provider # pyright: ignore[reportUnusedImport]

        flags |= InstalledProviders.QISKIT_IBM_RUNTIME
    except ImportError:
        pass

    if find_spec("cirq") is not None:
        flags |= InstalledProviders.CIRQ

    if find_spec("braket") is not None:
        flags |= InstalledProviders.BRAKET

    if find_spec("qat") is not None:
        flags |= InstalledProviders.MY_QLM

    return flags


_INSTALLED_MPQP_PROVIDERS = installed_providers()
