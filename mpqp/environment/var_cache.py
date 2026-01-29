from functools import lru_cache
from importlib.util import find_spec
from enum import Flag, auto

from mpqp.environment.env_manager import get_env_variable


@lru_cache(maxsize=1)
def translation_warning_enabled() -> bool:
    return not get_env_variable("MPQP_TRANSLATION_WARNING").lower() == "false"


class PackageInstall(Flag):
    NONE = 0
    QISKIT = auto()
    QISKIT_IBM_RUNTIME = auto()
    CIRQ = auto()
    BRAKET = auto()
    MY_QLM = auto()


def package_install() -> PackageInstall:
    flags = PackageInstall.NONE

    if find_spec("qiskit") is not None:
        flags |= PackageInstall.QISKIT

    try:
        from qiskit_ibm_runtime import fake_provider

        flags |= PackageInstall.QISKIT_IBM_RUNTIME
    except ImportError:
        pass

    if find_spec("cirq") is not None:
        flags |= PackageInstall.CIRQ

    if find_spec("braket") is not None:
        flags |= PackageInstall.BRAKET

    if find_spec("qat") is not None:
        flags |= PackageInstall.MY_QLM

    return flags


MPQP_PACKAGE_INSTALL = package_install()
