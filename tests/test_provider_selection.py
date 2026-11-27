"""Provider requirements are enforced through the public pytest selection option."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    'providers, passed, skipped',
    [(['all'], 4, 0), (['qiskit'], 1, 3), (['qiskit', 'braket'], 3, 1), ([], 0, 4)],
)
def test_provider_markers_require_every_declared_provider(
    tmp_path: Path, providers: list[str], passed: int, skipped: int
):
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / 'conftest.py', tmp_path / 'conftest.py')
    (tmp_path / 'test_probe.py').write_text(
        """import pytest
@pytest.mark.provider('qiskit')
def test_qiskit():
    pass
@pytest.mark.provider('qiskit', 'braket')
def test_multiple_arguments():
    pass
@pytest.mark.provider('qiskit')
@pytest.mark.provider('braket')
def test_multiple_markers():
    pass
@pytest.mark.provider('cirq')
def test_cirq():
    pass
""",
        encoding='utf-8',
    )
    environment = os.environ.copy()
    environment['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    environment['PYTHONPATH'] = os.pathsep.join(
        [str(root), environment.get('PYTHONPATH', '')]
    )
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            'test_probe.py',
            '-q',
            '--providers',
            *providers,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if passed:
        assert f'{passed} passed' in result.stdout
    if skipped:
        assert f'{skipped} skipped' in result.stdout
