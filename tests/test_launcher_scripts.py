"""Native shell launchers (audit HPC-18): run them against fake conda/voila.

The launchers are copied into a throwaway repo skeleton, so nothing here
touches this checkout (they ``cd`` to their parent folder, clear
``quantui/__pycache__`` and write ``.dev_install_stamp`` there).
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="runs bash"
)

PORT = 8867  # fixed in both native launchers


def _skeleton(tmp_path: Path, launcher: str) -> tuple[Path, dict]:
    root = tmp_path / "repo with 'apostrophe'"
    (root / "launchers").mkdir(parents=True)
    (root / "notebooks").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    shutil.copy(REPO / "launchers" / launcher, root / "launchers" / launcher)

    home = tmp_path / "home"
    conda_sh = home / "miniforge3" / "etc" / "profile.d" / "conda.sh"
    conda_sh.parent.mkdir(parents=True)
    # Real conda activation scripts read unset variables; under `set -u`
    # that used to abort the launch.
    conda_sh.write_text(
        ': "${QUANTUI_TEST_UNSET_AT_SOURCE}"\n'
        'conda() { : "${QUANTUI_TEST_UNSET_IN_ACTIVATE}"; }\n'
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in {
        "pip": "exit 0",
        # Exits at once, like a Voilà with a startup error.
        "voila": 'echo "fake voila $*"; exit 3',
        "open": 'echo "OPENED $*"',
        "xdg-open": 'echo "OPENED $*"',
    }.items():
        exe = bindir / name
        exe.write_text(f"#!/bin/sh\n{body}\n")
        exe.chmod(0o755)
    env = {
        "HOME": str(home),
        "PATH": f"{bindir}:{Path(sys.executable).parent}:/usr/bin:/bin",
        "QUANTUI_APP_TOKEN": "tok",
    }
    return root / "launchers" / launcher, env


def _run(script: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )


@pytest.mark.parametrize("launcher", ["launch-native.sh", "launch-native.command"])
def test_native_launcher_startup_checks(tmp_path, launcher):
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", PORT))
    except OSError:
        probe.close()
        pytest.skip(f"port {PORT} is in use on this machine")
    script, env = _skeleton(tmp_path, launcher)

    # 1. Something already listening on the port: refuse, never start Voilà.
    probe.listen(1)
    try:
        busy = _run(script, env)
    finally:
        probe.close()
    assert busy.returncode == 1
    assert f"port {PORT} is already in use" in busy.stdout
    assert "fake voila" not in busy.stdout
    assert "unbound variable" not in busy.stderr  # conda activation under set -u

    # 2. Port free, Voilà dies at startup: say so instead of opening a URL
    #    nothing serves.
    died = _run(script, env)
    assert died.returncode == 1, died.stderr
    assert "fake voila" in died.stdout
    assert "exited during startup" in died.stdout
    assert "OPENED" not in died.stdout


def test_command_launcher_has_no_conda_executable_candidate():
    # `source`-ing the conda *executable* (condabin/conda) is not conda.sh.
    text = (REPO / "launchers" / "launch-native.command").read_text()
    assert "condabin/conda" not in text
    assert 'open "${URL}" || true' in text
