"""Audit INFRA-11 — the cloud SessionStart hook must cope with an
unsupported default interpreter.

The hook used to run bare ``python -m pip install -e ...``. On the cloud image
``python`` is 3.13, outside QuantUI's supported 3.9-3.11 (pyscf-properties
does not build there), so sessions started with nothing installed. The hook
now detects that, builds a 3.11 venv in ``.venv`` (uv, else python3.11) and
installs into it, exporting the venv through ``$CLAUDE_ENV_FILE``.

The hook itself is never executed here (it pip-installs the world); these
tests check its syntax, that every pip call goes through the selected
interpreter, and the version gate's behaviour on real/stub interpreters.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_HOOK = (
    Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "session-start.sh"
)

needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="runs bash",
)


def _hook_text() -> str:
    return _HOOK.read_text(encoding="utf-8")


@needs_bash
def test_hook_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(_HOOK)], check=True)


def test_every_pip_call_uses_selected_interpreter() -> None:
    code_lines = [
        ln for ln in _hook_text().splitlines() if not ln.lstrip().startswith("#")
    ]
    pip_lines = [ln for ln in code_lines if "-m pip" in ln]
    assert pip_lines, "hook no longer installs anything?"
    for ln in pip_lines:
        assert ln.lstrip().startswith('"$PY" -m pip'), ln


def test_hook_creates_311_venv_and_exports_it() -> None:
    text = _hook_text()
    assert "uv venv --seed -p python3.11 .venv" in text
    assert "CLAUDE_ENV_FILE" in text
    assert 'PY="$PWD/.venv/bin/python"' in text


def _supported_fn() -> str:
    m = re.search(r"^_supported\(\) \{.*?^\}", _hook_text(), re.S | re.M)
    assert m, "_supported() gate missing from hook"
    return m.group(0)


@needs_bash
@pytest.mark.parametrize(
    "version, ok",
    [
        ((3, 9), True),
        ((3, 11), True),
        ((3, 8), False),
        ((3, 12), False),
        ((3, 13), False),
    ],
)
def test_version_gate(tmp_path: Path, version, ok) -> None:
    # Stub interpreter: a tiny script that runs the gate's -c snippet against a
    # faked sys.version_info.
    stub = tmp_path / "fakepy"
    stub.write_text(
        "#!/bin/sh\n"
        f'exec "{sys.executable}" -c '
        f'"import sys; sys.version_info = {version!r}; exec(sys.argv[1])" "$2"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    script = _supported_fn() + f'\n_supported "{stub}"\n'
    rc = subprocess.run(["bash", "-c", script]).returncode
    assert (rc == 0) is ok
