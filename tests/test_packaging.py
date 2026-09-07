"""Wheel-content smoke tests (AUDIT F20).

Every other test in this suite runs against an **editable** install
(``pip install -e .``, per CI and CLAUDE.md's setup instructions), which
never consults ``[tool.setuptools]``'s package list at all — the source
tree is used as-is. That is exactly why the bug this file guards against
went unnoticed: ``pyproject.toml`` explicitly listed
``packages = ["quantui", "quantui.backends"]``, silently omitting the real
``quantui.engines`` subpackage. A *built* wheel installed non-editably
contained no engine files at all, while CI — always editable — stayed
green.

These tests build a real wheel via the PEP 517 frontend and inspect its
contents directly, so a future subpackage that isn't picked up by
discovery (or a future explicit list that forgets one) fails here instead
of only in an installed deployment.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


def _build_wheel(tmp_path: Path) -> Path:
    """Build the project wheel into ``tmp_path`` and return its path."""
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one built wheel, found {wheels}"
    return wheels[0]


@pytest.mark.slow
class TestWheelContents:
    """Inspect the built wheel's file list directly (no install needed)."""

    def test_engines_subpackage_is_shipped(self, tmp_path):
        """AUDIT F20 — quantui/engines/* must be present in the wheel.

        This is the exact regression: the package list omitted
        quantui.engines, so a real (non-editable) install had no engine
        module at all.
        """
        pytest.importorskip("build")
        wheel_path = _build_wheel(tmp_path)
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
        engine_files = [n for n in names if n.startswith("quantui/engines/")]
        assert engine_files, f"no quantui/engines/* files in wheel: {names}"
        assert "quantui/engines/__init__.py" in engine_files
        assert "quantui/engines/base.py" in engine_files
        assert "quantui/engines/pyscf_engine.py" in engine_files
        assert "quantui/engines/pyfock_engine.py" in engine_files

    def test_backends_subpackage_is_shipped(self, tmp_path):
        """Same check for quantui.backends, which the old explicit list did
        include — a regression guard so a future edit can't drop it either."""
        pytest.importorskip("build")
        wheel_path = _build_wheel(tmp_path)
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
        assert "quantui/backends/__init__.py" in names
        assert "quantui/backends/worker.py" in names

    def test_tests_package_is_not_shipped(self, tmp_path):
        """``tests/`` has its own __init__.py (it's a real Python package),
        so package discovery must explicitly exclude it — otherwise the
        wheel ships the whole test suite as an importable top-level
        package, which is not what a distributed application wheel wants.
        """
        pytest.importorskip("build")
        wheel_path = _build_wheel(tmp_path)
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
        test_files = [n for n in names if n.startswith("tests/")]
        assert not test_files, f"tests/* leaked into the wheel: {test_files}"

    def test_bundled_data_files_are_shipped(self, tmp_path):
        """Package-data (the offline molecule library + vendored 3Dmol.js)
        must survive alongside the package-discovery change — this is a
        pre-existing guarantee (not part of F20), reasserted here since
        this file is the one place a wheel is actually built and
        inspected."""
        pytest.importorskip("build")
        wheel_path = _build_wheel(tmp_path)
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
        assert "quantui/data/library/library.sqlite" in names
        assert "quantui/data/js/3Dmol-min.js" in names


@pytest.mark.slow
class TestWheelIsolatedImport:
    """Install the built wheel into a throwaway venv and import from
    outside the checkout — the audit's own reproduction: "Importing
    quantui.engines from the extracted wheel, with the editable source
    finder removed from the probe process, raises ModuleNotFoundError."
    """

    def test_engines_importable_from_installed_wheel(self, tmp_path):
        pytest.importorskip("build")
        wheel_path = _build_wheel(tmp_path / "dist")

        venv_dir = tmp_path / "venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        venv_python = venv_dir / "bin" / "python"
        if not venv_python.exists():  # pragma: no cover - Windows layout
            venv_python = venv_dir / "Scripts" / "python.exe"

        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--quiet", str(wheel_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        # Run the import check from tmp_path, NOT the repo checkout, so
        # Python can't fall back to the source tree on sys.path[0] and
        # mask a genuinely broken installed package (the audit's own
        # methodology: "with the editable source finder removed from the
        # probe process").
        result = subprocess.run(
            [
                str(venv_python),
                "-c",
                "import quantui.engines as e; "
                "from quantui.engines import base, pyscf_engine, pyfock_engine; "
                "print('OK', e.__file__)",
            ],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"quantui.engines not importable from the installed wheel:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "OK" in result.stdout
