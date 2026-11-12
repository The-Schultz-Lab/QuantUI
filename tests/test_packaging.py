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

The wheel is built once per session (per xdist worker) from a ``git
ls-files`` copy of the tree, with ``--no-isolation`` so no network is needed
(audit INFRA-3). Only the fresh-venv ``pip install`` probe, which resolves
runtime dependencies from PyPI, is marked ``network``; an offline
unpacked-wheel import probe covers the same regression without it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


# Fallback only (no git checkout, e.g. an unpacked sdist): build droppings,
# VCS data and local environments that a private source copy must not carry
# over. A stale build/ or *.egg-info could leak old files into the wheel, and
# an in-repo .venv (which CLAUDE.md tells cloud sessions to create) is ~1.5 GB
# — copying it once per test filled /tmp with 7.8 GB (audit INFRA-3).
_COPY_IGNORE = shutil.ignore_patterns(
    ".git",
    "build",
    "dist",
    "*.egg-info",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.sif",
    ".plans",  # local symlink to the planning repo (CLAUDE.md)
    ".venv",
    "venv",
    "env",
    ".tox",
    "results",
    "site",
    "htmlcov",
)


def _git_source_files() -> list[str] | None:
    """Tracked + untracked-but-not-ignored files, or ``None`` without git."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    files = [f for f in out.decode("utf-8").split("\0") if f]
    return files or None


def _copy_source_tree(dest: Path) -> None:
    """Copy the project source into *dest*, leaving local junk behind.

    Uses ``git ls-files`` so ignored paths (``.venv``, ``results/``,
    ``site/``, build droppings …) are never copied; falls back to a
    ``copytree`` with an explicit ignore list outside a git checkout.
    """
    files = _git_source_files()
    if files is None:
        shutil.copytree(REPO_ROOT, dest, ignore=_COPY_IGNORE, symlinks=True)
        return
    for rel in files:
        src = REPO_ROOT / rel
        if not os.path.lexists(src):  # tracked but deleted in the work tree
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target, follow_symlinks=False)


def _require_offline_build_backend() -> None:
    """Skip unless this interpreter can build the wheel without a network.

    The build runs with ``--no-isolation`` so it never downloads the
    build-system requirements; that needs the frontend (``build``) and a
    setuptools that builds wheels on its own (>= 70.1 ships bdist_wheel, so
    the separate ``wheel`` package is not needed). The dev extra pins both.
    """
    pytest.importorskip("build")
    setuptools = pytest.importorskip("setuptools")
    major, minor = (int(x) for x in setuptools.__version__.split(".")[:2])
    if (major, minor) < (70, 1):
        pytest.importorskip(
            "wheel",
            reason=f"setuptools {setuptools.__version__} needs the 'wheel' "
            "package to build offline (or upgrade: pip install -e .[dev])",
        )


def _build_wheel(out_dir: Path) -> Path:
    """Build the project wheel into ``out_dir`` and return its path.

    setuptools stages every build in ``build/`` and ``*.egg-info`` inside the
    source tree, so parallel builds from the same checkout (pytest-xdist runs
    these tests on several workers) delete each other's staging directory
    mid-build ("No such file or directory: 'build/bdist.<plat>/wheel'").
    Building from a private copy of the tree gives each build its own. The
    copy keeps ``tests/``, so the "tests are not shipped" check still has
    something to exclude.

    ``--no-isolation`` uses this interpreter's setuptools instead of
    downloading one into a fresh build env, so the build works offline;
    ``--skip-dependency-check`` because ``wheel`` (listed in
    ``[build-system].requires``) is redundant with setuptools >= 70.1.
    """
    src = out_dir / "src"
    _copy_source_tree(src)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--skip-dependency-check",
            "--outdir",
            str(out_dir),
        ],
        cwd=src,
        capture_output=True,
        text=True,
    )
    assert (
        proc.returncode == 0
    ), f"wheel build failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one built wheel, found {wheels}"
    return wheels[0]


@pytest.fixture(scope="session")
def built_wheel(tmp_path_factory) -> Path:
    """One wheel per test session (per xdist worker), shared by every test."""
    _require_offline_build_backend()
    return _build_wheel(tmp_path_factory.mktemp("wheel"))


@pytest.fixture(scope="session")
def wheel_names(built_wheel) -> list[str]:
    with zipfile.ZipFile(built_wheel) as zf:
        return zf.namelist()


@pytest.mark.slow
class TestWheelContents:
    """Inspect the built wheel's file list directly (no install needed)."""

    def test_engines_subpackage_is_shipped(self, wheel_names):
        """AUDIT F20 — quantui/engines/* must be present in the wheel.

        This is the exact regression: the package list omitted
        quantui.engines, so a real (non-editable) install had no engine
        module at all.
        """
        names = wheel_names
        engine_files = [n for n in names if n.startswith("quantui/engines/")]
        assert engine_files, f"no quantui/engines/* files in wheel: {names}"
        assert "quantui/engines/__init__.py" in engine_files
        assert "quantui/engines/base.py" in engine_files
        assert "quantui/engines/pyscf_engine.py" in engine_files
        assert "quantui/engines/pyfock_engine.py" in engine_files

    def test_backends_subpackage_is_shipped(self, wheel_names):
        """Same check for quantui.backends, which the old explicit list did
        include — a regression guard so a future edit can't drop it either."""
        names = wheel_names
        assert "quantui/backends/__init__.py" in names
        assert "quantui/backends/worker.py" in names

    def test_tests_package_is_not_shipped(self, wheel_names):
        """``tests/`` has its own __init__.py (it's a real Python package),
        so package discovery must explicitly exclude it — otherwise the
        wheel ships the whole test suite as an importable top-level
        package, which is not what a distributed application wheel wants.
        """
        names = wheel_names
        test_files = [n for n in names if n.startswith("tests/")]
        assert not test_files, f"tests/* leaked into the wheel: {test_files}"

    def test_bundled_data_files_are_shipped(self, wheel_names):
        """Package-data (the offline molecule library + vendored 3Dmol.js)
        must survive alongside the package-discovery change — this is a
        pre-existing guarantee (not part of F20), reasserted here since
        this file is the one place a wheel is actually built and
        inspected."""
        names = wheel_names
        assert "quantui/data/library/library.sqlite" in names
        assert "quantui/data/js/3Dmol-min.js" in names


@pytest.mark.slow
class TestWheelUnpackedImport:
    """Offline counterpart of :class:`TestWheelIsolatedImport`.

    A pure-Python wheel installs by unpacking it onto ``sys.path``; do that
    and import ``quantui.engines`` with *only* the unpacked wheel on
    ``PYTHONPATH`` and the checkout out of the working directory. The
    dependencies come from this interpreter, so no network is needed, and
    the probe asserts the modules were loaded from the wheel — the editable
    finder (appended to ``sys.meta_path``) cannot mask a missing file.
    """

    def test_engines_importable_from_unpacked_wheel(self, tmp_path, built_wheel):
        with zipfile.ZipFile(built_wheel) as zf:
            wheel_meta = next(
                n for n in zf.namelist() if n.endswith(".dist-info/WHEEL")
            )
            assert "Root-Is-Purelib: true" in zf.read(wheel_meta).decode()
            site = tmp_path / "site-packages"
            zf.extractall(site)

        env = dict(os.environ, PYTHONPATH=str(site))
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import quantui, quantui.engines as e; "
                "from quantui.engines import base, pyscf_engine, pyfock_engine; "
                "print(quantui.__file__); print(e.__file__); "
                "print(pyscf_engine.__file__)",
            ],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"quantui.engines not importable from the unpacked wheel:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        for loaded in result.stdout.split():
            assert (
                Path(loaded).resolve().is_relative_to(site.resolve())
            ), f"{loaded} was not loaded from the wheel at {site}"


@pytest.mark.slow
class TestWheelIsolatedImport:
    """Install the built wheel into a throwaway venv and import from
    outside the checkout — the audit's own reproduction: "Importing
    quantui.engines from the extracted wheel, with the editable source
    finder removed from the probe process, raises ModuleNotFoundError."
    """

    @pytest.mark.network  # pip resolves the runtime dependencies from PyPI
    def test_engines_importable_from_installed_wheel(self, tmp_path, built_wheel):
        wheel_path = built_wheel

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


class TestSourceCopy:
    """Audit INFRA-3: the private build copy must not drag local junk along.

    The old ``copytree`` copied an in-repo ``.venv`` (~1.5 GB) once per test.
    """

    _JUNK = (".venv", "venv", "results", "site", "htmlcov", ".tox")

    def _fake_repo(self, root: Path) -> None:
        (root / "quantui").mkdir(parents=True)
        (root / "quantui" / "__init__.py").write_text("")
        (root / "pyproject.toml").write_text("")
        for junk in self._JUNK:
            (root / junk).mkdir()
            (root / junk / "big.bin").write_text("x")

    def _assert_clean_copy(self, dest: Path) -> None:
        assert (dest / "quantui" / "__init__.py").is_file()
        assert (dest / "pyproject.toml").is_file()
        leaked = [j for j in self._JUNK if (dest / j).exists()]
        assert not leaked, f"copied into the build tree: {leaked}"

    def test_fallback_copy_skips_local_envs_and_outputs(self, tmp_path, monkeypatch):
        import tests.test_packaging as tp

        repo = tmp_path / "repo"
        self._fake_repo(repo)
        monkeypatch.setattr(tp, "REPO_ROOT", repo)
        monkeypatch.setattr(tp, "_git_source_files", lambda: None)
        tp._copy_source_tree(tmp_path / "dest")
        self._assert_clean_copy(tmp_path / "dest")

    def test_git_copy_skips_ignored_paths(self, tmp_path, monkeypatch):
        import tests.test_packaging as tp

        if shutil.which("git") is None:
            pytest.skip("git not available")
        repo = tmp_path / "repo"
        self._fake_repo(repo)
        (repo / ".gitignore").write_text("".join(f"{j}/\n" for j in self._JUNK))
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        monkeypatch.setattr(tp, "REPO_ROOT", repo)
        tp._copy_source_tree(tmp_path / "dest")
        self._assert_clean_copy(tmp_path / "dest")
