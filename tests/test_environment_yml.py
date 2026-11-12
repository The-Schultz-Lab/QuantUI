"""Audit INFRA-10 — ``local-setup/environment.yml`` must not drift from
``pyproject.toml``.

The conda env had drifted: no ``pytest-xdist`` while pyproject's ``addopts``
carries ``-n=auto`` (so a bare ``pytest`` failed), ``mypy>=1.0.0`` instead of
CI's ``~=1.10.0``, ``voila`` uncapped against the ``[app]`` extra's ``<0.6``,
and no ``build`` / ``types-requests``. These tests compare the two files:

* every ``[dev]`` tool is present in the env file;
* for each package declared in both, the env file's range is no wider than
  pyproject's (every probed version the YAML accepts, pyproject accepts too).

Line-based YAML parsing (no PyYAML dependency); pyproject via ``tomllib`` /
``tomli``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.9/3.10
    tomllib = pytest.importorskip("tomli")

_ROOT = Path(__file__).resolve().parent.parent
_ENV_YML = _ROOT / "local-setup" / "environment.yml"
_PYPROJECT = _ROOT / "pyproject.toml"

# The env pins python=3.11; evaluate pyproject environment markers for that.
_MARKER_ENV = {"python_version": "3.11", "python_full_version": "3.11.0"}

# conda-forge name -> PyPI name, where they differ.
_CONDA_TO_PYPI = {"python-build": "build"}

# Extras whose packages the conda env is meant to provide.
_EXTRAS = ("pyscf", "ase", "ts", "app", "notebook", "dev")


def _norm(name: str) -> str:
    name = re.sub(r"[-_.]+", "-", name).lower()
    return _CONDA_TO_PYPI.get(name, name)


def _yaml_specs() -> Dict[str, SpecifierSet]:
    specs: Dict[str, SpecifierSet] = {}
    text = _ENV_YML.read_text(encoding="utf-8")
    for raw in text.split("\ndependencies:", 1)[1].splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line.startswith("- "):
            continue
        item = line[2:].strip()
        if not item or item.endswith(":") or item.startswith("-") or "@" in item:
            continue  # "pip:" header, "-e ..", git URLs, channel names
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", item)
        assert m, item
        name, spec = m.group(1), m.group(2).replace(" ", "")
        if spec.startswith("=") and not spec.startswith("=="):
            continue  # conda exact-ish pin such as python=3.11
        specs[_norm(name)] = SpecifierSet(spec)
    return specs


def _pyproject_reqs() -> List[Tuple[str, Requirement]]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    out = [("dependencies", Requirement(r)) for r in data["dependencies"]]
    for extra in _EXTRAS:
        out += [(extra, Requirement(r)) for r in data["optional-dependencies"][extra]]
    return [
        (group, r)
        for group, r in out
        if r.marker is None or r.marker.evaluate(_MARKER_ENV)
    ]


def _probes(*specsets: SpecifierSet) -> List[Version]:
    """Versions around every boundary mentioned in the given specifiers."""
    out = set()
    for ss in specsets:
        for spec in ss:
            v = Version(spec.version.replace(".*", ""))
            rel = list(v.release) + [0, 0]
            major, minor, micro = rel[0], rel[1], rel[2]
            out.update(
                {
                    v,
                    Version(f"{major}.{minor}.{micro + 1}"),
                    Version(f"{major}.{minor + 1}.0"),
                    Version(f"{major + 1}.0.0"),
                }
            )
            if micro:
                out.add(Version(f"{major}.{minor}.{micro - 1}"))
            if minor:
                out.add(Version(f"{major}.{minor - 1}.0"))
            if major:
                out.add(Version(f"{major - 1}.0.0"))
    return sorted(out)


def test_dev_tools_present_in_env_file() -> None:
    yml = _yaml_specs()
    missing = sorted(
        _norm(r.name)
        for group, r in _pyproject_reqs()
        if group == "dev" and _norm(r.name) not in yml
    )
    assert not missing, f"environment.yml lacks [dev] tools: {missing}"


def test_voila_and_mypy_pinned_like_pyproject() -> None:
    yml = _yaml_specs()
    assert not yml["voila"].contains("0.6.0")
    assert not yml["mypy"].contains("1.11.0")
    assert not yml["mypy"].contains("1.9.0")


def test_env_ranges_within_pyproject_ranges() -> None:
    yml = _yaml_specs()
    problems = []
    for group, req in _pyproject_reqs():
        name = _norm(req.name)
        if name not in yml or not req.specifier:
            continue
        y = yml[name]
        for v in _probes(y, req.specifier):
            if y.contains(v, prereleases=True) and not req.specifier.contains(
                v, prereleases=True
            ):
                problems.append(
                    f"{name}: env '{y}' allows {v}, [{group}] '{req.specifier}' does not"
                )
                break
    assert not problems, "\n".join(problems)
