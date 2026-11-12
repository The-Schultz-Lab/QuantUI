"""Audit INFRA-16 — the vendored 3Dmol.js must track py3Dmol's target build.

QuantUI serves a vendored 3Dmol.js (``quantui/data/js/3Dmol-min.js``,
``viz_assets.THREEDMOL_VERSION``) in place of the CDN build py3Dmol's
``view(js=...)`` default points at. The JS py3Dmol generates calls that
build's API, so the two must stay compatible. Nothing checked this: the
bundle was 2.5.4 while the installed py3Dmol 2.5.5 targets 3Dmol.js 2.5.5.

Patch skew is tolerated; a minor/major mismatch fails here, and pyproject's
py3Dmol cap must stop below the vendored bundle's next minor so an install
cannot silently pull a py3Dmol targeting a newer 3Dmol.js API.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from quantui import viz_assets

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.9/3.10
    tomllib = pytest.importorskip("tomli")

_ROOT = Path(__file__).resolve().parent.parent


def _minor(v: str) -> tuple:
    return Version(v).release[:2]


def test_py3dmol_default_targets_vendored_minor() -> None:
    py3Dmol = pytest.importorskip("py3Dmol")
    default_js = inspect.signature(py3Dmol.view.__init__).parameters["js"].default
    m = re.search(r"3dmol@([0-9][0-9.]*)/", default_js)
    assert m, f"unexpected py3Dmol default js URL: {default_js!r}"
    assert _minor(m.group(1)) == _minor(viz_assets.THREEDMOL_VERSION), (
        f"py3Dmol targets 3Dmol.js {m.group(1)} but QuantUI vendors "
        f"{viz_assets.THREEDMOL_VERSION}: re-vendor data/js/3Dmol-min.js"
    )


def test_pyproject_caps_py3dmol_below_next_vendored_minor() -> None:
    deps = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["dependencies"]
    (req,) = [Requirement(d) for d in deps if Requirement(d).name == "py3Dmol"]
    major, minor = _minor(viz_assets.THREEDMOL_VERSION)
    assert req.specifier.contains(viz_assets.THREEDMOL_VERSION)
    assert not req.specifier.contains(f"{major}.{minor + 1}.0")


def test_cdn_url_constant_matches_vendored_version() -> None:
    assert f"3dmol@{viz_assets.THREEDMOL_VERSION}/" in viz_assets.CDN_URL
