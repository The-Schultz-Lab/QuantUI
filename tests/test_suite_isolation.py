"""The test suite must never touch the developer's real ``~/.quantui``.

Audit INFRA-1 (2026-11-10): after a suite run ``~/.quantui`` held a
``calibration.json`` pointing at deleted temp dirs and test checkpoints the
app would offer to resume. ``tests/conftest.py`` now redirects every state
location for the whole session; these tests pin that, and the
``QUANTUI_CALIBRATION_PATH`` override it relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REAL_HOME = Path.home() / ".quantui"


def _outside_real_home(path: Path) -> bool:
    path = Path(path).expanduser().resolve()
    try:
        path.relative_to(REAL_HOME.resolve())
    except ValueError:
        return True
    return False


@pytest.mark.parametrize(
    "where",
    [
        "checkpoint_root",
        "default_jobs_root",
        "default_staging_root",
        "quantui_home",
        "calibration_json",
    ],
)
def test_state_locations_are_isolated_for_the_suite(where):
    from quantui import benchmarks, checkpoint
    from quantui.app_launcher import quantui_home
    from quantui.backends import cluster_config

    resolve = {
        "checkpoint_root": checkpoint.checkpoint_root,
        "default_jobs_root": cluster_config.default_jobs_root,
        "default_staging_root": cluster_config.default_staging_root,
        "quantui_home": quantui_home,
        "calibration_json": benchmarks._calibration_json_path,
    }[where]
    assert _outside_real_home(resolve()), f"{where} -> {resolve()}"


def test_staging_isolation_does_not_lock_the_job_root_field():
    # Isolating via QUANTUI_STAGING_DIR would flip the UI into its
    # "pinned by environment" mode for every test; conftest must not.
    from quantui.backends import cluster_config

    assert not cluster_config.staging_root_env_configured()


class TestCalibrationPathOverride:
    def test_env_var_overrides_the_default(self, tmp_path, monkeypatch):
        from quantui import benchmarks

        target = tmp_path / "cal" / "calibration.json"
        monkeypatch.setenv("QUANTUI_CALIBRATION_PATH", str(target))
        assert benchmarks._calibration_json_path() == target

    def test_default_lives_under_the_quantui_home(self, monkeypatch):
        from quantui import benchmarks

        monkeypatch.delenv("QUANTUI_CALIBRATION_PATH", raising=False)
        assert (
            benchmarks._calibration_json_path()
            == Path.home() / ".quantui" / "calibration.json"
        )

    def test_save_and_load_round_trip_through_the_override(self, tmp_path, monkeypatch):
        from quantui.benchmarks import (
            CalibrationResult,
            _save_calibration_json,
            load_last_calibration,
        )

        target = tmp_path / "calibration.json"
        monkeypatch.setenv("QUANTUI_CALIBRATION_PATH", str(target))
        result = CalibrationResult(timestamp="2026-11-12T00:00:00+00:00", mode="tier1")
        _save_calibration_json(result, tmp_path / "run.log")

        assert json.loads(target.read_text(encoding="utf-8"))["mode"] == "tier1"
        loaded = load_last_calibration()
        assert loaded is not None and loaded["log_path"] == str(tmp_path / "run.log")
