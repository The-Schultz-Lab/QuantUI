"""Tests for quantui.calc_log estimation behavior."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def isolated_log_dir(tmp_path, monkeypatch):
    """Point QUANTUI_LOG_DIR at a fresh temp directory for every test."""
    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    import quantui.calc_log as clog

    importlib.reload(clog)
    yield tmp_path


def test_estimate_time_scopes_by_calc_type(isolated_log_dir):
    import quantui.calc_log as clog

    # Fast single-point history
    for elapsed in (12.0, 14.0, 16.0):
        clog.log_calculation(
            formula="CH2O",
            n_atoms=4,
            n_electrons=16,
            method="B3LYP",
            basis="6-31G",
            n_iterations=12,
            elapsed_s=elapsed,
            converged=True,
            n_basis=44,
            n_cores=1,
            calc_type="single_point",
        )

    # Slow frequency history
    for elapsed in (118.0, 122.0):
        clog.log_calculation(
            formula="CH2O",
            n_atoms=4,
            n_electrons=16,
            method="B3LYP",
            basis="6-31G",
            n_iterations=12,
            elapsed_s=elapsed,
            converged=True,
            n_basis=44,
            n_cores=1,
            calc_type="frequency",
        )

    est_freq = clog.estimate_time(
        n_atoms=4,
        n_electrons=16,
        method="B3LYP",
        basis="6-31G",
        n_basis=44,
        calc_type="frequency",
    )
    est_sp = clog.estimate_time(
        n_atoms=4,
        n_electrons=16,
        method="B3LYP",
        basis="6-31G",
        n_basis=44,
        calc_type="single_point",
    )

    assert est_freq is not None
    assert est_sp is not None
    assert est_freq["n_samples"] == 2
    assert est_freq["seconds"] > 80
    assert est_sp["seconds"] < 30


def test_estimate_time_non_single_point_ignores_legacy_untyped_records(
    isolated_log_dir,
):
    import quantui.calc_log as clog

    # Legacy records with no calc_type should not be used for frequency estimates.
    for elapsed in (10.0, 12.0, 15.0):
        clog.log_calculation(
            formula="CH2O",
            n_atoms=4,
            n_electrons=16,
            method="B3LYP",
            basis="6-31G",
            n_iterations=12,
            elapsed_s=elapsed,
            converged=True,
            n_basis=44,
            n_cores=1,
        )

    est_freq = clog.estimate_time(
        n_atoms=4,
        n_electrons=16,
        method="B3LYP",
        basis="6-31G",
        n_basis=44,
        calc_type="frequency",
    )

    assert est_freq is None
