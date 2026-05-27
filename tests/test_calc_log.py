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
    """Legacy untyped records must not enter the freq pool as *direct* matches.

    Before M-EST / EST.2 (session 55) this asserted ``est_freq is None`` —
    a strict "no freq records → no freq estimate" rule. EST.2 added a
    structured cost-model fallback that intentionally reuses the SP
    history (where legacy untyped records DO count) to derive a freq
    estimate when no direct freq records exist. So the contract today
    is two-fold:

    1. Legacy records still don't count as frequency-typed (strategies
       1-4 produce no direct prediction).
    2. The cost-model fallback DOES fire — producing a structured
       SCF-anchor + Hessian + 6N IR estimate — and its value is much
       larger than the underlying SP time (otherwise we know the
       cost-model decomposition collapsed to just the SP anchor).
    """
    import quantui.calc_log as clog

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

    # EST.2 fallback fires: not None, and noticeably larger than the
    # bare SP median (~12 s) thanks to the +Hessian + 6×n_atoms × SP term.
    assert est_freq is not None
    assert est_freq["seconds"] > 100.0, (
        f"Expected freq estimate > 100 s (SP ~12 s × ~21 cost-model multiplier "
        f"for 4 atoms), got {est_freq['seconds']:.1f} s — suggests the cost "
        "model isn't firing on legacy SP records"
    )
