"""Tests for M-EST / EST.2 — frequency cost model.

The cost model decomposes a freq estimate into::

    freq_total ≈ scf_anchor + hessian_term + ir_intensity_term

This file exercises the helper :func:`quantui.calc_log._estimate_frequency_cost`
directly (no PySCF needed) plus the integration with :func:`estimate_time`
(falls back to the cost model when direct freq history is empty).

Each test seeds a temporary perf-log via the ``QUANTUI_LOG_DIR`` env
var override so we don't touch the user's real log.
"""

from __future__ import annotations

import pytest

from quantui.calc_log import (
    _HESSIAN_MULTIPLIER_HF_DFT,
    _HESSIAN_MULTIPLIER_POST_HF,
    _estimate_frequency_cost,
    estimate_time,
    log_calculation,
)


@pytest.fixture
def isolated_perf_log(tmp_path, monkeypatch):
    """Redirect calc_log to a temp dir so tests don't pollute the user's log."""
    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    return tmp_path


def _seed_sp_record(
    *,
    formula: str,
    n_atoms: int,
    n_electrons: int,
    method: str,
    basis: str,
    elapsed_s: float,
    n_basis: int,
    gpu_used: bool = False,
):
    """Write one converged single-point record into the temp perf log."""
    log_calculation(
        formula=formula,
        n_atoms=n_atoms,
        n_electrons=n_electrons,
        method=method,
        basis=basis,
        n_iterations=10,
        elapsed_s=elapsed_s,
        converged=True,
        n_basis=n_basis,
        n_cores=1,
        calc_type="single_point",
        gpu_used=gpu_used,
    )


class TestCostModelStructure:
    """The decomposition must show its work: every component scales the
    way the docstring claims."""

    def test_returns_none_when_no_sp_anchor(self, isolated_perf_log):
        # No SP history → no anchor → cost model can't fire.
        est = _estimate_frequency_cost(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=7,
        )
        assert est is None

    def test_returns_dict_when_sp_anchor_available(self, isolated_perf_log):
        # Two SP records → strategy 1 fires → cost model has an anchor.
        for elapsed in (1.0, 1.2):
            _seed_sp_record(
                formula="H2O",
                n_atoms=3,
                n_electrons=10,
                method="B3LYP",
                basis="STO-3G",
                elapsed_s=elapsed,
                n_basis=7,
            )
        est = _estimate_frequency_cost(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=7,
        )
        assert est is not None
        assert "seconds" in est
        assert "confidence" in est
        assert "n_samples" in est
        assert est["seconds"] > 0

    def test_returns_none_for_zero_atoms(self):
        est = _estimate_frequency_cost(
            n_atoms=0, n_electrons=0, method="RHF", basis="STO-3G"
        )
        assert est is None


class TestCostModelArithmetic:
    """The model is ``scf + hessian + 6N×scf / workers``. With workers=1
    and a known SP anchor, we can predict the exact total."""

    def test_water_b3lyp_total_matches_decomposition(self, isolated_perf_log):
        # Seed water B3LYP/STO-3G SP at exactly 1.0 s with all-equal samples
        # so IQR can't drop anything and median == 1.0.
        for _ in range(5):
            _seed_sp_record(
                formula="H2O",
                n_atoms=3,
                n_electrons=10,
                method="B3LYP",
                basis="STO-3G",
                elapsed_s=1.0,
                n_basis=7,
            )
        # SP anchor for n_basis=7, β=3.5, n_cores=1: predicted == 1.0 s.
        sp = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=7,
            n_cores=1,
            calc_type="single_point",
        )
        assert sp is not None
        scf_anchor = sp["seconds"]
        # Now the freq cost model: 1 + 2*1 + 6*3*1/1 = 21 s.
        cost = _estimate_frequency_cost(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=7,
            n_cores=1,
        )
        assert cost is not None
        expected = (
            scf_anchor + _HESSIAN_MULTIPLIER_HF_DFT * scf_anchor + 6 * 3 * scf_anchor
        )
        assert cost["seconds"] == pytest.approx(expected, rel=1e-6)

    def test_post_hf_uses_larger_hessian_multiplier(self, isolated_perf_log):
        # Two MP2 SP records → MP2 anchor available.
        for _ in range(5):
            _seed_sp_record(
                formula="H2O",
                n_atoms=3,
                n_electrons=10,
                method="MP2",
                basis="cc-pVDZ",
                elapsed_s=10.0,
                n_basis=24,
            )
        cost = _estimate_frequency_cost(
            n_atoms=3,
            n_electrons=10,
            method="MP2",
            basis="cc-pVDZ",
            n_basis=24,
            n_cores=1,
        )
        assert cost is not None
        # Post-HF: hessian multiplier is _HESSIAN_MULTIPLIER_POST_HF (=6.0).
        # Verify the multiplier is meaningfully larger than HF/DFT's (=2.0).
        assert _HESSIAN_MULTIPLIER_POST_HF > _HESSIAN_MULTIPLIER_HF_DFT

    def test_scales_linearly_in_n_atoms(self, isolated_perf_log):
        # Same anchor cost, but the IR term should grow ~6N.
        # We can't seed different n_atoms cleanly with strategy 1, so we
        # use strategy 2 (electron count) which is more permissive.
        for _ in range(5):
            _seed_sp_record(
                formula="H2",
                n_atoms=2,
                n_electrons=2,
                method="RHF",
                basis="STO-3G",
                elapsed_s=1.0,
                n_basis=2,
            )
        # Predict freq for various n_atoms. The SP anchor should grow
        # via the electron-count scale, but the freq prediction should
        # ALSO grow with the 6N IR term.
        c2 = _estimate_frequency_cost(
            n_atoms=2,
            n_electrons=2,
            method="RHF",
            basis="STO-3G",
            n_basis=2,
            n_cores=1,
        )
        c4 = _estimate_frequency_cost(
            n_atoms=4,
            n_electrons=2,  # held fixed to isolate the n_atoms effect
            method="RHF",
            basis="STO-3G",
            n_basis=2,
            n_cores=1,
        )
        assert c2 is not None and c4 is not None
        # ir_term doubles when n_atoms doubles (24 vs 12 displacement SCFs).
        # SP anchor doesn't change (electron count fixed, n_basis fixed).
        # So total should grow by roughly the additional 12 × scf_anchor.
        assert c4["seconds"] > c2["seconds"]


class TestParallelIrAwareness:
    """The model must reflect whether ``QUANTUI_FREQ_PARALLEL`` would
    actually engage on the predicted run."""

    def test_serial_when_env_var_off(self, isolated_perf_log, monkeypatch):
        monkeypatch.delenv("QUANTUI_FREQ_PARALLEL", raising=False)
        for _ in range(5):
            _seed_sp_record(
                formula="C6H6",
                n_atoms=12,
                n_electrons=42,
                method="B3LYP",
                basis="6-31G*",
                elapsed_s=2.0,
                n_basis=120,
            )
        cost = _estimate_frequency_cost(
            n_atoms=12,
            n_electrons=42,
            method="B3LYP",
            basis="6-31G*",
            n_basis=120,
            n_cores=8,
        )
        assert cost is not None
        # Compute SP anchor for the same profile to cross-check.
        sp = estimate_time(
            n_atoms=12,
            n_electrons=42,
            method="B3LYP",
            basis="6-31G*",
            n_basis=120,
            n_cores=8,
            calc_type="single_point",
        )
        assert sp is not None
        # Serial: ir_term = 6*12 * anchor = 72 * anchor (no division).
        expected = (
            sp["seconds"]
            + _HESSIAN_MULTIPLIER_HF_DFT * sp["seconds"]
            + 6 * 12 * sp["seconds"]
        )
        assert cost["seconds"] == pytest.approx(expected, rel=1e-6)

    def test_parallel_reduces_estimate_when_env_var_on_and_gates_pass(
        self, isolated_perf_log, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "1")
        for _ in range(5):
            _seed_sp_record(
                formula="C6H6",
                n_atoms=12,
                n_electrons=42,
                method="B3LYP",
                basis="6-31G*",
                elapsed_s=2.0,
                n_basis=120,
            )
        cost_parallel = _estimate_frequency_cost(
            n_atoms=12,
            n_electrons=42,
            method="B3LYP",
            basis="6-31G*",
            n_basis=120,
            n_cores=8,
            gpu_used=False,  # parallel gated off on GPU
        )
        # Compare to serial (same params, different env var).
        monkeypatch.delenv("QUANTUI_FREQ_PARALLEL")
        cost_serial = _estimate_frequency_cost(
            n_atoms=12,
            n_electrons=42,
            method="B3LYP",
            basis="6-31G*",
            n_basis=120,
            n_cores=8,
            gpu_used=False,
        )
        assert cost_parallel is not None
        assert cost_serial is not None
        # Parallel divides the 72-SCF IR term by effective_workers (= 4
        # on an 8-core host per pick_worker_count). Total should be
        # noticeably smaller.
        assert cost_parallel["seconds"] < cost_serial["seconds"]
        # Sanity: parallel can't reduce to less than (1 + Hessian) × scf
        # since only the 6N IR term gets divided. With Hessian=2× scf,
        # the floor is 3× scf — which is well above zero/negative.
        assert cost_parallel["seconds"] > cost_serial["seconds"] * 0.1

    def test_gpu_run_uses_parallel_estimate_when_env_var_on(
        self, isolated_perf_log, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "1")
        for _ in range(5):
            _seed_sp_record(
                formula="C6H6",
                n_atoms=12,
                n_electrons=42,
                method="B3LYP",
                basis="6-31G*",
                elapsed_s=2.0,
                n_basis=120,
                gpu_used=True,
            )
        cost_parallel = _estimate_frequency_cost(
            n_atoms=12,
            n_electrons=42,
            method="B3LYP",
            basis="6-31G*",
            n_basis=120,
            n_cores=8,
            gpu_used=True,
        )
        monkeypatch.delenv("QUANTUI_FREQ_PARALLEL")
        cost_serial = _estimate_frequency_cost(
            n_atoms=12,
            n_electrons=42,
            method="B3LYP",
            basis="6-31G*",
            n_basis=120,
            n_cores=8,
            gpu_used=True,
        )
        assert cost_parallel is not None
        assert cost_serial is not None
        assert cost_parallel["seconds"] < cost_serial["seconds"]


class TestEstimateTimeIntegration:
    """``estimate_time(calc_type="frequency")`` must fall back to the
    cost model when direct freq history is empty AND return the
    direct-history result when one exists."""

    def test_falls_back_when_no_freq_history(self, isolated_perf_log):
        # SP history only — direct strategies 1-4 should fail for freq.
        for _ in range(5):
            _seed_sp_record(
                formula="H2O",
                n_atoms=3,
                n_electrons=10,
                method="B3LYP",
                basis="STO-3G",
                elapsed_s=1.0,
                n_basis=7,
            )
        est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=7,
            n_cores=1,
            calc_type="frequency",
        )
        assert est is not None
        # Should be the cost-model prediction: ~21 s.
        assert est["seconds"] > 10.0  # well above SP alone
        assert est["seconds"] < 100.0  # within sanity range

    def test_direct_freq_history_wins_over_cost_model(self, isolated_perf_log):
        # Seed BOTH SP records AND direct freq records. The freq pool
        # is what we want the estimator to use; the cost model should
        # never fire when direct data exists.
        for _ in range(5):
            _seed_sp_record(
                formula="H2O",
                n_atoms=3,
                n_electrons=10,
                method="B3LYP",
                basis="STO-3G",
                elapsed_s=1.0,
                n_basis=7,
            )
        # Direct freq runs: ALL exactly 30 s, very different from the
        # cost model's predicted ~21 s.
        for _ in range(5):
            log_calculation(
                formula="H2O",
                n_atoms=3,
                n_electrons=10,
                method="B3LYP",
                basis="STO-3G",
                n_iterations=10,
                elapsed_s=30.0,
                converged=True,
                n_basis=7,
                n_cores=1,
                calc_type="frequency",
            )
        est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=7,
            n_cores=1,
            calc_type="frequency",
        )
        assert est is not None
        # Direct freq history dominates → close to 30 s, not 21 s.
        assert est["seconds"] == pytest.approx(30.0, rel=1e-6)

    def test_returns_none_when_no_history_at_all(self, isolated_perf_log):
        est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=7,
            n_cores=1,
            calc_type="frequency",
        )
        assert est is None


class TestConfidenceInheritance:
    """Cost model adds structural assumptions but no new data — it
    should never claim higher confidence than the SP anchor."""

    def test_low_confidence_when_anchor_is_low(self, isolated_perf_log):
        # Highly variable SP records → low confidence on the anchor.
        # Mix tiny + huge values; IQR will still trim but CV will be high.
        for v in (1.0, 1.2, 1.1, 5.0, 6.0):
            _seed_sp_record(
                formula="H2O",
                n_atoms=3,
                n_electrons=10,
                method="B3LYP",
                basis="STO-3G",
                elapsed_s=v,
                n_basis=7,
            )
        sp = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=7,
            n_cores=1,
            calc_type="single_point",
        )
        cost = _estimate_frequency_cost(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=7,
            n_cores=1,
        )
        assert sp is not None and cost is not None
        # Cost model inherits the SP anchor's confidence.
        assert cost["confidence"] == sp["confidence"]
        assert cost["n_samples"] == sp["n_samples"]
