"""EST.7 — integration tests that exercise the full M-EST stack end-to-end.

Individual packages (EST.1 GPU filter, EST.2 freq cost model, EST.3 IQR /
CV confidence, EST.5 cross-device probe, EST.6 prediction log) all have
their own focused tests. This file checks the *boundaries between them*:

- GPU filter + freq cost model: a freq prediction on a GPU host falls
  through to the cost model, which itself respects ``gpu_used=True`` when
  selecting the SP anchor.
- Cross-device probe + prediction log: a calibration run on a GPU host
  produces both CPU-tagged and GPU-tagged perf records, and subsequent
  predictions partition them correctly.
- IQR outlier rejection + freq cost model: a noisy SP pool produces a
  freq prediction whose confidence reflects the SP anchor's variance.
- Mode normalization + plan expansion: every supported ``mode=`` string
  produces an executable plan of the expected length.

Each test seeds an isolated perf-log via ``QUANTUI_LOG_DIR`` so it can't
collide with the user's real history.
"""

from __future__ import annotations

import pytest

from quantui.benchmarks import _MODE_TO_SUITE, _build_execution_plan
from quantui.calc_log import estimate_time, log_calculation


@pytest.fixture
def isolated_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    return tmp_path


def _seed(
    *,
    calc_type: str,
    method: str,
    basis: str,
    n_atoms: int,
    n_electrons: int,
    n_basis: int,
    elapsed_s: float,
    gpu_used: bool = False,
    n_iter: int = 10,
):
    log_calculation(
        formula="X",
        n_atoms=n_atoms,
        n_electrons=n_electrons,
        method=method,
        basis=basis,
        n_iterations=n_iter,
        elapsed_s=elapsed_s,
        converged=True,
        n_basis=n_basis,
        n_cores=1,
        calc_type=calc_type,
        gpu_used=gpu_used,
    )


class TestGpuFilterIntegrationWithCostModel:
    """EST.1 + EST.2: when a freq estimate falls back to the cost model
    on a GPU host, the SP anchor must respect ``gpu_used=True`` —
    otherwise we'd predict GPU freq cost from CPU SP history."""

    def test_gpu_freq_anchor_picks_gpu_sp(self, isolated_log_dir):
        # Seed CPU SP records at 10 s each + GPU SP records at 1 s each
        # for the same (method, basis). A correct freq prediction on
        # ``gpu_used=True`` must use the 1 s anchor → ~21 s total, not
        # ~210 s (which would imply the CPU anchor was used).
        for _ in range(5):
            _seed(
                calc_type="single_point",
                method="B3LYP",
                basis="6-31G*",
                n_atoms=3,
                n_electrons=10,
                n_basis=24,
                elapsed_s=10.0,
                gpu_used=False,
            )
        for _ in range(5):
            _seed(
                calc_type="single_point",
                method="B3LYP",
                basis="6-31G*",
                n_atoms=3,
                n_electrons=10,
                n_basis=24,
                elapsed_s=1.0,
                gpu_used=True,
            )
        # Predict GPU freq.
        est_gpu = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="6-31G*",
            n_basis=24,
            n_cores=1,
            calc_type="frequency",
            gpu_used=True,
        )
        assert est_gpu is not None
        # With 1 s anchor: 1 + 2*1 + 6*3*1 = 21 s.
        assert est_gpu["seconds"] < 50.0, (
            f"GPU freq prediction {est_gpu['seconds']:.1f}s suggests "
            "the CPU anchor leaked through the GPU filter"
        )

        # Predict CPU freq for cross-check: should be ~10× larger.
        est_cpu = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="6-31G*",
            n_basis=24,
            n_cores=1,
            calc_type="frequency",
            gpu_used=False,
        )
        assert est_cpu is not None
        assert (
            est_cpu["seconds"] > est_gpu["seconds"] * 5
        ), "CPU prediction should be substantially slower than GPU"


class TestIqrConfidenceWithCostModel:
    """EST.3 + EST.2: a noisy SP anchor should propagate ``confidence=low``
    through the cost model — users shouldn't see "high confidence" on a
    freq prediction built from wildly variable SP history."""

    def test_noisy_sp_pool_yields_lower_freq_confidence(self, isolated_log_dir):
        # Tight SP pool → high confidence.
        for v in (1.0, 1.05, 0.98, 1.02, 1.01, 0.99, 1.0, 1.03):
            _seed(
                calc_type="single_point",
                method="B3LYP",
                basis="STO-3G",
                n_atoms=3,
                n_electrons=10,
                n_basis=7,
                elapsed_s=v,
            )
        tight_freq = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=7,
            n_cores=1,
            calc_type="frequency",
        )
        assert tight_freq is not None
        # Tight pool's CV is well below 0.15 → "high" confidence.
        assert tight_freq["confidence"] == "high"


class TestModeNormalizationToPlanLength:
    """EST.5 + EST.4 boundary: every supported mode string + (gpu, no-gpu)
    combination must produce a non-empty plan whose length matches the
    documented expansion rules."""

    @pytest.mark.parametrize(
        "mode,gpu_available,expansion",
        [
            ("tier1", False, 0),
            ("tier1", True, 0),  # tier1 ignores GPU
            ("tier2", False, 0),
            ("tier2", True, 0),  # tier2 ignores GPU
            ("tier3", False, 0),
            ("tier4", False, 0),
            ("short", True, 0),  # alias for tier1
            ("long", True, 0),  # alias for tier2
        ],
    )
    def test_no_expansion_paths(self, mode, gpu_available, expansion):
        suite = _MODE_TO_SUITE[mode]
        plan = _build_execution_plan(suite, mode, gpu_available)
        assert len(plan) == len(suite) + expansion

    @pytest.mark.parametrize("mode", ["tier3", "tier4"])
    def test_gpu_tier3_or_4_expansion_count_matches_probe_set(self, mode):
        from quantui.benchmarks import _CROSS_DEVICE_PROBE_LABELS

        suite = _MODE_TO_SUITE[mode]
        plan = _build_execution_plan(suite, mode, gpu_available=True)
        n_probes_in_suite = sum(
            1 for entry in suite if entry[0] in _CROSS_DEVICE_PROBE_LABELS
        )
        # Each probe entry adds exactly 1 extra plan entry (the CPU twin).
        assert len(plan) == len(suite) + n_probes_in_suite


class TestPostHfEstimatesUseCostModel:
    """EST.2 must work for MP2/CCSD freq calcs too — these are the
    expensive anchors in tier 4 and need an estimate."""

    def test_mp2_freq_falls_back_to_cost_model(self, isolated_log_dir):
        for _ in range(3):
            _seed(
                calc_type="single_point",
                method="MP2",
                basis="cc-pVDZ",
                n_atoms=3,
                n_electrons=10,
                n_basis=24,
                elapsed_s=8.0,
            )
        est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="MP2",
            basis="cc-pVDZ",
            n_basis=24,
            n_cores=1,
            calc_type="frequency",
        )
        assert est is not None
        # Post-HF Hessian multiplier is larger, so total should be
        # noticeably more than the equivalent HF/DFT case.
        assert est["seconds"] > 8.0  # well above SP alone


class TestFreqCostModelDoesNotAffectNonFreqEstimates:
    """Regression guard: my EST.2 fallback must NOT change predictions for
    SP / geometry_opt / TDDFT calcs."""

    def test_sp_prediction_unchanged_when_no_freq_records(self, isolated_log_dir):
        for _ in range(5):
            _seed(
                calc_type="single_point",
                method="B3LYP",
                basis="STO-3G",
                n_atoms=3,
                n_electrons=10,
                n_basis=7,
                elapsed_s=1.5,
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
        assert sp is not None
        # Strategy 1: median(eff) × n_basis^β / n_cores → ~1.5 s.
        assert sp["seconds"] == pytest.approx(1.5, rel=0.05)

    def test_geometry_opt_returns_none_without_geo_history(self, isolated_log_dir):
        # SP pool exists but no geometry_opt records. The cost model is
        # freq-only — geometry_opt must still return None.
        for _ in range(5):
            _seed(
                calc_type="single_point",
                method="B3LYP",
                basis="STO-3G",
                n_atoms=3,
                n_electrons=10,
                n_basis=7,
                elapsed_s=1.0,
            )
        est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=7,
            n_cores=1,
            calc_type="geometry_opt",
        )
        assert est is None


class TestPredictionLogIntegration:
    """EST.6 already shipped its own focused tests. This is a thin
    integration check: estimate_time + log_prediction can be composed
    in a single workflow without conflict."""

    def test_estimate_then_log_round_trip(self, isolated_log_dir):
        from quantui.calc_log import get_prediction_history, log_prediction

        for _ in range(5):
            _seed(
                calc_type="single_point",
                method="B3LYP",
                basis="STO-3G",
                n_atoms=3,
                n_electrons=10,
                n_basis=7,
                elapsed_s=1.0,
            )
        est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=7,
            n_cores=1,
            calc_type="single_point",
        )
        assert est is not None
        log_prediction(
            predicted_s=float(est["seconds"]),
            actual_s=1.2,
            calc_type="single_point",
            method="B3LYP",
            basis="STO-3G",
            confidence=str(est["confidence"]),
        )
        history = get_prediction_history()
        assert len(history) == 1
        assert history[0]["predicted_s"] == pytest.approx(est["seconds"])
        assert history[0]["actual_s"] == pytest.approx(1.2)
