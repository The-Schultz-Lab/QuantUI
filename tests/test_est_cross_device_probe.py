"""Tests for M-EST / EST.5 — cross-device CPU/GPU probe in tier 3+4.

The goal of EST.5 is that a single tier-4 calibration run on a GPU host
populates the analytics dashboard's GPU-vs-CPU speedup table without
asking users to manually re-run the suite under ``QUANTUI_DISABLE_GPU=1``.
The mechanism is to expand the execution plan so a SMALL representative
subset of entries appears twice — once forced-CPU, once GPU — and the
worker process sets ``QUANTUI_DISABLE_GPU=1`` before any PySCF /
gpu4pyscf import on the CPU variant.

These tests are platform-independent: they exercise ``_build_execution_plan``
directly (a pure function) plus a smoke test on ``_calibration_worker``
to confirm the env-var toggle happens before quantui imports. The actual
GPU-vs-CPU wall-clock validation lives in manual WSL testing (EST.7).
"""

from __future__ import annotations

import os

import pytest

from quantui.benchmarks import (
    _CROSS_DEVICE_PROBE_LABELS,
    _MODE_TO_SUITE,
    _build_execution_plan,
)


class TestProbeLabelsExist:
    """The probe labels must actually match entries in the tier 3/4 suites
    — a typo here would silently disable the cross-device probe with no
    test failure if we only checked the expansion machinery."""

    def test_all_probe_labels_present_in_tier3(self):
        labels_in_suite = {entry[0] for entry in _MODE_TO_SUITE["tier3"]}
        missing = _CROSS_DEVICE_PROBE_LABELS - labels_in_suite
        assert not missing, (
            f"Probe labels not found in tier3 suite: {missing}. "
            f"Either add them to the suite or fix the labels."
        )

    def test_all_probe_labels_present_in_tier4(self):
        labels_in_suite = {entry[0] for entry in _MODE_TO_SUITE["tier4"]}
        missing = _CROSS_DEVICE_PROBE_LABELS - labels_in_suite
        assert not missing, f"Probe labels not found in tier4 suite: {missing}"

    def test_probe_set_is_short(self):
        # Doubling the whole suite would blow the time budget — keep this
        # set small (≤5) so cross-device pairs cost ~5-10 min, not 30+.
        assert 1 <= len(_CROSS_DEVICE_PROBE_LABELS) <= 5


class TestNoGpuHostBehavior:
    """On a CPU-only machine the plan must NEVER expand — cross-device
    pairs are meaningless without a GPU to compare against."""

    @pytest.mark.parametrize("mode", ["tier1", "tier2", "tier3", "tier4"])
    def test_no_expansion_on_cpu_only(self, mode):
        suite = _MODE_TO_SUITE[mode]
        plan = _build_execution_plan(suite, mode, gpu_available=False)
        assert len(plan) == len(suite)

    def test_no_force_cpu_flags_on_cpu_only(self):
        plan = _build_execution_plan(
            _MODE_TO_SUITE["tier4"], "tier4", gpu_available=False
        )
        assert all(p["force_cpu"] is False for p in plan)

    def test_no_label_suffixes_on_cpu_only(self):
        plan = _build_execution_plan(
            _MODE_TO_SUITE["tier4"], "tier4", gpu_available=False
        )
        for p in plan:
            assert "[GPU]" not in p["label"]
            assert "[CPU]" not in p["label"]


class TestGpuHostTier1And2:
    """Tier 1/2 are pure-SP smoke tests. Even on a GPU host they should
    NOT expand — the cross-device data lives in tier 3+4 only because
    those are the tiers users actually run when they want speedup data."""

    @pytest.mark.parametrize("mode", ["tier1", "tier2"])
    def test_no_expansion_for_tier1_or_2(self, mode):
        suite = _MODE_TO_SUITE[mode]
        plan = _build_execution_plan(suite, mode, gpu_available=True)
        assert len(plan) == len(suite)

    def test_legacy_aliases_no_expansion(self):
        # ``"short"`` / ``"long"`` are tier1/tier2 aliases — same rule.
        for legacy in ("short", "long"):
            suite = _MODE_TO_SUITE[legacy]
            plan = _build_execution_plan(suite, legacy, gpu_available=True)
            assert len(plan) == len(suite)


class TestGpuHostTier3And4Expansion:
    """The whole point of EST.5: GPU host + tier3/4 must produce CPU+GPU
    pairs for each probe label."""

    @pytest.mark.parametrize("mode", ["tier3", "tier4"])
    def test_expansion_increases_plan_size(self, mode):
        suite = _MODE_TO_SUITE[mode]
        plan = _build_execution_plan(suite, mode, gpu_available=True)
        n_probe_in_suite = sum(
            1 for entry in suite if entry[0] in _CROSS_DEVICE_PROBE_LABELS
        )
        # Each probe entry produces 2 plan entries (original count + n_probe extras).
        assert len(plan) == len(suite) + n_probe_in_suite

    @pytest.mark.parametrize("mode", ["tier3", "tier4"])
    def test_each_probe_label_appears_twice(self, mode):
        suite = _MODE_TO_SUITE[mode]
        plan = _build_execution_plan(suite, mode, gpu_available=True)
        for probe_label in _CROSS_DEVICE_PROBE_LABELS:
            # Probe entries are renamed to include [GPU] / [CPU] suffix.
            gpu_count = sum(1 for p in plan if p["label"] == f"{probe_label}  [GPU]")
            cpu_count = sum(1 for p in plan if p["label"] == f"{probe_label}  [CPU]")
            assert gpu_count == 1, f"Expected exactly 1 GPU variant of {probe_label}"
            assert cpu_count == 1, f"Expected exactly 1 CPU variant of {probe_label}"

    def test_cpu_variants_carry_force_cpu_flag(self):
        plan = _build_execution_plan(
            _MODE_TO_SUITE["tier4"], "tier4", gpu_available=True
        )
        cpu_entries = [p for p in plan if "[CPU]" in p["label"]]
        gpu_entries = [p for p in plan if "[GPU]" in p["label"]]
        assert cpu_entries, "Expected at least one CPU-tagged plan entry"
        assert gpu_entries, "Expected at least one GPU-tagged plan entry"
        assert all(p["force_cpu"] is True for p in cpu_entries)
        assert all(p["force_cpu"] is False for p in gpu_entries)

    def test_non_probe_entries_keep_original_label_and_no_force_cpu(self):
        suite = _MODE_TO_SUITE["tier4"]
        plan = _build_execution_plan(suite, "tier4", gpu_available=True)
        non_probe_originals = [
            entry[0] for entry in suite if entry[0] not in _CROSS_DEVICE_PROBE_LABELS
        ]
        for label in non_probe_originals:
            matching = [p for p in plan if p["label"] == label]
            assert len(matching) == 1, (
                f"Non-probe entry {label!r} should appear exactly once "
                f"(unchanged), got {len(matching)}"
            )
            assert matching[0]["force_cpu"] is False

    def test_plan_entries_preserve_calc_type(self):
        # The freq probe must keep calc_type="frequency"; the SP probes
        # must keep "single_point". A bug that defaults everything to
        # SP would silently break the freq-on-CPU vs freq-on-GPU pair.
        plan = _build_execution_plan(
            _MODE_TO_SUITE["tier4"], "tier4", gpu_available=True
        )
        freq_probe = [
            p for p in plan if p["label"].startswith("H₂O  B3LYP/STO-3G  [Freq]")
        ]
        assert len(freq_probe) == 2  # GPU + CPU variants
        assert all(p["calc_type"] == "frequency" for p in freq_probe)

        sp_probe = [p for p in plan if p["label"].startswith("H₂O  B3LYP/6-31G*  [")]
        assert len(sp_probe) == 2
        assert all(p["calc_type"] == "single_point" for p in sp_probe)


class TestPlanEntryShape:
    """Plan entries must have all the fields the worker's positional args
    expect — adding a field to one path but forgetting the other has
    bitten us before."""

    def test_all_required_fields_present(self):
        required = {
            "label",
            "atoms",
            "coords",
            "charge",
            "multiplicity",
            "method",
            "basis",
            "calc_type",
            "force_cpu",
        }
        plan = _build_execution_plan(
            _MODE_TO_SUITE["tier4"], "tier4", gpu_available=True
        )
        for p in plan:
            missing = required - p.keys()
            assert not missing, f"Plan entry missing fields {missing}: {p}"


class TestWorkerEnvVarToggle:
    """The worker must set QUANTUI_DISABLE_GPU=1 BEFORE any quantui /
    gpu4pyscf import, otherwise the cached ``is_gpu_available()`` probe
    sees the parent's environment and the CPU variant ends up using GPU.

    We can't easily test the import-order property without an actual
    subprocess spawn, but we can confirm the env var IS set by the time
    the worker's body executes. The worker accepts a ``result_queue``;
    we monkeypatch ``Molecule`` to capture the env state at call time
    and skip the rest of the calc."""

    def test_force_cpu_true_sets_disable_gpu_env(self, monkeypatch, tmp_path):
        # Strip any pre-existing value so we can see the worker set it.
        monkeypatch.delenv("QUANTUI_DISABLE_GPU", raising=False)

        # Sentinel raise to short-circuit the worker after env-setup.
        class _StopEarly(Exception):
            pass

        captured_env: dict = {}

        def _spy_molecule(*args, **kwargs):
            captured_env["QUANTUI_DISABLE_GPU"] = os.environ.get(
                "QUANTUI_DISABLE_GPU", ""
            )
            raise _StopEarly("captured")

        monkeypatch.setattr("quantui.molecule.Molecule", _spy_molecule)

        from quantui.benchmarks import _calibration_worker

        class _StubQueue:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        q = _StubQueue()
        log_path = tmp_path / "cal.log"
        log_path.write_text("")

        _calibration_worker(
            ["H", "H"],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            0,
            1,
            "RHF",
            "STO-3G",
            "single_point",
            str(log_path),
            q,
            "test-cal-id",
            True,  # force_cpu
        )
        assert captured_env.get("QUANTUI_DISABLE_GPU") == "1"

    def test_force_cpu_false_does_not_touch_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("QUANTUI_DISABLE_GPU", raising=False)

        class _StopEarly(Exception):
            pass

        captured_env: dict = {}

        def _spy_molecule(*args, **kwargs):
            captured_env["QUANTUI_DISABLE_GPU"] = os.environ.get(
                "QUANTUI_DISABLE_GPU", "<unset>"
            )
            raise _StopEarly("captured")

        monkeypatch.setattr("quantui.molecule.Molecule", _spy_molecule)

        from quantui.benchmarks import _calibration_worker

        class _StubQueue:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        q = _StubQueue()
        log_path = tmp_path / "cal.log"
        log_path.write_text("")

        _calibration_worker(
            ["H", "H"],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            0,
            1,
            "RHF",
            "STO-3G",
            "single_point",
            str(log_path),
            q,
            "test-cal-id",
            False,  # force_cpu
        )
        # No env var set by the worker → still unset (== "<unset>" sentinel).
        assert captured_env.get("QUANTUI_DISABLE_GPU") == "<unset>"


class TestCalibrationResultTotal:
    """The dataclass's ``n_total`` property must reflect the expanded
    plan length, not just the raw suite size, so the UI's progress
    denominator stays correct on a GPU-host tier-4 run."""

    def test_default_falls_back_to_suite_size(self):
        from quantui.benchmarks import CalibrationResult

        r = CalibrationResult(timestamp="t", mode="tier4")
        assert r.n_total == len(_MODE_TO_SUITE["tier4"])

    def test_expected_steps_overrides_suite_size(self):
        from quantui.benchmarks import CalibrationResult

        r = CalibrationResult(timestamp="t", mode="tier4", expected_steps=42)
        assert r.n_total == 42

    def test_expected_steps_zero_falls_back(self):
        from quantui.benchmarks import CalibrationResult

        # 0 is the "no override" sentinel — must NOT shadow the suite size.
        r = CalibrationResult(timestamp="t", mode="tier3", expected_steps=0)
        assert r.n_total == len(_MODE_TO_SUITE["tier3"])
