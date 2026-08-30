"""Windows-safe tests for the freq_ir_workers opt-in parallel IR loop.

The actual ProcessPoolExecutor + PySCF integration lives in the
PySCF-gated ``test_freq_calc.py::TestIRIntensities`` path and runs on
WSL. These tests pin the contracts that don't require PySCF:

- ``parallel_enabled_for_run`` gate logic (env-var opt-in, core
  threshold, displacement threshold).
- ``pick_worker_count`` heuristic.
- ``threads_per_worker`` BLAS budgeting math.
- ``_truthy`` env-var parser conventions.
"""

from __future__ import annotations

import pytest

from quantui.freq_ir_workers import (
    _truthy,
    parallel_enabled_for_run,
    pick_worker_count,
    threads_per_worker,
)
from quantui.user_settings import UserSettings


class TestParallelEnabledGate:
    """``parallel_enabled_for_run`` must be False unless every condition is met."""

    def test_off_by_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("QUANTUI_FREQ_PARALLEL", raising=False)
        assert parallel_enabled_for_run(cpu_count=16, displacement_count=60) is False

    def test_off_when_env_falsy(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "0")
        assert parallel_enabled_for_run(cpu_count=16, displacement_count=60) is False

    def test_on_when_env_truthy_and_conditions_met(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "1")
        assert parallel_enabled_for_run(cpu_count=8, displacement_count=18) is True

    def test_env_truthy_string_variants_accepted(self, monkeypatch):
        for val in ("1", "true", "True", "yes", "on"):
            monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", val)
            assert parallel_enabled_for_run(
                cpu_count=8, displacement_count=18
            ), f"value {val!r} should be truthy"

    def test_gpu_available_does_not_veto_parallel(self, monkeypatch):
        # NCShare-style nodes: one GPU, many CPU cores. Opt-in parallel
        # uses CPU workers for displacements while the reference SCF/Hessian
        # may still have used gpu4pyscf.
        monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "1")
        assert parallel_enabled_for_run(cpu_count=16, displacement_count=60) is True

    def test_too_few_cores_vetoes_parallel(self, monkeypatch):
        # Below 4 cores the BLAS-oversubscription tradeoff doesn't pay off.
        monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "1")
        assert parallel_enabled_for_run(cpu_count=2, displacement_count=60) is False

    def test_too_few_displacements_vetoes_parallel(self, monkeypatch):
        # For a diatomic (2 atoms → 12 displacements? No, 2*3*2=12) we still
        # parallelize; for a single atom (3*2=6 exactly) we hit the floor.
        # For a hypothetical 5-displacement case (not real, but the gate is
        # generic) we'd skip parallel.
        monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "1")
        assert parallel_enabled_for_run(cpu_count=16, displacement_count=4) is False
        # 6 displacements is exactly at the threshold and should pass.
        assert parallel_enabled_for_run(cpu_count=16, displacement_count=6) is True

    def test_settings_checkbox_opt_in_without_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("QUANTUI_FREQ_PARALLEL", raising=False)
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
        settings = UserSettings()
        settings.compute.freq_parallel = True
        settings.save()
        assert parallel_enabled_for_run(cpu_count=8, displacement_count=18) is True

    def test_env_var_overrides_settings_off(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
        settings = UserSettings()
        settings.compute.freq_parallel = True
        settings.save()
        monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "0")
        assert parallel_enabled_for_run(cpu_count=16, displacement_count=60) is False


class TestPickWorkerCount:
    """Worker count = ``min(cpu // 2, displacement_count)``, floored at 1."""

    def test_uses_half_of_cpu_when_displacements_plenty(self):
        assert pick_worker_count(cpu_count=16, displacement_count=60) == 8

    def test_capped_by_displacement_count_when_few_tasks(self):
        # 18 displacements, 16 cores: half would be 8 but we only have 18
        # tasks — that's fine, 8 workers each get ~2 tasks. But with 4
        # displacements we cap to 4 workers (more would idle).
        assert pick_worker_count(cpu_count=16, displacement_count=4) == 4

    def test_minimum_one_worker(self):
        assert pick_worker_count(cpu_count=1, displacement_count=60) == 1

    def test_zero_displacement_returns_zero(self):
        # Degenerate input; consumers should gate on this anyway.
        assert pick_worker_count(cpu_count=8, displacement_count=0) == 0


class TestThreadsPerWorker:
    """BLAS-thread budget per worker must always be >= 1."""

    def test_divides_evenly_when_possible(self):
        assert threads_per_worker(cpu_count=16, n_workers=4) == 4

    def test_floors_to_integer(self):
        # 8 // 3 = 2 (integer floor); no oversubscription guarantee.
        assert threads_per_worker(cpu_count=8, n_workers=3) == 2

    def test_floor_at_one_for_huge_worker_count(self):
        # If the parent picked more workers than cores, give each 1 thread
        # rather than 0 (which BLAS would interpret as "use default").
        assert threads_per_worker(cpu_count=4, n_workers=8) == 1

    def test_zero_workers_returns_one(self):
        assert threads_per_worker(cpu_count=16, n_workers=0) == 1


class TestTruthyParser:
    """``_truthy`` matches the convention used by other QuantUI env vars."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "True"])
    def test_recognised_truthy(self, value):
        assert _truthy(value) is True

    @pytest.mark.parametrize("value", ["0", "", "false", "no", "off", "anything"])
    def test_recognised_falsy(self, value):
        assert _truthy(value) is False

    def test_whitespace_stripped(self):
        assert _truthy("  1  ") is True
        assert _truthy("\ttrue\n") is True
