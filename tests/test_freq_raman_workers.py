"""Tests for parallel Raman displacement workers."""

from __future__ import annotations

from quantui import freq_ir_workers as irw


class TestRamanParallelGate:
    def test_parallel_enabled_uses_same_gate_as_ir(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "1")
        assert irw.parallel_enabled_for_run(cpu_count=8, displacement_count=12) is True
        assert irw.parallel_enabled_for_run(cpu_count=2, displacement_count=12) is False

    def test_raman_calc_imports_raman_workers(self):
        import quantui.freq_raman_workers as rw

        assert callable(rw.init_raman_worker)
        assert callable(rw.run_displaced_polarizability)
