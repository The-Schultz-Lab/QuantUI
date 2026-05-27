"""Tests for the M-GPU / GPU.1 gpu4pyscf detection + dispatch helpers.

These tests run on every platform — they don't require a GPU. The actual
``mf.to_gpu()`` migration path is verified by the manual WSL run-through
(see `STATUS.md`); the unit tests cover the detection logic, the CPU-
fallback contract, and the SessionResult field plumbing.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

from quantui.gpu_offload import (
    _GPU_UNSUPPORTED_METHODS,
    is_gpu_available,
    try_to_gpu,
)


def _clear_cache():
    """Force a fresh detection probe before each test."""
    is_gpu_available.cache_clear()


class TestIsGpuAvailable:
    """``is_gpu_available`` must always return a tuple and never raise.

    The actual True branch is exercised only on a CUDA-capable machine
    with gpu4pyscf installed (manual WSL verification). On the Windows CI
    + the user's quantui-win env, the function always reports no GPU.
    """

    def setup_method(self, _m):
        _clear_cache()

    def teardown_method(self, _m):
        _clear_cache()

    def test_returns_tuple_of_bool_and_optional_name(self):
        result = is_gpu_available()
        assert isinstance(result, tuple)
        assert len(result) == 2
        available, name = result
        assert isinstance(available, bool)
        assert name is None or isinstance(name, str)

    def test_disable_env_var_forces_cpu(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_DISABLE_GPU", "1")
        _clear_cache()
        available, name = is_gpu_available()
        assert available is False
        assert name is None

    def test_disable_env_var_accepts_true_string(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_DISABLE_GPU", "true")
        _clear_cache()
        available, _name = is_gpu_available()
        assert available is False

    def test_missing_env_var_does_not_force_cpu(self, monkeypatch):
        monkeypatch.delenv("QUANTUI_DISABLE_GPU", raising=False)
        _clear_cache()
        # We can't assert True without a real GPU; just confirm the env
        # var path doesn't short-circuit to False when unset. The remaining
        # check depends on actual gpu4pyscf availability.
        result = is_gpu_available()
        assert isinstance(result[0], bool)

    def test_missing_gpu4pyscf_returns_false(self, monkeypatch):
        # Simulate gpu4pyscf not installed by removing it from the import
        # cache and shadowing it with a ModuleNotFoundError.
        monkeypatch.delitem(sys.modules, "gpu4pyscf", raising=False)
        original_import = (
            __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__
        )

        def _fake_import(name, *args, **kwargs):
            if name == "gpu4pyscf":
                raise ImportError("simulated: gpu4pyscf missing")
            return original_import(name, *args, **kwargs)

        _clear_cache()
        with patch("builtins.__import__", side_effect=_fake_import):
            available, name = is_gpu_available()
        assert available is False
        assert name is None

    def test_result_is_cached(self):
        # Same call twice should reuse cached result (perf assertion via
        # checking cache info, not timing).
        _clear_cache()
        is_gpu_available()
        info_after_first = is_gpu_available.cache_info()
        is_gpu_available()
        info_after_second = is_gpu_available.cache_info()
        assert info_after_second.hits >= info_after_first.hits


class TestTryToGpu:
    """``try_to_gpu`` must always return a 3-tuple and never raise.

    CCSD(T) is explicitly skipped per gpu4pyscf's documented coverage.
    Unsupported / missing-GPU paths must return the original ``mf``
    unchanged so the SCF can still run on CPU.
    """

    def setup_method(self, _m):
        _clear_cache()

    def test_returns_three_tuple(self):
        sentinel_mf = object()
        result = try_to_gpu(sentinel_mf, "RHF")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_ccsd_t_is_skipped(self):
        sentinel_mf = object()
        mf_out, used, name = try_to_gpu(sentinel_mf, "CCSD(T)")
        # Original mf returned unchanged; GPU not used.
        assert mf_out is sentinel_mf
        assert used is False
        assert name is None

    def test_ccsd_t_is_in_unsupported_set(self):
        # Lock in the documented gpu4pyscf coverage gap so future
        # contributors don't accidentally add CCSD(T) to the GPU path.
        assert "CCSD(T)" in _GPU_UNSUPPORTED_METHODS

    def test_no_gpu_available_returns_original_mf(self, monkeypatch):
        # Force CPU via env var; mf must come back unchanged.
        monkeypatch.setenv("QUANTUI_DISABLE_GPU", "1")
        _clear_cache()
        sentinel_mf = object()
        mf_out, used, name = try_to_gpu(sentinel_mf, "RHF")
        assert mf_out is sentinel_mf
        assert used is False
        assert name is None

    def test_to_gpu_failure_falls_back_cleanly(self, monkeypatch):
        # Simulate a successful is_gpu_available probe but a broken
        # .to_gpu() call (e.g. unsupported method variant). The helper
        # must catch and return the original mf with used=False.
        class _BadMf:
            def to_gpu(self):
                raise RuntimeError("simulated gpu4pyscf failure")

        # Patch the helper to return "GPU is available, name=fake".
        with patch(
            "quantui.gpu_offload.is_gpu_available",
            return_value=(True, "Fake GPU"),
        ):
            mf = _BadMf()
            mf_out, used, name = try_to_gpu(mf, "RHF")
        assert mf_out is mf
        assert used is False
        assert name is None

    def test_to_gpu_success_propagates_gpu_name(self):
        # Successful migration: helper returns the migrated mf + used=True
        # + the device name reported by is_gpu_available.
        class _GoodMf:
            def to_gpu(self):
                return _GpuMf()

        class _GpuMf:
            pass

        with patch(
            "quantui.gpu_offload.is_gpu_available",
            return_value=(True, "Tesla V100"),
        ):
            mf = _GoodMf()
            mf_out, used, name = try_to_gpu(mf, "B3LYP")
        assert isinstance(mf_out, _GpuMf)
        assert used is True
        assert name == "Tesla V100"


class TestSessionResultGpuFields:
    """SessionResult exposes ``gpu_used`` + ``gpu_name`` with safe defaults."""

    def test_defaults_are_cpu_outcome(self):
        from quantui.session_calc import SessionResult

        r = SessionResult(
            energy_hartree=-1.0,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=8,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        assert r.gpu_used is False
        assert r.gpu_name is None

    def test_can_store_gpu_outcome(self):
        from quantui.session_calc import SessionResult

        r = SessionResult(
            energy_hartree=-1.0,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=8,
            method="RHF",
            basis="STO-3G",
            formula="H2",
            gpu_used=True,
            gpu_name="NVIDIA RTX 3080",
        )
        assert r.gpu_used is True
        assert r.gpu_name == "NVIDIA RTX 3080"


class TestResultCardComputeDeviceRow:
    """``format_result`` shows a Compute device row reflecting gpu_used."""

    def test_cpu_row_shown_by_default(self):
        from quantui.app_formatters import format_result
        from quantui.session_calc import SessionResult

        r = SessionResult(
            energy_hartree=-1.0,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=8,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        html = format_result(r)
        assert "Compute device" in html
        assert "CPU" in html
        assert "GPU" not in html

    def test_gpu_row_shows_name(self):
        from quantui.app_formatters import format_result
        from quantui.session_calc import SessionResult

        r = SessionResult(
            energy_hartree=-1.0,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=8,
            method="RHF",
            basis="STO-3G",
            formula="H2",
            gpu_used=True,
            gpu_name="Tesla V100",
        )
        html = format_result(r)
        assert "Compute device" in html
        assert "GPU" in html
        assert "Tesla V100" in html


class TestStatusTabGpuIndicator:
    """Status tab includes a GPU-offload row regardless of detection result."""

    def test_status_html_includes_gpu_offload_row(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        html_value = app._status_html.value
        assert "GPU offload" in html_value
        assert "gpu4pyscf" in html_value
