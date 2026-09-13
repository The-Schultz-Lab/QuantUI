"""Platform-independent tests for the guarded PyFock CuPy path."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from quantui.pyfock_gpu import (
    clear_pyfock_gpu_probe_cache,
    probe_pyfock_gpu,
    resolve_pyfock_gpu,
)


def _fake_cupy(count=1, name=b"NVIDIA H200"):
    runtime = SimpleNamespace(
        getDeviceCount=lambda: count,
        getDeviceProperties=lambda _index: {"name": name},
    )
    module = ModuleType("cupy")
    module.cuda = SimpleNamespace(runtime=runtime)
    return module


def test_probe_accepts_cupy_device_without_gpu4pyscf(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    clear_pyfock_gpu_probe_cache()

    assert probe_pyfock_gpu() == (True, "NVIDIA H200", "")


def test_probe_rejects_no_cuda_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(count=0))
    clear_pyfock_gpu_probe_cache()

    available, name, reason = probe_pyfock_gpu()
    assert available is False
    assert name is None
    assert "0 CUDA devices" in reason


def test_process_opt_out_wins_over_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    monkeypatch.setenv("QUANTUI_DISABLE_GPU", "1")
    clear_pyfock_gpu_probe_cache()

    assert probe_pyfock_gpu()[0] is False
    assert "QUANTUI_DISABLE_GPU" in probe_pyfock_gpu()[2]


def test_per_calculation_opt_out_does_not_probe(monkeypatch):
    def fail_probe():
        raise AssertionError("GPU probe should not run for explicit opt-out")

    monkeypatch.setattr("quantui.pyfock_gpu.probe_pyfock_gpu", fail_probe)
    assert resolve_pyfock_gpu(False) == (
        False,
        None,
        "GPU acceleration disabled for this calculation",
    )
