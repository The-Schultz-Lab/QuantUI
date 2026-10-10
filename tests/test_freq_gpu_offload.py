"""
Frequency jobs offload the reference SCF and the analytical Hessian to the GPU
(GOTCHAS 2026-09-26: on the GPU image both ran on CPU).

There is no GPU here, so ``try_to_gpu`` is patched to hand back a stand-in
"GPU" object that delegates to the real CPU SCF. That checks the wiring: the
offload is attempted, the Hessian runs on the offloaded object (or falls back
to ``to_cpu()`` when it raises), and everything after the Hessian sees the
CPU object again. Real GPU numbers need an H200 run (manual checklist).
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

pytest.importorskip("pyscf")

from quantui.freq_calc import run_freq_calc  # noqa: E402
from quantui.molecule import Molecule  # noqa: E402

WATER = Molecule(
    ["O", "H", "H"],
    [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
)


class FakeGPU:
    """Delegates to a real CPU SCF object; records how it was used."""

    def __init__(self, cpu, *, fail_hessian=False, calls=None):
        object.__setattr__(self, "_cpu", cpu)
        object.__setattr__(self, "_fail", fail_hessian)
        object.__setattr__(self, "calls", calls if calls is not None else [])

    def __getattr__(self, name):
        return getattr(self._cpu, name)

    def __setattr__(self, name, value):
        setattr(self._cpu, name, value)

    def Hessian(self):  # noqa: N802 — PySCF's name
        self.calls.append("gpu_hessian")
        if self._fail:
            raise RuntimeError("ECP Hessian not implemented on GPU")
        return self._cpu.Hessian()

    def to_cpu(self):
        self.calls.append("to_cpu")
        return self._cpu


def _run(fail_hessian=False):
    calls: list = []
    first = {"done": False}

    def fake_try_to_gpu(mf, method_upper):
        if first["done"]:  # IR displacement SCFs: leave them on CPU
            return mf, False, None
        first["done"] = True
        calls.append(("offload", method_upper))
        return FakeGPU(mf, fail_hessian=fail_hessian, calls=calls), True, "Fake H200"

    log = io.StringIO()
    with patch("quantui.gpu_offload.try_to_gpu", side_effect=fake_try_to_gpu):
        result = run_freq_calc(WATER, method="RHF", basis="STO-3G", progress_stream=log)
    return result, calls, log.getvalue()


def test_reference_scf_and_hessian_run_on_the_offloaded_object():
    result, calls, log = _run()
    assert calls[0] == ("offload", "RHF")
    assert "gpu_hessian" in calls
    # Back on CPU right after the Hessian, before IR/thermo use mf.
    assert calls.index("to_cpu") > calls.index("gpu_hessian")
    assert result.gpu_used is True and result.gpu_name == "Fake H200"
    assert "GPU offload active" in log
    assert result.converged and len(result.frequencies_cm1) == 3
    assert all(f > 1000 for f in result.frequencies_cm1)
    assert result.ir_intensities and result.thermo is not None


def test_failed_gpu_hessian_is_recomputed_on_cpu():
    result, calls, log = _run(fail_hessian=True)
    assert "gpu_hessian" in calls and "to_cpu" in calls
    assert "GPU Hessian failed" in log and "recomputing the Hessian on CPU" in log
    assert result.converged and len(result.frequencies_cm1) == 3


def test_cpu_run_is_unchanged_without_a_gpu():
    with patch(
        "quantui.gpu_offload.try_to_gpu", side_effect=lambda mf, m: (mf, False, None)
    ):
        result = run_freq_calc(
            WATER, method="RHF", basis="STO-3G", progress_stream=io.StringIO()
        )
    assert result.gpu_used is False and result.gpu_name is None
    assert result.converged and len(result.frequencies_cm1) == 3
