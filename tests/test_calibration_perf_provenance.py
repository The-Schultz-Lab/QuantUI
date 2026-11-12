"""Calibration perf records carry GPU / density-fitting provenance (audit DATA-5).

Without ``gpu_used`` / ``gpu_name`` / ``density_fit`` in the record, the
cross-device probe's forced-CPU and GPU runs of the same entry logged
identically: the analytics speedup table stayed empty and GPU timings fed
the CPU estimate pool.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from quantui import benchmarks


class _StubQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_worker_reports_device_and_df(monkeypatch, tmp_path):
    fake = SimpleNamespace(
        formula="H2",
        converged=True,
        n_iterations=3,
        gpu_used=True,
        gpu_name="Fake GPU",
        density_fit=True,
    )
    monkeypatch.setattr(
        "quantui.session_calc.run_in_session", lambda *a, **k: fake, raising=True
    )
    monkeypatch.setattr(benchmarks, "_save_calibration_step", lambda *a, **k: None)
    log = tmp_path / "cal.log"
    log.write_text("")
    q = _StubQueue()
    benchmarks._calibration_worker(
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        0,
        1,
        "RHF",
        "STO-3G",
        "single_point",
        str(log),
        q,
        "run-id",
        False,
    )
    (msg,) = q.items
    assert msg["status"] == "ok"
    assert msg["gpu_used"] is True
    assert msg["gpu_name"] == "Fake GPU"
    assert msg["density_fit"] is True


class _InlineProcess:
    """Runs the target synchronously; stands in for a spawn Process."""

    def __init__(self, target, args, daemon=True):
        self._target, self._args = target, args
        self.exitcode = 0

    def start(self):
        self._target(*self._args)

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False

    def terminate(self):
        return None


class _InlineCtx:
    Process = _InlineProcess

    @staticmethod
    def Queue():
        return queue.Queue()


def test_run_calibration_logs_device_and_df(monkeypatch, tmp_path):
    pytest.importorskip("pyscf")  # run_calibration skips steps without it
    import multiprocessing

    entry = benchmarks._MODE_TO_SUITE["tier1"][0]
    monkeypatch.setitem(benchmarks._MODE_TO_SUITE, "tier1", [entry])
    monkeypatch.setattr(multiprocessing, "get_context", lambda *_a: _InlineCtx())
    monkeypatch.setattr(
        benchmarks, "_calibration_log_path", lambda ts: tmp_path / "cal.log"
    )
    monkeypatch.setattr(benchmarks, "_save_calibration_json", lambda *a, **k: None)

    def _fake_worker(*args):
        result_queue = args[8]
        result_queue.put(
            {
                "status": "ok",
                "formula": "H2",
                "converged": True,
                "n_iterations": 3,
                "elapsed_s": 0.5,
                "import_s": 0.1,
                "gpu_used": True,
                "gpu_name": "Fake GPU",
                "density_fit": True,
                "result_dir": None,
            }
        )

    monkeypatch.setattr(benchmarks, "_calibration_worker", _fake_worker)
    logged = []
    monkeypatch.setattr(
        "quantui.calc_log.log_calculation", lambda **kw: logged.append(kw)
    )
    monkeypatch.setattr(
        "quantui.gpu_offload.is_gpu_available", lambda: (False, None), raising=False
    )

    result = benchmarks.run_calibration(mode="tier1")

    assert [s.status for s in result.steps] == ["ok"]
    (rec,) = logged
    assert rec["source"] == "calibration"
    assert rec["gpu_used"] is True
    assert rec["gpu_name"] == "Fake GPU"
    assert rec["density_fit"] is True
