"""Regression tests for the Low-priority data-layer audit findings (DATA-10..17)."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import time

import pytest

from quantui import benchmarks

# ---------------------------------------------------------------------------
# DATA-10 — calibration worker must be able to start the parallel IR pool
# ---------------------------------------------------------------------------


class _RecordingProcess:
    created: list = []

    def __init__(self, target, args, daemon=None):
        self.target, self.args, self.daemon = target, args, daemon
        self.exitcode = 0
        self.pid = None
        _RecordingProcess.created.append(self)

    def start(self):
        self.args[8].put(
            {
                "status": "ok",
                "formula": "H2",
                "converged": True,
                "n_iterations": 3,
                "elapsed_s": 0.5,
                "result_dir": None,
            }
        )

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False

    def terminate(self):
        return None


class _RecordingCtx:
    Process = _RecordingProcess

    @staticmethod
    def Queue():
        return queue.Queue()


def test_calibration_worker_is_not_daemonic(monkeypatch, tmp_path):
    """A daemonic worker cannot have children, so freq steps never got the
    parallel IR pool (audit DATA-10)."""
    pytest.importorskip("pyscf")
    import multiprocessing

    entry = benchmarks._MODE_TO_SUITE["tier1"][0]
    monkeypatch.setitem(benchmarks._MODE_TO_SUITE, "tier1", [entry])
    monkeypatch.setattr(multiprocessing, "get_context", lambda *_a: _RecordingCtx())
    monkeypatch.setattr(
        benchmarks, "_calibration_log_path", lambda ts: tmp_path / "cal.log"
    )
    monkeypatch.setattr(benchmarks, "_save_calibration_json", lambda *a, **k: None)
    monkeypatch.setattr("quantui.calc_log.log_calculation", lambda **kw: None)
    monkeypatch.setattr(
        "quantui.gpu_offload.is_gpu_available", lambda: (False, None), raising=False
    )
    _RecordingProcess.created = []

    result = benchmarks.run_calibration(mode="tier1")

    assert [s.status for s in result.steps] == ["ok"]
    (proc,) = _RecordingProcess.created
    assert proc.daemon is False


def _group_leader_with_child(pids):  # pragma: no cover - runs in a fork child
    os.setpgrp()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pids.put(child.pid)
    time.sleep(60)


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
def test_terminate_worker_tree_kills_grandchildren():
    """Stop/Skip/timeout must not orphan the worker's pool processes."""
    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    pids = ctx.Queue()
    worker = ctx.Process(target=_group_leader_with_child, args=(pids,))
    worker.start()
    try:
        kids = [pids.get(timeout=20)]
        benchmarks._terminate_worker_tree(worker)
        worker.join(10)
        assert not worker.is_alive()
        deadline = time.time() + 10
        while time.time() < deadline:
            alive = [
                k
                for k in kids
                if os.path.exists(f"/proc/{k}")
                and "Z" not in open(f"/proc/{k}/stat").read().split()[2]
            ]
            if not alive:
                break
            time.sleep(0.1)
        assert not alive
    finally:
        if worker.is_alive():
            os.killpg(worker.pid, 9)
