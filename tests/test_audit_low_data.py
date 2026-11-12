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


# ---------------------------------------------------------------------------
# DATA-11 — save_result / save_trajectory robustness
# ---------------------------------------------------------------------------


def _fake_result(**kw):
    from types import SimpleNamespace

    base = dict(
        formula="H2",
        method="RHF",
        basis="STO-3G",
        energy_hartree=-1.117,
        converged=True,
        n_iterations=5,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_save_result_numpy_fields_are_serialised(tmp_path):
    import json

    import numpy as np

    from quantui.results_storage import save_result

    res = _fake_result(cc_converged=np.bool_(True), solvent=np.str_("water"))
    d = save_result(
        res,
        results_dir=tmp_path,
        spectra={"ir": {"freqs": np.array([1.0, 2.0], dtype=np.float32)}},
        extras={"flag": np.bool_(False), "where": tmp_path},
    )
    data = json.loads((d / "result.json").read_text())
    assert data["cc_converged"] is True
    assert data["spectra"]["ir"]["freqs"] == [1.0, 2.0]
    assert data["flag"] is False
    assert data["where"] == str(tmp_path)


def test_save_result_failure_leaves_no_empty_dir(tmp_path, monkeypatch):
    from quantui import results_storage

    def _boom(*_a, **_k):
        raise TypeError("unserialisable")

    monkeypatch.setattr(results_storage, "_dumps_json_safe", _boom)
    with pytest.raises(TypeError):
        results_storage.save_result(_fake_result(), results_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_save_result_nan_written_as_null(tmp_path):
    import json

    from quantui.results_storage import save_result

    res = _fake_result(energy_hartree=float("nan"), homo_lumo_gap_ev=float("inf"))
    d = save_result(res, results_dir=tmp_path)
    text = (d / "result.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    data = json.loads(text)
    assert data["energy_hartree"] is None
    assert data["homo_lumo_gap_ev"] is None


def test_save_result_truncates_long_basis_name(tmp_path):
    from quantui.results_storage import save_result

    d = save_result(_fake_result(basis="x" * 300), results_dir=tmp_path)
    assert len(d.name) < 255
    assert (d / "result.json").is_file()
    assert not list(d.glob("*.tmp"))


def test_save_trajectory_float32(tmp_path):
    import numpy as np

    from quantui.molecule import Molecule
    from quantui.results_storage import load_trajectory, save_trajectory

    mol = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    mol.coordinates = np.asarray(mol.coordinates, dtype=np.float32)
    save_trajectory(tmp_path, [mol, mol], [np.float32(-1.1), np.float32(-1.12)])
    mols, energies = load_trajectory(tmp_path)
    assert len(mols) == 2
    assert energies[1] == pytest.approx(-1.12, abs=1e-6)


# ---------------------------------------------------------------------------
# DATA-12 — corrupt settings.json must not crash; unknown sections survive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [b"\xff\xfe\x00garbage\x80", b"[" * 100000],
    ids=["non-utf8", "deeply-nested"],
)
def test_corrupt_settings_fall_back_to_defaults(tmp_path, raw):
    from quantui.user_settings import UserSettings

    p = tmp_path / "settings.json"
    p.write_bytes(raw)
    assert UserSettings.load(p) == UserSettings()


def test_settings_save_preserves_unknown_sections(tmp_path):
    import json

    from quantui.user_settings import UserSettings

    p = tmp_path / "settings.json"
    data = UserSettings().to_dict()
    data["future_section"] = {"x": 1}
    data["compute"]["future_key"] = "keep"
    p.write_text(json.dumps(data), encoding="utf-8")

    s = UserSettings.load(p)
    s.compute.density_fit = True
    s.save(p)

    saved = json.loads(p.read_text(encoding="utf-8"))
    assert saved["future_section"] == {"x": 1}
    assert saved["compute"]["future_key"] == "keep"
    assert saved["compute"]["density_fit"] is True


# ---------------------------------------------------------------------------
# DATA-13 — calc_log writers/readers robustness
# ---------------------------------------------------------------------------


@pytest.fixture
def _calc_log_dir(tmp_path, monkeypatch):
    from quantui import calc_log

    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    calc_log._READ_ALL_CACHE.clear()
    yield tmp_path
    calc_log._READ_ALL_CACHE.clear()


def test_log_event_accepts_numpy_and_path(_calc_log_dir):
    import numpy as np

    from quantui import calc_log

    calc_log.log_event(
        "x", "msg", n=np.int64(3), ok=np.bool_(True), where=_calc_log_dir
    )
    (rec,) = calc_log.get_recent_events()
    assert rec["n"] == 3 and rec["ok"] is True
    assert rec["where"] == str(_calc_log_dir)


def test_non_object_lines_are_skipped(_calc_log_dir):
    from quantui import calc_log

    calc_log.log_event("a", "first")
    with open(calc_log._event_path(), "a", encoding="utf-8") as fh:
        fh.write("123\n")
    calc_log.prune_events()  # used to raise TypeError on the ``123`` line
    events = calc_log.get_recent_events()
    assert [e["event"] for e in events] == ["a"]

    with open(calc_log._perf_path(), "a", encoding="utf-8") as fh:
        fh.write("42\n")
    assert calc_log.get_perf_history() == []
    calc_log.estimate_time(n_atoms=2, n_electrons=2, method="RHF", basis="STO-3G")


def test_prune_keeps_bytes_appended_after_read(_calc_log_dir, monkeypatch):
    """An append from another process between read and rewrite survives."""
    from quantui import calc_log

    calc_log.log_event("a", "first")
    path = calc_log._event_path()
    real_open = open
    state = {"n": 0}

    def _open(file, mode="r", *a, **k):
        fh = real_open(file, mode, *a, **k)
        if str(file) == str(path) and mode == "rb":
            state["n"] += 1
            if state["n"] == 1:  # after the prune's first read
                with real_open(path, "a", encoding="utf-8") as other:
                    other.write(
                        '{"timestamp": "2999-01-01T00:00:00+00:00",'
                        ' "event": "foreign", "message": ""}\n'
                    )
        return fh

    monkeypatch.setattr(calc_log, "open", _open, raising=False)
    calc_log.prune_events()
    monkeypatch.delattr(calc_log, "open")
    calc_log._READ_ALL_CACHE.clear()
    assert [e["event"] for e in calc_log.get_recent_events()] == ["a", "foreign"]


def test_read_all_cache_not_poisoned_by_caller_mutation(_calc_log_dir):
    from quantui import calc_log

    calc_log.log_event("a", "first")
    calc_log.get_recent_events()[0]["event"] = "mutated"
    assert calc_log.get_recent_events()[0]["event"] == "a"
