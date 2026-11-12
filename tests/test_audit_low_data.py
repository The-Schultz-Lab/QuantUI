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


class _PopenWorker:
    """The slice of ``multiprocessing.Process`` that ``_terminate_worker_tree`` uses.

    The worker is a fresh interpreter in its own session (so it leads its
    process group, as the calibration worker does after ``os.setpgrp()``).
    Forking the pytest process instead can hang the child before it reaches
    ``setpgrp`` when xdist/coverage threads hold a lock at fork time.
    """

    def __init__(self, popen):
        self._p = popen
        self.pid = popen.pid

    def is_alive(self):
        return self._p.poll() is None

    def terminate(self):
        self._p.terminate()

    def join(self, timeout=None):
        try:
            self._p.wait(timeout)
        except subprocess.TimeoutExpired:
            pass


def _running(pid):
    """True while ``pid`` exists and is not a zombie (tolerates it vanishing)."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


@pytest.mark.skipif(
    not hasattr(os, "killpg") or not os.path.isdir("/proc"),
    reason="POSIX process groups + /proc",
)
def test_terminate_worker_tree_kills_grandchildren(tmp_path):
    """Stop/Skip/timeout must not orphan the worker's pool processes."""
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(c.pid))\n"
        "time.sleep(60)\n"
    )
    worker = _PopenWorker(
        subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    )
    try:
        deadline = time.time() + 30
        while not (pid_file.exists() and pid_file.read_text().strip()):
            assert time.time() < deadline, "worker never started its child"
            assert worker.is_alive(), "worker exited before starting its child"
            time.sleep(0.05)
        kid = int(pid_file.read_text())
        assert os.getpgid(worker.pid) == worker.pid

        benchmarks._terminate_worker_tree(worker)
        worker.join(10)
        assert not worker.is_alive()
        deadline = time.time() + 10
        while _running(kid) and time.time() < deadline:
            time.sleep(0.1)
        assert not _running(kid)
    finally:
        try:
            os.killpg(worker.pid, 9)
        except ProcessLookupError:
            pass
        worker.join(5)


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


# ---------------------------------------------------------------------------
# DATA-14 — time estimator
# ---------------------------------------------------------------------------


def _perf(**kw):
    rec = dict(
        converged=True,
        method="RHF",
        basis="STO-3G",
        n_atoms=3,
        n_electrons=10,
        n_basis=7,
        calc_type="single_point",
        source="app",
        gpu_used=False,
    )
    rec.update(kw)
    return rec


def test_frequency_cost_model_partitions_by_density_fit():
    from quantui import calc_log

    records = [_perf(elapsed_s=2.0, density_fit=True) for _ in range(3)] + [
        _perf(elapsed_s=200.0, density_fit=False) for _ in range(3)
    ]
    kw = dict(
        n_atoms=3,
        n_electrons=10,
        method="RHF",
        basis="STO-3G",
        n_basis=7,
        calc_type="frequency",
        gpu_used=False,
        source="app",
    )
    fast = calc_log.estimate_time_from_records(records, density_fit=True, **kw)
    slow = calc_log.estimate_time_from_records(records, density_fit=False, **kw)
    assert fast is not None and slow is not None
    assert slow["seconds"] > 50 * fast["seconds"]


def test_replay_passes_density_fit(monkeypatch):
    from quantui import calc_log, estimator_eval

    seen = []

    def _spy(history, **kw):
        seen.append(kw.get("density_fit"))
        return None

    monkeypatch.setattr(calc_log, "estimate_time_from_records", _spy)
    estimator_eval.replay(
        [_perf(elapsed_s=3.0, density_fit=True, timestamp="2026-01-01")]
    )
    assert seen == [True]


def test_sub_second_records_do_not_drive_power_law():
    """Overhead-dominated runs must not be extrapolated by n_basis^3.5."""
    from quantui import calc_log

    records = [
        _perf(elapsed_s=0.05, n_basis=2, n_electrons=2),
        _perf(elapsed_s=0.06, n_basis=7, n_electrons=10),
        _perf(elapsed_s=0.07, n_basis=7, n_electrons=10),
        _perf(elapsed_s=0.08, n_basis=2, n_electrons=2),
    ]
    est = calc_log.estimate_time_from_records(
        records,
        n_atoms=40,
        n_electrons=160,
        method="RHF",
        basis="STO-3G",
        n_basis=200,
        calc_type="single_point",
    )
    assert est is not None
    # Power law gave ~51 min here; electron-count scaling is minutes at most.
    assert est["seconds"] < 600


def test_log_calculation_records_n_threads(tmp_path, monkeypatch):
    from quantui import calc_log

    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    calc_log._READ_ALL_CACHE.clear()
    calc_log.log_calculation(
        formula="H2",
        n_atoms=2,
        n_electrons=2,
        method="RHF",
        basis="STO-3G",
        n_iterations=3,
        elapsed_s=1.0,
        converged=True,
        n_threads=4,
    )
    (rec,) = calc_log.get_perf_history()
    assert rec["n_threads"] == 4
    assert "n_cores" not in rec
    calc_log._READ_ALL_CACHE.clear()


def test_calibration_logs_opt_steps_and_threads(monkeypatch, tmp_path):
    pytest.importorskip("pyscf")
    import multiprocessing

    entry = tuple(benchmarks._MODE_TO_SUITE["tier1"][0]) + ("geometry_opt",)
    monkeypatch.setitem(benchmarks._MODE_TO_SUITE, "tier1", [entry])

    class _Proc(_RecordingProcess):
        def start(self):
            self.args[8].put(
                {
                    "status": "ok",
                    "formula": "H2",
                    "converged": True,
                    "n_iterations": None,
                    "n_steps": 4,
                    "n_threads": 2,
                    "elapsed_s": 1.5,
                    "result_dir": None,
                }
            )

    class _Ctx(_RecordingCtx):
        Process = _Proc

    monkeypatch.setattr(multiprocessing, "get_context", lambda *_a: _Ctx())
    monkeypatch.setattr(
        benchmarks, "_calibration_log_path", lambda ts: tmp_path / "cal.log"
    )
    monkeypatch.setattr(benchmarks, "_save_calibration_json", lambda *a, **k: None)
    monkeypatch.setattr(
        "quantui.gpu_offload.is_gpu_available", lambda: (False, None), raising=False
    )
    logged = []
    monkeypatch.setattr(
        "quantui.calc_log.log_calculation", lambda **kw: logged.append(kw)
    )
    benchmarks.run_calibration(mode="tier1")
    (rec,) = logged
    assert rec["n_steps"] == 4
    assert rec["n_iterations"] is None
    assert rec["n_threads"] == 2
    assert "n_cores" not in rec


def test_calibration_worker_reports_opt_steps(monkeypatch, tmp_path):
    from types import SimpleNamespace

    fake = SimpleNamespace(
        molecule=SimpleNamespace(get_formula=lambda: "H2"),
        converged=True,
        n_steps=5,
    )
    monkeypatch.setattr("quantui.optimizer.optimize_geometry", lambda **k: fake)
    monkeypatch.setattr(benchmarks, "_save_calibration_step", lambda *a, **k: None)
    log = tmp_path / "cal.log"
    log.write_text("")

    class _Q:
        items: list = []

        def put(self, item):
            self.items.append(item)

    q = _Q()
    benchmarks._calibration_worker(
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        0,
        1,
        "RHF",
        "STO-3G",
        "geometry_opt",
        str(log),
        q,
        "rid",
        False,
    )
    (msg,) = q.items
    assert msg["status"] == "ok", msg
    assert msg["n_steps"] == 5
    assert msg["n_iterations"] is None


# ---------------------------------------------------------------------------
# DATA-15 — checkpoint retention / resume key
# ---------------------------------------------------------------------------


def test_coords_digest_ignores_sign_of_zero():
    from quantui.checkpoint import _coords_digest

    a = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
    b = [[-0.0, 0.0, -0.0], [0.0, -0.0000001, 0.74]]
    assert _coords_digest(a) == _coords_digest(b)
    assert _coords_digest(a) != _coords_digest([[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]])


def test_prune_removes_old_unreadable_checkpoint_dirs(tmp_path):
    from quantui import checkpoint as C

    old = time.time() - 60 * 86400
    missing = tmp_path / ("a" * 16)
    missing.mkdir()
    corrupt = tmp_path / ("b" * 16)
    corrupt.mkdir()
    (corrupt / "meta.json").write_text("{not json")
    fresh = tmp_path / ("c" * 16)
    fresh.mkdir()  # e.g. a checkpoint being created right now
    foreign = tmp_path / "not-a-checkpoint"
    foreign.mkdir()
    for p in (missing, corrupt / "meta.json", corrupt, foreign):
        os.utime(p, (old, old))

    assert C.prune(tmp_path, max_age_days=14) == 2
    assert not missing.exists() and not corrupt.exists()
    assert fresh.exists() and foreign.exists()


# ---------------------------------------------------------------------------
# DATA-16 — parser gaps for real-world files
# ---------------------------------------------------------------------------

_WATER = "3\nwater\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"


@pytest.mark.parametrize(
    "text",
    [
        "﻿" + _WATER,
        _WATER.replace("\n", "\r"),
        _WATER.replace("0.117", "1.17D-01"),
        _WATER.replace("-0.467", "−0.467"),
        "O 0 0 0.117\nD 0 0.757 -0.467\nT 0 -0.757 -0.467\n",
    ],
    ids=["bom", "old-mac-cr", "fortran-d", "unicode-minus", "isotopes"],
)
def test_paste_parser_accepts_real_world_quirks(text):
    from quantui.molecule import parse_xyz_input

    atoms, coords = parse_xyz_input(text)
    assert atoms == ["O", "H", "H"]
    assert coords[0][2] == pytest.approx(0.117)
    assert coords[1][2] == pytest.approx(-0.467)


def test_has_isotope_labels():
    from quantui.molecule import has_isotope_labels

    assert has_isotope_labels("O 0 0 0\nD 0 0 1\n")
    assert not has_isotope_labels(_WATER)


@pytest.mark.parametrize(
    "content",
    [
        b"\xef\xbb\xbf" + _WATER.encode(),
        _WATER.encode("utf-16"),
        _WATER.replace("\n", "\r").encode(),
        _WATER.replace("0.117", "1.17D-01").encode(),
        _WATER.replace("-0.467", "−0.467").encode(),
        b"O 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n",
    ],
    ids=["bom", "utf16", "old-mac-cr", "fortran-d", "unicode-minus", "headerless"],
)
def test_upload_xyz_accepts_real_world_quirks(content):
    pytest.importorskip("ase")
    from quantui.structure_upload import read_uploaded_structure

    up = read_uploaded_structure("water.xyz", content)
    assert up.molecule.atoms == ["O", "H", "H"]
    assert up.molecule.coordinates[0][2] == pytest.approx(0.117)


def test_upload_xyz_isotopes_mapped_with_note():
    pytest.importorskip("ase")
    from quantui.molecule import ISOTOPE_NOTE
    from quantui.structure_upload import read_uploaded_structure

    up = read_uploaded_structure(
        "d2o.xyz", _WATER.replace("H 0 0.757", "D 0 0.757").encode()
    )
    assert up.molecule.atoms == ["O", "H", "H"]
    assert ISOTOPE_NOTE in up.notes


def test_upload_xyz_charge_comment_from_last_frame():
    pytest.importorskip("ase")
    from quantui.structure_upload import read_uploaded_structure

    content = (
        b"2\ncharge=0 multiplicity=1\nH 0 0 0\nH 0 0 0.74\n"
        b"2\ncharge=1 multiplicity=2\nH 0 0 0\nH 0 0 1.0\n"
    )
    up = read_uploaded_structure("traj.xyz", content)
    assert up.molecule.coordinates[1][2] == pytest.approx(1.0)
    assert (up.molecule.charge, up.molecule.multiplicity) == (1, 2)


def test_upload_pdb_altlocs_keep_one_conformer():
    pytest.importorskip("ase")
    from quantui.structure_upload import read_uploaded_structure

    pdb = (
        "ATOM      1  O  AHOH A   1       0.000   0.000   0.117  0.50  0.00"
        "           O\n"
        "ATOM      2  O  BHOH A   1       0.100   0.000   0.117  0.50  0.00"
        "           O\n"
        "ATOM      3  H1  HOH A   1       0.000   0.757  -0.467  1.00  0.00"
        "           H\n"
        "ATOM      4  H2  HOH A   1       0.000  -0.757  -0.467  1.00  0.00"
        "           H\n"
        "END\n"
    )
    up = read_uploaded_structure("w.pdb", pdb.encode())
    assert sorted(up.molecule.atoms) == ["H", "H", "O"]
    assert any("alternate" in n for n in up.notes)


def test_upload_overlong_filename():
    pytest.importorskip("ase")
    from quantui.structure_upload import read_uploaded_structure

    up = read_uploaded_structure("x" * 300 + ".xyz", _WATER.encode())
    assert up.molecule.atoms == ["O", "H", "H"]


# ---------------------------------------------------------------------------
# DATA-17 — smaller items
# ---------------------------------------------------------------------------


def test_package_exports():
    import quantui

    assert quantui.student_friendly_resolve.__module__ == (
        "quantui.structure_providers"
    )
    assert "run_pes_scan" in quantui.__all__
    assert "PESScanResult" in quantui.__all__


def test_get_spin_doublet_is_half():
    from quantui.molecule import Molecule

    assert Molecule(["H"], [[0, 0, 0]], multiplicity=2).get_spin() == 0.5


def test_parse_atom_list_bounds_checked_before_expanding():
    from quantui.structure_edit import parse_atom_list

    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="only 3 atoms"):
        parse_atom_list("1-30000000", 3)
    with pytest.raises(ValueError):
        parse_atom_list("1-300000000000")
    with pytest.raises(ValueError, match="start at 1"):
        parse_atom_list("0-2", 3)
    assert time.perf_counter() - t0 < 1.0
    assert parse_atom_list("3-1, 5", 5) == [0, 1, 2, 4]


def test_smiles_without_optimisation_and_atom_cap():
    pytest.importorskip("rdkit")
    from quantui.pubchem import MAX_LOCAL_EMBED_ATOMS, smiles_to_xyz

    xyz, meta = smiles_to_xyz("CCO", optimize_3d=False)
    assert xyz.splitlines()[0] == "9"
    with pytest.raises(ValueError, match=str(MAX_LOCAL_EMBED_ATOMS)):
        smiles_to_xyz("C" * 200)


def test_issue_tracker_closes_connections(tmp_path, monkeypatch):
    import sqlite3

    from quantui import issue_tracker

    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    opened = []
    real_connect = sqlite3.connect

    def _connect(*a, **k):
        conn = real_connect(*a, **k)
        opened.append(conn)
        return conn

    monkeypatch.setattr(issue_tracker.sqlite3, "connect", _connect)
    issue_tracker.log_issue("x")
    assert issue_tracker.get_issues()[0]["description"] == "x"
    issue_tracker.clear_issues()
    assert opened
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


def test_save_thumbnail_keeps_matplotlib_backend(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    from quantui.results_storage import save_thumbnail

    orig = matplotlib.get_backend()
    matplotlib.use("svg")
    try:
        save_thumbnail(tmp_path, {"calc_type": "single_point", "formula": "H2"})
        assert matplotlib.get_backend() == "svg"
        assert (tmp_path / "thumbnail.png").stat().st_size > 0
    finally:
        matplotlib.use(orig)


def test_warnings_digest_matches_whole_words():
    from quantui.log_utils import _extract_warnings

    log = "\n".join(
        [
            "conv_tol = 1e-09   error_tol = 1e-05",
            "WARN: HOMO-LUMO gap small",
            "SCF not converged.",
            "Error in DIIS",
            "UserWarning: Module EFG is under testing",
        ]
    )
    found = _extract_warnings(log)
    assert "conv_tol = 1e-09   error_tol = 1e-05" not in found
    assert len(found) == 4


def test_library_search_treats_wildcards_literally():
    from quantui import molecule_library

    assert molecule_library.search("%") == []
    assert molecule_library.search("_") == []
    assert molecule_library.search("water")
