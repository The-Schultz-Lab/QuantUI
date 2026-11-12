"""Displaced-geometry SCF convergence in the IR / Raman finite-difference
loops (audit SCF-8 / LEAD-3).

The serial loops and the parallel workers took the dipole / polarizability
of every displaced SCF without reading ``mf.converged`` and banked it to
the checkpoint, so one unconverged displacement silently corrupted the
derivative (and every resume after it). Now an unconverged displaced SCF
abandons that property with a clear status, and nothing is banked.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("pyscf")

from quantui.molecule import Molecule  # noqa: E402
from quantui.scf_robust import DisplacedSCFNotConverged  # noqa: E402


def _water() -> Molecule:
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]],
    )


@pytest.fixture
def displaced_scfs_fail(monkeypatch):
    """The reference SCF converges; every later (displaced) SCF reports
    not converged, as a real non-converging displacement would."""
    import quantui.scf_robust as sr

    real = sr.run_scf_with_rescue
    calls = {"n": 0}

    def fake(mf, *a, **kw):
        e = real(mf, *a, **kw)
        calls["n"] += 1
        if calls["n"] > 1:
            mf.converged = False
            mf.scf_rescue_stage = sr.SCF_RESCUE_FAILED
        return e

    monkeypatch.setattr(sr, "run_scf_with_rescue", fake)
    monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "0")
    return calls


class _RecordingCheckpoint:
    """Just the checkpoint surface run_freq_calc / Raman touch."""

    def __init__(self, tmp_path):
        self.root = tmp_path
        self.banked: list = []

    def completed_items(self, name):
        return {}

    def log_resumed(self, msg):
        pass

    def items_dir(self, name):
        return self.root / name

    def mark_item_done(self, name, item_id, payload):
        self.banked.append((name, item_id))

    def mark_complete(self):
        pass


class TestSerialLoops:
    def test_ir_and_raman_abandoned_not_banked(self, displaced_scfs_fail, tmp_path):
        from quantui.freq_calc import run_freq_calc

        log = io.StringIO()
        ckpt = _RecordingCheckpoint(tmp_path)
        r = run_freq_calc(
            _water(), "RHF", "STO-3G", progress_stream=log, checkpoint=ckpt
        )
        # Frequencies (analytic Hessian) are unaffected ...
        assert len(r.frequencies_cm1) == 3
        # ... but the derivative-based properties are marked unavailable.
        assert r.ir_intensities == []
        assert r.raman_activities == []
        text = log.getvalue()
        assert "IR intensities unavailable" in text
        assert "Raman activities unavailable" in text
        assert "did not converge" in text
        assert ckpt.banked == []

    def test_converged_run_still_has_intensities(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "0")
        from quantui.freq_calc import run_freq_calc

        r = run_freq_calc(_water(), "RHF", "STO-3G", progress_stream=io.StringIO())
        assert len(r.ir_intensities) == 3


class _InlineExecutor:
    """ProcessPoolExecutor stand-in that runs the workers in-process."""

    created = 0

    def __init__(
        self, max_workers=None, mp_context=None, initializer=None, initargs=()
    ):
        type(self).created += 1
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        import concurrent.futures as cf

        fut: cf.Future = cf.Future()
        try:
            fut.set_result(fn(*args))
        except BaseException as exc:  # noqa: BLE001 — mirror a real Future
            fut.set_exception(exc)
        return fut


class TestParallelPath:
    def test_parallel_failure_is_not_retried_serially(
        self, displaced_scfs_fail, monkeypatch
    ):
        import concurrent.futures as cf

        from quantui import freq_ir_workers
        from quantui.freq_calc import run_freq_calc

        for v in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYSCF_NUM_THREADS",
        ):
            monkeypatch.setenv(v, "1")
        monkeypatch.setattr(
            freq_ir_workers, "parallel_enabled_for_run", lambda **_: True
        )
        monkeypatch.setattr(cf, "ProcessPoolExecutor", _InlineExecutor)
        _InlineExecutor.created = 0
        log = io.StringIO()
        r = run_freq_calc(_water(), "RHF", "STO-3G", progress_stream=log)
        assert r.ir_intensities == []
        assert r.raman_activities == []
        text = log.getvalue()
        assert "IR intensities unavailable" in text
        assert "Raman activities unavailable" in text
        assert "falling back to serial" not in text
        assert _InlineExecutor.created == 2  # the IR and Raman pools both ran


class TestWorkers:
    """The parallel workers, run in-process."""

    def _dm0_file(self, tmp_path):
        import pickle

        from pyscf import gto, scf

        mol = gto.M(atom=_water().to_pyscf_format(), basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        path = tmp_path / "dm0.pkl"
        path.write_bytes(pickle.dumps(mf.make_rdm1()))
        return mol, str(path)

    @pytest.fixture(autouse=True)
    def _threads(self, monkeypatch):
        for v in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYSCF_NUM_THREADS",
        ):
            monkeypatch.setenv(v, "1")

    @pytest.fixture
    def never_converges(self, monkeypatch):
        import quantui.scf_robust as sr

        real = sr.run_scf_with_rescue

        def fake(mf, *a, **kw):
            e = real(mf, *a, **kw)
            mf.converged = False
            return e

        monkeypatch.setattr(sr, "run_scf_with_rescue", fake)

    def test_ir_worker_raises_and_does_not_bank(self, tmp_path, never_converges):
        from quantui import freq_ir_workers as w

        mol, dm0 = self._dm0_file(tmp_path)
        items = tmp_path / "items"
        w.init_worker(
            _water().to_pyscf_format(), "STO-3G", 0, 0, None, dm0, 1, str(items)
        )
        coords = mol.atom_coords().flatten().tolist()
        with pytest.raises(DisplacedSCFNotConverged, match="d000_x_\\+"):
            w.run_displaced_scf("d000_x_+", coords)
        assert not items.exists() or not any(items.rglob("*"))

    def test_raman_worker_raises_and_does_not_bank(self, tmp_path, never_converges):
        pytest.importorskip("pyscf.prop.polarizability")
        from quantui import freq_raman_workers as w

        mol, dm0 = self._dm0_file(tmp_path)
        items = tmp_path / "items"
        w.init_raman_worker(
            _water().to_pyscf_format(),
            "STO-3G",
            0,
            0,
            None,
            dm0,
            1,
            False,
            False,
            str(items),
        )
        coords = mol.atom_coords().flatten().tolist()
        with pytest.raises(DisplacedSCFNotConverged):
            w.run_displaced_polarizability("d000_x_+", coords)
        assert not items.exists() or not any(items.rglob("*"))
