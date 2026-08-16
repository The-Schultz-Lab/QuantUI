"""M-CHECKPOINT CHK.7 — Reorganization Energy checkpointing.

Reorganization Energy is really 2-3 independent geometry optimizations (the
neutral reference plus one per ion channel) run under a single Calculate-tab
"calculation". A single shared checkpoint would let one leg's trajectory and
BFGS Hessian overwrite another's, so each leg needs its own resume identity —
that's what ``Checkpoint.sub()`` exists for.

No PySCF/ASE here: ``optimize_geometry``/``run_in_session`` are monkeypatched
with fakes so the wiring is tested directly, mirroring
``tests/test_checkpoint_wiring.py``'s no-SCF style.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from quantui import checkpoint as C
from quantui import reorganization_energy as R
from quantui.molecule import Molecule

# ══ Fixtures ═════════════════════════════════════════════════════════════════


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("QUANTUI_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    return tmp_path / "ckpt"


def _identity(**overrides) -> C.CalcIdentity:
    base = dict(
        calc_type="reorganization_energy",
        method="B3LYP",
        basis="6-31G*",
        charge=0,
        multiplicity=1,
        atom_symbols=("O", "H", "H"),
        coords=((0.0, 0.0, 0.0), (0.76, 0.59, 0.0), (-0.76, 0.59, 0.0)),
    )
    base.update(overrides)
    return C.CalcIdentity(**base)


def _neutral_molecule() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]],
        charge=0,
        multiplicity=1,
    )


# ══ Checkpoint.sub() — nested leg identity ══════════════════════════════════


class TestCheckpointSub:
    def test_a_leg_has_its_own_resume_key(self, root):
        parent = C.Checkpoint(_identity())
        leg = parent.sub(
            "neutral_opt", charge=0, multiplicity=1, coords=[[0.0, 0.0, 0.0]]
        )
        assert leg.identity.resume_key != parent.identity.resume_key

    def test_two_different_tags_never_collide(self, root):
        parent = C.Checkpoint(_identity())
        hole = parent.sub("hole_opt", charge=1, multiplicity=2, coords=[[0, 0, 0]])
        electron = parent.sub(
            "electron_opt", charge=-1, multiplicity=2, coords=[[0, 0, 0]]
        )
        assert hole.identity.resume_key != electron.identity.resume_key

    def test_the_same_tag_and_inputs_give_the_same_key(self, root):
        parent = C.Checkpoint(_identity())
        a = parent.sub("neutral_opt", charge=0, multiplicity=1, coords=[[0, 0, 0]])
        b = parent.sub("neutral_opt", charge=0, multiplicity=1, coords=[[0, 0, 0]])
        assert a.identity.resume_key == b.identity.resume_key

    def test_a_leg_nests_under_the_parent_directory(self, root):
        parent = C.Checkpoint(_identity())
        leg = parent.sub("neutral_opt", charge=0, multiplicity=1, coords=[[0, 0, 0]])
        assert leg.dir.parent.parent == parent.dir
        assert leg.dir.parent.name == "legs"

    def test_method_basis_and_atoms_are_inherited_from_the_parent(self, root):
        parent = C.Checkpoint(_identity(method="B3LYP", basis="6-31G*"))
        leg = parent.sub("hole_opt", charge=1, multiplicity=2, coords=[[0, 0, 0]])
        assert leg.identity.method == "B3LYP"
        assert leg.identity.basis == "6-31G*"
        assert leg.identity.atom_symbols == ("O", "H", "H")

    def test_a_leg_writes_a_real_usable_checkpoint(self, root):
        parent = C.Checkpoint(_identity())
        leg = parent.sub("neutral_opt", charge=0, multiplicity=1, coords=[[0, 0, 0]])
        assert leg.begin() is True
        assert leg.exists()


class TestParentProgressReflectsLegs:
    """The parent checkpoint never writes a trajectory of its own — all the
    real state lives in its legs. Without _has_leg_progress, the parent would
    always look empty and the run would never be offered as resumable."""

    def test_no_progress_when_no_leg_has_run(self, root):
        parent = C.Checkpoint(_identity())
        parent.begin()
        assert parent.has_progress() is False

    def test_progress_appears_once_a_leg_has_a_trajectory(self, root):
        parent = C.Checkpoint(_identity())
        parent.begin()
        leg = parent.sub("neutral_opt", charge=0, multiplicity=1, coords=[[0, 0, 0]])
        leg.begin()
        leg.trajectory_path.write_bytes(b"not empty")
        assert parent.has_progress() is True

    def test_an_empty_leg_trajectory_is_not_progress(self, root):
        parent = C.Checkpoint(_identity())
        parent.begin()
        leg = parent.sub("neutral_opt", charge=0, multiplicity=1, coords=[[0, 0, 0]])
        leg.begin()
        leg.trajectory_path.write_bytes(b"")
        assert parent.has_progress() is False

    def test_a_leg_with_no_begin_never_created_a_directory(self, root):
        # begin() never called -> no legs/ dir at all -> must not raise.
        parent = C.Checkpoint(_identity())
        parent.begin()
        assert parent.has_progress() is False


# ══ run_reorganization_energy wiring ════════════════════════════════════════


class _FakeOptResult:
    def __init__(self, molecule, n_steps=3, converged=True):
        self.molecule = molecule
        self.n_steps = n_steps
        self.converged = converged


def _make_optimize_geometry(calls: list):
    """A fake optimize_geometry that records every call's checkpoint/resume
    and returns the input geometry unchanged (converged)."""

    def _fake(*, molecule, checkpoint=None, resume=False, **kw):
        calls.append(
            {
                "molecule": molecule,
                "checkpoint": checkpoint,
                "resume": resume,
                "status_label": kw.get("status_label"),
            }
        )
        return _FakeOptResult(molecule)

    return _fake


def _fake_run_in_session(*, molecule, method, basis, **kw):
    return SimpleNamespace(converged=True, energy_hartree=-1.0)


class TestReorgCheckpointWiring:
    def test_without_a_checkpoint_every_leg_runs_uncheckpointed(
        self, root, monkeypatch
    ):
        calls: list = []
        monkeypatch.setattr(R, "optimize_geometry", _make_optimize_geometry(calls))
        monkeypatch.setattr(R, "run_in_session", _fake_run_in_session)

        R.run_reorganization_energy(
            _neutral_molecule(), mode="both", method="B3LYP", basis="6-31G*"
        )

        assert len(calls) == 3  # neutral + hole + electron
        assert all(c["checkpoint"] is None for c in calls)
        assert all(c["resume"] is False for c in calls)

    def test_each_leg_gets_its_own_begun_checkpoint(self, root, monkeypatch):
        calls: list = []
        monkeypatch.setattr(R, "optimize_geometry", _make_optimize_geometry(calls))
        monkeypatch.setattr(R, "run_in_session", _fake_run_in_session)

        parent = C.Checkpoint(_identity())
        R.run_reorganization_energy(
            _neutral_molecule(),
            mode="both",
            method="B3LYP",
            basis="6-31G*",
            checkpoint=parent,
        )

        assert len(calls) == 3
        legs = [c["checkpoint"] for c in calls]
        assert all(leg is not None for leg in legs)
        assert all(leg.exists() for leg in legs)
        # No two legs share a resume key.
        keys = {leg.identity.resume_key for leg in legs}
        assert len(keys) == 3

    def test_leg_calc_type_tags_are_distinct_and_prefixed_by_the_parent(
        self, root, monkeypatch
    ):
        calls: list = []
        monkeypatch.setattr(R, "optimize_geometry", _make_optimize_geometry(calls))
        monkeypatch.setattr(R, "run_in_session", _fake_run_in_session)

        parent = C.Checkpoint(_identity(calc_type="reorganization_energy"))
        R.run_reorganization_energy(
            _neutral_molecule(),
            mode="both",
            method="B3LYP",
            basis="6-31G*",
            checkpoint=parent,
        )

        tags = {c["checkpoint"].identity.calc_type for c in calls}
        assert tags == {
            "reorganization_energy:neutral_opt",
            "reorganization_energy:hole_opt",
            "reorganization_energy:electron_opt",
        }

    def test_the_parent_checkpoint_is_marked_complete_on_success(
        self, root, monkeypatch
    ):
        # Without this, a fully successful run would linger forever in the
        # "unfinished calculations" listing — resumable_state() only excludes
        # STATUS_COMPLETE, and nothing else ever marks the parent done.
        calls: list = []
        monkeypatch.setattr(R, "optimize_geometry", _make_optimize_geometry(calls))
        monkeypatch.setattr(R, "run_in_session", _fake_run_in_session)

        parent = C.Checkpoint(_identity())
        parent.begin()
        R.run_reorganization_energy(
            _neutral_molecule(),
            mode="hole",
            method="B3LYP",
            basis="6-31G*",
            checkpoint=parent,
        )

        assert parent.load_state()["status"] == C.STATUS_COMPLETE
        assert parent.resumable_state() is None

    def test_the_parent_is_not_marked_complete_when_a_single_point_fails(
        self, root, monkeypatch
    ):
        calls: list = []
        monkeypatch.setattr(R, "optimize_geometry", _make_optimize_geometry(calls))

        def _failing_run_in_session(*, molecule, method, basis, **kw):
            return SimpleNamespace(converged=False, energy_hartree=0.0)

        monkeypatch.setattr(R, "run_in_session", _failing_run_in_session)

        parent = C.Checkpoint(_identity())
        parent.begin()
        with pytest.raises(RuntimeError):
            R.run_reorganization_energy(
                _neutral_molecule(),
                mode="hole",
                method="B3LYP",
                basis="6-31G*",
                checkpoint=parent,
            )

        assert parent.load_state()["status"] != C.STATUS_COMPLETE

    def test_hole_only_mode_only_checkpoints_two_legs(self, root, monkeypatch):
        calls: list = []
        monkeypatch.setattr(R, "optimize_geometry", _make_optimize_geometry(calls))
        monkeypatch.setattr(R, "run_in_session", _fake_run_in_session)

        parent = C.Checkpoint(_identity())
        R.run_reorganization_energy(
            _neutral_molecule(),
            mode="hole",
            method="B3LYP",
            basis="6-31G*",
            checkpoint=parent,
        )

        tags = {c["checkpoint"].identity.calc_type for c in calls}
        assert tags == {
            "reorganization_energy:neutral_opt",
            "reorganization_energy:hole_opt",
        }

    def test_resume_is_threaded_to_every_leg(self, root, monkeypatch):
        calls: list = []
        monkeypatch.setattr(R, "optimize_geometry", _make_optimize_geometry(calls))
        monkeypatch.setattr(R, "run_in_session", _fake_run_in_session)

        parent = C.Checkpoint(_identity())
        R.run_reorganization_energy(
            _neutral_molecule(),
            mode="both",
            method="B3LYP",
            basis="6-31G*",
            checkpoint=parent,
            resume=True,
        )

        assert all(c["resume"] is True for c in calls)

    def test_a_leg_that_fails_to_open_runs_uncheckpointed_not_broken(
        self, root, monkeypatch
    ):
        # A checkpoint is an optimisation for the failure case, never the
        # reason a leg doesn't run — mirrors Checkpoint.begin()'s own contract.
        calls: list = []
        monkeypatch.setattr(R, "optimize_geometry", _make_optimize_geometry(calls))
        monkeypatch.setattr(R, "run_in_session", _fake_run_in_session)
        monkeypatch.setattr(C.Checkpoint, "begin", lambda self, **kw: False)

        parent = C.Checkpoint(_identity())
        result = R.run_reorganization_energy(
            _neutral_molecule(),
            mode="hole",
            method="B3LYP",
            basis="6-31G*",
            checkpoint=parent,
        )

        assert result.converged
        assert all(c["checkpoint"] is None for c in calls)


# ══ calc_type_key regression (found while implementing CHK.7) ══════════════


class TestReorgCalcTypeKey:
    """``_CALC_TYPE_KEYS`` (app_runflow.py) is the label->key map that feeds
    both the checkpoint identity and the runtime estimator. It listed every
    calc type except Reorganization Energy, which silently fell back to
    "single_point" — colliding a reorg run's checkpoint identity with an
    actual single-point run on the same molecule/method/basis, and feeding
    the estimator single-point history for a run that is 2-3 full geometry
    optimizations. Fixed alongside CHK.7 since the leg tags
    (``f"{parent.identity.calc_type}:{tag}"``) are meaningless if the parent
    key itself is wrong.
    """

    def test_the_dropdown_label_maps_to_the_canonical_key(self):
        from quantui import app_runflow

        app = SimpleNamespace(
            calc_type_dd=SimpleNamespace(value="Reorganization Energy")
        )
        assert app_runflow.calc_type_key(app) == "reorganization_energy"
