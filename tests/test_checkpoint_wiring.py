"""Checkpoint wiring into the calculation modules and the app — M-CHECKPOINT.

``tests/test_checkpoint.py`` covers the storage layer in isolation. These tests
cover the seams, which is where this feature can fail quietly: a checkpoint
that is created but never passed to the calculation, a resume offer that is
built but never added to the layout, or a warm start that silently reuses a
density from the wrong basis set.

Two of those have precedent in this project — a panel built, registered, and
never added to its container, caught only by a screenshot — so the layout and
plumbing assertions here are structural rather than behavioural on purpose.

Platform-independent: no PySCF, no ASE.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from quantui import app_runflow, optimizer, pes_scan, session_calc
from quantui import checkpoint as C


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("QUANTUI_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    return tmp_path / "ckpt"


class _FakeMolecule:
    def __init__(self, atoms=("O", "H", "H"), coords=None, charge=0, multiplicity=1):
        self.atoms = list(atoms)
        self.coordinates = coords or [
            [0.0, 0.0, 0.0],
            [0.76, 0.59, 0.0],
            [-0.76, 0.59, 0.0],
        ]
        self.charge = charge
        self.multiplicity = multiplicity


class _FakeWidget:
    """Minimal stand-in for the ipywidgets used by the resume notice."""

    def __init__(self, value=None):
        self.value = value
        self.layout = type("_L", (), {"display": ""})()


class _FakeApp:
    def __init__(self, molecule=None):
        self._molecule = molecule
        self.method_dd = _FakeWidget("RHF")
        self.basis_dd = _FakeWidget("6-31G")
        self.calc_type_dd = _FakeWidget("Geometry Opt")
        self._resume_notice_html = _FakeWidget("")
        self._resume_cb = _FakeWidget(True)


# ══ The calc modules accept and use a checkpoint ═════════════════════════════


class TestCalcModuleSignatures:
    """A checkpoint that never reaches the calculation saves nothing."""

    @pytest.mark.parametrize(
        "func,params",
        [
            (optimizer.optimize_geometry, ("checkpoint", "resume")),
            (pes_scan.run_pes_scan, ("checkpoint", "resume")),
            (session_calc.run_in_session, ("checkpoint", "warm_start")),
        ],
    )
    def test_accepts_checkpoint_parameters(self, func, params):
        signature = inspect.signature(func)
        for name in params:
            assert name in signature.parameters

    @pytest.mark.parametrize(
        "func,name,default",
        [
            (optimizer.optimize_geometry, "checkpoint", None),
            (optimizer.optimize_geometry, "resume", False),
            (pes_scan.run_pes_scan, "checkpoint", None),
            (pes_scan.run_pes_scan, "resume", False),
            (session_calc.run_in_session, "checkpoint", None),
        ],
    )
    def test_defaults_preserve_the_previous_behaviour(self, func, name, default):
        """Callers that know nothing about checkpoints must be unaffected."""
        assert inspect.signature(func).parameters[name].default == default

    def test_warm_start_is_on_by_default(self):
        """Reusing a converged density is free; opting out is the special case."""
        assert (
            inspect.signature(session_calc.run_in_session)
            .parameters["warm_start"]
            .default
            is True
        )

    def test_app_passes_the_checkpoint_to_every_long_calc(self):
        """The three calc types that can be interrupted must all receive one."""
        import quantui.app as A

        src = Path(A.__file__).read_text(encoding="utf-8")
        assert src.count("checkpoint=_ckpt") >= 3

    def test_app_passes_resume_to_the_resumable_calc_types(self):
        import quantui.app as A

        src = Path(A.__file__).read_text(encoding="utf-8")
        assert src.count("resume=_resume") >= 2


# ══ Warm-start selection ═════════════════════════════════════════════════════


class TestPrepareScfCheckpoint:
    """``_prepare_scf_checkpoint`` decides which density seeds the SCF."""

    class _FakeMf:
        def __init__(self, dm="density"):
            self.chkfile = None
            self._dm = dm

        def from_chk(self, path):
            return self._dm

    def _call(self, mf, molecule=None, **kwargs):
        params = {
            "molecule": molecule or _FakeMolecule(),
            "method": "RHF",
            "basis": "6-31G",
            "checkpoint": None,
            "warm_start": True,
        }
        params.update(kwargs)
        return session_calc._prepare_scf_checkpoint(mf, **params)

    def _seed_chkfile(self, molecule, *, method="RHF", basis="6-31G"):
        identity = C.CalcIdentity.from_molecule(
            molecule, calc_type="single_point", method=method, basis=basis
        )
        ckpt = C.Checkpoint(identity)
        ckpt.begin()
        ckpt.scf_chkfile.write_bytes(b"density")
        return ckpt

    def test_points_pyscf_at_the_checkpoint_chkfile(self, root):
        mf = self._FakeMf()
        ckpt = C.Checkpoint(
            C.CalcIdentity.from_molecule(
                _FakeMolecule(), calc_type="single_point", method="RHF", basis="6-31G"
            )
        )
        self._call(mf, checkpoint=ckpt)
        assert mf.chkfile == str(ckpt.scf_chkfile)

    def test_returns_none_without_history(self, root):
        assert self._call(self._FakeMf()) is None

    def test_returns_a_density_when_a_match_exists(self, root):
        molecule = _FakeMolecule()
        self._seed_chkfile(molecule)
        assert self._call(self._FakeMf(), molecule=molecule) == "density"

    def test_warm_start_disabled_returns_none(self, root):
        molecule = _FakeMolecule()
        self._seed_chkfile(molecule)
        assert self._call(self._FakeMf(), molecule=molecule, warm_start=False) is None

    def test_does_not_reuse_the_file_it_is_about_to_overwrite(self, root):
        """That file is this run's previous attempt — possibly a partial write."""
        molecule = _FakeMolecule()
        ckpt = self._seed_chkfile(molecule)
        assert self._call(self._FakeMf(), molecule=molecule, checkpoint=ckpt) is None

    def test_a_different_basis_is_not_reused(self, root):
        """A density in one basis is not a density in another."""
        molecule = _FakeMolecule()
        self._seed_chkfile(molecule, basis="STO-3G")
        assert self._call(self._FakeMf(), molecule=molecule, basis="cc-pVDZ") is None

    def test_a_failing_from_chk_degrades_to_no_warm_start(self, root):
        """A bad guess must cost the warm start, not the calculation."""
        molecule = _FakeMolecule()
        self._seed_chkfile(molecule)

        class _Exploding(self._FakeMf):
            def from_chk(self, path):
                raise RuntimeError("basis mismatch")

        assert self._call(_Exploding(), molecule=molecule) is None

    def test_an_object_without_from_chk_degrades_to_no_warm_start(self, root):
        """gpu4pyscf-migrated objects may not expose the PySCF chkfile API."""
        molecule = _FakeMolecule()
        self._seed_chkfile(molecule)

        class _Bare:
            chkfile = None

        assert self._call(_Bare(), molecule=molecule) is None

    def test_an_unsettable_chkfile_attribute_is_not_fatal(self, root):
        class _Frozen:
            __slots__ = ()

            def from_chk(self, path):
                return None

        self._call(_Frozen())  # must not raise


# ══ Scan-point reuse ═════════════════════════════════════════════════════════


class TestReuseScanPoint:
    class _FakeAtoms:
        def __init__(self):
            self.positions = None

        def set_positions(self, coords):
            self.positions = coords

    def _record(self, **overrides):
        base = {
            "index": 1,
            "value": 1.25,
            "energy_hartree": -1.5,
            "ok": True,
            "atoms": ["H", "H"],
            "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.25]],
        }
        base.update(overrides)
        return base

    def _call(self, record, value=1.25):
        atoms = self._FakeAtoms()
        result = pes_scan._reuse_scan_point(record, value, atoms, _FakeMolecule())
        return result, atoms

    def test_reuses_a_matching_point(self):
        result, _ = self._call(self._record())
        assert result is not None
        assert result[0] == -1.5

    def test_moves_the_live_geometry_onto_the_stored_one(self):
        """Each point relaxes from where the last one finished.

        Skipping a point without moving the atoms would start the next
        computed point from the wrong geometry and quietly change the profile.
        """
        _, atoms = self._call(self._record())
        assert atoms.positions == [[0.0, 0.0, 0.0], [0.0, 0.0, 1.25]]

    def test_rejects_a_point_computed_at_a_different_value(self):
        """This is what makes the cache self-validating when the grid changes."""
        result, _ = self._call(self._record(value=1.25), value=1.75)
        assert result is None

    def test_rejects_a_missing_record(self):
        assert self._call(None)[0] is None

    def test_rejects_a_record_with_no_energy(self):
        record = self._record()
        del record["energy_hartree"]
        assert self._call(record)[0] is None

    def test_rejects_mismatched_atoms_and_coordinates(self):
        assert self._call(self._record(atoms=["H", "H", "H"]))[0] is None

    def test_rejects_an_empty_geometry(self):
        assert self._call(self._record(atoms=[], coordinates=[]))[0] is None

    def test_carries_charge_and_multiplicity_from_the_live_molecule(self):
        atoms = self._FakeAtoms()
        molecule = _FakeMolecule(charge=-1, multiplicity=2)
        result = pes_scan._reuse_scan_point(self._record(), 1.25, atoms, molecule)
        assert result[1].charge == -1
        assert result[1].multiplicity == 2


# ══ Resume offer ═════════════════════════════════════════════════════════════


class TestResumeNotice:
    def _seed(self, app, *, points=0, steps=None, total=None):
        identity = app_runflow.checkpoint_identity(app)
        ckpt = C.Checkpoint(identity)
        ckpt.begin(**({"total_points": total} if total else {}))
        for i in range(points):
            ckpt.append_point({"index": i + 1, "value": float(i), "ok": True})
        if steps is not None:
            ckpt.trajectory_path.write_bytes(b"frames")
            ckpt.update(steps_done=steps)
        return ckpt

    def test_hidden_when_no_checkpoint_exists(self, root):
        app = _FakeApp(_FakeMolecule())
        app_runflow.refresh_resume_notice(app)
        assert app._resume_notice_html.layout.display == "none"
        assert app._resume_cb.layout.display == "none"

    def test_hidden_without_a_molecule(self, root):
        app = _FakeApp(None)
        app_runflow.refresh_resume_notice(app)
        assert app._resume_cb.layout.display == "none"

    def test_hidden_when_the_checkpoint_has_no_progress(self, root):
        """A bare directory is not a saving worth offering."""
        app = _FakeApp(_FakeMolecule())
        self._seed(app)
        app_runflow.refresh_resume_notice(app)
        assert app._resume_cb.layout.display == "none"

    def test_shown_when_there_is_stored_progress(self, root):
        app = _FakeApp(_FakeMolecule())
        self._seed(app, steps=6)
        app_runflow.refresh_resume_notice(app)
        assert app._resume_cb.layout.display == "block"
        assert "interrupted run" in app._resume_notice_html.value

    def test_reports_how_many_optimizer_steps_are_banked(self, root):
        app = _FakeApp(_FakeMolecule())
        self._seed(app, steps=6)
        app_runflow.refresh_resume_notice(app)
        assert "6 optimizer steps" in app._resume_notice_html.value

    def test_reports_scan_points_out_of_the_total(self, root):
        app = _FakeApp(_FakeMolecule())
        app.calc_type_dd.value = "PES Scan"
        self._seed(app, points=8, total=20)
        app_runflow.refresh_resume_notice(app)
        assert "8 of 20 scan points" in app._resume_notice_html.value

    def test_singular_wording_for_one_point(self, root):
        app = _FakeApp(_FakeMolecule())
        app.calc_type_dd.value = "PES Scan"
        self._seed(app, points=1)
        app_runflow.refresh_resume_notice(app)
        assert "1 scan point already" in app._resume_notice_html.value

    def test_hidden_after_the_run_completes(self, root):
        app = _FakeApp(_FakeMolecule())
        ckpt = self._seed(app, steps=6)
        ckpt.mark_complete()
        app_runflow.refresh_resume_notice(app)
        assert app._resume_cb.layout.display == "none"

    def test_hidden_when_the_configured_calc_type_changes(self, root):
        """The offer must never describe a calculation the user moved on from."""
        app = _FakeApp(_FakeMolecule())
        self._seed(app, steps=6)
        app.calc_type_dd.value = "Single Point"
        app_runflow.refresh_resume_notice(app)
        assert app._resume_cb.layout.display == "none"

    def test_hidden_when_the_basis_changes(self, root):
        app = _FakeApp(_FakeMolecule())
        self._seed(app, steps=6)
        app.basis_dd.value = "cc-pVDZ"
        app_runflow.refresh_resume_notice(app)
        assert app._resume_cb.layout.display == "none"

    def test_a_broken_checkpoint_hides_the_offer_rather_than_raising(self, root):
        app = _FakeApp(_FakeMolecule())
        ckpt = self._seed(app, steps=6)
        ckpt.meta_path.write_text("{ truncated", encoding="utf-8")
        app_runflow.refresh_resume_notice(app)
        assert app._resume_cb.layout.display == "none"

    def test_missing_widgets_are_tolerated(self, root):
        """Called from update_estimate, which runs before the run panel exists."""

        class _Early:
            _molecule = None

        app_runflow.refresh_resume_notice(_Early())  # must not raise


class TestCheckpointIdentity:
    def test_uses_the_configured_method_and_basis(self, root):
        app = _FakeApp(_FakeMolecule())
        identity = app_runflow.checkpoint_identity(app)
        assert identity.method == "RHF"
        assert identity.basis == "6-31G"

    def test_maps_the_dropdown_label_to_the_canonical_key(self, root):
        app = _FakeApp(_FakeMolecule())
        app.calc_type_dd.value = "UV-Vis (TD-DFT)"
        assert app_runflow.checkpoint_identity(app).calc_type == "tddft"

    def test_none_without_a_molecule(self, root):
        assert app_runflow.checkpoint_identity(_FakeApp(None)) is None

    def test_calc_type_key_falls_back_to_single_point(self):
        app = _FakeApp(_FakeMolecule())
        app.calc_type_dd.value = "Something New"
        assert app_runflow.calc_type_key(app) == "single_point"


class TestResumeIsOnlyRequestedWhenThereIsProgress:
    """The checkbox defaults to ticked and hides when nothing is resumable.

    So consulting it alone asks *every* ordinary run to resume, and the
    optimizer answers with a "no usable checkpoint" warning on a calculation
    the user started from scratch. The run must gate on real stored progress
    as well as on the checkbox.
    """

    def test_run_gates_resume_on_stored_progress(self):
        import quantui.app as A

        src = Path(A.__file__).read_text(encoding="utf-8")
        assert "_checkpoint_resumable" in src
        start = src.index("_resume = bool(")
        clause = src[start : start + 260]
        assert "_resume_cb.value" in clause
        assert "_checkpoint_resumable" in clause

    def test_resumability_is_read_before_begin_rewrites_the_metadata(self):
        """begin() stamps a fresh "running" status over the interrupted one.

        Asking afterwards would describe the run about to start rather than
        the one that stopped, and every resume offer would evaluate false.
        """
        import quantui.app as A

        src = Path(A.__file__).read_text(encoding="utf-8")
        body = src[src.index("def _begin_run_checkpoint") :]
        body = body[: body.index("\n    def ", 1)]
        assert body.index("_checkpoint_resumable = ") < body.index("ckpt.begin(")


class TestCalcTypeChangeRefreshesTheOffer:
    """A resume offer must never describe a calculation the user left behind.

    ``refresh_resume_notice`` hides the offer when the calc type no longer
    matches — but only if something calls it. The calc-type observer did not,
    so the function was correct and unreachable on the one change most likely
    to invalidate the offer.
    """

    def test_calc_type_handler_refreshes_the_estimate_and_offer(self):
        src = Path(app_runflow.__file__).read_text(encoding="utf-8")
        handler = src[src.index("def on_calc_type_changed") :]
        handler = handler[: handler.index("\ndef ", 1)]
        assert "update_estimate(" in handler

    def test_update_estimate_refreshes_the_offer(self):
        """The single hook the offer hangs off — it must not be removed."""
        src = Path(app_runflow.__file__).read_text(encoding="utf-8")
        body = src[src.index("def update_estimate") :]
        body = body[: body.index("\ndef ", 1)]
        assert "refresh_resume_notice(app)" in body


class TestFailureCardPointsAtTheResume:
    """The offer sits by the Run button, which is not where anyone looks
    after a failure. Without a pointer there, the feature is undiscoverable
    in exactly the situation it was built for."""

    def _ckpt_with_progress(self, points=0, steps=False):
        identity = C.CalcIdentity.from_molecule(
            _FakeMolecule(), calc_type="pes_scan", method="RHF", basis="6-31G"
        )
        ckpt = C.Checkpoint(identity)
        ckpt.begin()
        for i in range(points):
            ckpt.append_point({"index": i + 1, "value": float(i), "ok": True})
        if steps:
            ckpt.trajectory_path.write_bytes(b"frames")
        return ckpt

    def _hint(self, ckpt):
        import quantui.app as A

        return A.QuantUIApp._resume_hint_html(object(), ckpt)

    def test_no_hint_without_a_checkpoint(self, root):
        assert self._hint(None) == ""

    def test_no_hint_when_nothing_was_saved(self, root):
        """A run that died in its first seconds has nothing to offer."""
        assert self._hint(self._ckpt_with_progress()) == ""

    def test_hint_names_the_saved_scan_points(self, root):
        hint = self._hint(self._ckpt_with_progress(points=8))
        assert "8 completed scan points" in hint
        assert "Resume from checkpoint" in hint

    def test_hint_is_singular_for_one_point(self, root):
        assert "1 completed scan point " in self._hint(
            self._ckpt_with_progress(points=1)
        )

    def test_hint_covers_optimizer_progress_without_points(self, root):
        hint = self._hint(self._ckpt_with_progress(steps=True))
        assert "steps completed so far" in hint

    def test_a_broken_checkpoint_yields_no_hint_rather_than_raising(self, root):
        """A hint must never mask the error it is printed beside."""
        ckpt = self._ckpt_with_progress(points=3)
        ckpt.meta_path.write_text("{ truncated", encoding="utf-8")
        assert self._hint(ckpt) == ""

    def test_the_failure_card_includes_the_hint(self):
        import quantui.app as A

        src = Path(A.__file__).read_text(encoding="utf-8")
        assert "_resume_hint_html(_ckpt)" in src


class TestHelpTopic:
    """Docs users can reach without leaving the app."""

    def test_a_resume_help_topic_exists(self):
        from quantui.help_content import HELP_TOPICS

        assert "resuming_calculations" in HELP_TOPICS

    def test_it_explains_that_settings_must_match(self):
        """The most likely confusion: "why has the offer disappeared?"."""
        from quantui.help_content import HELP_TOPICS

        body = HELP_TOPICS["resuming_calculations"]["body"]
        assert "identical" in body

    def test_it_names_the_control_the_user_has_to_find(self):
        from quantui.help_content import HELP_TOPICS

        assert "Resume from checkpoint" in HELP_TOPICS["resuming_calculations"]["body"]

    def test_it_states_which_calc_types_are_resumable(self):
        from quantui.help_content import HELP_TOPICS

        body = HELP_TOPICS["resuming_calculations"]["body"]
        for calc_type in ("Geometry Opt", "PES Scan", "Frequency"):
            assert calc_type in body

    def test_it_covers_finding_unfinished_work_after_a_restart(self):
        """The same-session path is easy; the cross-session one is the
        question users will actually have."""
        from quantui.help_content import HELP_TOPICS

        body = HELP_TOPICS["resuming_calculations"]["body"]
        assert "Unfinished calculations" in body
        assert "Load these settings" in body

    def test_it_says_where_checkpoints_live_and_that_deleting_is_safe(self):
        from quantui.help_content import HELP_TOPICS

        body = HELP_TOPICS["resuming_calculations"]["body"]
        assert "~/.quantui/checkpoints" in body
        assert "safe" in body


class _FakeListApp(_FakeApp):
    """A fake app carrying the CHK.6 listing widgets as well."""

    def __init__(self, molecule=None):
        super().__init__(molecule)
        self._resume_list_box = _FakeWidget()
        self._resume_list_dd = _FakeWidget("")
        self._resume_list_html = _FakeWidget("")
        self._resume_restore_btn = _FakeWidget()
        self._resume_restore_btn.disabled = False
        self.charge_si = _FakeWidget(0)
        self.mult_si = _FakeWidget(1)
        self.set_molecule_calls = []

    def _set_molecule(self, mol, label=""):
        self._molecule = mol
        self.set_molecule_calls.append(label)


def _seed_checkpoint(calc_type="geometry_opt", *, points=0, steps=None, molecule=None):
    identity = C.CalcIdentity.from_molecule(
        molecule or _FakeMolecule(),
        calc_type=calc_type,
        method="RHF",
        basis="6-31G",
    )
    ckpt = C.Checkpoint(identity)
    ckpt.begin()
    for i in range(points):
        ckpt.append_point({"index": i + 1, "value": float(i), "ok": True})
    if steps is not None:
        ckpt.trajectory_path.write_bytes(b"frames")
        ckpt.update(steps_done=steps)
    return ckpt


class TestGeometryIsStored:
    """Without coordinates a listing can report work it cannot restore."""

    def test_begin_records_the_starting_geometry(self, root):
        ckpt = _seed_checkpoint()
        assert ckpt.load_state()["coords"] == [
            [0.0, 0.0, 0.0],
            [0.76, 0.59, 0.0],
            [-0.76, 0.59, 0.0],
        ]

    def test_spec_round_trips_to_a_matching_resume_key(self, root):
        """The whole point of restoring: the rebuilt calculation must be
        recognised as the same one, or the offer never appears."""
        original = C.CalcIdentity.from_molecule(
            _FakeMolecule(charge=-1, multiplicity=2),
            calc_type="geometry_opt",
            method="RHF",
            basis="6-31G",
        )
        ckpt = C.Checkpoint(original)
        ckpt.begin()
        spec = C.restorable_molecule_spec(ckpt.load_state())
        rebuilt = C.CalcIdentity.from_molecule(
            _FakeMolecule(
                atoms=spec["atoms"],
                coords=spec["coordinates"],
                charge=spec["charge"],
                multiplicity=spec["multiplicity"],
            ),
            calc_type="geometry_opt",
            method="RHF",
            basis="6-31G",
        )
        assert rebuilt.resume_key == original.resume_key

    def test_spec_is_none_without_stored_coordinates(self, root):
        assert C.restorable_molecule_spec({"atom_symbols": ["H", "H"]}) is None

    def test_spec_is_none_when_counts_disagree(self, root):
        assert (
            C.restorable_molecule_spec(
                {"atom_symbols": ["H", "H"], "coords": [[0.0, 0.0, 0.0]]}
            )
            is None
        )

    def test_spec_is_none_for_malformed_rows(self, root):
        assert (
            C.restorable_molecule_spec({"atom_symbols": ["H"], "coords": [[0.0, 0.0]]})
            is None
        )


class TestResumableCheckpointsListing:
    """The path that works after a restart, when nothing is configured."""

    def test_lists_unfinished_work_regardless_of_configuration(self, root):
        _seed_checkpoint(steps=5)
        assert len(C.resumable_checkpoints()) == 1

    def test_omits_checkpoints_with_no_progress(self, root):
        _seed_checkpoint()
        assert C.resumable_checkpoints() == []

    def test_omits_completed_runs(self, root):
        _seed_checkpoint(steps=5).mark_complete()
        assert C.resumable_checkpoints() == []

    def test_counts_stored_points(self, root):
        _seed_checkpoint(calc_type="pes_scan", points=6)
        assert C.resumable_checkpoints()[0]["n_points"] == 6

    def test_lists_several_at_once(self, root):
        _seed_checkpoint(calc_type="geometry_opt", steps=3)
        _seed_checkpoint(calc_type="pes_scan", points=2)
        assert len(C.resumable_checkpoints()) == 2

    def test_a_truncated_points_tail_is_not_counted(self, root):
        ckpt = _seed_checkpoint(calc_type="pes_scan", points=2)
        with open(ckpt.points_path, "a", encoding="utf-8") as fh:
            fh.write('{"index": 3, "val')
        assert C.resumable_checkpoints()[0]["n_points"] == 2


class TestResumeListUi:
    def test_hidden_when_there_is_no_unfinished_work(self, root):
        app = _FakeListApp()
        app_runflow.refresh_resume_list(app)
        assert app._resume_list_box.layout.display == "none"

    def test_shown_when_unfinished_work_exists(self, root):
        _seed_checkpoint(steps=4)
        app = _FakeListApp()
        app_runflow.refresh_resume_list(app)
        assert app._resume_list_box.layout.display == "block"

    def test_the_label_names_the_molecule_and_theory(self, root):
        _seed_checkpoint(steps=4)
        app = _FakeListApp()
        app_runflow.refresh_resume_list(app)
        label = app._resume_list_dd.options[0][0]
        assert "H2O" in label
        assert "RHF/6-31G" in label
        assert "Geometry Opt" in label

    def test_the_description_reports_saved_progress(self, root):
        _seed_checkpoint(calc_type="pes_scan", points=7)
        app = _FakeListApp()
        app_runflow.refresh_resume_list(app)
        assert "7 scan points computed" in app._resume_list_html.value

    def test_the_description_reports_optimizer_steps(self, root):
        _seed_checkpoint(steps=9)
        app = _FakeListApp()
        app_runflow.refresh_resume_list(app)
        assert "9 optimizer steps completed" in app._resume_list_html.value

    def test_selection_survives_a_refresh(self, root):
        """The list refreshes after every run; silently moving the selection
        would be a fine way to discard the wrong checkpoint."""
        _seed_checkpoint(calc_type="geometry_opt", steps=3)
        second = _seed_checkpoint(calc_type="pes_scan", points=2)
        app = _FakeListApp()
        app_runflow.refresh_resume_list(app)
        app._resume_list_dd.value = str(second.dir)
        app_runflow.refresh_resume_list(app)
        assert app._resume_list_dd.value == str(second.dir)

    def test_restore_is_disabled_when_geometry_is_missing(self, root):
        """Offering a button that cannot work is worse than disabling it."""
        ckpt = _seed_checkpoint(steps=3)
        state = ckpt.load_state()
        del state["coords"]
        ckpt.meta_path.write_text(json.dumps(state), encoding="utf-8")
        app = _FakeListApp()
        app_runflow.refresh_resume_list(app)
        assert app._resume_restore_btn.disabled is True
        assert "cannot be loaded" in app._resume_list_html.value

    def test_missing_widgets_are_tolerated(self, root):
        class _Early:
            pass

        app_runflow.refresh_resume_list(_Early())  # must not raise


class TestRestoreResumeEntry:
    def _prepared(self, root, **kwargs):
        _seed_checkpoint(**kwargs)
        app = _FakeListApp()
        app_runflow.refresh_resume_list(app)
        return app

    def test_restores_the_molecule(self, root):
        app = self._prepared(root, steps=3)
        assert app_runflow.restore_resume_entry(app) is True
        assert app._molecule is not None
        assert app._molecule.atoms == ["O", "H", "H"]

    def test_restores_method_basis_and_calc_type(self, root):
        app = self._prepared(root, steps=3)
        app_runflow.restore_resume_entry(app)
        assert app.method_dd.value == "RHF"
        assert app.basis_dd.value == "6-31G"
        assert app.calc_type_dd.value == "Geometry Opt"

    def test_restores_charge_and_multiplicity(self, root):
        """Both are part of the resume key — a restore that skipped them
        would rebuild a calculation the checkpoint no longer matches."""
        _seed_checkpoint(steps=3, molecule=_FakeMolecule(charge=-1, multiplicity=2))
        app = _FakeListApp()
        app_runflow.refresh_resume_list(app)
        app_runflow.restore_resume_entry(app)
        assert app.charge_si.value == -1
        assert app.mult_si.value == 2

    def test_maps_the_stored_key_back_to_the_dropdown_label(self, root):
        app = self._prepared(root, calc_type="tddft", points=0, steps=2)
        app_runflow.restore_resume_entry(app)
        assert app.calc_type_dd.value == "UV-Vis (TD-DFT)"

    def test_returns_false_with_nothing_selected(self, root):
        app = _FakeListApp()
        assert app_runflow.restore_resume_entry(app) is False

    def test_returns_false_when_geometry_is_missing(self, root):
        ckpt = _seed_checkpoint(steps=3)
        state = ckpt.load_state()
        del state["coords"]
        ckpt.meta_path.write_text(json.dumps(state), encoding="utf-8")
        app = _FakeListApp()
        app_runflow.refresh_resume_list(app)
        assert app_runflow.restore_resume_entry(app) is False

    def test_a_value_the_widget_rejects_does_not_abort_the_restore(self, root):
        """A basis may vanish from the options across an upgrade. The rest of
        the restore should still land."""

        class _Picky(_FakeWidget):
            def __setattr__(self, name, value):
                if name == "value" and value == "6-31G":
                    raise ValueError("not an option")
                super().__setattr__(name, value)

        app = self._prepared(root, steps=3)
        app.basis_dd = _Picky("STO-3G")
        assert app_runflow.restore_resume_entry(app) is True
        assert app.method_dd.value == "RHF"

    def test_restore_does_not_start_the_run(self, root):
        """The user should see what they are about to continue."""
        src = Path(app_runflow.__file__).read_text(encoding="utf-8")
        body = src[src.index("def restore_resume_entry") :]
        body = body[: body.index("\ndef ", 1)]
        assert "_do_run" not in body
        assert "run_btn.click" not in body


class TestPesScanSettingsRestore:
    def test_scan_geometry_is_stored_with_the_checkpoint(self):
        """Scan range isn't part of the resume key, so a restore that skipped
        it would reinstate a different scan and miss every stored point."""
        import quantui.app as A

        src = Path(A.__file__).read_text(encoding="utf-8")
        assert '"scan_start"' in src
        assert '"scan_steps"' in src

    def test_stored_settings_are_applied(self, root):
        identity = C.CalcIdentity.from_molecule(
            _FakeMolecule(), calc_type="pes_scan", method="RHF", basis="6-31G"
        )
        ckpt = C.Checkpoint(identity)
        ckpt.begin(
            total_points=12,
            settings={"scan_start": 0.7, "scan_stop": 2.4, "scan_steps": 12},
        )
        ckpt.append_point({"index": 1, "value": 0.7, "ok": True})

        app = _FakeListApp()
        app._scan_start = _FakeWidget(0.0)
        app._scan_stop = _FakeWidget(0.0)
        app._scan_steps = _FakeWidget(0)
        app_runflow.refresh_resume_list(app)
        app_runflow.restore_resume_entry(app)
        assert app._scan_start.value == 0.7
        assert app._scan_stop.value == 2.4
        assert app._scan_steps.value == 12


class TestResumeListIsRefreshedAtTheRightTimes:
    def test_refreshed_at_startup_on_both_paths(self):
        """After a restart the targeted offer cannot fire — nothing is
        configured yet — so startup is the moment this list matters.

        Startup refreshes through an io_loop when one exists and directly
        otherwise. Both branches need the call: covering only one leaves the
        list empty on whichever path that deployment happens to take, which
        is invisible until someone reports the feature "not working".
        """
        import quantui.app as A

        src = Path(A.__file__).read_text(encoding="utf-8")
        block = src[src.index("loop.add_callback(self._refresh_results_browser)") :]
        block = block[: block.index("def display")]
        assert block.count("_refresh_resume_list") == 2, (
            "both the io_loop and the direct startup branch must refresh "
            "the unfinished-calculations list"
        )

    def test_refreshed_after_a_run_finishes(self):
        import quantui.app as A

        src = Path(A.__file__).read_text(encoding="utf-8")
        body = src[src.index("def _finish_run_checkpoint") :][:2200]
        assert "_refresh_resume_list" in body

    def test_the_listing_widgets_are_in_the_history_panel(self):
        import quantui.app_builders as B

        src = Path(B.__file__).read_text(encoding="utf-8")
        assert "app._resume_list_box," in src
        assert "history_panel.children" in src


class TestResumedRunsGetTheirOwnResultDirectory:
    """A resumed run must not overwrite the interrupted run's saved output.

    ``save_result`` builds a microsecond timestamp plus a collision counter
    and calls ``mkdir(parents=True)`` **without** ``exist_ok`` — so a second
    run always lands in a new directory and can never write over an earlier
    ``pyscf.log`` or ``result.json``. Asserted here because it is a property
    the checkpoint feature now depends on, and a well-meaning change to the
    naming scheme elsewhere could silently remove it.
    """

    def test_directory_name_is_timestamped(self):
        import quantui.results_storage as R

        src = Path(R.__file__).read_text(encoding="utf-8")
        body = src[src.index("def save_result") :][:4000]
        assert "%Y-%m-%d_%H-%M-%S-%f" in body

    def test_a_collision_never_reuses_an_existing_directory(self):
        import quantui.results_storage as R

        src = Path(R.__file__).read_text(encoding="utf-8")
        body = src[src.index("def save_result") :][:4000]
        assert "while dest.exists():" in body
        assert "dest.mkdir(parents=True)" in body
        assert "exist_ok" not in body.split("dest.mkdir(parents=True)")[0][-200:]

    def test_two_saves_of_the_same_calculation_land_in_different_dirs(
        self, tmp_path, monkeypatch
    ):
        from types import SimpleNamespace

        from quantui.results_storage import save_result

        result = SimpleNamespace(
            formula="H2O",
            method="RHF",
            basis="6-31G",
            energy_hartree=-76.0,
            converged=True,
            n_iterations=8,
        )
        first = save_result(result, results_dir=tmp_path, pyscf_log="first run")
        second = save_result(result, results_dir=tmp_path, pyscf_log="second run")
        assert first != second
        assert (first / "pyscf.log").read_text() == "first run"
        assert (second / "pyscf.log").read_text() == "second run"


class TestResumeIsAnnouncedInTheLog:
    def test_optimizer_announces_a_resume(self):
        src = Path(optimizer.__file__).read_text(encoding="utf-8")
        assert "log_resumed(" in src

    def test_pes_scan_announces_a_resume(self):
        src = Path(pes_scan.__file__).read_text(encoding="utf-8")
        assert "log_resumed(" in src

    def test_the_checkpoint_is_given_the_run_log(self):
        """Without a stream attached, every checkpoint line is discarded."""
        import quantui.app as A

        src = Path(A.__file__).read_text(encoding="utf-8")
        assert "_begin_run_checkpoint(log)" in src
        assert "log_stream=log_stream" in src

    def test_warm_start_names_its_source_in_the_log(self):
        """The SCF iteration count is only interpretable with the starting
        density identified."""
        src = Path(session_calc.__file__).read_text(encoding="utf-8")
        assert "Warm start — initial guess read from" in src


class TestResumeControlsAreInTheLayout:
    """Built, registered, and never added to the container has happened here."""

    def test_both_widgets_are_placed_in_the_run_panel(self):
        import quantui.app_builders as B

        src = Path(B.__file__).read_text(encoding="utf-8")
        run_section = src[src.index("def build_run_section") :]
        assert "app._resume_notice_html," in run_section
        assert "app._resume_cb," in run_section
