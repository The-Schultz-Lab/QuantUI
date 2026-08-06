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
from pathlib import Path

import pytest

from quantui import app_runflow, checkpoint as C, optimizer, pes_scan, session_calc


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
        assert inspect.signature(session_calc.run_in_session).parameters[
            "warm_start"
        ].default is True

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


class TestResumeControlsAreInTheLayout:
    """Built, registered, and never added to the container has happened here."""

    def test_both_widgets_are_placed_in_the_run_panel(self):
        import quantui.app_builders as B

        src = Path(B.__file__).read_text(encoding="utf-8")
        run_section = src[src.index("def build_run_section") :]
        assert "app._resume_notice_html," in run_section
        assert "app._resume_cb," in run_section
