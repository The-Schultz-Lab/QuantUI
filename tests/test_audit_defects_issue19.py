"""Remaining parity-audit § 5 defects (M-ISSUES ISSUE.19)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quantui.backends.dispatch import build_calculation_request
from quantui.molecule import Molecule
from tests.test_backends_dispatch import _fake_app

H2 = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])


@pytest.fixture
def staging(tmp_path):
    """A worker staging folder holding an H2 single-point request."""
    from quantui.backends.base import CalculationRequest

    staging_dir = tmp_path / "staging" / "job1"
    staging_dir.mkdir(parents=True)
    request = CalculationRequest(
        request_id="job1",
        calc_type="single_point",
        method="RHF",
        basis="STO-3G",
        charge=0,
        multiplicity=1,
        molecule={"atoms": H2.atoms, "coordinates": H2.coordinates},
    )
    (staging_dir / "request.json").write_text(json.dumps(request.to_dict()))
    return staging_dir


def _freq_result_dir(tmp_path):
    """A saved H2 frequency result with one normal mode."""
    from quantui.results_storage import save_result

    result = SimpleNamespace(
        formula="H2", method="RHF", basis="STO-3G", energy_hartree=-1.1, converged=True
    )
    return save_result(
        result,
        results_dir=tmp_path,
        calc_type="frequency",
        spectra={
            "ir": {
                "frequencies_cm1": [-300.0],
                "ir_intensities": [0.0],
                "displacements": [[[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]],
            },
            "molecule": {
                "atoms": H2.atoms,
                "coords": H2.coordinates,
                "charge": 0,
                "multiplicity": 1,
            },
        },
    )


class TestPreoptOnSlurm:
    """#3 — SP/NMR pre-opt was dropped on SLURM; #8 — mode seeds skipped it."""

    @pytest.mark.parametrize(
        "ct,calc_type", [("Single Point", "single_point"), ("NMR Shielding", "nmr")]
    )
    def test_single_point_and_nmr_send_the_preopt_flag(self, ct, calc_type):
        app = _fake_app(
            calc_type_dd=SimpleNamespace(value=ct),
            _freq_preopt_cb=SimpleNamespace(value=True),
        )
        req = build_calculation_request(app, request_id="p")
        assert req.calc_type == calc_type
        assert req.options.get("preopt_before_run") is True

    def test_unticked_sends_nothing(self):
        req = build_calculation_request(_fake_app(), request_id="p")
        assert "preopt_before_run" not in req.options

    def test_mode_displaced_frequency_seed_keeps_the_preopt(self, tmp_path):
        seed = f"freq:{_freq_result_dir(tmp_path)}"
        app = _fake_app(
            calc_type_dd=SimpleNamespace(value="Frequency"),
            _seed_dd=SimpleNamespace(value=seed),
            _freq_preopt_cb=SimpleNamespace(value=True),
            _freq_perturb_mode_dd=SimpleNamespace(value=1),
            _freq_perturb_fraction=SimpleNamespace(value=0.5),
        )
        req = build_calculation_request(app, request_id="m")
        # Displaced along the mode, and re-optimized before the frequencies.
        assert req.molecule["coordinates"][1][2] != pytest.approx(0.74)
        assert req.options.get("preopt_before_run") is True

    @pytest.mark.parametrize("calc_type", ["single_point", "nmr"])
    def test_worker_preoptimizes_before_sp_and_nmr(self, staging, calc_type):
        moved = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.71]])
        opt = SimpleNamespace(
            molecule=moved,
            trajectory=[moved],
            energies_hartree=[-1.117],
            converged=True,
            n_steps=3,
        )
        seen = {}

        def _fake_calc(**kw):
            seen["z"] = kw["molecule"].coordinates[1][2]
            return SimpleNamespace(
                energy_hartree=-1.117,
                homo_lumo_gap_ev=None,
                converged=True,
                n_iterations=4,
                method="RHF",
                basis="STO-3G",
                formula="H2",
                shielding_iso=[30.0, 30.0],
                atom_symbols=["H", "H"],
                reference_shielding={},
                chemical_shifts={},
            )

        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = calc_type
        data["options"] = {"preopt_before_run": True}
        (staging / "request.json").write_text(json.dumps(data))
        target = (
            "quantui.session_calc.run_in_session"
            if calc_type == "single_point"
            else "quantui.nmr_calc.run_nmr_calc"
        )
        from quantui.backends.base import CalculationRequest
        from quantui.backends.worker import _run_nmr, _run_single_point

        req = CalculationRequest.from_dict(data)
        runner = _run_single_point if calc_type == "single_point" else _run_nmr
        with patch("quantui.optimizer.optimize_geometry", return_value=opt) as m_opt:
            with patch(target, side_effect=_fake_calc):
                with open(staging / "live.log", "w") as log:
                    runner(req, staging, log)
        m_opt.assert_called_once()
        assert seen["z"] == pytest.approx(0.71)
        # B2.5 tightens only the frequency pre-opt.
        from quantui.optimizer import DEFAULT_FMAX

        assert m_opt.call_args.kwargs["fmax"] == DEFAULT_FMAX


class TestSeedGating:
    """#8 — the checkbox follows the seed kind."""

    def _app(self, seed):
        return SimpleNamespace(
            _freq_preopt_cb=SimpleNamespace(value=False, disabled=True),
            _seed_dd=SimpleNamespace(value=seed),
        )

    def test_mode_seed_ticks_and_enables(self):
        from quantui.app_runflow import _gate_preopt_for_seed

        app = self._app("freq:/x")
        _gate_preopt_for_seed(app, "freq:/x")
        assert app._freq_preopt_cb.value is True
        assert app._freq_preopt_cb.disabled is False

    def test_optimized_seed_disables(self):
        from quantui.app_runflow import _gate_preopt_for_seed

        app = self._app("/results/opt")
        app._freq_preopt_cb.value = True
        _gate_preopt_for_seed(app, "/results/opt")
        assert app._freq_preopt_cb.value is False
        assert app._freq_preopt_cb.disabled is True

    def test_no_seed_enables(self):
        from quantui.app_runflow import _gate_preopt_for_seed

        app = self._app("")
        _gate_preopt_for_seed(app, "")
        assert app._freq_preopt_cb.disabled is False


class TestReorgShowsItsOptimizerFields:
    """#4 — the run reads fmax / max steps, so the panel must show them."""

    def test_fields_are_on_the_reorg_panel(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        app.calc_type_dd.value = "Reorganization Energy"

        def _walk(w):
            yield w
            for c in getattr(w, "children", ()):
                yield from _walk(c)

        shown = list(_walk(app.calc_extra_opts))
        assert app.fmax_fi in shown and app.max_steps_si in shown
        assert app._reorg_mode_dd in shown


class TestRamanWithoutActivities:
    """#5 — no fake equal-height spectrum when activities are missing."""

    def _app(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        captured = {}
        orig = app._set_html_output

        def _grab(out, html):
            if out is app._raman_fig:
                captured["html"] = html
            return orig(out, html)

        app._set_html_output = _grab
        return app, captured

    def test_missing_activities_explain_instead_of_plotting(self):
        app, captured = self._app()
        stub = SimpleNamespace(frequencies_cm1=[1600.0, 3700.0], raman_activities=[])
        assert app._show_raman_spectrum(stub) is True
        assert "not computed" in app._raman_accordion.get_title(0)
        assert "pyscf-properties" in captured["html"]
        assert app._last_raman_fig is None
        # A later re-render (theme switch, mode toggle) must not draw peaks.
        app._update_raman_figure("Stick", 20.0)
        assert app._last_raman_fig is None

    def test_pcm_result_names_the_solvent(self):
        app, captured = self._app()
        stub = SimpleNamespace(
            frequencies_cm1=[1600.0], raman_activities=None, solvent="Water"
        )
        app._show_raman_spectrum(stub)
        assert "implicit solvent (Water)" in captured["html"]

    def test_real_activities_still_plot(self):
        app, _ = self._app()
        stub = SimpleNamespace(frequencies_cm1=[1600.0], raman_activities=[5.0])
        app._show_raman_spectrum(stub)
        assert app._raman_accordion.get_title(0) == "Raman Spectrum"
        assert app._last_raman_fig is not None


class TestMultiFrameXyzPaste:
    """#10 — a pasted trajectory gets a teaching message."""

    FRAME = "3\nwater\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"

    def test_two_frames(self):
        from quantui.molecule import parse_xyz_input

        with pytest.raises(ValueError, match="multi-frame XYZ file \\(2 structures"):
            parse_xyz_input(self.FRAME + self.FRAME)

    def test_blank_line_between_frames(self):
        from quantui.molecule import parse_xyz_input

        with pytest.raises(ValueError, match="Upload File tab"):
            parse_xyz_input(self.FRAME + "\n" + self.FRAME)

    def test_single_frame_and_headerless_still_parse(self):
        from quantui.molecule import parse_xyz_input

        assert parse_xyz_input(self.FRAME)[0] == ["O", "H", "H"]
        assert parse_xyz_input("O 0 0 0\nH 0 0 1\n")[0] == ["O", "H"]

    def test_truncated_second_frame_is_an_ordinary_error(self):
        from quantui.molecule import parse_xyz_input

        with pytest.raises(ValueError) as exc:
            parse_xyz_input(self.FRAME + "3\nx\nO 0 0 0\n")
        assert "multi-frame" not in str(exc.value)


class TestFragmentNoteOnEveryLoad:
    """#9 — disconnected structures are flagged whatever their source."""

    DIMER = Molecule(
        ["O", "H", "H", "O", "H", "H"],
        [
            [0, 0, 0.117],
            [0, 0.757, -0.469],
            [0, -0.757, -0.469],
            [0, 0, 3.117],
            [0, 0.757, 2.531],
            [0, -0.757, 2.531],
        ],
    )

    def test_note_text(self):
        from quantui.connectivity import disconnection_note

        note = disconnection_note(self.DIMER.atoms, self.DIMER.coordinates)
        assert note.startswith("2 separate fragments (2×H2O)")
        assert disconnection_note(H2.atoms, H2.coordinates) is None

    def test_summary_shows_it_for_a_pasted_or_uploaded_structure(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        app._set_molecule(self.DIMER, "XYZ input")
        assert "2 separate fragments" in app.mol_info_html.value
        app._set_molecule(H2, "XYZ input")
        assert "separate fragments" not in app.mol_info_html.value

    def test_search_message_is_unchanged(self):
        from quantui.connectivity import describe_disconnection

        msg = describe_disconnection(self.DIMER.atoms, self.DIMER.coordinates)
        assert "resolved to 2 separate fragments (2×H2O)" in msg


WATER = Molecule(
    ["O", "H", "H"], [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]]
)


@pytest.fixture(scope="module")
def water_freq():
    pytest.importorskip("pyscf")
    import io

    from quantui.freq_calc import run_freq_calc

    return run_freq_calc(WATER, "RHF", "STO-3G", progress_stream=io.StringIO())


@pytest.fixture(scope="module")
def water_tddft():
    pytest.importorskip("pyscf")
    import io

    from quantui.tddft_calc import run_tddft_calc

    return run_tddft_calc(
        WATER, "B3LYP", "STO-3G", nstates=2, progress_stream=io.StringIO()
    )


class TestGroundStatePanelsForFreqAndTddft:
    """#6 — Frequency and TD-DFT results fill Isosurface and Populations."""

    def test_registry_order(self):
        from quantui.app import QuantUIApp

        for ct in ("frequency", "tddft"):
            names = [n for n, _, _ in QuantUIApp._PANEL_REGISTRY[ct]]
            assert "Isosurface" in names and "Populations" in names
            # Energies loads the orbital state Isosurface checks.
            assert names.index("Energies") < names.index("Isosurface")

    def test_results_carry_ground_state_data(self, water_freq, water_tddft):
        for r in (water_freq, water_tddft):
            assert len(r.mulliken_charges) == 3 and r.mulliken_charges[0] < 0
            assert r.dipole_moment_debye > 1.0
            assert r.mo_coeff.shape == (7, 7)
            assert r.spin_square is None  # closed shell

    @pytest.mark.parametrize("which", ["frequency", "tddft"])
    def test_history_replay_enables_both_panels(
        self, which, water_freq, water_tddft, tmp_path
    ):
        from quantui.app import QuantUIApp
        from quantui.results_storage import save_orbitals, save_result

        r = water_freq if which == "frequency" else water_tddft
        spectra = (
            {"ir": {"frequencies_cm1": list(r.frequencies_cm1)}}
            if which == "frequency"
            else {
                "uv_vis": {
                    "excitation_energies_ev": list(r.excitation_energies_ev),
                    "oscillator_strengths": list(r.oscillator_strengths),
                    "wavelengths_nm": list(r.wavelengths_nm()),
                }
            }
        )
        saved = save_result(r, results_dir=tmp_path, calc_type=which, spectra=spectra)
        save_orbitals(saved, r)
        app = QuantUIApp()
        app._set_molecule(WATER, "test")
        app._apply_analysis_context(app._build_history_context(saved))
        assert "Populations" in app._ana_available
        assert "Isosurface" in app._ana_available

    def test_worker_payloads_and_tddft_orbitals(
        self, water_freq, water_tddft, tmp_path
    ):
        from quantui.backends.worker_payload import (
            freq_result_payload,
            tddft_result_payload,
            write_analysis_artifacts,
        )

        fp = freq_result_payload(water_freq, WATER)
        tp = tddft_result_payload(water_tddft)
        for p in (fp, tp):
            assert len(p["mulliken_charges"]) == 3
            assert p["dipole_moment_debye"] > 1.0
            json.dumps(p)  # JSON-safe
        write_analysis_artifacts(tmp_path, "tddft", water_tddft)
        assert (tmp_path / "orbitals.npz").exists()


class TestExportScriptFollowsTheCalculation:
    """#7 — Export Script used to be a gas-phase single point every time."""

    def _script(self, tmp_path, calc_type="single_point", method="RHF", **kw):
        from quantui.calculator import PySCFCalculation

        return PySCFCalculation(
            WATER, method=method, basis="STO-3G"
        ).generate_calculation_script(
            tmp_path / f"{calc_type}.py", calc_type=calc_type, **kw
        )

    @pytest.mark.parametrize(
        "calc_type",
        [
            "single_point",
            "geometry_opt",
            "frequency",
            "tddft",
            "nmr",
            "pes_scan",
            "reorganization_energy",
        ],
    )
    @pytest.mark.parametrize("solvent", [None, "water"])
    def test_every_variant_is_valid_python(self, tmp_path, calc_type, solvent):
        src = self._script(tmp_path, calc_type, solvent=solvent, density_fit=True)
        compile(src, "exported.py", "exec")

    def test_single_point_has_no_extras(self, tmp_path):
        src = self._script(tmp_path)
        assert "PCM" not in src and "density_fit()" not in src
        assert "NOTE: QuantUI ran" not in src

    def test_solvent_and_density_fitting(self, tmp_path):
        src = self._script(tmp_path, solvent="water", density_fit=True)
        assert "mf = PCM(mf)" in src and "mf.with_solvent.eps = 78.39" in src
        assert "mf = mf.density_fit()" in src

    def test_unknown_solvent_is_an_error(self, tmp_path):
        with pytest.raises(ValueError):
            self._script(tmp_path, solvent="Unobtainium")

    def test_tddft_uses_the_optical_dielectric(self, tmp_path):
        src = self._script(
            tmp_path, "tddft", method="B3LYP", solvent="Ethanol", nstates=4
        )
        assert "td.nstates = 4" in src and "td.with_solvent.eps = 1.853" in src

    def test_partial_workflows_say_so(self, tmp_path):
        src = self._script(tmp_path, "geometry_opt")
        assert "QuantUI ran a Geometry Opt calculation" in src

    def test_app_export_passes_the_settings(self, tmp_path):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        app._set_molecule(WATER, "test")
        app._last_result_dir = tmp_path
        app.calc_type_dd.value = "Geometry Opt"
        app.solvent_cb.value = True
        app.solvent_dd.value = "Water"
        app._on_export(None)
        src = next(tmp_path.glob("*.py")).read_text(encoding="utf-8")
        assert "mf = PCM(mf)" in src
        assert "SCF at this geometry only" in app.export_status.value

    def test_frequency_script_matches_the_app(self, tmp_path, water_freq):
        import re
        import subprocess
        import sys

        self._script(tmp_path, "frequency")
        run = subprocess.run(
            [sys.executable, str(tmp_path / "frequency.py")],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=tmp_path,
        )
        assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
        m = re.search(r"Frequencies \(cm\^-1\):\n\[([^\]]+)\]", run.stdout)
        got = sorted(float(x) for x in m.group(1).split())
        want = sorted(f for f in water_freq.frequencies_cm1 if f > 0)
        assert got == pytest.approx(want, abs=1.0)
