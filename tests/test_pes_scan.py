"""Tests for quantui.pes_scan — PES scan module and app integration."""

from __future__ import annotations

import pytest

# ── Helpers ──────────────────────────────────────────────────────────────────


def _water():
    from quantui.molecule import Molecule

    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


def _h2():
    from quantui.molecule import Molecule

    return Molecule(atoms=["H", "H"], coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


# ── PESScanResult dataclass ───────────────────────────────────────────────────


class TestPESScanResult:
    """Unit tests for PESScanResult dataclass properties."""

    def _make(self, energies=(0.0, -0.1, -0.2, -0.1, 0.0), scan_type="bond"):
        from quantui.pes_scan import PESScanResult

        n = len(energies)
        mol = _h2()
        return PESScanResult(
            formula="H2",
            method="RHF",
            basis="STO-3G",
            scan_type=scan_type,
            atom_indices=[0, 1],
            scan_parameter_values=[0.5 + i * 0.3 for i in range(n)],
            energies_hartree=list(energies),
            coordinates_list=[mol] * n,
            converged_all=True,
        )

    def test_energy_hartree_returns_minimum(self):
        r = self._make(energies=[-0.5, -1.0, -0.8])
        assert r.energy_hartree == pytest.approx(-1.0)

    def test_energy_ev_scales_correctly(self):
        from quantui.session_calc import HARTREE_TO_EV

        r = self._make(energies=[-1.0, -1.0])
        assert r.energy_ev == pytest.approx(-1.0 * HARTREE_TO_EV)

    def test_converged_property(self):
        r = self._make()
        assert r.converged is True
        r.converged_all = False
        assert r.converged is False

    def test_n_steps(self):
        r = self._make(energies=[-0.1, -0.2, -0.3])
        assert r.n_steps == 3

    def test_energies_relative_kcal_minimum_is_zero(self):
        r = self._make(energies=[-1.0, -1.1, -1.05])
        rel = r.energies_relative_kcal
        assert min(rel) == pytest.approx(0.0, abs=1e-9)

    def test_energies_relative_kcal_length_matches(self):
        r = self._make(energies=[-0.1, -0.2, -0.15])
        assert len(r.energies_relative_kcal) == 3

    def test_scan_unit_bond(self):
        r = self._make(scan_type="bond")
        assert r.scan_unit == "Å"

    def test_scan_unit_angle(self):
        from quantui.pes_scan import PESScanResult

        r = PESScanResult(
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            scan_type="angle",
            atom_indices=[0, 1, 2],
            scan_parameter_values=[90.0, 100.0, 110.0],
            energies_hartree=[-75.0, -75.1, -75.0],
            coordinates_list=[_water()] * 3,
            converged_all=True,
        )
        assert r.scan_unit == "°"

    def test_scan_coordinate_label_bond(self):
        r = self._make()
        label = r.scan_coordinate_label
        assert "1" in label and "2" in label and "Å" in label

    def test_summary_contains_formula(self):
        r = self._make()
        assert "H2" in r.summary()

    def test_empty_energies_returns_nan(self):
        from quantui.pes_scan import PESScanResult

        r = PESScanResult(
            formula="X",
            method="RHF",
            basis="STO-3G",
            scan_type="bond",
            atom_indices=[0, 1],
            scan_parameter_values=[],
            energies_hartree=[],
            coordinates_list=[],
            converged_all=False,
        )
        import math

        assert math.isnan(r.energy_hartree)
        assert r.energies_relative_kcal == []


class TestPESScanResultNanHandling:
    """M6 audit fix (2026-07-14): NaN from failed scan points must not
    poison min()/max() based on ordering.

    Regression: Python's min()/max() are order-dependent when NaN is
    present — a NaN as the first element "wins" (every comparison
    against it is False) and poisons the result; a NaN later in the list
    is correctly skipped. energy_hartree, energies_relative_kcal, and
    summary() all called min()/max() directly on energies_hartree
    without filtering, so whether a scan came out usable depended on
    *which* point happened to fail.
    """

    def _make(self, energies, scan_type="bond"):
        from quantui.pes_scan import PESScanResult

        n = len(energies)
        mol = _h2()
        return PESScanResult(
            formula="H2",
            method="RHF",
            basis="STO-3G",
            scan_type=scan_type,
            atom_indices=[0, 1],
            scan_parameter_values=[0.5 + i * 0.3 for i in range(n)],
            energies_hartree=list(energies),
            coordinates_list=[mol] * n,
            converged_all=False,
        )

    def test_energy_hartree_ignores_leading_nan(self):
        r = self._make([float("nan"), -1.10, -1.117, -1.05])
        assert r.energy_hartree == pytest.approx(-1.117)

    def test_energy_hartree_ignores_middle_nan(self):
        r = self._make([-1.10, -1.117, float("nan"), -1.05])
        assert r.energy_hartree == pytest.approx(-1.117)

    def test_energy_hartree_ignores_trailing_nan(self):
        r = self._make([-1.10, -1.117, -1.05, float("nan")])
        assert r.energy_hartree == pytest.approx(-1.117)

    def test_energies_relative_kcal_not_poisoned_by_leading_nan(self):
        r = self._make([float("nan"), -1.10, -1.117, -1.05])
        rel = r.energies_relative_kcal
        import math

        assert math.isnan(rel[0])
        assert min(v for v in rel[1:]) == pytest.approx(0.0, abs=1e-9)

    def test_summary_has_no_nan_with_leading_failed_point(self):
        r = self._make([float("nan"), -1.10, -1.117, -1.05])
        summary = r.summary()
        assert "nan" not in summary.lower()
        assert "-1.11700000 Ha" in summary


# ── run_pes_scan validation (no PySCF needed) ─────────────────────────────────


class TestRunPesScanValidation:
    """Error-handling paths that do not require PySCF."""

    def test_wrong_scan_type_raises(self):
        from quantui.pes_scan import run_pes_scan

        with pytest.raises((ImportError, ValueError)):
            run_pes_scan(_h2(), scan_type="invalid")

    def test_wrong_atom_count_for_bond_raises(self):
        from quantui.pes_scan import run_pes_scan

        with pytest.raises((ImportError, ValueError)):
            run_pes_scan(_h2(), scan_type="bond", atom_indices=[0, 1, 2])

    def test_out_of_range_atom_index_raises(self):
        from quantui.pes_scan import run_pes_scan

        with pytest.raises((ImportError, ValueError)):
            run_pes_scan(_h2(), scan_type="bond", atom_indices=[0, 99])

    def test_duplicate_atom_indices_raises(self):
        from quantui.pes_scan import run_pes_scan

        with pytest.raises((ImportError, ValueError)):
            run_pes_scan(_h2(), scan_type="bond", atom_indices=[0, 0])

    def test_steps_less_than_2_raises(self):
        from quantui.pes_scan import run_pes_scan

        with pytest.raises((ImportError, ValueError)):
            run_pes_scan(_h2(), scan_type="bond", atom_indices=[0, 1], steps=1)

    @pytest.mark.parametrize("method", ["MP2", "CCSD", "CCSD(T)"])
    def test_post_hf_method_raises(self, method):
        """M2 audit fix (2026-07-14): post-HF methods raise a clear error.

        Regression: run_pes_scan() had no special-casing for MP2/CCSD/
        CCSD(T) — _QuantUIPySCFCalc.calculate() (shared with optimizer.py)
        silently treated them as a DFT xc functional, failing deep inside
        PySCF with a cryptic "LibXCFunctional" error instead of a clear one.
        """
        from quantui.pes_scan import run_pes_scan

        with pytest.raises((ImportError, ValueError), match="post-HF|ASE"):
            run_pes_scan(_h2(), method=method, scan_type="bond", atom_indices=[0, 1])


# ── App widget integration ────────────────────────────────────────────────────


class TestPesScanWidgets:
    """PES scan UI widgets are wired correctly in QuantUIApp."""

    def test_pes_scan_in_calc_type_options(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        assert "PES Scan" in app.calc_type_dd.options

    def test_scan_type_dropdown_defaults_to_bond(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        assert app._scan_type_dd.value == "Bond"

    def test_scan_start_stop_defaults(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        # Populated from geometry when PES Scan is first selected
        app.calc_type_dd.value = "PES Scan"
        assert app._scan_start.value != 0  # bond default from suggest
        assert app._scan_stop.value > app._scan_start.value

    def test_scan_atoms_are_dropdowns(self):
        from ipywidgets import Dropdown

        from quantui.app import QuantUIApp

        app = QuantUIApp()
        assert isinstance(app._scan_atom1, Dropdown)
        assert isinstance(app._scan_atom2, Dropdown)

    def test_angle_scan_accepts_negative_start(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        app.calc_type_dd.value = "PES Scan"
        app._scan_type_dd.value = "Angle"
        app._scan_start.value = -45.0
        app._scan_stop.value = 45.0
        assert app._scan_start.value == pytest.approx(-45.0)

    def test_pes_scan_shows_atom_reference(self):
        from quantui.app import QuantUIApp
        from quantui.molecule import Molecule

        app = QuantUIApp()
        app._set_molecule(
            Molecule(
                atoms=["O", "H", "H"],
                coordinates=[
                    [0.0, 0.0, 0.0],
                    [0.757, 0.587, 0.0],
                    [-0.757, 0.587, 0.0],
                ],
            ),
            label="test",
        )
        app.calc_type_dd.value = "PES Scan"
        assert "O1" in app._scan_atom_list_html.value

    def test_scan_steps_default(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        assert app._scan_steps.value == 10

    def test_pes_scan_accordion_visible_collapsed_initially(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        assert app._pes_scan_accordion.layout.display == ""
        assert app._pes_scan_accordion.selected_index is None

    def test_pes_plot_html_empty_initially(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        assert len(app._pes_plot_html.outputs) == 0

    def test_on_calc_type_changed_to_pes_scan_populates_extras(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        app.calc_type_dd.value = "PES Scan"
        assert len(app.calc_extra_opts.children) > 0

    def test_pes_scan_accordion_collapsed_on_run_clicked(self):
        from IPython.display import HTML

        from quantui.app import QuantUIApp

        app = QuantUIApp()
        app._pes_scan_accordion.selected_index = 0
        app._pes_plot_html.append_display_data(HTML("<div>old</div>"))
        assert len(app._pes_plot_html.outputs) == 1
        app._on_run_clicked(None)
        assert app._pes_scan_accordion.selected_index is None
        assert len(app._pes_plot_html.outputs) == 0


# ── Format method ─────────────────────────────────────────────────────────────


class TestFormatPesScanResult:
    def _make_result(self):
        from quantui.pes_scan import PESScanResult

        mol = _h2()
        return PESScanResult(
            formula="H2",
            method="RHF",
            basis="STO-3G",
            scan_type="bond",
            atom_indices=[0, 1],
            scan_parameter_values=[0.5, 1.0, 1.5, 2.0],
            energies_hartree=[-1.0, -1.1, -1.05, -0.9],
            coordinates_list=[mol] * 4,
            converged_all=True,
        )

    def test_format_returns_string(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        html = app._format_pes_scan_result(self._make_result())
        assert isinstance(html, str)

    def test_format_contains_formula(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        html = app._format_pes_scan_result(self._make_result())
        assert "H2" in html

    def test_format_contains_scan_type(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        html = app._format_pes_scan_result(self._make_result())
        assert "bond" in html.lower() or "Bond" in html

    def test_format_contains_range(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        html = app._format_pes_scan_result(self._make_result())
        assert "0.500" in html

    def test_format_shows_converged_yes(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        html = app._format_pes_scan_result(self._make_result())
        assert "Yes" in html


# ── PySCF-gated integration test ─────────────────────────────────────────────

_pyscf_available = pytest.mark.skipif(
    not __import__("sys").platform.startswith("linux"),
    reason="PySCF only available on Linux/WSL",
)


@_pyscf_available
@pytest.mark.slow
class TestRunPesScanIntegration:
    def test_h2_bond_scan_returns_result(self):
        from quantui.pes_scan import run_pes_scan

        result = run_pes_scan(
            _h2(),
            method="RHF",
            basis="STO-3G",
            scan_type="bond",
            atom_indices=[0, 1],
            start=0.6,
            stop=1.4,
            steps=4,
        )
        assert result.n_steps == 4
        assert len(result.energies_hartree) == 4
        assert all(e < 0 for e in result.energies_hartree)

    def test_h2_bond_scan_minimum_near_equilibrium(self):
        from quantui.pes_scan import run_pes_scan

        result = run_pes_scan(
            _h2(),
            method="RHF",
            basis="STO-3G",
            scan_type="bond",
            atom_indices=[0, 1],
            start=0.5,
            stop=2.0,
            steps=6,
        )
        # Minimum energy should be near the equilibrium bond length (~0.74 Å)
        e_rel = result.energies_relative_kcal
        min_idx = e_rel.index(min(e_rel))
        min_val = result.scan_parameter_values[min_idx]
        assert 0.5 <= min_val <= 1.5  # broad tolerance for 6-step coarse scan

    def test_failed_scan_point_falls_back_to_last_good_geometry(self, monkeypatch):
        """M6 audit fix (2026-07-14): a failed point's frame must be the
        last successful geometry, not always the original input molecule.

        Forces the 2nd scan point's BFGS relaxation to raise, then checks
        that point's coordinates match point 1's (the last good frame) —
        not the raw input geometry, which is what the bug used to record
        regardless of how far the scan had already progressed.
        """
        import math

        import ase.optimize as ase_optimize

        from quantui.pes_scan import run_pes_scan

        real_bfgs = ase_optimize.BFGS
        call_count = {"n": 0}

        class _FlakyBFGS:
            def __init__(self, *args, **kwargs):
                call_count["n"] += 1
                self._real = real_bfgs(*args, **kwargs)

            def run(self, *args, **kwargs):
                if call_count["n"] == 2:
                    raise RuntimeError("simulated optimizer failure")
                return self._real.run(*args, **kwargs)

        monkeypatch.setattr(ase_optimize, "BFGS", _FlakyBFGS)

        result = run_pes_scan(
            _water(),
            method="RHF",
            basis="STO-3G",
            scan_type="bond",
            atom_indices=[0, 1],  # O-H bond; 3 atoms so BFGS actually runs
            start=0.85,
            stop=1.15,
            steps=3,
        )

        assert result.converged_all is False
        assert math.isnan(result.energies_hartree[1])
        failed_frame = result.coordinates_list[1]
        last_good_frame = result.coordinates_list[0]
        input_frame = _water()
        assert failed_frame.coordinates == last_good_frame.coordinates
        assert failed_frame.coordinates != input_frame.coordinates


@_pyscf_available
@pytest.mark.slow
class TestRunPesScanAngleDihedral:
    """M7 audit fix (2026-07-14): angle/dihedral scans must actually run.

    Regression: FixInternals(angles=[...]) / FixInternals(dihedrals=[...])
    (the radian-based, deprecated kwargs) don't just emit a FutureWarning
    with the ASE version QuantUI targets (verified against 3.29.0) — they
    raise "setting an array element with a sequence" from an internal
    np.asarray reshape, unconditionally, for every angle/dihedral
    constraint. Every angle and dihedral PES scan failed at 100% of its
    points as a result (silently, caught by the per-point try/except).
    Switching to angles_deg/dihedrals_deg (plain degrees, no radian
    conversion) fixes this.
    """

    def test_angle_scan_converges_without_nan(self):
        from quantui.pes_scan import run_pes_scan

        result = run_pes_scan(
            _water(),
            method="RHF",
            basis="STO-3G",
            scan_type="angle",
            atom_indices=[1, 0, 2],  # H-O-H angle
            start=95.0,
            stop=115.0,
            steps=3,
        )
        assert result.converged_all is True
        assert all(e == e for e in result.energies_hartree)  # no NaN

    def test_dihedral_scan_converges_without_nan(self):
        from quantui.molecule import Molecule
        from quantui.pes_scan import run_pes_scan

        # Hydrogen peroxide (H-O-O-H) — smallest molecule with a real
        # dihedral degree of freedom.
        h2o2 = Molecule(
            atoms=["H", "O", "O", "H"],
            coordinates=[
                [0.9, 0.0, 0.9],
                [0.0, 0.0, 0.75],
                [0.0, 0.0, -0.75],
                [-0.9, 0.3, -0.9],
            ],
        )
        result = run_pes_scan(
            h2o2,
            method="RHF",
            basis="STO-3G",
            scan_type="dihedral",
            atom_indices=[0, 1, 2, 3],
            start=60.0,
            stop=120.0,
            steps=3,
        )
        assert result.converged_all is True
        assert all(e == e for e in result.energies_hartree)  # no NaN
