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
        assert app._scan_start.value == pytest.approx(0.5)
        assert app._scan_stop.value == pytest.approx(2.0)

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
