"""Tests for ECD (electronic circular dichroism) — roadmap SPC2.3.

Covers:
  * unit constants and the Gaussian Δε broadening (analytic checks);
  * the rotatory-strength formula against PySCF's sign conventions;
  * physics: mirror-image molecules give exactly opposite R_n, an achiral
    molecule gives R ≈ 0, and length/velocity gauges agree for a decent
    basis (PySCF-gated);
  * persistence (result.json additive keys, batch worker payload) and the
    ECD Analysis panel for live and History contexts.
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quantui.ecd_plot import (
    ECD_INTEGRAL_CONST,
    ecd_delta_epsilon,
    is_effectively_achiral,
    plot_ecd_spectrum,
)
from quantui.molecule import Molecule
from quantui.tddft_calc import ROTATORY_AU_TO_1E40_CGS, rotatory_strengths

try:
    import pyscf  # noqa: F401

    _PYSCF = sys.platform != "win32"
except ImportError:
    _PYSCF = False

pyscf_only = pytest.mark.skipif(not _PYSCF, reason="PySCF not available")

# numpy 2 renamed trapz → trapezoid (trapz is deprecated there).
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------


def _h2o2(dihedral_deg: float = 120.0) -> Molecule:
    """Non-planar (hence chiral, C2) hydrogen peroxide."""
    roo, roh, ang = 1.45, 0.97, math.radians(100.0)
    phi = math.radians(dihedral_deg)
    c, s = math.cos(math.pi - ang), math.sin(math.pi - ang)
    coords = [
        [0.0, 0.0, 0.0],
        [roo, 0.0, 0.0],
        [-roh * c, roh * s, 0.0],
        [roo + roh * c, roh * s * math.cos(phi), roh * s * math.sin(phi)],
    ]
    return Molecule(["O", "O", "H", "H"], coords)


def _mirror(mol: Molecule) -> Molecule:
    """Reflect through the xy plane — the other enantiomer."""
    return Molecule(list(mol.atoms), [[x, y, -z] for x, y, z in mol.coordinates])


def _water() -> Molecule:
    return Molecule(
        ["O", "H", "H"], [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]]
    )


# ---------------------------------------------------------------------------
# Constants and broadening (pure, analytic)
# ---------------------------------------------------------------------------


class TestUnitConstants:
    def test_au_to_cgs_from_codata(self):
        # e·a0 · eħ/(m_e c) in esu²·cm², CODATA 2018 (Gaussian units)
        e = 4.803204712570263e-10
        a0 = 0.529177210903e-8
        hbar = 1.054571817e-27
        me = 9.1093837015e-28
        c = 2.99792458e10
        assert ROTATORY_AU_TO_1E40_CGS == pytest.approx(
            e * a0 * e * hbar / (me * c) / 1e-40, rel=1e-6
        )

    def test_integral_constant_from_definition(self):
        # R = 3 h c 10³ ln10 / (32 π³ N_A) · ∫ Δε/ν̃ dν̃   (cgs)
        h, c, na = 6.62607015e-27, 2.99792458e10, 6.02214076e23
        expected = 3 * h * c * 1000 * math.log(10) / (32 * math.pi**3 * na)
        assert ECD_INTEGRAL_CONST == pytest.approx(expected, rel=1e-4)


class TestDeltaEpsilonBroadening:
    def test_peak_height_is_analytic(self):
        e0, r, sigma = 6.0, 10.0, 0.3
        peak = ecd_delta_epsilon([e0], [r], [e0], sigma_ev=sigma)[0]
        expected = e0 * r * 1e-40 / (ECD_INTEGRAL_CONST * math.sqrt(math.pi) * sigma)
        assert peak == pytest.approx(expected, rel=1e-12)

    def test_band_integral_recovers_rotatory_strength(self):
        """∫ Δε/E dE = R / 2.296e-39 — the definition of R (narrow band)."""
        e0, r, sigma = 6.0, 25.0, 0.1
        grid = np.linspace(4.0, 8.0, 200001)
        deps = ecd_delta_epsilon([e0], [r], grid, sigma_ev=sigma)
        integral = _trapz(deps / grid, grid)
        # Δε/E = (prefactor/E)·E0·exp(...) integrates to R/C up to a
        # (1 + σ²/(2E0²)) Gaussian-moment correction ≈ 1.00014 here.
        assert integral * ECD_INTEGRAL_CONST / 1e-40 == pytest.approx(r, rel=1e-3)

    def test_area_independent_of_width(self):
        grid = np.linspace(2.0, 10.0, 80001)
        a1 = _trapz(ecd_delta_epsilon([6.0], [5.0], grid, 0.2) / grid, grid)
        a2 = _trapz(ecd_delta_epsilon([6.0], [5.0], grid, 0.4) / grid, grid)
        # Equal up to the O(σ²/2E0²) moment correction (0.06 % vs 0.22 %).
        assert a1 == pytest.approx(a2, rel=3e-3)

    def test_sign_flip_and_linearity(self):
        grid = np.linspace(4.0, 9.0, 101)
        e, r = [5.5, 6.7], [3.0, -8.0]
        plus = ecd_delta_epsilon(e, r, grid, 0.3)
        minus = ecd_delta_epsilon(e, [-x for x in r], grid, 0.3)
        assert np.allclose(plus, -minus)
        single = ecd_delta_epsilon([5.5], [3.0], grid, 0.3) + ecd_delta_epsilon(
            [6.7], [-8.0], grid, 0.3
        )
        assert np.allclose(plus, single)

    def test_rejects_non_positive_width(self):
        with pytest.raises(ValueError):
            ecd_delta_epsilon([6.0], [1.0], [6.0], sigma_ev=0.0)

    def test_achiral_detection(self):
        assert is_effectively_achiral([1e-9, -3e-10, 0.0])
        assert not is_effectively_achiral([1e-9, 0.2])
        assert not is_effectively_achiral([])


class TestPlotEcdSpectrum:
    def test_broadened_axis_is_delta_epsilon(self):
        fig = plot_ecd_spectrum([5.0, 6.0], [4.0, -2.0], mode="broadened")
        assert "Δε" in fig.layout.yaxis.title.text
        lo, hi = fig.layout.yaxis.range
        assert lo == pytest.approx(-hi)  # symmetric about zero

    def test_stick_axis_is_rotatory_strength(self):
        fig = plot_ecd_spectrum([5.0, 6.0], [4.0, -2.0], mode="stick")
        assert fig.layout.yaxis.title.text.startswith("R")
        ys = [y for y in fig.data[1].y]
        assert ys == pytest.approx([4.0, -2.0])

    def test_achiral_spectrum_gets_unit_axis(self):
        fig = plot_ecd_spectrum([5.0, 6.0], [0.0, 0.0], mode="broadened")
        assert tuple(fig.layout.yaxis.range) == (-1.0, 1.0)


# ---------------------------------------------------------------------------
# Rotatory-strength formula against PySCF conventions (no PySCF needed)
# ---------------------------------------------------------------------------


class TestRotatoryStrengthFormula:
    def test_gauges_agree_when_hypervirial_holds(self):
        """V = ωD exactly ⇒ R_vel = R_len = −½ D·M (in a.u.)."""
        omega = np.array([0.2, 0.3])
        d = np.array([[0.1, 0.2, -0.3], [0.0, 0.5, 0.1]])
        m = np.array([[0.4, -0.1, 0.2], [0.3, 0.0, -0.2]])
        td = SimpleNamespace(
            e=omega,
            transition_dipole=lambda: d,
            transition_velocity_dipole=lambda: omega[:, None] * d,
            transition_magnetic_dipole=lambda: m,
        )
        r_vel, r_len = rotatory_strengths(td)
        expected = -0.5 * np.einsum("sx,sx->s", d, m) * ROTATORY_AU_TO_1E40_CGS
        assert r_vel == pytest.approx(list(expected))
        assert r_len == pytest.approx(list(expected))

    def test_eigenvector_phase_cancels(self):
        omega = np.array([0.25])
        d = np.array([[0.1, 0.2, -0.3]])
        v = np.array([[0.03, 0.04, -0.07]])
        m = np.array([[0.4, -0.1, 0.2]])

        def _td(sign):
            return SimpleNamespace(
                e=omega,
                transition_dipole=lambda: sign * d,
                transition_velocity_dipole=lambda: sign * v,
                transition_magnetic_dipole=lambda: sign * m,
            )

        assert rotatory_strengths(_td(1)) == rotatory_strengths(_td(-1))


# ---------------------------------------------------------------------------
# Physics with real PySCF TD-DFT
# ---------------------------------------------------------------------------


@pyscf_only
@pytest.mark.slow  # real TD-DFT runs (CI still runs these: -m "not network")
class TestRotatoryStrengthPhysics:
    def test_mirror_images_flip_sign_exactly(self):
        from quantui.tddft_calc import run_tddft_calc

        a = run_tddft_calc(_h2o2(), method="B3LYP", basis="6-31G", nstates=4)
        b = run_tddft_calc(_mirror(_h2o2()), method="B3LYP", basis="6-31G", nstates=4)
        ra, rb = a.rotatory_strengths_velocity, b.rotatory_strengths_velocity
        assert len(ra) == len(a.excitation_energies_ev) == 4
        assert a.excitation_energies_ev == pytest.approx(b.excitation_energies_ev)
        assert max(abs(x) for x in ra) > 1.0  # genuinely chiral
        assert ra == pytest.approx([-x for x in rb], abs=1e-6)
        assert a.rotatory_strengths_length == pytest.approx(
            [-x for x in b.rotatory_strengths_length], abs=1e-6
        )

    def test_achiral_water_has_zero_rotatory_strength(self):
        from quantui.tddft_calc import run_tddft_calc

        r = run_tddft_calc(_water(), method="B3LYP", basis="6-31G*", nstates=4)
        assert r.rotatory_strengths_velocity
        assert max(abs(x) for x in r.rotatory_strengths_velocity) < 1e-6
        assert max(abs(x) for x in r.rotatory_strengths_length) < 1e-6
        assert is_effectively_achiral(r.rotatory_strengths_velocity)

    def test_pyscf_velocity_dipole_parallel_to_length_dipole(self):
        """Pins PySCF's phase convention the length-gauge formula relies on:
        transition_velocity_dipole ≈ +ω·transition_dipole (hypervirial)."""
        from pyscf import dft, gto

        m = _h2o2()
        mol = gto.M(atom=m.to_pyscf_format(), basis="6-31G", verbose=0)
        mf = dft.RKS(mol)
        mf.xc = "B3LYP"
        mf.run()
        td = mf.TDDFT()
        td.nstates = 4
        td.kernel()
        dots = np.einsum(
            "sx,sx->s", td.transition_velocity_dipole(), td.transition_dipole()
        )
        assert np.all(dots > 0)

    def test_length_and_velocity_gauges_agree_in_diffuse_basis(self):
        from quantui.tddft_calc import run_tddft_calc

        r = run_tddft_calc(_h2o2(), method="B3LYP", basis="aug-cc-pVDZ", nstates=4)
        for rv, rl in zip(r.rotatory_strengths_velocity, r.rotatory_strengths_length):
            if abs(rv) > 1.0:
                assert math.copysign(1, rv) == math.copysign(1, rl)
                assert abs(rv - rl) / abs(rv) < 0.2


# ---------------------------------------------------------------------------
# Persistence + panel
# ---------------------------------------------------------------------------


def _tddft_ns(r_vel, r_len=None):
    energies = [5.6, 6.2, 7.1]
    return SimpleNamespace(
        formula="H2O2",
        method="B3LYP",
        basis="6-31G",
        energy_hartree=-151.0,
        energy_ev=-4108.9,
        homo_lumo_gap_ev=7.0,
        converged=True,
        n_iterations=10,
        excitation_energies_ev=energies,
        oscillator_strengths=[0.001, 0.01, 0.02],
        wavelengths_nm=lambda: [1239.84193 / e for e in energies],
        rotatory_strengths_velocity=r_vel,
        rotatory_strengths_length=r_len if r_len is not None else list(r_vel),
    )


def _spectra(res):
    return {
        "uv_vis": {
            "excitation_energies_ev": res.excitation_energies_ev,
            "oscillator_strengths": res.oscillator_strengths,
            "wavelengths_nm": res.wavelengths_nm(),
            "rotatory_strengths_velocity": res.rotatory_strengths_velocity,
            "rotatory_strengths_length": res.rotatory_strengths_length,
        }
    }


@pytest.fixture
def app():
    from quantui.app import QuantUIApp

    a = QuantUIApp()
    a._set_molecule(_h2o2())
    return a


def _output_html(out) -> str:
    return "".join(o.get("data", {}).get("text/html", "") for o in out.outputs)


class TestEcdPanel:
    def test_registered_after_uv_vis(self):
        from quantui.app import QuantUIApp

        names = [n for n, _, _ in QuantUIApp._PANEL_META]
        assert names.index("ECD") == names.index("UV-Vis") + 1
        entries = [n for n, _, _ in QuantUIApp._PANEL_REGISTRY["tddft"]]
        assert entries[:2] == ["UV-Vis", "ECD"]

    def test_history_round_trip_activates_ecd(self, tmp_path, app):
        from quantui.results_storage import load_result, save_result

        res = _tddft_ns([-9.6, 31.1, -14.9], [-0.2, 14.2, -6.4])
        saved = save_result(
            res, results_dir=tmp_path, calc_type="tddft", spectra=_spectra(res)
        )
        uv = load_result(saved)["spectra"]["uv_vis"]
        assert uv["rotatory_strengths_velocity"] == [-9.6, 31.1, -14.9]
        assert uv["rotatory_strengths_length"] == [-0.2, 14.2, -6.4]
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert {"UV-Vis", "ECD"} <= app._ana_available
        assert app._last_ecd_rot_velocity == [-9.6, 31.1, -14.9]
        assert app._last_ecd_fig is not None
        html = _output_html(app._ecd_fig)
        assert "not chiral" not in html
        assert 'include_plotlyjs="cdn"' not in html

    def test_old_result_without_rotatory_strengths(self, tmp_path, app):
        """Results saved before SPC2.3 still open: UV-Vis yes, ECD placeholder."""
        from quantui.results_storage import save_result

        res = _tddft_ns([])
        spectra = _spectra(res)
        for k in ("rotatory_strengths_velocity", "rotatory_strengths_length"):
            spectra["uv_vis"].pop(k)
        saved = save_result(
            res, results_dir=tmp_path, calc_type="tddft", spectra=spectra
        )
        app._apply_analysis_context(app._build_history_context(saved))
        assert "UV-Vis" in app._ana_available
        assert "ECD" not in app._ana_available
        assert "rotatory strengths" in app._ana_unavail_msgs["ECD"].value

    def test_live_achiral_shows_not_chiral_note(self, app):
        from quantui.app import _AnalysisContext

        res = _tddft_ns([1e-10, -2e-11, 0.0])
        ctx = _AnalysisContext(
            calc_type="tddft",
            formula="H2O",
            method="B3LYP",
            basis="6-31G",
            live_result=res,
        )
        assert app._pop_ecd(ctx) is True
        assert "not chiral" in _output_html(app._ecd_fig)

    def test_stick_mode_and_copy_data(self, app):
        from quantui.app import _AnalysisContext

        ctx = _AnalysisContext(
            calc_type="tddft",
            formula="H2O2",
            method="B3LYP",
            basis="6-31G",
            live_result=_tddft_ns([-9.6, 31.1, -14.9]),
        )
        assert app._pop_ecd(ctx)
        app._update_ecd_figure("Stick", 0.3)
        assert app._last_ecd_fig.layout.yaxis.title.text.startswith("R")
        csv = app._fig_to_csv(app._last_ecd_fig, title="ECD Spectrum")
        assert "31.1" in csv

    def test_mismatched_lengths_are_rejected(self, app):
        from quantui.app import _AnalysisContext

        ctx = _AnalysisContext(
            calc_type="tddft",
            formula="H2O2",
            method="B3LYP",
            basis="6-31G",
            live_result=_tddft_ns([1.0]),
        )
        assert app._pop_ecd(ctx) is False


class TestWorkerPayloadAndHelp:
    def test_worker_payload_carries_rotatory_strengths(self):
        from quantui.backends.worker_payload import tddft_result_payload

        res = _tddft_ns([-9.6, 31.1, -14.9], [-0.2, 14.2, -6.4])
        uv = tddft_result_payload(res)["spectra"]["uv_vis"]
        assert uv["rotatory_strengths_velocity"] == [-9.6, 31.1, -14.9]
        assert uv["rotatory_strengths_length"] == [-0.2, 14.2, -6.4]

    def test_worker_payload_without_rotatory_strengths(self):
        from quantui.backends.worker_payload import tddft_result_payload

        res = _tddft_ns([])
        del res.rotatory_strengths_velocity
        del res.rotatory_strengths_length
        uv = tddft_result_payload(res)["spectra"]["uv_vis"]
        assert uv["rotatory_strengths_velocity"] == []

    def test_help_topic(self):
        from quantui.help_content import HELP_TOPICS

        body = HELP_TOPICS["ecd"]["body"]
        for phrase in (
            "Rotatory strength",
            "Velocity gauge",
            "mirror-image",
            "Conformers",
        ):
            assert phrase in body
