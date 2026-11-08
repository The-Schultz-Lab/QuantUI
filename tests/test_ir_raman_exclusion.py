"""M-SPECTRA2 SPC2.5: IR/Raman selection rules, the automated half.

Teaching point: in a molecule with a centre of inversion, a mode is IR
active only if it is *u* (antisymmetric under inversion: it changes the
dipole) and Raman active only if it is *g* (it changes the polarizability),
so no mode is both — the rule of mutual exclusion. CO2 (D∞h, linear, 3N − 5
= 4 modes) shows it: the symmetric stretch (σg⁺) is Raman-only, the
antisymmetric stretch (σu⁺) and the degenerate bend (πu) are IR-only.
Water (C2v, no inversion centre) is the contrast: all three modes (2a1 + b2)
are both IR and Raman active.

Live RHF/STO-3G Hessians through :func:`quantui.freq_calc.run_freq_calc`
(numerical IR intensities and Raman activities), the SPC2.4 mode irreps, the
IR/Raman figure builders with zero activities, and History replay. The
Voilà visual half of SPC2.5 is a local manual check, not covered here.
"""

from __future__ import annotations

import io
import json
import re
import sys

import numpy as np
import pytest

try:
    import pyscf  # noqa: F401

    _HAS_PYSCF = sys.platform != "win32"
except ImportError:  # pragma: no cover - depends on the environment
    _HAS_PYSCF = False

needs_pyscf = pytest.mark.skipif(
    not _HAS_PYSCF, reason="PySCF not available or not supported on native Windows"
)

# "Active" = above 1 % of the strongest band of that spectrum; "inactive" =
# below 0.1 % of it (the symmetry-forbidden values come out at ~1e-6 of the
# allowed ones; finite-difference noise, not zero).
ACTIVE = 1e-2
INACTIVE = 1e-3


def _co2():
    from quantui.molecule import Molecule

    # Near the RHF/STO-3G C=O bond length; exactly linear and centrosymmetric.
    return Molecule(["O", "C", "O"], [[0, 0, -1.188], [0, 0, 0], [0, 0, 1.188]])


def _water():
    from quantui.molecule import Molecule

    return Molecule(
        ["O", "H", "H"],
        [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]],
    )


def _freq(mol):
    from quantui.freq_calc import run_freq_calc

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("QUANTUI_RAMAN", "1")
        return run_freq_calc(
            mol, method="RHF", basis="STO-3G", progress_stream=io.StringIO()
        )


@pytest.fixture(scope="module")
def co2_freq():
    if not _HAS_PYSCF:
        pytest.skip("PySCF not available")
    r = _freq(_co2())
    assert r.converged
    return r


@pytest.fixture(scope="module")
def water_freq():
    if not _HAS_PYSCF:
        pytest.skip("PySCF not available")
    r = _freq(_water())
    assert r.converged
    return r


def _activity(values):
    """Each value relative to the strongest band of its spectrum."""
    v = np.asarray(values, dtype=float)
    return v / v.max()


# ---------------------------------------------------------------------------
# CO2: mutual exclusion
# ---------------------------------------------------------------------------


@needs_pyscf
@pytest.mark.slow
class TestCO2MutualExclusion:
    def test_linear_molecule_has_3n_minus_5_real_modes(self, co2_freq):
        f = co2_freq.frequencies_cm1
        assert len(f) == 3 * 3 - 5
        assert all(x > 0 for x in f)
        assert len(co2_freq.ir_intensities) == len(co2_freq.raman_activities) == 4
        # Doubly degenerate bend, then symmetric < antisymmetric stretch
        # (experiment: 667 < 1333 < 2349 cm⁻¹).
        assert f[0] == pytest.approx(f[1], abs=0.5)
        assert f[1] < f[2] < f[3]

    def test_no_mode_is_both_ir_and_raman_active(self, co2_freq):
        ir = _activity(co2_freq.ir_intensities)
        raman = _activity(co2_freq.raman_activities)
        for i, (a, b) in enumerate(zip(ir, raman)):
            ir_on, raman_on = a > ACTIVE, b > ACTIVE
            assert ir_on != raman_on, f"mode {i}: IR {a:.3g}, Raman {b:.3g}"
            # …and the forbidden one is essentially zero, not just small.
            assert min(a, b) < INACTIVE, f"mode {i}: IR {a:.3g}, Raman {b:.3g}"

    def test_mode_irreps(self, co2_freq):
        from quantui.app_visualization import freq_mode_irreps
        from quantui.symmetry import pretty_group

        irreps = freq_mode_irreps(co2_freq)
        assert irreps is not None
        assert pretty_group(irreps.group) == "D∞h" and not irreps.is_subgroup
        assert irreps.labels == ["πu", "πu", "σg⁺", "σu⁺"]
        assert irreps.gamma == "Σg⁺ + Σu⁺ + Πu"

    def test_g_modes_are_raman_only_u_modes_are_ir_only(self, co2_freq):
        from quantui.app_visualization import freq_mode_irreps

        labels = freq_mode_irreps(co2_freq).labels
        ir = _activity(co2_freq.ir_intensities)
        raman = _activity(co2_freq.raman_activities)
        for lbl, a, b in zip(labels, ir, raman):
            if "g" in lbl:
                assert b > ACTIVE and a < INACTIVE, lbl
            else:
                assert "u" in lbl
                assert a > ACTIVE and b < INACTIVE, lbl

    def test_modes_have_the_textbook_shapes(self, co2_freq):
        # Ties each irrep label to the motion it names, independently of the
        # symmetry code: z is the molecular axis, atoms are O, C, O.
        d = np.asarray(co2_freq.displacements, dtype=float)
        assert d.shape == (4, 3, 3)
        for k in (0, 1):  # πu bend: every atom moves perpendicular to the axis
            assert np.abs(d[k][:, 2]).max() < 1e-6
            assert np.abs(d[k][1]).max() > 1e-3
        sym = d[2]  # σg⁺: the O atoms move apart, C stays put
        assert np.abs(sym[1]).max() < 1e-6
        assert sym[0, 2] == pytest.approx(-sym[2, 2], rel=1e-4)
        asym = d[3]  # σu⁺: the O atoms move together, C against them
        assert asym[0, 2] == pytest.approx(asym[2, 2], rel=1e-4)
        assert np.sign(asym[1, 2]) == -np.sign(asym[0, 2])
        assert np.abs(asym[:, :2]).max() < 1e-6


# ---------------------------------------------------------------------------
# Water: no inversion centre, every mode active in both
# ---------------------------------------------------------------------------


@needs_pyscf
@pytest.mark.slow
class TestWaterAllModesActive:
    def test_three_modes_labelled_2a1_plus_b2(self, water_freq):
        from quantui.app_visualization import freq_mode_irreps

        assert len(water_freq.frequencies_cm1) == 3 * 3 - 6
        irreps = freq_mode_irreps(water_freq)
        assert irreps.group == "C2v"
        # Bend and symmetric stretch a1, antisymmetric stretch b2.
        assert irreps.labels == ["a1", "a1", "b2"]
        assert irreps.gamma == "2A1 + B2"

    def test_every_mode_is_both_ir_and_raman_active(self, water_freq):
        ir = _activity(water_freq.ir_intensities)
        raman = _activity(water_freq.raman_activities)
        assert (ir > ACTIVE).all(), ir
        assert (raman > ACTIVE).all(), raman


# ---------------------------------------------------------------------------
# The app's IR / Raman panels on the CO2 result
# ---------------------------------------------------------------------------


@needs_pyscf
@pytest.mark.slow
class TestExclusionInThePanels:
    @pytest.fixture
    def app(self):
        from quantui.app import QuantUIApp

        return QuantUIApp()

    def test_raman_panel_plots_ir_only_modes_at_zero(self, app, co2_freq):
        assert app._show_raman_spectrum(co2_freq)
        assert app._raman_accordion.get_title(0) == "Raman Spectrum"
        assert app._last_raman_labels == ["πu", "πu", "σg⁺", "σu⁺"]
        peaks = app._last_raman_fig.data[-1]
        y = _activity(peaks.y)
        assert list(y > ACTIVE) == [False, False, True, False]
        assert "σ<sub>g</sub>⁺" in peaks.customdata[2]

    def test_ir_panel_plots_the_raman_only_mode_at_zero(self, app, co2_freq):
        assert app._show_ir_spectrum(co2_freq)
        assert app._ir_intensities_real
        peaks = app._last_ir_fig.data[-1]
        assert len(peaks.x) == 4
        # The σg⁺ peak is drawn, at (essentially) zero height.
        y = _activity(peaks.y)
        assert list(y > ACTIVE) == [True, True, False, True]
        assert "σ<sub>u</sub>⁺" in peaks.customdata[3]

    def test_mode_table_shows_the_exclusion(self, co2_freq):
        from quantui.app_visualization import freq_mode_irreps, vib_mode_table_html

        html = vib_mode_table_html(co2_freq, freq_mode_irreps(co2_freq))
        assert "IR (km/mol)" in html and "Raman (Å⁴/amu)" in html
        assert "Γ<sub>vib</sub> = Σ<sub>g</sub>⁺ + Σ<sub>u</sub>⁺ + Π<sub>u</sub>" in (
            html
        )
        rows = [
            re.findall(r"<td[^>]*>(.*?)</td>", row) for row in html.split("<tr")[2:]
        ]
        assert len(rows) == 4
        # Columns: mode, ν, symmetry, IR, Raman.
        ir = [float(r[3]) for r in rows]
        raman = [float(r[4]) for r in rows]
        assert [r[2] for r in rows] == [
            "π<sub>u</sub>",
            "π<sub>u</sub>",
            "σ<sub>g</sub>⁺",
            "σ<sub>u</sub>⁺",
        ]
        assert ir[2] == 0.0 and min(ir[0], ir[1], ir[3]) > 1.0
        assert raman[2] > 1.0 and raman[0] == raman[1] == raman[3] == 0.0


@needs_pyscf
@pytest.mark.slow
class TestExclusionSurvivesHistory:
    def test_history_replay_restores_activities_and_labels(
        self, co2_freq, tmp_path, monkeypatch
    ):
        from quantui.app import QuantUIApp
        from quantui.results_storage import save_result

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        mol = _co2()
        spectra = {
            # Same shape as QuantUIApp._do_run's Frequency branch.
            "ir": {
                "frequencies_cm1": co2_freq.frequencies_cm1,
                "ir_intensities": co2_freq.ir_intensities,
                "raman_activities": co2_freq.raman_activities,
                "zpve_hartree": co2_freq.zpve_hartree,
                "displacements": np.asarray(co2_freq.displacements).tolist(),
            },
            "molecule": {
                "atoms": list(mol.atoms),
                "coords": [list(map(float, row)) for row in mol.coordinates],
                "charge": 0,
                "multiplicity": 1,
            },
        }
        saved = save_result(
            co2_freq, results_dir=tmp_path, calc_type="frequency", spectra=spectra
        )
        json.loads((saved / "result.json").read_text())  # valid JSON on disk

        app = QuantUIApp()
        app._history_load_analysis(saved)
        assert {"IR Spectrum", "Raman Spectrum"} <= set(app._ana_available)
        assert app._last_raman_labels == ["πu", "πu", "σg⁺", "σu⁺"]
        assert app._last_ir_labels == ["πu", "πu", "σg⁺", "σu⁺"]
        ir = _activity(app._last_ir_ints)
        raman = _activity(app._last_raman_acts)
        assert list(ir > ACTIVE) == [True, True, False, True]
        assert list(raman > ACTIVE) == [False, False, True, False]


# ---------------------------------------------------------------------------
# Figure builders with zero activities (no PySCF)
# ---------------------------------------------------------------------------


class TestZeroActivityFigures:
    FREQS = [566.2, 566.2, 1434.9, 2534.5]
    LABELS = ["πu", "πu", "σg⁺", "σu⁺"]

    def test_raman_stick_keeps_zero_peaks_with_labels(self):
        from quantui.raman_plot import plot_raman_spectrum

        fig = plot_raman_spectrum(
            self.FREQS, [0.0, 0.0, 7.87, 0.0], mode="stick", labels=self.LABELS
        )
        peaks = fig.data[-1]
        assert list(peaks.y) == [0.0, 0.0, 7.87, 0.0]
        assert len(peaks.customdata) == 4

    @pytest.mark.parametrize("mode", ["stick", "broadened"])
    def test_all_zero_raman_is_a_flat_finite_trace(self, mode):
        from quantui.raman_plot import plot_raman_spectrum

        fig = plot_raman_spectrum(self.FREQS, [0.0] * 4, mode=mode)
        assert fig.data
        y = np.array([v for v in fig.data[0].y if v is not None], dtype=float)
        assert np.isfinite(y).all() and (y == 0).all()

    @pytest.mark.parametrize("mode", ["stick", "broadened"])
    def test_ir_with_a_forbidden_band(self, mode):
        from quantui.ir_plot import plot_ir_spectrum

        ints = [33.95, 33.95, 0.0, 342.05]
        fig = plot_ir_spectrum(self.FREQS, ints, mode=mode, labels=self.LABELS)
        y = np.array([v for v in fig.data[0].y if v is not None], dtype=float)
        assert np.isfinite(y).all() and y.max() > 0

    def test_all_zero_raman_is_still_a_computed_spectrum(self):
        # Zero activities are a result (every mode Raman-forbidden), not
        # "Raman not computed".
        from types import SimpleNamespace

        from quantui.app import QuantUIApp

        app = QuantUIApp()
        r = SimpleNamespace(frequencies_cm1=[1000.0, 2000.0], raman_activities=[0, 0])
        assert app._show_raman_spectrum(r)
        assert app._raman_accordion.get_title(0) == "Raman Spectrum"
        assert app._last_raman_fig is not None
