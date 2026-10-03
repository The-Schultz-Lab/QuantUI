"""Vibration panel polish and thermochemistry at any T/P (DEC-023 Tier-1 #5/#6)."""

from __future__ import annotations

import io
import math
from types import SimpleNamespace

import pytest

WATER = (
    ["O", "H", "H"],
    [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]],
)
# 4-decimal coordinates: exact symmetry is only Cs at PySCF's strict tolerance.
NH3 = (
    ["N", "H", "H", "H"],
    [
        [0, 0, 0.1],
        [0.9377, 0, -0.27],
        [-0.4689, 0.8121, -0.27],
        [-0.4689, -0.8121, -0.27],
    ],
)
R = 8.314462618


class TestModeTable:
    def test_columns_follow_available_data(self):
        from quantui.app_visualization import vib_mode_table_html

        fr = SimpleNamespace(
            frequencies_cm1=[-120.0, 1650.0, 3700.0],
            ir_intensities=[5.0, 70.2, 12.0],
            raman_activities=[],
        )
        html = vib_mode_table_html(fr)
        assert "IR (km/mol)" in html and "Raman" not in html
        assert "120.0i" in html  # imaginary shown with i
        assert "70.2" in html

    def test_skips_translation_rotation_noise(self):
        from quantui.app_visualization import vib_mode_table_html

        fr = SimpleNamespace(frequencies_cm1=[3.0, 1650.0], ir_intensities=[])
        html = vib_mode_table_html(fr)
        assert "<td style='padding:2px 10px;text-align:right'>1</td>" not in html
        assert "1650.0" in html

    def test_empty(self):
        from quantui.app_visualization import vib_mode_table_html

        assert vib_mode_table_html(SimpleNamespace(frequencies_cm1=[])) == ""


class TestVibViewerJs:
    def test_arrows_and_live_amplitude_hooks(self):
        import numpy as np

        from quantui.app_visualization import build_vib_viewer_html
        from quantui.molecule import Molecule

        mol = Molecule(atoms=WATER[0], coordinates=WATER[1])
        disp = np.zeros((3, 3, 3))
        disp[0, 1, 1] = 0.7
        fr = SimpleNamespace(displacements=disp.tolist())
        html = build_vib_viewer_html(mol, fr, [1, 2, 3], 1, amplitude=0.6, arrows=True)
        compact = html.replace(" ", "")
        assert "varARROWS=true" in compact
        assert "AMP=0.6" in compact
        assert "__quantuiVibSetAmp" in html and "__quantuiVibSetArrows" in html
        assert "addArrow" in html
        off = build_vib_viewer_html(mol, fr, [1], 1)
        assert "ARROWS=false" in off.replace(" ", "")


@pytest.mark.slow
class TestThermochemistry:
    @pytest.fixture(autouse=True)
    def _pyscf(self):
        pytest.importorskip("pyscf")

    @staticmethod
    def _freq(geom):
        from quantui.freq_calc import run_freq_calc
        from quantui.molecule import Molecule

        mol = Molecule(atoms=geom[0], coordinates=geom[1])
        return mol, run_freq_calc(mol, "RHF", "STO-3G", progress_stream=io.StringIO())

    def test_recompute_matches_the_calculation_at_298(self):
        from quantui.freq_calc import compute_thermochemistry

        mol, fr = self._freq(WATER)
        td = compute_thermochemistry(
            mol.atoms,
            mol.coordinates,
            energy_hartree=fr.energy_hartree,
            frequencies_cm1=fr.frequencies_cm1,
        )
        assert td.G_hartree == pytest.approx(fr.thermo.G_hartree, abs=1e-8)
        assert td.H_hartree == pytest.approx(fr.thermo.H_hartree, abs=1e-8)
        assert td.S_jmol == pytest.approx(fr.thermo.S_jmol, abs=1e-6)
        assert td.zpve_hartree == pytest.approx(fr.thermo.zpve_hartree, abs=1e-9)

    @pytest.mark.parametrize("geom, sigma", [(WATER, 2), (NH3, 3)])
    def test_rotational_symmetry_number_uses_teaching_tolerance(self, geom, sigma):
        # PySCF's strict tolerance gives sigma = 1 for the rounded NH3,
        # overstating S_rot by R ln 3.
        _, fr = self._freq(geom)
        assert fr.thermo.symmetry_number == sigma

    def test_temperature_and_pressure_dependence(self):
        from quantui.freq_calc import compute_thermochemistry

        mol, fr = self._freq(WATER)

        def at(temp, pres):
            return compute_thermochemistry(
                mol.atoms,
                mol.coordinates,
                energy_hartree=fr.energy_hartree,
                frequencies_cm1=fr.frequencies_cm1,
                temperature_k=temp,
                pressure_atm=pres,
            )

        t298, t500, t298_10 = at(298.15, 1.0), at(500.0, 1.0), at(298.15, 10.0)
        assert t500.S_jmol > t298.S_jmol
        assert t500.H_hartree > t298.H_hartree
        # Ideal gas: S(P2) - S(P1) = -R ln(P2/P1); H is pressure-independent.
        assert t298.S_jmol - t298_10.S_jmol == pytest.approx(R * math.log(10), rel=1e-4)
        assert t298_10.H_hartree == pytest.approx(t298.H_hartree, abs=1e-10)
        # Ideal gas: Cp - Cv = R.
        assert t298.Cp_jmolk - t298.Cv_jmolk == pytest.approx(R, rel=1e-4)
        assert t298.E_thermal_hartree < t298.H_hartree  # H = U + RT

    def test_rejects_nonphysical_inputs(self):
        from quantui.freq_calc import compute_thermochemistry

        with pytest.raises(ValueError):
            compute_thermochemistry(
                WATER[0],
                WATER[1],
                energy_hartree=-75.0,
                frequencies_cm1=[1600.0],
                temperature_k=0,
            )

    def test_app_thermo_box_recomputes_on_input(self):
        from quantui.app import QuantUIApp
        from quantui.app_visualization import show_thermo_box

        mol, fr = self._freq(WATER)
        app = QuantUIApp()
        show_thermo_box(app, mol, fr.energy_hartree, fr.frequencies_cm1)
        assert app._thermo_box.layout.display == ""
        assert "At 298.15 K" in app._thermo_html.value
        assert "Gibbs" in app._thermo_html.value and "σ" in app._thermo_html.value
        app._thermo_T.value = 400.0
        assert "At 400.00 K" in app._thermo_html.value

    def test_app_thermo_box_hidden_without_energy(self):
        from quantui.app import QuantUIApp
        from quantui.app_visualization import show_thermo_box

        app = QuantUIApp()
        show_thermo_box(app, None, None, [])
        assert app._thermo_box.layout.display == "none"
