"""Unit tests for extracted app formatter helpers."""

from __future__ import annotations

from types import SimpleNamespace

from quantui.app_formatters import (
    format_freq_result,
    format_nmr_result,
    format_opt_result,
    format_past_result,
    format_pes_scan_result,
    format_result,
    format_tddft_result,
)


class _FreqStub(SimpleNamespace):
    def n_real_modes(self) -> int:
        return sum(1 for f in self.frequencies_cm1 if f > 0)

    def n_imaginary_modes(self) -> int:
        return sum(1 for f in self.frequencies_cm1 if f < 0)


class _NMRStub(SimpleNamespace):
    def h_shifts(self) -> list[tuple[int, float]]:
        return [
            (i, d)
            for i, d in self.chemical_shifts_ppm.items()
            if self.atom_symbols[i] == "H"
        ]

    def c_shifts(self) -> list[tuple[int, float]]:
        return [
            (i, d)
            for i, d in self.chemical_shifts_ppm.items()
            if self.atom_symbols[i] == "C"
        ]


class _TDDFTStub(SimpleNamespace):
    def wavelengths_nm(self) -> list[float]:
        return [1239.841984 / e for e in self.excitation_energies_ev]


def test_format_result_includes_mp2_rows():
    result = SimpleNamespace(
        converged=True,
        homo_lumo_gap_ev=None,
        energy_hartree=-76.3,
        energy_ev=-2076.2,
        n_iterations=12,
        mp2_correlation_hartree=-0.3,
        solvent="Water",
        dipole_moment_debye=1.2,
        mulliken_charges=[-0.5, 0.25, 0.25],
        atom_symbols=["O", "H", "H"],
        formula="H2O",
        method="MP2",
        basis="def2-SVP",
    )
    html = format_result(result)
    assert "HF reference" in html
    assert "MP2 correlation" in html
    assert "Solvent (PCM)" in html


def test_format_opt_result_contains_geometry_fields():
    result = SimpleNamespace(
        converged=True,
        energy_hartree=-40.0,
        energy_change_hartree=-0.1,
        n_steps=7,
        rmsd_angstrom=0.03,
        formula="NH3",
        method="RHF",
        basis="6-31G",
    )
    html = format_opt_result(result)
    assert "Geometry Optimisation" in html
    assert "Steps taken" in html


def test_format_freq_result_highlights_imaginary_modes():
    result = _FreqStub(
        converged=True,
        frequencies_cm1=[-50.0, 1200.0, 1600.0],
        energy_hartree=-75.0,
        zpve_hartree=0.021,
        formula="H2O",
        method="B3LYP",
        basis="def2-SVP",
        thermo=None,
    )
    html = format_freq_result(result)
    assert "Frequency Analysis" in html
    assert "Imaginary modes" in html


def test_format_tddft_result_lists_excitations():
    result = _TDDFTStub(
        converged=True,
        energy_hartree=-100.0,
        excitation_energies_ev=[2.0, 3.0],
        oscillator_strengths=[0.1, 0.02],
        formula="C2H4",
        method="CAM-B3LYP",
        basis="def2-SVP",
    )
    html = format_tddft_result(result)
    assert "Vertical excitations" in html
    assert "S1" in html


def test_format_nmr_result_warns_on_small_basis():
    result = _NMRStub(
        converged=True,
        reference_compound="TMS",
        atom_symbols=["C", "H", "H", "H", "H"],
        chemical_shifts_ppm={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},
        formula="CH4",
        method="B3LYP",
        basis="STO-3G",
    )
    html = format_nmr_result(result)
    assert "qualitative NMR only" in html


def test_format_pes_scan_result_reports_range_and_convergence():
    result = SimpleNamespace(
        converged_all=True,
        energies_hartree=[-40.0, -39.95, -39.98],
        atom_indices=[0, 1],
        scan_type="bond",
        scan_parameter_values=[1.0, 1.1, 1.2],
        scan_unit="A",
        n_steps=3,
        formula="H2",
        method="RHF",
        basis="STO-3G",
    )
    html = format_pes_scan_result(result)
    assert "PES Scan" in html
    assert "All converged" in html


def test_format_past_result_contains_calc_type_badge():
    data = {
        "calc_type": "single_point",
        "converged": True,
        "homo_lumo_gap_ev": 10.0,
        "energy_hartree": -75.0,
        "energy_ev": -2040.0,
        "n_iterations": 10,
        "timestamp": "2026-05-02_12-00-00-000001",
        "formula": "H2O",
        "method": "RHF",
        "basis": "STO-3G",
    }
    html = format_past_result(data)
    assert "Single Point" in html
    assert "H2O" in html
