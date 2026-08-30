"""Tests for normal-mode geometry perturbation (imaginary-frequency follow)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantui.freq_calc import (
    DEFAULT_MODE_PERTURBATION_FRACTION,
    FREQ_SEED_PREFIX,
    VIB_MODE_DISPLAY_AMPLITUDE_ANGSTROM,
    freq_mode_seed_result_dir,
    is_freq_mode_seed,
    load_frequency_mode_seed_data,
    molecule_from_freq_mode_seed,
    perturb_along_mode,
)
from quantui.molecule import Molecule
from quantui.results_storage import save_result


def _water_mol() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        charge=0,
        multiplicity=1,
    )


def _stretch_mode_displacements() -> list:
    # Mode 0: symmetric O-H stretch on a 3-atom toy system
    return [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        [[0.0, 1.0, 0.0], [0.0, -0.5, 0.0], [0.0, -0.5, 0.0]],
    ]


class TestPerturbAlongMode:
    def test_applies_fraction_of_animation_amplitude(self):
        mol = _water_mol()
        disps = _stretch_mode_displacements()
        out = perturb_along_mode(mol, disps, 0, fraction=0.75)
        expected_scale = 0.75 * VIB_MODE_DISPLAY_AMPLITUDE_ANGSTROM
        base = np.array(mol.coordinates, dtype=float)
        disp = np.array(disps[0], dtype=float)
        expected = base + expected_scale * disp
        assert np.allclose(out.coordinates, expected, atol=1e-9)

    def test_sign_flips_direction(self):
        mol = _water_mol()
        disps = _stretch_mode_displacements()
        pos = perturb_along_mode(mol, disps, 0, fraction=1.0, sign=1.0)
        neg = perturb_along_mode(mol, disps, 0, fraction=1.0, sign=-1.0)
        mid = np.array(mol.coordinates, dtype=float)
        assert np.allclose(
            np.array(pos.coordinates) - mid, -(np.array(neg.coordinates) - mid)
        )

    def test_rejects_shape_mismatch(self):
        mol = _water_mol()
        with pytest.raises(ValueError, match="shape"):
            perturb_along_mode(mol, [[[0.0, 0.0]]], 0)

    def test_rejects_bad_mode_index(self):
        mol = _water_mol()
        with pytest.raises(ValueError, match="out of range"):
            perturb_along_mode(mol, _stretch_mode_displacements(), 5)


class TestFreqModeSeedHelpers:
    def test_seed_prefix_roundtrip(self):
        path = f"{FREQ_SEED_PREFIX}/tmp/result123"
        assert is_freq_mode_seed(path)
        assert freq_mode_seed_result_dir(path) == Path("/tmp/result123")

    def test_load_from_saved_frequency_result(self, tmp_path):
        mol = _water_mol()
        disps = _stretch_mode_displacements()
        result = SimpleNamespace(
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-76.0,
            converged=True,
        )
        out_dir = save_result(
            result,
            results_dir=tmp_path,
            calc_type="frequency",
            spectra={
                "ir": {
                    "frequencies_cm1": [-500.0, 1600.0],
                    "ir_intensities": [0.0, 1.0],
                    "displacements": disps,
                },
                "molecule": {
                    "atoms": mol.atoms,
                    "coords": mol.coordinates,
                    "charge": mol.charge,
                    "multiplicity": mol.multiplicity,
                },
            },
        )
        loaded_mol, loaded_disps, freqs = load_frequency_mode_seed_data(out_dir)
        assert loaded_mol.atoms == mol.atoms
        assert np.allclose(loaded_mol.coordinates, mol.coordinates)
        assert len(loaded_disps) == 2
        assert freqs[0] == -500.0

    def test_molecule_from_freq_mode_seed(self, tmp_path):
        mol = _water_mol()
        disps = _stretch_mode_displacements()
        result = SimpleNamespace(
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-76.0,
            converged=True,
        )
        out_dir = save_result(
            result,
            results_dir=tmp_path,
            calc_type="frequency",
            spectra={
                "ir": {
                    "frequencies_cm1": [-500.0, 1600.0],
                    "displacements": disps,
                },
                "molecule": {
                    "atoms": mol.atoms,
                    "coords": mol.coordinates,
                    "charge": 0,
                    "multiplicity": 1,
                },
            },
        )
        perturbed, meta = molecule_from_freq_mode_seed(
            out_dir,
            mode_number=1,
            fraction=DEFAULT_MODE_PERTURBATION_FRACTION,
        )
        assert meta["mode_number"] == 1
        assert meta["frequency_cm1"] == -500.0
        assert perturbed.atoms == mol.atoms
        assert not np.allclose(perturbed.coordinates, mol.coordinates)
