"""Tests for quantui/results_storage.py.

Covers save_result, load_result, list_results, save_orbitals, load_orbitals,
save_trajectory, load_trajectory, and save_thumbnail end-to-end using
tmp_path fixtures (no mocking of the storage layer itself).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantui.molecule import Molecule
from quantui.results_storage import (
    geometry_payload_for_result,
    list_results,
    load_orbitals,
    load_result,
    load_trajectory,
    molecule_from_geometry_payload,
    save_orbitals,
    save_result,
    save_thumbnail,
    save_trajectory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(**overrides):
    defaults = dict(
        formula="H2O",
        method="RHF",
        basis="STO-3G",
        energy_hartree=-75.0,
        energy_ev=-2040.5,
        homo_lumo_gap_ev=10.5,
        converged=True,
        n_iterations=8,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _water_traj():
    mol = Molecule(["O", "H", "H"], [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
    return [mol, mol], [-75.0, -75.1]


# ---------------------------------------------------------------------------
# save_result / load_result
# ---------------------------------------------------------------------------


class TestSaveResult:
    def test_creates_result_directory(self, tmp_path):
        saved = save_result(_make_result(), results_dir=tmp_path)
        assert saved.is_dir()

    def test_writes_result_json(self, tmp_path):
        saved = save_result(_make_result(), results_dir=tmp_path)
        assert (saved / "result.json").exists()

    def test_result_json_fields(self, tmp_path):
        saved = save_result(
            _make_result(), results_dir=tmp_path, calc_type="single_point"
        )
        data = json.loads((saved / "result.json").read_text())
        assert data["formula"] == "H2O"
        assert data["method"] == "RHF"
        assert data["basis"] == "STO-3G"
        assert data["energy_hartree"] == pytest.approx(-75.0)
        assert data["converged"] is True
        assert data["calc_type"] == "single_point"
        assert data["_schema_version"] == 2

    def test_geometry_persisted_when_molecule_provided(self, tmp_path):
        mol = Molecule(
            ["O", "H", "H"],
            [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
        )
        saved = save_result(
            _make_result(), results_dir=tmp_path, calc_type="single_point", molecule=mol
        )
        data = json.loads((saved / "result.json").read_text())
        assert data["geometry"]["atoms"] == ["O", "H", "H"]
        assert len(data["geometry"]["coordinates"]) == 3
        restored = molecule_from_geometry_payload(data["geometry"])
        assert restored.atoms == mol.atoms
        for row, expected in zip(restored.coordinates, mol.coordinates):
            assert row == pytest.approx(expected)

    def test_geometry_payload_from_pyscf_mol_atom(self, tmp_path):
        result = _make_result(
            pyscf_mol_atom=[
                ("O", [0.0, 0.0, 0.0]),
                ("H", [0.757, 0.586, 0.0]),
                ("H", [-0.757, 0.586, 0.0]),
            ]
        )
        payload = geometry_payload_for_result(result)
        assert payload is not None
        assert payload["atoms"] == ["O", "H", "H"]
        saved = save_result(result, results_dir=tmp_path)
        data = json.loads((saved / "result.json").read_text())
        assert "geometry" in data

    def test_spectra_stored_in_json(self, tmp_path):
        spectra = {
            "ir": {"frequencies_cm1": [1000.0, 2000.0], "ir_intensities": [1.0, 2.0]}
        }
        saved = save_result(_make_result(), results_dir=tmp_path, spectra=spectra)
        data = json.loads((saved / "result.json").read_text())
        assert data["spectra"]["ir"]["frequencies_cm1"] == [1000.0, 2000.0]

    def test_scf_variant_persisted(self, tmp_path):
        """M-UX2 UXP2.10 — a saved result records which SCF variant ran."""
        saved = save_result(_make_result(scf_variant="UKS"), results_dir=tmp_path)
        data = json.loads((saved / "result.json").read_text())
        assert data["scf_variant"] == "UKS"

    def test_scf_variant_null_when_absent(self, tmp_path):
        """A calc type that never sets scf_variant (e.g. geometry_opt)
        serializes null, not a missing key or an empty string."""
        saved = save_result(_make_result(), results_dir=tmp_path)
        data = json.loads((saved / "result.json").read_text())
        assert data["scf_variant"] is None

    def test_pyscf_log_written_when_provided(self, tmp_path):
        saved = save_result(
            _make_result(),
            pyscf_log="converged SCF energy = -75.0",
            results_dir=tmp_path,
        )
        log_path = saved / "pyscf.log"
        assert log_path.exists()
        assert "converged" in log_path.read_text()

    def test_no_pyscf_log_when_empty(self, tmp_path):
        saved = save_result(_make_result(), pyscf_log="", results_dir=tmp_path)
        assert not (saved / "pyscf.log").exists()

    def test_directory_name_contains_formula_method_basis(self, tmp_path):
        saved = save_result(_make_result(), results_dir=tmp_path)
        assert "H2O" in saved.name
        assert "RHF" in saved.name
        assert "STO-3G" in saved.name

    def test_missing_optional_fields_use_defaults(self, tmp_path):
        minimal = SimpleNamespace(
            formula="H2", method="RHF", basis="STO-3G", energy_hartree=-1.0
        )
        saved = save_result(minimal, results_dir=tmp_path)
        data = json.loads((saved / "result.json").read_text())
        assert data["homo_lumo_gap_ev"] is None
        assert data["converged"] is None
        assert data["n_iterations"] == -1

    def test_returns_path_to_created_directory(self, tmp_path):
        saved = save_result(_make_result(), results_dir=tmp_path)
        assert isinstance(saved, type(tmp_path))
        assert saved.parent == tmp_path

    def test_each_call_creates_unique_directory(self, tmp_path):
        d1 = save_result(_make_result(), results_dir=tmp_path)
        d2 = save_result(_make_result(), results_dir=tmp_path)
        assert d1 != d2


class TestSaveResultJsonSafeCoercion:
    """L audit fix: save_result must coerce every numeric/boolean field to a
    plain Python type before json.dumps.

    numpy.float64 happens to subclass float (so it round-trips through
    json.dumps by accident), but numpy.float32, numpy.int64, and
    numpy.bool_ do NOT subclass float/int/bool and raise TypeError
    unconverted — a real risk for any duck-typed result object (e.g. one
    carrying GPU-offloaded float32 values) since only
    dipole_moment_debye/mulliken_charges/atom_symbols were coerced before,
    while energy_hartree, homo_lumo_gap_ev, converged, n_iterations, and
    the post-HF correlation fields were written straight from getattr().
    """

    def test_numpy_scalar_result_saves_without_raising(self, tmp_path):
        result = SimpleNamespace(
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=np.float32(-74.5),
            energy_ev=np.float32(-2027.3),
            homo_lumo_gap_ev=np.float32(12.3),
            converged=np.bool_(True),
            n_iterations=np.int64(12),
            mp2_correlation_hartree=np.float32(-0.01),
        )
        saved = save_result(result, results_dir=tmp_path)
        data = json.loads((saved / "result.json").read_text())
        assert data["energy_hartree"] == pytest.approx(-74.5, rel=1e-5)
        assert data["converged"] is True
        assert data["n_iterations"] == 12
        assert data["mp2_correlation_hartree"] == pytest.approx(-0.01, rel=1e-3)

    def test_numpy_bool_and_int64_fields_are_plain_python_types(self, tmp_path):
        result = _make_result(converged=np.bool_(False), n_iterations=np.int64(5))
        saved = save_result(result, results_dir=tmp_path)
        # Read the raw file text (not json.loads, which normalizes types)
        # to confirm no numpy repr ever reached the serialized bytes.
        raw = (saved / "result.json").read_text()
        assert "np.bool_" not in raw
        assert "np.int64" not in raw
        data = json.loads(raw)
        assert isinstance(data["converged"], bool)
        assert isinstance(data["n_iterations"], int)


class TestLoadResult:
    def test_roundtrip(self, tmp_path):
        saved = save_result(_make_result(), results_dir=tmp_path, calc_type="frequency")
        data = load_result(saved)
        assert data["formula"] == "H2O"
        assert data["calc_type"] == "frequency"
        assert data["_schema_version"] == 2

    def test_spectra_roundtrip(self, tmp_path):
        spectra = {
            "nmr": {"atom_symbols": ["H", "H"], "shielding_iso_ppm": [30.1, 30.2]}
        }
        saved = save_result(_make_result(), results_dir=tmp_path, spectra=spectra)
        data = load_result(saved)
        assert data["spectra"]["nmr"]["atom_symbols"] == ["H", "H"]


# ---------------------------------------------------------------------------
# list_results
# ---------------------------------------------------------------------------


class TestListResults:
    def test_empty_when_directory_missing(self, tmp_path):
        assert list_results(tmp_path / "nonexistent") == []

    def test_empty_when_directory_is_empty(self, tmp_path):
        assert list_results(tmp_path) == []

    def test_returns_directories_that_have_result_json(self, tmp_path):
        r1 = save_result(_make_result(), results_dir=tmp_path)
        r2 = save_result(_make_result(), results_dir=tmp_path)
        found = list_results(tmp_path)
        assert r1 in found
        assert r2 in found

    def test_excludes_directories_without_result_json(self, tmp_path):
        empty_dir = tmp_path / "2026-no-json"
        empty_dir.mkdir()
        found = list_results(tmp_path)
        assert empty_dir not in found

    def test_sorted_newest_first(self, tmp_path):
        r1 = save_result(_make_result(), results_dir=tmp_path)
        r2 = save_result(_make_result(), results_dir=tmp_path)
        found = list_results(tmp_path)
        assert found.index(r2) < found.index(r1)

    def test_collision_suffixes_sort_numerically_not_lexicographically(self, tmp_path):
        """L audit fix: same-microsecond collision counters (…_1 … _10) must
        sort newest-first by numeric value, not lexicographic string order
        (which put "…_10" before "…_2").
        """
        base = tmp_path / "2026-07-14_00-00-00-000000_H2O_RHF_STO-3G"
        dirs = [base] + [Path(f"{base}_{n}") for n in (1, 2, 9, 10, 11)]
        for d in dirs:
            d.mkdir(parents=True)
            (d / "result.json").write_text("{}")

        found = list_results(tmp_path)
        # Newest first: highest collision counter was created most recently.
        expected_order = [
            dirs[i] for i in [5, 4, 3, 2, 1, 0]
        ]  # _11, _10, _9, _2, _1, base
        assert found == expected_order


# ---------------------------------------------------------------------------
# save_orbitals / load_orbitals
# ---------------------------------------------------------------------------


class TestSaveOrbitals:
    def test_creates_npz_file(self, tmp_path):
        result = SimpleNamespace(
            mo_energy_hartree=np.array([-1.0, -0.5, 0.2]),
            mo_occ=np.array([2.0, 2.0, 0.0]),
            mo_coeff=None,
            pyscf_mol_atom=None,
            pyscf_mol_basis=None,
        )
        save_orbitals(tmp_path, result)
        assert (tmp_path / "orbitals.npz").exists()

    def test_skips_when_no_mo_data(self, tmp_path):
        result = SimpleNamespace(mo_energy_hartree=None, mo_occ=None)
        save_orbitals(tmp_path, result)
        assert not (tmp_path / "orbitals.npz").exists()

    def test_writes_meta_json_when_mol_data_present(self, tmp_path):
        result = SimpleNamespace(
            mo_energy_hartree=np.array([-1.0]),
            mo_occ=np.array([2.0]),
            mo_coeff=None,
            pyscf_mol_atom=[("O", [0.0, 0.0, 0.0])],
            pyscf_mol_basis="sto-3g",
        )
        save_orbitals(tmp_path, result)
        assert (tmp_path / "orbitals_meta.json").exists()
        meta = json.loads((tmp_path / "orbitals_meta.json").read_text())
        assert meta["mol_basis"] == "sto-3g"


class TestLoadOrbitals:
    def test_roundtrip(self, tmp_path):
        mo_e = np.array([-1.0, -0.5, 0.2])
        result = SimpleNamespace(
            mo_energy_hartree=mo_e,
            mo_occ=np.array([2.0, 2.0, 0.0]),
            mo_coeff=None,
            pyscf_mol_atom=[("O", [0.0, 0.0, 0.0])],
            pyscf_mol_basis="sto-3g",
        )
        save_orbitals(tmp_path, result)
        loaded = load_orbitals(tmp_path)
        np.testing.assert_array_almost_equal(loaded.mo_energy_hartree, mo_e)
        assert loaded.pyscf_mol_basis == "sto-3g"

    def test_raises_file_not_found_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_orbitals(tmp_path / "nonexistent")

    def test_handles_missing_meta_json_gracefully(self, tmp_path):
        result = SimpleNamespace(
            mo_energy_hartree=np.array([-1.0]),
            mo_occ=np.array([2.0]),
            mo_coeff=None,
            pyscf_mol_atom=None,
            pyscf_mol_basis=None,
        )
        save_orbitals(tmp_path, result)
        loaded = load_orbitals(tmp_path)
        assert loaded.pyscf_mol_atom is None
        assert loaded.pyscf_mol_basis is None


# ---------------------------------------------------------------------------
# save_trajectory / load_trajectory
# ---------------------------------------------------------------------------


class TestSaveTrajectory:
    def test_creates_trajectory_json(self, tmp_path):
        traj, energies = _water_traj()
        save_trajectory(tmp_path, traj, energies)
        assert (tmp_path / "trajectory.json").exists()

    def test_skips_empty_trajectory(self, tmp_path):
        save_trajectory(tmp_path, [], [])
        assert not (tmp_path / "trajectory.json").exists()

    def test_stores_atom_symbols_and_coords(self, tmp_path):
        traj, energies = _water_traj()
        save_trajectory(tmp_path, traj, energies)
        data = json.loads((tmp_path / "trajectory.json").read_text())
        assert data["atoms"] == ["O", "H", "H"]
        assert len(data["steps"]) == 2


class TestLoadTrajectory:
    def test_roundtrip(self, tmp_path):
        traj, energies = _water_traj()
        save_trajectory(tmp_path, traj, energies)
        loaded_traj, loaded_e = load_trajectory(tmp_path)
        assert len(loaded_traj) == 2
        assert loaded_e[0] == pytest.approx(-75.0)
        assert loaded_e[1] == pytest.approx(-75.1)

    def test_loaded_molecules_have_correct_atoms(self, tmp_path):
        traj, energies = _water_traj()
        save_trajectory(tmp_path, traj, energies)
        loaded_traj, _ = load_trajectory(tmp_path)
        assert list(loaded_traj[0].atoms) == ["O", "H", "H"]

    def test_raises_file_not_found_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_trajectory(tmp_path / "nonexistent")

    def test_all_none_energies_returns_empty_list(self, tmp_path):
        mol = Molecule(["H", "H"], [[0, 0, 0], [0.74, 0, 0]])
        traj = [mol]
        save_trajectory(tmp_path, traj, [None])
        _, energies = load_trajectory(tmp_path)
        assert energies == []


# ---------------------------------------------------------------------------
# save_thumbnail
# ---------------------------------------------------------------------------


class TestSaveThumbnail:
    def test_creates_png(self, tmp_path):
        data = {
            "calc_type": "single_point",
            "formula": "H2O",
            "method": "RHF",
            "basis": "STO-3G",
            "energy_hartree": -75.0,
            "converged": True,
        }
        save_thumbnail(tmp_path, data)
        assert (tmp_path / "thumbnail.png").exists()

    def test_does_not_raise_for_unknown_calc_type(self, tmp_path):
        save_thumbnail(tmp_path, {"calc_type": "unknown", "formula": "X"})

    @pytest.mark.parametrize(
        "calc_type", ["single_point", "geometry_opt", "frequency", "tddft", "nmr"]
    )
    def test_creates_png_for_all_calc_types(self, tmp_path, calc_type):
        result_dir = tmp_path / calc_type
        result_dir.mkdir()
        data = {
            "calc_type": calc_type,
            "formula": "H2O",
            "method": "RHF",
            "basis": "STO-3G",
            "energy_hartree": -75.0,
            "converged": True,
        }
        save_thumbnail(result_dir, data)
        assert (result_dir / "thumbnail.png").exists()

    def test_savefig_failure_does_not_raise(self, tmp_path):
        """M9 audit fix (2026-07-14): the docstring's "silently skips ...
        any error" promise must hold for failures past the matplotlib
        import — a fig.savefig() failure (disk full, permission denied,
        a bad font cache, ...) is a real possibility since it's an actual
        filesystem write, not just figure construction.

        Regression: only the `import matplotlib` block was guarded by
        try/except; a failure anywhere else in the function (including
        fig.savefig itself) propagated straight out.
        """
        from unittest.mock import MagicMock, patch

        import quantui.results_storage as rs

        fake_fig = MagicMock()
        fake_fig.savefig.side_effect = OSError("simulated disk full")
        fake_fig.get_facecolor.return_value = "#ffffff"

        with patch.object(rs, "_build_thumbnail_figure", return_value=fake_fig):
            rs.save_thumbnail(
                tmp_path, {"calc_type": "single_point", "formula": "H2O"}
            )  # must not raise

        assert fake_fig.savefig.called
        # plt.close() must still run on the figure even though savefig failed.
        import matplotlib.pyplot as plt

        with patch.object(plt, "close") as mock_close:
            with patch.object(rs, "_build_thumbnail_figure", return_value=fake_fig):
                rs.save_thumbnail(
                    tmp_path, {"calc_type": "single_point", "formula": "H2O"}
                )
            mock_close.assert_called_once_with(fake_fig)
