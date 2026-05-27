"""Tests for the M-EXPORT / EXPORT.1+2 save_molden helper.

Coverage is two-tier:

1. **Platform-independent contract tests**: ``save_molden`` returns ``None``
   when given insufficient inputs (no orbitals AND no vibrations, or
   missing atom string / basis); never raises on those paths. Caller-
   safe by design.

2. **PySCF-gated round-trip tests**: when PySCF is available, the writer
   produces a Molden file that round-trips via ``pyscf.tools.molden.load``
   and the Frequency variant contains the ``[FREQ]`` + ``[FR-NORM-COORD]``
   blocks that Avogadro animates.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantui.results_storage import save_molden

_PYSCF_AVAILABLE = False
try:
    import pyscf as _pyscf  # noqa: F401

    _PYSCF_AVAILABLE = True
except ImportError:
    pass

pyscf_only = pytest.mark.skipif(
    not _PYSCF_AVAILABLE,
    reason="PySCF not installed (Linux / macOS / WSL only)",
)


def _water_atom_list():
    return [
        ("O", [0.0, 0.0, 0.0]),
        ("H", [0.957, 0.0, 0.0]),
        ("H", [-0.24, 0.927, 0.0]),
    ]


class TestSaveMoldenContract:
    """Inputs-validation / no-op paths. No PySCF required."""

    def test_returns_none_when_no_data_at_all(self, tmp_path):
        # Neither orbitals nor vibrations given → nothing meaningful
        # to export. Helper must return None, NOT raise.
        result = save_molden(tmp_path, pyscf_mol_atom=_water_atom_list())
        assert result is None

    def test_returns_none_when_atom_list_missing(self, tmp_path):
        # mo_coeff present but no atom list → cannot build the Mole.
        result = save_molden(
            tmp_path,
            mo_coeff=np.eye(2),
            mo_energy_hartree=np.array([0.0, 0.0]),
            mo_occ=np.array([2.0, 0.0]),
            pyscf_mol_atom=None,
            pyscf_mol_basis="sto-3g",
        )
        assert result is None

    def test_returns_none_when_basis_missing(self, tmp_path):
        result = save_molden(
            tmp_path,
            mo_coeff=np.eye(2),
            mo_energy_hartree=np.array([0.0, 0.0]),
            mo_occ=np.array([2.0, 0.0]),
            pyscf_mol_atom=_water_atom_list(),
            pyscf_mol_basis=None,
        )
        assert result is None


@pyscf_only
class TestSaveMoldenWithOrbitals:
    """Full Molden write path: SP / GeoOpt result with mo_coeff present."""

    def _run_water_rhf_sto3g(self):
        # Real RHF/STO-3G on water — produces the MO arrays we need.
        from pyscf import gto, scf

        mol = gto.Mole()
        mol.atom = _water_atom_list()
        mol.basis = "sto-3g"
        mol.verbose = 0
        mol.build()
        mf = scf.RHF(mol)
        mf.kernel()
        return mol, mf

    def test_writes_molden_file_with_mo_block(self, tmp_path):
        mol, mf = self._run_water_rhf_sto3g()
        out = save_molden(
            tmp_path,
            mo_coeff=mf.mo_coeff,
            mo_energy_hartree=mf.mo_energy,
            mo_occ=mf.mo_occ,
            pyscf_mol_atom=_water_atom_list(),
            pyscf_mol_basis="sto-3g",
            charge=0,
            multiplicity=1,
        )
        assert out is not None
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "[Molden Format]" in text
        assert "[Atoms]" in text
        assert "[MO]" in text
        # No vibrations were passed → no [FREQ] block.
        assert "[FREQ]" not in text

    def test_round_trips_via_molden_load(self, tmp_path):
        from pyscf.tools import molden as _molden

        mol, mf = self._run_water_rhf_sto3g()
        out = save_molden(
            tmp_path,
            mo_coeff=mf.mo_coeff,
            mo_energy_hartree=mf.mo_energy,
            mo_occ=mf.mo_occ,
            pyscf_mol_atom=_water_atom_list(),
            pyscf_mol_basis="sto-3g",
        )
        # The Molden writer should produce a file that PySCF's own parser
        # accepts. Returns (mol, mo_energy, mo_coeff, mo_occ, irrep, spins).
        parsed = _molden.load(str(out))
        loaded_mol = parsed[0]
        loaded_mo_energy = np.asarray(parsed[1])
        loaded_mo_occ = np.asarray(parsed[3])
        assert loaded_mol.natm == 3  # water
        assert loaded_mo_energy.shape == mf.mo_energy.shape
        # MO energies should match within float precision after the
        # text round-trip (Molden writes ~6 decimal places).
        np.testing.assert_allclose(loaded_mo_energy, mf.mo_energy, atol=1e-5)
        np.testing.assert_allclose(loaded_mo_occ, mf.mo_occ, atol=1e-6)


@pyscf_only
class TestSaveMoldenWithVibrations:
    """Structure-only Molden + [FREQ] block for Frequency results.

    Mirrors the path where ``FreqResult`` has no ``mo_coeff`` but does
    have frequencies + normal modes — Avogadro can still animate
    vibrations from this file.
    """

    def test_writes_freq_block_when_no_orbitals(self, tmp_path):
        frequencies = [1500.0, 2000.0, 3500.0]
        # 3 modes × 3 atoms × (x, y, z) — values arbitrary, just need
        # the right shape so the writer doesn't reject.
        normal_modes = [
            [[0.1, 0.0, 0.0], [-0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]],
            [[0.0, 0.1, 0.0], [0.0, -0.05, 0.0], [0.0, -0.05, 0.0]],
            [[0.0, 0.0, 0.1], [0.0, 0.0, -0.05], [0.0, 0.0, -0.05]],
        ]
        out = save_molden(
            tmp_path,
            pyscf_mol_atom=_water_atom_list(),
            pyscf_mol_basis="sto-3g",
            frequencies_cm1=frequencies,
            normal_modes=normal_modes,
        )
        assert out is not None
        text = out.read_text(encoding="utf-8")
        # Header sections present even without orbitals.
        assert "[Molden Format]" in text
        assert "[Atoms]" in text
        # Vibration sections appended.
        assert "[FREQ]" in text
        assert "[FR-COORD]" in text
        assert "[FR-NORM-COORD]" in text
        # Frequencies serialized exactly as floats.
        assert "1500.000000" in text
        assert "3500.000000" in text
        # vibration N markers.
        assert "vibration   1" in text
        assert "vibration   3" in text
        # No [MO] block since mo_coeff was None.
        assert "[MO]" not in text

    def test_writes_freq_block_alongside_orbitals_when_both_present(self, tmp_path):
        # The combined case: a freq result with persisted MOs gets both
        # the orbital block AND the vibration blocks in one file.
        from pyscf import gto, scf

        mol = gto.Mole()
        mol.atom = _water_atom_list()
        mol.basis = "sto-3g"
        mol.verbose = 0
        mol.build()
        mf = scf.RHF(mol)
        mf.kernel()

        frequencies = [1500.0, 2000.0, 3500.0]
        normal_modes = [
            [[0.1, 0.0, 0.0], [-0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]],
            [[0.0, 0.1, 0.0], [0.0, -0.05, 0.0], [0.0, -0.05, 0.0]],
            [[0.0, 0.0, 0.1], [0.0, 0.0, -0.05], [0.0, 0.0, -0.05]],
        ]
        out = save_molden(
            tmp_path,
            mo_coeff=mf.mo_coeff,
            mo_energy_hartree=mf.mo_energy,
            mo_occ=mf.mo_occ,
            pyscf_mol_atom=_water_atom_list(),
            pyscf_mol_basis="sto-3g",
            frequencies_cm1=frequencies,
            normal_modes=normal_modes,
        )
        assert out is not None
        text = out.read_text(encoding="utf-8")
        assert "[MO]" in text
        assert "[FREQ]" in text
        assert "[FR-NORM-COORD]" in text
