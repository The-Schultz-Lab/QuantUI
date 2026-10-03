"""PCM implicit solvent across calc types (DEC-023, M-BATCH2 B2.2).

``session_calc.apply_pcm`` is the single PCM entry point. Single point,
geometry optimization (every step), frequency (reference SCF, PCM Hessian,
IR displacement SCFs) and TD-DFT (equilibrium ground state, non-equilibrium
excitations) all go through it.
"""

from __future__ import annotations

import io

import pytest

from quantui import config
from quantui.session_calc import apply_pcm, resolve_solvent


class TestResolveSolvent:
    def test_none_and_blank_mean_gas_phase(self):
        assert resolve_solvent(None) is None
        assert resolve_solvent("") is None
        assert resolve_solvent("  ") is None

    @pytest.mark.parametrize("spelling", ["water", "WATER", " Water "])
    def test_case_insensitive(self, spelling):
        # A batch request JSON may say "water" where the app says "Water";
        # that used to silently run gas-phase.
        assert resolve_solvent(spelling) == "Water"

    def test_unknown_solvent_raises(self):
        with pytest.raises(ValueError, match="seawater"):
            resolve_solvent("seawater")

    def test_every_solvent_has_an_optical_dielectric(self):
        assert set(config.SOLVENT_OPTICAL_EPS) == set(config.SOLVENT_OPTIONS)
        for name, eps_inf in config.SOLVENT_OPTICAL_EPS.items():
            # n^2 of ordinary liquids: between 1.7 and 2.3, and always far
            # below the static dielectric constant.
            assert 1.7 < eps_inf < 2.3
            assert eps_inf < config.SOLVENT_OPTIONS[name]


class _FakeMf:
    pass


class TestApplyPcm:
    def test_gas_phase_is_a_no_op(self):
        mf = _FakeMf()
        out, applied = apply_pcm(mf, None)
        assert out is mf
        assert applied is None

    def test_wraps_and_sets_static_dielectric(self):
        pytest.importorskip("pyscf")
        from pyscf import gto, scf

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf, applied = apply_pcm(scf.RHF(mol), "ethanol")
        assert applied == "Ethanol"
        assert mf.with_solvent.eps == config.SOLVENT_OPTIONS["Ethanol"]


def _water():
    from quantui.molecule import Molecule

    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]],
        charge=0,
        multiplicity=1,
    )


@pytest.mark.slow
class TestPcmCalculations:
    """Real PySCF runs (water, STO-3G) — solvated vs gas phase."""

    def test_solvated_optimization(self):
        pytest.importorskip("pyscf")
        pytest.importorskip("ase")
        from quantui.optimizer import optimize_geometry

        gas = optimize_geometry(
            _water(), "RHF", "STO-3G", progress_stream=io.StringIO()
        )
        wet = optimize_geometry(
            _water(),
            "RHF",
            "STO-3G",
            progress_stream=io.StringIO(),
            solvent="Water",
        )
        assert gas.solvent is None
        assert wet.solvent == "Water"
        assert wet.converged
        # Solvation stabilizes a polar molecule.
        assert wet.energies_hartree[-1] < gas.energies_hartree[-1]

    def test_solvated_frequency(self):
        pytest.importorskip("pyscf")
        from quantui.freq_calc import run_freq_calc

        log = io.StringIO()
        gas = run_freq_calc(_water(), "RHF", "STO-3G", progress_stream=io.StringIO())
        wet = run_freq_calc(
            _water(), "RHF", "STO-3G", progress_stream=log, solvent="Water"
        )
        assert gas.solvent is None
        assert wet.solvent == "Water"
        assert wet.converged
        assert len(wet.frequencies_cm1) == 3
        # PCM Hessian: frequencies move, but not wildly, for water in water.
        for f_gas, f_wet in zip(gas.frequencies_cm1, wet.frequencies_cm1):
            assert f_wet != pytest.approx(f_gas, abs=0.01)
            assert abs(f_wet - f_gas) < 150
        # IR intensities come from solvated displacement SCFs.
        assert len(wet.ir_intensities) == 3
        assert wet.ir_intensities != pytest.approx(gas.ir_intensities, abs=0.01)
        # Raman is gas-phase only, so it is skipped rather than mixed in.
        assert wet.raman_activities == []
        assert "skipping Raman" in log.getvalue()

    def test_solvated_tddft_uses_the_solvents_optical_dielectric(self):
        pytest.importorskip("pyscf")
        from quantui.tddft_calc import run_tddft_calc

        gas = run_tddft_calc(
            _water(), "B3LYP", "STO-3G", nstates=2, progress_stream=io.StringIO()
        )
        log = io.StringIO()
        thf = run_tddft_calc(
            _water(),
            "B3LYP",
            "STO-3G",
            nstates=2,
            progress_stream=log,
            solvent="THF",
        )
        assert gas.solvent is None
        assert thf.solvent == "THF"
        assert thf.converged
        assert thf.excitation_energies_ev[0] != pytest.approx(
            gas.excitation_energies_ev[0], abs=1e-3
        )
        text = log.getvalue()
        # Not PySCF's fixed water value (1.78): THF's own n^2.
        assert "eps_inf = 1.980" in text
        assert "non-equilibrium solvation with eps=1.98" in text

    def test_parallel_ir_worker_applies_the_same_solvent(self, monkeypatch, tmp_path):
        """The opt-in parallel IR pool must solvate its displacement SCFs
        exactly like the serial loop does."""
        pytest.importorskip("pyscf")
        import pickle

        import numpy as np
        from pyscf import gto, scf

        from quantui import freq_ir_workers
        from quantui.session_calc import apply_pcm as _apply

        # init_worker writes BLAS thread vars into os.environ; let
        # monkeypatch restore them so nothing leaks to later tests.
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYSCF_NUM_THREADS",
        ):
            monkeypatch.setenv(var, "1")
        monkeypatch.setattr(freq_ir_workers, "_WORKER_STATE", {})

        atom = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
        mol = gto.M(atom=atom, basis="sto-3g", verbose=0)
        ref, _ = _apply(scf.RHF(mol), "Water")
        ref.kernel()
        dm_path = tmp_path / "dm0.pkl"
        dm_path.write_bytes(pickle.dumps(ref.make_rdm1()))

        coords = mol.atom_coords().copy()
        coords[0, 2] += 0.01
        flat = coords.flatten().tolist()

        def _dipole(solvent):
            freq_ir_workers.init_worker(
                atom, "sto-3g", 0, 0, None, str(dm_path), 1, solvent=solvent
            )
            return freq_ir_workers.run_displaced_scf("d000_z_+", flat)

        wet = _dipole("Water")
        gas = _dipole(None)

        expected_mol = mol.copy()
        expected_mol.set_geom_(coords, unit="Bohr")
        expected, _ = _apply(scf.RHF(expected_mol), "Water")
        expected.kernel()
        np.testing.assert_allclose(wet, expected.dip_moment(verbose=0), atol=1e-5)
        assert not np.allclose(wet, gas, atol=1e-4)
