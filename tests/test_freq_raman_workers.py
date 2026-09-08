"""Tests for parallel Raman displacement workers."""

from __future__ import annotations

import pytest

from quantui import freq_ir_workers as irw


class TestRamanParallelGate:
    def test_parallel_enabled_uses_same_gate_as_ir(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_FREQ_PARALLEL", "1")
        assert irw.parallel_enabled_for_run(cpu_count=8, displacement_count=12) is True
        assert irw.parallel_enabled_for_run(cpu_count=2, displacement_count=12) is False

    def test_raman_calc_imports_raman_workers(self):
        import quantui.freq_raman_workers as rw

        assert callable(rw.init_raman_worker)
        assert callable(rw.run_displaced_polarizability)


class TestRamanWorkerEcp:
    def test_ecp_omission_gives_a_different_hamiltonian(self):
        """AUDIT F05 regression — mirrors the IR worker's ECP fix
        (test_freq_calc.py::TestIrIntensityUhfClosedShellDispatch), for the
        Raman polarizability worker. Without ``ecp``, NaH/LANL2DZ runs as
        an all-electron (12-electron) calculation instead of the correct
        2-explicit-electron ECP one, giving a materially different
        polarizability tensor.
        """
        pytest.importorskip("pyscf")
        import os
        import pickle
        import tempfile

        from pyscf import gto

        from quantui.freq_raman_workers import (
            init_raman_worker,
            run_displaced_polarizability,
        )
        from quantui.inorganic_guards import ecp_for_basis

        atom_str = "Na 0 0 0; H 0 0 2.0"
        basis = "LANL2DZ"
        ecp = ecp_for_basis(basis, ["Na", "H"])
        assert ecp == {"Na": "LANL2DZ"}

        mol = gto.M(atom=atom_str, basis=basis, ecp=ecp, charge=0, spin=0, verbose=0)
        coords = mol.atom_coords(unit="Bohr").flatten().tolist()

        def _run(ecp_arg):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            try:
                pickle.dump(None, tmp)
                tmp.close()
                init_raman_worker(
                    atom_str,
                    basis,
                    0,
                    0,
                    None,
                    tmp.name,
                    1,
                    False,
                    False,
                    None,
                    ecp_arg,
                )
                return run_displaced_polarizability("d000_x_+", coords)
            finally:
                os.unlink(tmp.name)

        alpha_with_ecp = _run(ecp)
        alpha_without_ecp = _run({})
        assert abs(alpha_with_ecp[2][2] - alpha_without_ecp[2][2]) > 1.0


class TestRamanWorkerScfRescue:
    def test_scf_rescue_option_honored(self, monkeypatch):
        """AUDIT F19 regression — the Raman worker always called
        ``run_scf_with_rescue(mf, dm0=dm0)``, taking its default
        ``rescue=True`` regardless of what the caller (raman_calc.py's
        ``scf_rescue`` parameter) requested. Confirms the worker now
        threads that choice through.
        """
        pytest.importorskip("pyscf")
        import os
        import pickle
        import tempfile

        from pyscf import gto

        import quantui.scf_robust as scf_robust
        from quantui.freq_raman_workers import (
            init_raman_worker,
            run_displaced_polarizability,
        )

        atom_str = "O 0 0 0.119; H 0 0.763 -0.477; H 0 -0.763 -0.477"
        mol = gto.M(atom=atom_str, basis="sto-3g", spin=0, charge=0, verbose=0)
        coords = mol.atom_coords(unit="Bohr").flatten().tolist()

        seen_rescue = []
        _orig = scf_robust.run_scf_with_rescue

        def _spy(mf_arg, **kwargs):
            seen_rescue.append(kwargs.get("rescue", True))
            return _orig(mf_arg, **kwargs)

        monkeypatch.setattr(scf_robust, "run_scf_with_rescue", _spy)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        try:
            pickle.dump(None, tmp)
            tmp.close()
            init_raman_worker(
                atom_str,
                "sto-3g",
                0,
                0,
                None,
                tmp.name,
                1,
                False,  # dm0_is_unrestricted
                False,  # density_fit_used
                None,  # checkpoint_items_dir
                None,  # ecp
                False,  # scf_rescue
            )
            run_displaced_polarizability("d000_x_+", coords)
        finally:
            os.unlink(tmp.name)

        assert seen_rescue == [False]

    def test_scf_rescue_defaults_true_for_an_older_caller(self, monkeypatch):
        """Backward compatibility: a caller predating this fix (only
        positional args through ``ecp``) must still get rescue=True,
        matching the old always-on behavior exactly."""
        pytest.importorskip("pyscf")
        import os
        import pickle
        import tempfile

        from pyscf import gto

        import quantui.scf_robust as scf_robust
        from quantui.freq_raman_workers import (
            init_raman_worker,
            run_displaced_polarizability,
        )

        atom_str = "O 0 0 0.119; H 0 0.763 -0.477; H 0 -0.763 -0.477"
        mol = gto.M(atom=atom_str, basis="sto-3g", spin=0, charge=0, verbose=0)
        coords = mol.atom_coords(unit="Bohr").flatten().tolist()

        seen_rescue = []
        _orig = scf_robust.run_scf_with_rescue

        def _spy(mf_arg, **kwargs):
            seen_rescue.append(kwargs.get("rescue", True))
            return _orig(mf_arg, **kwargs)

        monkeypatch.setattr(scf_robust, "run_scf_with_rescue", _spy)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        try:
            pickle.dump(None, tmp)
            tmp.close()
            init_raman_worker(atom_str, "sto-3g", 0, 0, None, tmp.name, 1, False, False)
            run_displaced_polarizability("d000_x_+", coords)
        finally:
            os.unlink(tmp.name)

        assert seen_rescue == [True]
