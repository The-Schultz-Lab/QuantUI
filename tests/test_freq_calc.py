"""
Tests for quantui.freq_calc — ThermoData dataclass and FreqResult thermo field.

Test strategy
-------------
* ThermoData and FreqResult dataclass tests run unconditionally — no PySCF needed.
* run_freq_calc() tests are marked pyscf_only and skipped on Windows.
"""

from __future__ import annotations

import pytest

from quantui.freq_calc import FreqResult, ThermoData

# ---------------------------------------------------------------------------
# PySCF availability
# ---------------------------------------------------------------------------

_PYSCF_AVAILABLE = False
try:
    import pyscf as _pyscf  # noqa: F401

    _PYSCF_AVAILABLE = True
except ImportError:
    pass

pyscf_only = pytest.mark.skipif(
    not _PYSCF_AVAILABLE,
    reason="PySCF not installed (Linux/macOS/WSL only)",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HARTREE_TO_JMOL = 2625499.6


def _make_thermo(**overrides) -> ThermoData:
    defaults = dict(
        zpve_hartree=0.020734,
        H_hartree=-76.003456,
        S_jmol=198.7,
        G_hartree=-76.032952,
    )
    defaults.update(overrides)
    return ThermoData(**defaults)


def _make_freq_result(**overrides) -> FreqResult:
    defaults = dict(
        energy_hartree=-76.023190,
        homo_lumo_gap_ev=9.5,
        converged=True,
        n_iterations=10,
        method="RHF",
        basis="STO-3G",
        formula="H2O",
        frequencies_cm1=[1600.0, 3600.0, 3800.0],
        zpve_hartree=0.020734,
    )
    defaults.update(overrides)
    return FreqResult(**defaults)


def _water():
    from quantui.molecule import Molecule

    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


# ============================================================================
# ThermoData dataclass
# ============================================================================


class TestThermoData:
    def test_fields_stored(self):
        td = _make_thermo()
        assert td.zpve_hartree == pytest.approx(0.020734)
        assert td.H_hartree == pytest.approx(-76.003456)
        assert td.S_jmol == pytest.approx(198.7)
        assert td.G_hartree == pytest.approx(-76.032952)

    def test_default_temperature(self):
        td = _make_thermo()
        assert td.temperature_k == pytest.approx(298.15)

    def test_g_less_than_h(self):
        """G = H - T*S, so G < H for positive entropy."""
        td = _make_thermo()
        assert td.G_hartree < td.H_hartree

    def test_g_consistent_with_h_and_s(self):
        """Verify G ≈ H - T*S within floating-point tolerance."""
        td = _make_thermo()
        expected_g = td.H_hartree - td.temperature_k * td.S_jmol / _HARTREE_TO_JMOL
        assert td.G_hartree == pytest.approx(expected_g, abs=0.01)


# ============================================================================
# FreqResult.thermo field
# ============================================================================


class TestFreqResultThermoField:
    def test_thermo_defaults_to_none(self):
        result = _make_freq_result()
        assert result.thermo is None

    def test_thermo_stored_when_provided(self):
        td = _make_thermo()
        result = _make_freq_result(thermo=td)
        assert result.thermo is td

    def test_thermo_h_accessible(self):
        td = _make_thermo(H_hartree=-76.003456)
        result = _make_freq_result(thermo=td)
        assert result.thermo.H_hartree == pytest.approx(-76.003456)  # type: ignore[union-attr]

    def test_thermo_s_accessible(self):
        td = _make_thermo(S_jmol=198.7)
        result = _make_freq_result(thermo=td)
        assert result.thermo.S_jmol == pytest.approx(198.7)  # type: ignore[union-attr]

    def test_thermo_g_accessible(self):
        td = _make_thermo(G_hartree=-76.032952)
        result = _make_freq_result(thermo=td)
        assert result.thermo.G_hartree == pytest.approx(-76.032952)  # type: ignore[union-attr]


# ============================================================================
# run_freq_calc() — PySCF required
# ============================================================================


class TestRunFreqCalcThermo:
    @pyscf_only
    @pytest.mark.slow
    def test_thermo_populated_for_rhf(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert result.thermo is not None

    @pyscf_only
    @pytest.mark.slow
    def test_scf_variant_reports_rks_for_closed_shell_dft(self):
        """M-UX2 UXP2.10 — confirms the wiring, mirroring the identical
        RHF/UHF/RKS/UKS dispatch already thoroughly tested in
        test_session_calc.py::TestScfVariantProvenance."""
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="B3LYP", basis="STO-3G")
        assert result.scf_variant == "RKS"

    @pyscf_only
    @pytest.mark.slow
    def test_thermo_h_is_finite(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        if result.thermo is not None:
            assert abs(result.thermo.H_hartree) < 1e6

    @pyscf_only
    @pytest.mark.slow
    def test_thermo_s_positive(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        if result.thermo is not None:
            assert result.thermo.S_jmol > 0

    @pyscf_only
    @pytest.mark.slow
    def test_thermo_matches_independent_pyscf_reference(self):
        """AUDIT F01 regression — S_jmol/G_hartree against an independent
        PySCF reference (not merely S > 0 / G < H), for RHF/STO-3G water at
        the fixed geometry in ``_water()``.

        Before the F01 fix, PySCF's S_tot (returned in Eh/K) was stored
        directly as S_jmol without converting to J/(mol*K), then divided by
        _HARTREE_TO_JMOL a second time when forming G — deflating S_jmol by
        ~2.6e6x and leaving G ~= H. Reference values below (S=188.538424
        J/(mol*K), G=-74.954908540 Eh) come from calling
        pyscf.hessian.thermo.thermo() directly on the same RHF/STO-3G water
        SCF object/frequencies, independent of quantui.freq_calc.
        """
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert result.thermo is not None
        assert result.thermo.S_jmol == pytest.approx(188.538424, abs=0.01)
        assert result.thermo.G_hartree == pytest.approx(-74.954908540, abs=1e-6)
        # The old bug's error was ~8e-9 Eh (S_jmol deflated to ~7.18e-5); a
        # correct calculation differs from H by orders of magnitude more.
        assert result.thermo.H_hartree - result.thermo.G_hartree > 1e-3

    @pyscf_only
    @pytest.mark.slow
    def test_thermo_g_less_than_h(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        if result.thermo is not None:
            assert result.thermo.G_hartree < result.thermo.H_hartree


# ============================================================================
# pyscf_mol_atom unit convention (H2 audit fix, 2026-07-14)
# ============================================================================


class TestPyscfMolAtomUnits:
    """``pyscf_mol_atom`` must be Angstrom, matching session_calc/optimizer.

    Regression for a bug where freq_calc built this field from PySCF's
    internal ``mol._atom`` (always Bohr), while every consumer (Molden
    export, cube generation, orbital replay) assumes Angstrom — silently
    inflating exported geometries ~1.89x for Frequency results only.
    """

    @pyscf_only
    @pytest.mark.slow
    def test_pyscf_mol_atom_matches_input_geometry_in_angstrom(self):
        from quantui.freq_calc import run_freq_calc

        molecule = _water()
        result = run_freq_calc(molecule, method="RHF", basis="STO-3G")
        assert result.pyscf_mol_atom is not None
        for (sym, coords), (orig_sym, orig_coords) in zip(
            result.pyscf_mol_atom, zip(molecule.atoms, molecule.coordinates)
        ):
            assert sym == orig_sym
            for c, orig_c in zip(coords, orig_coords):
                assert c == pytest.approx(orig_c, abs=1e-9)


# ============================================================================
# Post-HF method guard (M2 audit fix, 2026-07-14)
# ============================================================================


class TestRunFreqCalcPostHfGuard:
    """Post-HF methods raise a clear ValueError instead of a cryptic LibXC error.

    Regression: run_freq_calc() had no special-casing for MP2/CCSD/CCSD(T)
    — the SCF-selection branch silently treated them as a DFT xc functional
    (mf.xc = "CCSD"), failing deep inside PySCF with "LibXCFunctional: name
    'CCSD' not found" instead of a clear message. The guard fires before any
    PySCF import, so it needs neither PySCF nor ASE.
    """

    @pytest.mark.parametrize("method", ["MP2", "CCSD", "CCSD(T)"])
    def test_post_hf_method_raises_value_error(self, method):
        from quantui.freq_calc import run_freq_calc

        with pytest.raises(ValueError, match="post-HF"):
            run_freq_calc(_water(), method=method, basis="STO-3G")


# ============================================================================
# AUDIT F15 — a failed Hessian must not read as a converged result
# ============================================================================


class TestFreqResultReflectsHessianCompletion:
    """A ROHF reference's analytic Hessian is unavailable on this path
    (PySCF has no ROHF Hessian implementation here); the caught exception
    used to leave FreqResult.converged reading whatever the reference SCF
    alone reported, with frequencies_cm1=[] — a frequency calculation with
    no computed Hessian is not a successful frequency analysis.
    """

    @pyscf_only
    @pytest.mark.slow
    def test_rohf_hessian_failure_reports_unconverged(self):
        from quantui.freq_calc import run_freq_calc
        from quantui.molecule import Molecule

        # Real OH doublet — RHF/STO-3G dispatches to ROHF for this
        # open-shell molecule, matching the audit's exact reproduction.
        oh = Molecule(
            ["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]], charge=0, multiplicity=2
        )
        result = run_freq_calc(oh, method="RHF", basis="STO-3G")

        assert result.scf_variant == "ROHF"
        assert result.frequencies_cm1 == []
        assert result.converged is False


# ============================================================================
# IR intensities — PySCF required
# ============================================================================


class TestIRIntensities:
    """make_ir_intensity() should return real km/mol values for H₂O / RHF.

    H₂O has 3 vibrational modes: bending (~1600 cm⁻¹), symmetric stretch
    (~3700 cm⁻¹), antisymmetric stretch (~3800 cm⁻¹).  All three are
    IR-active (A1 and B2 symmetry), so all intensities must be positive.
    """

    @pyscf_only
    @pytest.mark.slow
    def test_ir_intensities_non_empty(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert result.ir_intensities, "ir_intensities should be non-empty for H₂O/RHF"

    @pyscf_only
    @pytest.mark.slow
    def test_ir_intensities_length_matches_frequencies(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert len(result.ir_intensities) == len(result.frequencies_cm1)

    @pyscf_only
    @pytest.mark.slow
    def test_ir_intensities_all_non_negative(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        for i, inten in enumerate(result.ir_intensities):
            assert inten >= 0, f"mode {i}: intensity {inten:.3f} < 0"

    @pyscf_only
    @pytest.mark.slow
    def test_ir_intensities_physically_reasonable(self):
        """All H₂O modes are IR-active; max intensity should be > 1 km/mol."""
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        if result.ir_intensities:
            assert max(result.ir_intensities) > 1.0


# ============================================================================
# IR-intensity inner-loop dm0/method dispatch (M5 audit fix, 2026-07-14)
# ============================================================================


class TestIrIntensityUhfClosedShellDispatch:
    """The inner displaced-SCF loop must dispatch on dm0's actual shape.

    Regression: the serial (_displaced_scf_dipole in freq_calc.py) and
    parallel (run_displaced_scf in freq_ir_workers.py) inner loops both
    picked RHF/UHF based on mol.spin == 0 alone. That agrees with the
    parent SCF's actual type only when the user's method choice matches
    the molecule's natural spin state. Explicitly selecting UHF for a
    closed-shell molecule (mol.spin == 0, but the parent mf — and its
    dm0 — is still UHF-shaped, a legitimate technique for probing
    symmetry-broken solutions) used to raise a shape-mismatch ValueError
    deep in PySCF, silently dropping IR intensities for the whole run
    (caught by the broad except around the entire IR-intensity block).
    """

    @pyscf_only
    @pytest.mark.slow
    def test_uhf_on_closed_shell_molecule_still_gets_ir_intensities(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="UHF", basis="STO-3G")
        assert result.ir_intensities, (
            "UHF on a closed-shell molecule should still produce IR "
            "intensities via the serial inner-loop dispatch fix"
        )
        assert len(result.ir_intensities) == len(result.frequencies_cm1)

    def test_freq_ir_workers_dispatches_on_dm0_shape_not_spin(self):
        """Unit-level check of the parallel worker's dispatch logic directly.

        Builds a UHF parent for a closed-shell (spin=0) molecule — the
        exact scenario that used to crash — and confirms the worker
        picks UHF (matching the (2, nao, nao) dm0) rather than RHF
        (which would raise on that dm0 shape).
        """
        pytest.importorskip("pyscf")
        import os
        import pickle
        import tempfile

        from pyscf import gto, scf

        from quantui.freq_ir_workers import init_worker, run_displaced_scf

        atom_str = "O 0 0 0.119; H 0 0.763 -0.477; H 0 -0.763 -0.477"
        mol = gto.M(atom=atom_str, basis="sto-3g", spin=0, charge=0, verbose=0)
        mf = scf.UHF(mol)
        mf.kernel()
        dm0 = mf.make_rdm1()
        assert dm0.ndim == 3  # UHF-shaped despite mol.spin == 0

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        try:
            pickle.dump(dm0, tmp)
            tmp.close()
            init_worker(atom_str, "sto-3g", 0, 0, None, tmp.name, 1)
            coords = mol.atom_coords(unit="Bohr").flatten().tolist()
            dip = run_displaced_scf("d000_x_+", coords)  # must not raise
            assert len(dip) == 3
        finally:
            os.unlink(tmp.name)

    def test_ecp_omission_gives_a_different_hamiltonian(self):
        """AUDIT F05 regression — the worker must run the reference's ECP
        (e.g. LANL2DZ on Na), not silently fall back to all-electron.

        Without ``ecp``, NaH/LANL2DZ is an all-electron (12-electron)
        calculation instead of the correct 2-explicit-electron ECP one —
        a different Hamiltonian, not numerical noise. Confirms both the
        electron-count claim directly (via the same ecp_for_basis mapping
        the worker now receives) and that the worker's own SCF result
        (the dipole it returns) differs materially between the two cases.
        """
        pytest.importorskip("pyscf")
        import os
        import pickle
        import tempfile

        from pyscf import gto

        from quantui.freq_ir_workers import init_worker, run_displaced_scf
        from quantui.inorganic_guards import ecp_for_basis

        atom_str = "Na 0 0 0; H 0 0 2.0"
        basis = "LANL2DZ"
        ecp = ecp_for_basis(basis, ["Na", "H"])
        assert ecp == {"Na": "LANL2DZ"}

        mol_with_ecp = gto.M(
            atom=atom_str, basis=basis, ecp=ecp, charge=0, spin=0, verbose=0
        )
        mol_without_ecp = gto.M(
            atom=atom_str, basis=basis, ecp={}, charge=0, spin=0, verbose=0
        )
        assert mol_with_ecp.nelectron == 2
        assert mol_without_ecp.nelectron == 12

        coords = mol_with_ecp.atom_coords(unit="Bohr").flatten().tolist()

        def _run(ecp_arg):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            try:
                pickle.dump(None, tmp)
                tmp.close()
                init_worker(atom_str, basis, 0, 0, None, tmp.name, 1, None, ecp_arg)
                return run_displaced_scf("d000_x_+", coords)
            finally:
                os.unlink(tmp.name)

        dip_with_ecp = _run(ecp)
        dip_without_ecp = _run({})
        # A 2-electron vs 12-electron calculation on the same geometry
        # produces a substantially different dipole, not a small
        # numerical discrepancy.
        assert abs(dip_with_ecp[2] - dip_without_ecp[2]) > 1.0

    def test_density_fit_option_applied_to_displaced_scf(self, monkeypatch):
        """AUDIT F19 regression — the worker had no density_fit parameter at
        all; every displaced SCF ran without density fitting even when the
        reference (and the serial fallback in freq_calc.py, which calls
        ``_try_density_fit(_mf_d, enabled=_density_fit_used)``) used it —
        silently changing the numerical approximation for parallel IR runs
        on a fitted reference, not merely its speed.
        """
        pytest.importorskip("pyscf")
        import os
        import pickle
        import tempfile

        from pyscf import gto, scf

        import quantui.density_fitting as density_fitting
        from quantui.freq_ir_workers import init_worker, run_displaced_scf

        atom_str = "O 0 0 0.119; H 0 0.763 -0.477; H 0 -0.763 -0.477"
        mol = gto.M(atom=atom_str, basis="sto-3g", spin=0, charge=0, verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        dm0 = mf.make_rdm1()

        seen_enabled = []
        _orig = density_fitting.try_density_fit

        def _spy(mf_arg, *, enabled=None, auxbasis=None):
            seen_enabled.append(enabled)
            return _orig(mf_arg, enabled=enabled, auxbasis=auxbasis)

        monkeypatch.setattr(density_fitting, "try_density_fit", _spy)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        try:
            pickle.dump(dm0, tmp)
            tmp.close()
            init_worker(
                atom_str,
                "sto-3g",
                0,
                0,
                None,
                tmp.name,
                1,
                None,  # checkpoint_items_dir
                None,  # ecp
                True,  # density_fit
                True,  # scf_rescue
            )
            coords = mol.atom_coords(unit="Bohr").flatten().tolist()
            dip = run_displaced_scf("d000_x_+", coords)
            assert len(dip) == 3
        finally:
            os.unlink(tmp.name)

        assert seen_enabled == [True]

    def test_density_fit_defaults_off_for_an_older_caller(self, monkeypatch):
        """Backward compatibility: a caller that predates this fix (only
        positional args through ``ecp``) must still get density_fit=False,
        matching the old always-off behavior exactly."""
        pytest.importorskip("pyscf")
        import os
        import pickle
        import tempfile

        from pyscf import gto, scf

        import quantui.density_fitting as density_fitting
        from quantui.freq_ir_workers import init_worker, run_displaced_scf

        atom_str = "O 0 0 0.119; H 0 0.763 -0.477; H 0 -0.763 -0.477"
        mol = gto.M(atom=atom_str, basis="sto-3g", spin=0, charge=0, verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        dm0 = mf.make_rdm1()

        seen_enabled = []
        _orig = density_fitting.try_density_fit

        def _spy(mf_arg, *, enabled=None, auxbasis=None):
            seen_enabled.append(enabled)
            return _orig(mf_arg, enabled=enabled, auxbasis=auxbasis)

        monkeypatch.setattr(density_fitting, "try_density_fit", _spy)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        try:
            pickle.dump(dm0, tmp)
            tmp.close()
            init_worker(atom_str, "sto-3g", 0, 0, None, tmp.name, 1)
            coords = mol.atom_coords(unit="Bohr").flatten().tolist()
            run_displaced_scf("d000_x_+", coords)
        finally:
            os.unlink(tmp.name)

        assert seen_enabled == [False]

    def test_scf_rescue_option_honored_in_displaced_scf(self, monkeypatch):
        """AUDIT F19 regression — the worker always called
        ``run_scf_with_rescue(mf, dm0=dm0)``, taking its default
        ``rescue=True`` regardless of what the caller requested. The
        serial loop threads the caller's choice through as
        ``rescue=scf_rescue``; this confirms the worker now does too.
        """
        pytest.importorskip("pyscf")
        import os
        import pickle
        import tempfile

        from pyscf import gto, scf

        import quantui.scf_robust as scf_robust
        from quantui.freq_ir_workers import init_worker, run_displaced_scf

        atom_str = "O 0 0 0.119; H 0 0.763 -0.477; H 0 -0.763 -0.477"
        mol = gto.M(atom=atom_str, basis="sto-3g", spin=0, charge=0, verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        dm0 = mf.make_rdm1()

        seen_rescue = []
        _orig = scf_robust.run_scf_with_rescue

        def _spy(mf_arg, **kwargs):
            seen_rescue.append(kwargs.get("rescue", True))
            return _orig(mf_arg, **kwargs)

        monkeypatch.setattr(scf_robust, "run_scf_with_rescue", _spy)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        try:
            pickle.dump(dm0, tmp)
            tmp.close()
            init_worker(
                atom_str,
                "sto-3g",
                0,
                0,
                None,
                tmp.name,
                1,
                None,  # checkpoint_items_dir
                None,  # ecp
                False,  # density_fit
                False,  # scf_rescue
            )
            coords = mol.atom_coords(unit="Bohr").flatten().tolist()
            run_displaced_scf("d000_x_+", coords)
        finally:
            os.unlink(tmp.name)

        assert seen_rescue == [False]

    def test_worker_writes_its_own_checkpoint_record(self, tmp_path):
        """M-CHECKPOINT CHK.4.4's crash-safety claim, at the unit level: the
        worker itself durably records completion — not just the parent
        after collecting the Future — so a parent crash mid-wave never
        loses a sibling that already finished. Exercised in-process here
        (no real ProcessPoolExecutor), since ``mark_item_done_at`` only
        needs a directory path — the same one a real spawned worker would
        get via ``initargs``.
        """
        pytest.importorskip("pyscf")
        import os
        import pickle
        import tempfile

        from pyscf import gto, scf

        from quantui.checkpoint import completed_items_at
        from quantui.freq_ir_workers import init_worker, run_displaced_scf

        atom_str = "O 0 0 0.119; H 0 0.763 -0.477; H 0 -0.763 -0.477"
        mol = gto.M(atom=atom_str, basis="sto-3g", spin=0, charge=0, verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        dm0 = mf.make_rdm1()

        items_dir = tmp_path / "items" / "freq_displacements"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        try:
            pickle.dump(dm0, tmp)
            tmp.close()
            init_worker(atom_str, "sto-3g", 0, 0, None, tmp.name, 1, str(items_dir))
            coords = mol.atom_coords(unit="Bohr").flatten().tolist()
            dip = run_displaced_scf("d000_x_+", coords)

            recorded = completed_items_at(items_dir)
            assert set(recorded) == {"d000_x_+"}
            assert recorded["d000_x_+"]["dipole"] == pytest.approx(list(dip), abs=1e-9)
        finally:
            os.unlink(tmp.name)

    def test_worker_with_no_checkpoint_dir_writes_nothing(self, tmp_path):
        """checkpointing must stay fully optional — a worker with
        ``checkpoint_items_dir=None`` (the default) must not create
        anything on disk."""
        pytest.importorskip("pyscf")
        import os
        import pickle
        import tempfile

        from pyscf import gto, scf

        from quantui.freq_ir_workers import init_worker, run_displaced_scf

        atom_str = "O 0 0 0.119; H 0 0.763 -0.477; H 0 -0.763 -0.477"
        mol = gto.M(atom=atom_str, basis="sto-3g", spin=0, charge=0, verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        dm0 = mf.make_rdm1()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        try:
            pickle.dump(dm0, tmp)
            tmp.close()
            init_worker(
                atom_str, "sto-3g", 0, 0, None, tmp.name, 1
            )  # no checkpoint dir
            coords = mol.atom_coords(unit="Bohr").flatten().tolist()
            run_displaced_scf("d000_x_+", coords)
            assert not (tmp_path / "items").exists()
        finally:
            os.unlink(tmp.name)

    @pyscf_only
    @pytest.mark.slow
    def test_parallel_failure_falls_back_to_serial(self, monkeypatch):
        """A parallel-path failure must fall back to serial, not give up.

        Regression: run_displaced_scf's own docstring claims "The
        freq_calc driver catches such failures and falls back to the
        serial loop so the user's calc still completes" — but no such
        fallback existed; any single worker failure propagated out to
        the broad except around the whole IR-intensity block, dropping
        IR intensities entirely. Forces the parallel path to be
        selected and to fail immediately, then checks IR intensities
        still come out via the serial fallback.
        """
        import concurrent.futures as cf
        import io as _io

        import quantui.freq_ir_workers as ir_workers
        from quantui.freq_calc import run_freq_calc

        monkeypatch.setattr(ir_workers, "parallel_enabled_for_run", lambda **kw: True)

        class _FailingExecutor:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                raise RuntimeError("simulated worker pool failure")

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(cf, "ProcessPoolExecutor", _FailingExecutor)

        buf = _io.StringIO()
        result = run_freq_calc(
            _water(), method="RHF", basis="STO-3G", progress_stream=buf
        )
        log = buf.getvalue()
        assert "falling back to serial" in log
        assert result.ir_intensities, (
            "IR intensities should still be populated via the serial "
            "fallback after a simulated parallel-path failure"
        )


# ============================================================================
# M-CHECKPOINT CHK.4 — displacement-level checkpoint/resume
# ============================================================================


@pyscf_only
@pytest.mark.slow
class TestChk4DisplacementCheckpointing:
    """The concurrency-safe, set-based design (roadmap 34's CHK.4 section):
    resume diffs a *set* of completed displacement ids, not a prefix.

    Raman is disabled in every test here (``QUANTUI_RAMAN=0``) so the call
    count asserted below is exactly the IR loop's — otherwise it would also
    depend on whether pyscf-properties happens to be installed.
    """

    def _checkpoint(self, tmp_path, molecule):
        from quantui.checkpoint import CalcIdentity, Checkpoint

        identity = CalcIdentity.from_molecule(
            molecule, calc_type="frequency", method="RHF", basis="STO-3G"
        )
        # Matches production: backends/worker.py's _begin_worker_checkpoint
        # always calls .begin() before handing a checkpoint to a calc
        # function — a checkpoint object is never "live" (has a meta.json
        # to update) until this runs.
        ckpt = Checkpoint(identity, root=tmp_path / "ckpt")
        ckpt.begin()
        return ckpt

    def _count_rescue_calls(self, monkeypatch):
        """Count real run_scf_with_rescue calls without changing behavior.

        Patched at the source (quantui.scf_robust), not at
        quantui.freq_calc — the call sites use a *local* ``from .scf_robust
        import run_scf_with_rescue`` re-executed on every call to
        _run_freq_calc_body, so patching the source module is what actually
        takes effect on the next run.
        """
        import quantui.scf_robust as scf_robust_mod

        real_rescue = scf_robust_mod.run_scf_with_rescue
        calls: list = []

        def _counting(*args, **kwargs):
            calls.append(1)
            return real_rescue(*args, **kwargs)

        monkeypatch.setattr(scf_robust_mod, "run_scf_with_rescue", _counting)
        return calls

    def test_every_displacement_is_recorded(self, tmp_path, monkeypatch):
        from quantui.freq_calc import run_freq_calc

        monkeypatch.setenv("QUANTUI_RAMAN", "0")
        molecule = _water()
        ckpt = self._checkpoint(tmp_path, molecule)
        run_freq_calc(molecule, method="RHF", basis="STO-3G", checkpoint=ckpt)
        # 3 atoms x 3 axes x 2 signs = 18.
        assert len(ckpt.completed_item_ids("freq_displacements")) == 18

    def test_successful_run_marks_the_checkpoint_complete(self, tmp_path, monkeypatch):
        """A successful frequency run must not linger forever in the
        "unfinished calculations" listing — resumable_checkpoints() filters
        on STATUS_COMPLETE, so this has to actually be set on success."""
        from quantui.checkpoint import STATUS_COMPLETE
        from quantui.freq_calc import run_freq_calc

        monkeypatch.setenv("QUANTUI_RAMAN", "0")
        molecule = _water()
        ckpt = self._checkpoint(tmp_path, molecule)
        run_freq_calc(molecule, method="RHF", basis="STO-3G", checkpoint=ckpt)
        assert ckpt.load_state()["status"] == STATUS_COMPLETE
        assert ckpt.resumable_state() is None

    def test_fully_banked_resume_recomputes_nothing_but_the_reference_scf(
        self, tmp_path, monkeypatch
    ):
        """The strong CHK.4 claim, proven by call count, not just a
        plausible-looking answer: every displacement already banked means
        zero new displacement SCFs on resume."""
        from quantui.freq_calc import run_freq_calc

        monkeypatch.setenv("QUANTUI_RAMAN", "0")
        molecule = _water()
        ckpt = self._checkpoint(tmp_path, molecule)

        calls = self._count_rescue_calls(monkeypatch)
        baseline = run_freq_calc(
            molecule, method="RHF", basis="STO-3G", checkpoint=ckpt
        )
        assert len(calls) == 1 + 18  # reference SCF + all 18 displacements

        calls.clear()
        # A real resubmission calls .begin() again (backends/worker.py's
        # _begin_worker_checkpoint runs on every attempt) — it resets status
        # to "running" but must not touch the already-banked item files.
        ckpt.begin()
        resumed = run_freq_calc(
            molecule, method="RHF", basis="STO-3G", checkpoint=ckpt, resume=True
        )
        assert len(calls) == 1  # only the reference SCF — zero displacement recompute
        assert resumed.ir_intensities == pytest.approx(
            baseline.ir_intensities, abs=1e-9
        )

    def test_partial_resume_recomputes_only_the_missing_displacements(
        self, tmp_path, monkeypatch
    ):
        """A checkpoint interrupted partway through: resume must recompute
        exactly the missing ids, reuse the rest, and land on the same
        answer as an uninterrupted run."""
        from quantui.freq_calc import run_freq_calc
        from quantui.freq_displacement_ids import required_displacement_ids

        monkeypatch.setenv("QUANTUI_RAMAN", "0")
        molecule = _water()
        ckpt = self._checkpoint(tmp_path, molecule)

        baseline = run_freq_calc(
            molecule, method="RHF", basis="STO-3G", checkpoint=ckpt
        )
        all_ids = required_displacement_ids(3)
        assert ckpt.completed_item_ids("freq_displacements") == set(all_ids)

        # Simulate an interrupted run: keep only the first 3 banked.
        keep = set(all_ids[:3])
        items_dir = ckpt.items_dir("freq_displacements")
        for item_id in all_ids:
            if item_id not in keep:
                (items_dir / f"{item_id}.json").unlink()
        assert ckpt.completed_item_ids("freq_displacements") == keep

        ckpt.begin()  # a real resubmission calls .begin() again — must not erase items
        calls = self._count_rescue_calls(monkeypatch)
        resumed = run_freq_calc(
            molecule, method="RHF", basis="STO-3G", checkpoint=ckpt, resume=True
        )
        # reference SCF + the 15 displacements that were NOT kept.
        assert len(calls) == 1 + (len(all_ids) - len(keep))
        assert ckpt.completed_item_ids("freq_displacements") == set(all_ids)
        assert resumed.ir_intensities == pytest.approx(
            baseline.ir_intensities, abs=1e-6
        )

    def test_resume_false_ignores_a_populated_checkpoint(self, tmp_path, monkeypatch):
        """resume=False must behave like no checkpoint was ever passed for
        *reading* progress — even though the run still writes into it, so a
        later resume has something to build on. Mirrors optimizer.py's
        CHK.2 convention (resume is opt-in per call, not implied by merely
        passing a checkpoint object)."""
        from quantui.freq_calc import run_freq_calc

        monkeypatch.setenv("QUANTUI_RAMAN", "0")
        molecule = _water()
        ckpt = self._checkpoint(tmp_path, molecule)
        run_freq_calc(molecule, method="RHF", basis="STO-3G", checkpoint=ckpt)
        assert len(ckpt.completed_item_ids("freq_displacements")) == 18

        ckpt.begin()
        calls = self._count_rescue_calls(monkeypatch)
        run_freq_calc(
            molecule, method="RHF", basis="STO-3G", checkpoint=ckpt, resume=False
        )
        # Started fresh despite 18 already banked — same as the first run.
        assert len(calls) == 1 + 18

    def test_a_corrupt_banked_item_is_recomputed_not_trusted(
        self, tmp_path, monkeypatch
    ):
        """checkpoint.py's "never break a calculation" rule, exercised
        through the actual freq_calc resume path rather than checkpoint.py
        in isolation."""
        from quantui.freq_calc import run_freq_calc
        from quantui.freq_displacement_ids import required_displacement_ids

        monkeypatch.setenv("QUANTUI_RAMAN", "0")
        molecule = _water()
        ckpt = self._checkpoint(tmp_path, molecule)
        run_freq_calc(molecule, method="RHF", basis="STO-3G", checkpoint=ckpt)

        all_ids = required_displacement_ids(3)
        items_dir = ckpt.items_dir("freq_displacements")
        corrupt_id = all_ids[0]
        (items_dir / f"{corrupt_id}.json").write_text("{not json", encoding="utf-8")
        assert corrupt_id not in ckpt.completed_item_ids("freq_displacements")

        ckpt.begin()
        calls = self._count_rescue_calls(monkeypatch)
        resumed = run_freq_calc(
            molecule, method="RHF", basis="STO-3G", checkpoint=ckpt, resume=True
        )
        # reference SCF + the one corrupted (and therefore recomputed) displacement.
        assert len(calls) == 2
        assert resumed.ir_intensities, "must still produce a usable result"
        assert corrupt_id in ckpt.completed_item_ids("freq_displacements")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
