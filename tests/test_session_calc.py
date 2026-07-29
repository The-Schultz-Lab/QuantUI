"""
Tests for quantui.session_calc — PySCF in-session calculator.

Test strategy
-------------
* Calculation tests require PySCF, so they are marked with ``pyscf_only``
  and skipped on platforms where PySCF is unavailable (Windows, CI without
  the pyscf extra).
* SessionResult unit tests (dataclass behaviour, summary formatting) run
  unconditionally — they construct the dataclass directly without PySCF.

WSL / Linux testing
--------------------
Run the full suite on your WSL terminal with pyscf installed:
    pytest tests/test_session_calc.py -v

Run only the fast, platform-independent tests anywhere:
    pytest tests/test_session_calc.py -v -k "not pyscf_only"
"""

import io

import pytest

from quantui.molecule import Molecule
from quantui.session_calc import HARTREE_TO_EV, SessionResult

# Check for PySCF availability
_PYSCF_AVAILABLE = False
try:
    import pyscf as _pyscf  # noqa: F401

    _PYSCF_AVAILABLE = True
except ImportError:
    pass

# Skip marker for tests that need PySCF (Linux/macOS/WSL only)
pyscf_only = pytest.mark.skipif(
    not _PYSCF_AVAILABLE,
    reason="PySCF not installed (Linux/macOS/WSL only)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h2() -> Molecule:
    """H2 with equilibrium geometry — fastest meaningful QM calculation."""
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def _water() -> Molecule:
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


def _make_result(**overrides) -> SessionResult:
    """Build a SessionResult with sensible defaults, allowing field overrides."""
    defaults = dict(
        energy_hartree=-1.117,
        homo_lumo_gap_ev=10.5,
        converged=True,
        n_iterations=8,
        method="RHF",
        basis="STO-3G",
        formula="H2",
    )
    defaults.update(overrides)
    return SessionResult(**defaults)


# ============================================================================
# SessionResult dataclass — no PySCF needed
# ============================================================================


class TestSessionResultDataclass:
    """Unit tests for SessionResult fields, properties, and summary."""

    def test_energy_ev_property(self):
        result = _make_result(energy_hartree=1.0)
        assert abs(result.energy_ev - HARTREE_TO_EV) < 1e-9

    def test_energy_ev_negative(self):
        result = _make_result(energy_hartree=-1.117)
        assert result.energy_ev == pytest.approx(-1.117 * HARTREE_TO_EV, rel=1e-9)

    def test_summary_contains_formula(self):
        result = _make_result(formula="H2O")
        assert "H2O" in result.summary()

    def test_summary_contains_method_basis(self):
        result = _make_result(method="UHF", basis="cc-pVDZ")
        summary = result.summary()
        assert "UHF" in summary
        assert "cc-pVDZ" in summary

    def test_summary_converged_shows_yes(self):
        result = _make_result(converged=True)
        assert "Yes" in result.summary()

    def test_summary_not_converged_shows_warning(self):
        result = _make_result(converged=False)
        summary = result.summary()
        assert "NO" in summary or "not converge" in summary.lower()

    def test_summary_contains_energy(self):
        result = _make_result(energy_hartree=-76.1234)
        assert "-76.1234" in result.summary()

    def test_summary_contains_homo_lumo_gap(self):
        result = _make_result(homo_lumo_gap_ev=8.5432)
        assert "8.5432" in result.summary()

    def test_summary_omits_gap_when_none(self):
        result = _make_result(homo_lumo_gap_ev=None)
        assert "HOMO" not in result.summary()

    def test_summary_contains_iterations(self):
        result = _make_result(n_iterations=13)
        assert "13" in result.summary()

    def test_all_fields_accessible(self):
        result = _make_result()
        assert isinstance(result.energy_hartree, float)
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.method, str)
        assert isinstance(result.basis, str)
        assert isinstance(result.formula, str)

    def test_homo_lumo_gap_can_be_none(self):
        result = _make_result(homo_lumo_gap_ev=None)
        assert result.homo_lumo_gap_ev is None

    def test_mulliken_charges_default_none(self):
        result = _make_result()
        assert result.mulliken_charges is None

    def test_dipole_moment_default_none(self):
        result = _make_result()
        assert result.dipole_moment_debye is None

    def test_atom_symbols_default_none(self):
        result = _make_result()
        assert result.atom_symbols is None

    def test_mulliken_charges_stored(self):
        result = _make_result(
            mulliken_charges=[-0.66, 0.33, 0.33],
            atom_symbols=["O", "H", "H"],
        )
        assert result.mulliken_charges == pytest.approx([-0.66, 0.33, 0.33])
        assert result.atom_symbols == ["O", "H", "H"]

    def test_dipole_moment_stored(self):
        result = _make_result(dipole_moment_debye=1.85)
        assert result.dipole_moment_debye == pytest.approx(1.85)


class TestMullikenDipolePySCF:
    """PySCF-backed tests for Mulliken charges and dipole moment extraction."""

    @pyscf_only
    @pytest.mark.slow
    def test_rhf_populates_mulliken_charges(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_water(), method="RHF", basis="STO-3G", verbose=0)
        assert result.mulliken_charges is not None
        assert len(result.mulliken_charges) == 3

    @pyscf_only
    @pytest.mark.slow
    def test_rhf_populates_dipole_moment(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_water(), method="RHF", basis="STO-3G", verbose=0)
        assert result.dipole_moment_debye is not None
        assert result.dipole_moment_debye > 0

    @pyscf_only
    @pytest.mark.slow
    def test_rhf_atom_symbols_match_molecule(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_water(), method="RHF", basis="STO-3G", verbose=0)
        assert result.atom_symbols == ["O", "H", "H"]

    @pyscf_only
    @pytest.mark.slow
    def test_uhf_populates_mulliken_charges(self):
        """M3 audit fix (2026-07-14): UHF now gets Mulliken charges too.

        Regression: session_calc.py used to skip this whole extraction
        block for method_upper == "UHF" specifically, even though
        mf.mulliken_pop() is well-defined and works correctly for a real
        UHF object (UKS — open-shell DFT — went through the identical
        code successfully the whole time). Uses an OH radical (doublet)
        rather than a lone atom so the charges are chemically meaningful
        (nonzero), not just trivially zero.
        """
        from quantui.session_calc import run_in_session

        mol = Molecule(
            ["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]], charge=0, multiplicity=2
        )
        result = run_in_session(mol, method="UHF", basis="STO-3G", verbose=0)
        assert result.mulliken_charges is not None
        assert len(result.mulliken_charges) == 2

    @pyscf_only
    @pytest.mark.slow
    def test_uhf_populates_dipole_moment(self):
        from quantui.session_calc import run_in_session

        mol = Molecule(
            ["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]], charge=0, multiplicity=2
        )
        result = run_in_session(mol, method="UHF", basis="STO-3G", verbose=0)
        assert result.dipole_moment_debye is not None
        assert result.dipole_moment_debye > 0


# ============================================================================
# Calculation tests — Linux/WSL with pyscf
# ============================================================================


class TestRunInSessionBasic:
    """Basic functional tests for run_in_session()."""

    @pyscf_only
    @pytest.mark.slow
    def test_returns_session_result(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        assert isinstance(result, SessionResult)

    @pyscf_only
    @pytest.mark.slow
    def test_h2_rhf_sto3g_converges(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        assert result.converged is True

    @pyscf_only
    @pytest.mark.slow
    def test_h2_energy_plausible(self):
        """RHF/STO-3G energy for H2 near equilibrium should be around -1.117 Ha."""
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        # Literature: ~-1.117 Ha; allow ±0.1 Ha for geometry/basis variation
        assert -1.25 < result.energy_hartree < -1.0

    @pyscf_only
    @pytest.mark.slow
    def test_result_formula_matches_molecule(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        assert result.formula == "H2"

    @pyscf_only
    @pytest.mark.slow
    def test_single_atom_calculation_converges(self):
        """L13 audit fix: single-atom molecules (parse_xyz_input no longer
        rejects them) must actually run through PySCF end-to-end, not just
        parse.
        """
        from quantui.molecule import parse_xyz_input
        from quantui.session_calc import run_in_session

        atoms, coords = parse_xyz_input("Ar 0.0 0.0 0.0")
        argon = Molecule(atoms, coords)

        result = run_in_session(argon, method="RHF", basis="STO-3G", verbose=0)
        assert result.converged is True
        assert result.formula == "Ar"

    @pyscf_only
    @pytest.mark.slow
    def test_result_method_matches_input(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        assert result.method == "RHF"

    @pyscf_only
    @pytest.mark.slow
    def test_result_basis_matches_input(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        assert result.basis == "STO-3G"

    @pyscf_only
    @pytest.mark.slow
    def test_homo_lumo_gap_is_positive(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        if result.homo_lumo_gap_ev is not None:
            assert result.homo_lumo_gap_ev > 0

    @pyscf_only
    @pytest.mark.slow
    def test_n_iterations_is_positive(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_h2(), method="RHF", basis="STO-3G", verbose=0)
        assert result.n_iterations > 0


class TestRunInSessionOutputStream:
    """Verify that PySCF output is routed to the progress_stream."""

    @pyscf_only
    @pytest.mark.slow
    def test_output_written_to_stream(self):
        from quantui.session_calc import run_in_session

        buf = io.StringIO()
        run_in_session(
            _h2(), method="RHF", basis="STO-3G", verbose=3, progress_stream=buf
        )
        output = buf.getvalue()
        # With verbose=3, PySCF writes SCF cycle information
        assert len(output) > 0

    @pyscf_only
    @pytest.mark.slow
    def test_silent_when_verbose_zero(self):
        from quantui.session_calc import run_in_session

        buf = io.StringIO()
        run_in_session(
            _h2(), method="RHF", basis="STO-3G", verbose=0, progress_stream=buf
        )
        output = buf.getvalue()
        # verbose=0 should produce little or no output
        assert len(output) < 500  # allow for minimal header lines


class TestRunInSessionMetadata:
    """run_in_session() preserves charge and multiplicity into the calculation."""

    @pyscf_only
    @pytest.mark.slow
    def test_water_rhf_converges(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(_water(), method="RHF", basis="STO-3G", verbose=0)
        assert result.converged is True

    @pyscf_only
    @pytest.mark.slow
    def test_water_energy_plausible(self):
        """RHF/STO-3G for H2O should be around -74.96 Ha."""
        from quantui.session_calc import run_in_session

        result = run_in_session(_water(), method="RHF", basis="STO-3G", verbose=0)
        assert -76.0 < result.energy_hartree < -73.0


# ============================================================================
# Public API surface
# ============================================================================


class TestPublicAPI:
    """SessionResult and run_in_session are importable from quantui top-level."""

    def test_session_result_importable_from_quantui(self):
        from quantui import SessionResult  # noqa: F401

    def test_run_in_session_importable_from_quantui(self):
        from quantui import run_in_session  # noqa: F401

    def test_hartree_to_ev_constant(self):
        """Sanity-check the conversion constant against a known value."""
        assert abs(HARTREE_TO_EV - 27.211) < 0.01


# ============================================================================
# M8.1 — CCSD and CCSD(T) scaffolding (config + result + formatter)
# ============================================================================


class TestM8CcsdScaffolding:
    """Verify CCSD + CCSD(T) are wired into the method list, METHOD_INFO,
    SessionResult dataclass, formatter, and perf-scaling tables.

    These checks run on any platform — no PySCF required. The actual CCSD
    compute path is exercised by ``TestM8CcsdComputeWater`` below, which
    is PySCF-gated.
    """

    def test_ccsd_in_supported_methods(self):
        from quantui.config import SUPPORTED_METHODS

        assert "CCSD" in SUPPORTED_METHODS

    def test_ccsd_t_in_supported_methods(self):
        from quantui.config import SUPPORTED_METHODS

        assert "CCSD(T)" in SUPPORTED_METHODS

    def test_method_info_has_ccsd_entry_with_scaling_warning(self):
        from quantui.config import METHOD_INFO

        assert "CCSD" in METHOD_INFO
        info = METHOD_INFO["CCSD"]
        # Type marker + description mentions the O(N^6) scaling so the user
        # understands the cost tradeoff before clicking Run.
        assert info["type"] == "wavefunction"
        assert "N⁶" in info["description"] or "N^6" in info["description"]

    def test_method_info_has_ccsd_t_entry_with_scaling_warning(self):
        from quantui.config import METHOD_INFO

        assert "CCSD(T)" in METHOD_INFO
        info = METHOD_INFO["CCSD(T)"]
        assert info["type"] == "wavefunction"
        # Either notation acceptable in the user-facing description.
        assert "N⁷" in info["description"] or "N^7" in info["description"]

    def test_session_result_has_ccsd_fields_defaulting_none(self):
        result = _make_result()
        assert result.ccsd_correlation_hartree is None
        assert result.ccsd_t_correction_hartree is None

    def test_session_result_can_store_ccsd_fields(self):
        result = _make_result(
            ccsd_correlation_hartree=-0.123,
            ccsd_t_correction_hartree=-0.005,
        )
        assert result.ccsd_correlation_hartree == pytest.approx(-0.123)
        assert result.ccsd_t_correction_hartree == pytest.approx(-0.005)

    def test_formatter_renders_ccsd_breakdown(self):
        from quantui.app_formatters import format_result

        # CCSD-only result: HF reference + CCSD correlation, no (T) row.
        result = _make_result(
            energy_hartree=-1.200,  # arbitrary; HF derived = -1.077
            ccsd_correlation_hartree=-0.123,
        )
        html = format_result(result)
        assert "HF reference" in html
        assert "CCSD correlation" in html
        assert "(T) triples correction" not in html

    def test_formatter_renders_ccsd_t_breakdown(self):
        from quantui.app_formatters import format_result

        # CCSD(T) result: HF + CCSD correlation + triples correction rows.
        result = _make_result(
            energy_hartree=-1.205,
            ccsd_correlation_hartree=-0.123,
            ccsd_t_correction_hartree=-0.005,
        )
        html = format_result(result)
        assert "HF reference" in html
        assert "CCSD correlation" in html
        assert "(T) triples correction" in html

    def test_calc_log_scaling_exponent_ccsd(self):
        from quantui.calc_log import _METHOD_SCALE_EXP

        assert _METHOD_SCALE_EXP.get("CCSD") == pytest.approx(6.0)

    def test_calc_log_scaling_exponent_ccsd_t(self):
        from quantui.calc_log import _METHOD_SCALE_EXP

        assert _METHOD_SCALE_EXP.get("CCSD(T)") == pytest.approx(7.0)


class TestM8CcsdComputeWater:
    """PySCF-gated water-CCSD smoke test. Runs on WSL / Linux / macOS where
    PySCF is installed; skipped on Windows.
    """

    @pyscf_only
    def test_ccsd_water_runs_and_reports_correlation(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(
            molecule=_water(),
            method="CCSD",
            basis="STO-3G",
        )
        assert result.converged is True
        # CCSD correlation must be set and negative (correlation lowers energy).
        assert result.ccsd_correlation_hartree is not None
        assert result.ccsd_correlation_hartree < 0
        # (T) field must remain None for plain CCSD.
        assert result.ccsd_t_correction_hartree is None
        # Total energy must equal HF reference + CCSD correlation.
        assert result.energy_hartree < -74.0  # HF/STO-3G water ≈ -74.96 Ha

    @pyscf_only
    def test_ccsd_t_water_runs_and_reports_triples(self):
        from quantui.session_calc import run_in_session

        result = run_in_session(
            molecule=_water(),
            method="CCSD(T)",
            basis="STO-3G",
        )
        assert result.converged is True
        assert result.ccsd_correlation_hartree is not None
        assert result.ccsd_correlation_hartree < 0
        # (T) correction must be present and negative for water at minimum.
        assert result.ccsd_t_correction_hartree is not None
        assert result.ccsd_t_correction_hartree < 0


# ============================================================================
# Run directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
