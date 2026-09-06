"""Tests for quantui.tddft_calc.

Unit-level tests for run_tddft_calc(). Broader integration coverage (history
replay, panel activation) lives in test_tddft_analysis_history.py.
"""

from __future__ import annotations

import pytest

from quantui.molecule import Molecule

_PYSCF_AVAILABLE = False
try:
    import pyscf as _pyscf  # noqa: F401

    _PYSCF_AVAILABLE = True
except ImportError:
    pass

pyscf_only = pytest.mark.skipif(
    not _PYSCF_AVAILABLE, reason="PySCF not installed (Linux/macOS/WSL only)"
)


def _water() -> Molecule:
    return Molecule(
        ["O", "H", "H"], [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]]
    )


# ============================================================================
# Post-HF method guard (M2 audit fix, 2026-07-14)
# ============================================================================


class TestRunTddftCalcPostHfGuard:
    """Post-HF methods raise a clear ValueError instead of a cryptic LibXC error.

    Regression: run_tddft_calc() had no special-casing for MP2/CCSD/CCSD(T)
    — the SCF-selection branch silently treated them as a DFT xc functional
    (mf.xc = "CCSD"), failing deep inside PySCF with "LibXCFunctional: name
    'CCSD' not found" instead of a clear message. The guard fires before any
    PySCF import, so it needs no PySCF.
    """

    @pytest.mark.parametrize("method", ["MP2", "CCSD", "CCSD(T)"])
    def test_post_hf_method_raises_value_error(self, method):
        from quantui.tddft_calc import run_tddft_calc

        with pytest.raises(ValueError, match="post-HF"):
            run_tddft_calc(_water(), method=method, basis="STO-3G", nstates=2)


# ============================================================================
# Basic run — PySCF-gated
# ============================================================================


class TestRunTddftCalcBasic:
    @pyscf_only
    @pytest.mark.slow
    def test_returns_tddft_result(self):
        from quantui.tddft_calc import run_tddft_calc

        result = run_tddft_calc(_water(), method="RHF", basis="STO-3G", nstates=2)
        assert result.formula == "H2O"
        assert len(result.excitation_energies_ev) <= 2

    @pyscf_only
    @pytest.mark.slow
    def test_scf_variant_reports_rks_for_closed_shell_dft(self):
        """M-UX2 UXP2.10 — confirms the wiring, mirroring the identical
        RHF/UHF/RKS/UKS dispatch already thoroughly tested in
        test_session_calc.py::TestScfVariantProvenance."""
        from quantui.tddft_calc import run_tddft_calc

        result = run_tddft_calc(_water(), method="B3LYP", basis="STO-3G", nstates=2)
        assert result.scf_variant == "RKS"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
