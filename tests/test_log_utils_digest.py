"""Warnings Digest filtering — M-UX2 UXP2.6.

PySCF's ``get_occ`` emits ``HOMO x == LUMO y`` on every density it evaluates,
including the ``minao`` initial guess before SCF iteration 1. A transition-metal
run therefore surfaces that degeneracy warning even when the SCF converges to a
healthy gap. The digest should drop it as transient when the converged gap is
clearly non-degenerate, while keeping it for a genuinely (near-)degenerate
converged state and keeping all other warnings unconditionally.
"""

from __future__ import annotations

from types import SimpleNamespace

from quantui.log_utils import _extract_warnings, format_log_footer

# The exact shape PySCF writes (%.15g floats), as captured on the ferrocene run.
_FERROCENE_LOG = """\
Initial guess from minao.
init E= -1650.00339122848

WARN: HOMO -0.0401210572232631 == LUMO -0.0400479397532248

cycle= 1 E= -1636.07 delta_E= 13.9 |g|= 5.79
converged SCF energy = -1649.69023957
"""


class TestDegeneracyWarningFilter:
    def test_transient_degeneracy_dropped_when_converged_gap_wide(self):
        # Ferrocene: warning is from the initial guess; converged gap is 2.98 eV.
        warnings = _extract_warnings(_FERROCENE_LOG, converged_gap_ev=2.98)
        assert not any("== LUMO" in w for w in warnings)

    def test_degeneracy_kept_when_converged_gap_is_small(self):
        # A genuinely (near-)degenerate converged state keeps the warning.
        warnings = _extract_warnings(_FERROCENE_LOG, converged_gap_ev=0.002)
        assert any("== LUMO" in w for w in warnings)

    def test_degeneracy_kept_when_gap_unknown(self):
        # No converged gap (e.g. failed/UHF path) — conservative: keep it.
        warnings = _extract_warnings(_FERROCENE_LOG, converged_gap_ev=None)
        assert any("== LUMO" in w for w in warnings)

    def test_other_warnings_always_survive(self):
        log = (
            "WARN: HOMO -0.04 == LUMO -0.04\n"
            "WARN: ECP not specified for something\n"
            "SCF did not converge\n"
        )
        warnings = _extract_warnings(log, converged_gap_ev=5.0)
        joined = "\n".join(warnings)
        assert "== LUMO" not in joined  # transient degeneracy dropped
        assert "ECP not specified" in joined  # unrelated warning kept
        assert any("not converge" in w for w in warnings)  # kept

    def test_threshold_boundary(self):
        # Just above the 0.1 eV threshold → dropped; at/below → kept.
        assert not any(
            "== LUMO" in w
            for w in _extract_warnings(_FERROCENE_LOG, converged_gap_ev=0.11)
        )
        assert any(
            "== LUMO" in w
            for w in _extract_warnings(_FERROCENE_LOG, converged_gap_ev=0.10)
        )


class TestFooterEndToEnd:
    def test_footer_drops_transient_degeneracy_for_healthy_result(self):
        result = SimpleNamespace(
            converged=True,
            n_iterations=25,
            energy_hartree=-1649.69023957,
            homo_lumo_gap_ev=2.9802,
        )
        footer = format_log_footer(
            result=result,
            wall_time=85.8,
            cpu_time=1635.9,
            log_text=_FERROCENE_LOG,
            success=True,
        )
        # The healthy gap is reported; the transient degeneracy warning is gone.
        assert "HOMO-LUMO gap: 2.9802 eV" in footer
        assert "== LUMO" not in footer
