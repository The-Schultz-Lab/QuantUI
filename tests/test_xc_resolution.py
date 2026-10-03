"""Tests for the session-55 xc-alias / D3-dispersion resolution helpers.

The user's tier-3 calibration output showed ``H₂O wB97X-D/6-31G*`` erroring
at 0.01 s — PySCF rejects ``mf.xc = "wb97x-d"`` because that composite
name is on the dftd3 black-list (pyscf/pyscf#2069). Session 55's original
fix aliased ``wB97X-D`` to bare ``wb97x`` and applied external Grimme D3 —
but that silently calculates a *different* functional (bare wb97x has
range-separation omega=0.3; the real wB97X-D has omega=0.2 and different
short-range exact exchange — AUDIT F03). The corrected fix:

- Alias ``wB97X-D`` to its full LibXC name ``hyb_gga_xc_wb97x_d`` — the
  actual Chai & Head-Gordon (2008) functional, whose own empirical
  dispersion is baked into the fit. PySCF's short-alias black-list
  (``pyscf.scf.dispersion.parse_dft``) intercepts "wb97x-d"/"wb97x_d" but
  not the full LibXC name, so this avoids the original error without
  substituting a different functional.
- ``wB97X-D`` does NOT go in ``_NEEDS_D3`` — wrapping it in
  ``pyscf.dftd3`` would double-count dispersion under a method that
  already includes its own.
- Extract ``resolve_xc()`` + ``maybe_apply_d3()`` so every DFT entry
  point (session_calc / freq_calc / tddft_calc / optimizer / nmr_calc /
  the script-export template) shares the same resolution logic. Before
  session 55 only ``session_calc`` had the alias lookup, meaning
  wB97X-D would have errored in EVERY non-SP workflow too.

All tests here are platform-independent. PySCF-gated round-trip tests
live in the other module suites that already gate on ``_PYSCF_AVAILABLE``.
"""

from __future__ import annotations

import inspect

import pytest

from quantui.session_calc import (
    _NEEDS_D3,
    _XC_ALIAS,
    maybe_apply_d3,
    needs_d3,
    resolve_xc,
)

# =====================================================================
# resolve_xc — the core mapping
# =====================================================================


class TestResolveXc:
    def test_wb97x_d_resolves_to_true_functional(self):
        # AUDIT F03: PySCF rejects "wb97x-d" (short-alias black-list), but
        # the fix must not substitute a different functional (bare wb97x)
        # to work around that — it must resolve to the actual wB97X-D
        # (Chai & Head-Gordon 2008) functional under its full LibXC name.
        assert resolve_xc("wB97X-D") == "hyb_gga_xc_wb97x_d"
        assert resolve_xc("wB97X-D") != "wb97x"

    def test_wb97x_d_case_insensitive(self):
        # Users sometimes type "WB97X-D" or "wb97x-d" — all should resolve.
        for spelling in ("wB97X-D", "WB97X-D", "wb97x-d", "Wb97x-D"):
            assert resolve_xc(spelling) == "hyb_gga_xc_wb97x_d"

    def test_wb97x_d_is_a_distinct_functional_from_bare_wb97x(self):
        """AUDIT F03 numeric regression: resolve_xc("wB97X-D") must resolve
        to a functional with different range-separation parameters than
        bare wb97x, confirmed against PySCF/LibXC directly (independent of
        the alias table's own claims)."""
        pytest.importorskip("pyscf.dft")
        from pyscf.dft import libxc

        resolved = resolve_xc("wB97X-D")
        assert resolved != "wb97x"
        wb97xd_omega = libxc.rsh_coeff(resolved)[0]
        wb97x_omega = libxc.rsh_coeff("wb97x")[0]
        assert wb97xd_omega == pytest.approx(0.2, abs=1e-6)
        assert wb97x_omega == pytest.approx(0.3, abs=1e-6)
        assert wb97xd_omega != wb97x_omega

    def test_pbe_d3_resolves_to_bare_pbe(self):
        # PBE-D3 is the long-standing pattern this fix mirrors.
        assert resolve_xc("PBE-D3") == "pbe"

    def test_m06_l_aliased(self):
        assert resolve_xc("M06-L") == "m06l"

    def test_cam_b3lyp_aliased(self):
        assert resolve_xc("CAM-B3LYP") == "camb3lyp"

    def test_unaliased_methods_pass_through(self):
        # B3LYP, PBE0, M06-2X, HSE06 — PySCF accepts them as-is.
        for method in ("B3LYP", "PBE0", "M06-2X", "HSE06", "PBE", "B3PW91"):
            assert resolve_xc(method) == method

    def test_unknown_method_passes_through(self):
        # Forward-compat: a new method not in the table returns unchanged
        # so PySCF gets to decide whether to accept it.
        assert resolve_xc("FUTURE-METHOD") == "FUTURE-METHOD"


# =====================================================================
# needs_d3 — gates external dispersion wrapping
# =====================================================================


class TestNeedsD3:
    def test_wb97x_d_does_not_need_external_d3(self):
        # AUDIT F03: wB97X-D's dispersion is baked into the XC functional
        # itself (hyb_gga_xc_wb97x_d) — wrapping it in pyscf.dftd3 would
        # double-count dispersion, so it must NOT be in _NEEDS_D3.
        assert needs_d3("wB97X-D") is False

    def test_pbe_d3_needs_d3(self):
        assert needs_d3("PBE-D3") is True

    def test_case_insensitive(self):
        assert needs_d3("WB97X-D") is False
        assert needs_d3("pbe-d3") is True

    def test_dispersion_free_methods_dont_need_d3(self):
        for method in ("RHF", "UHF", "B3LYP", "PBE0", "M06-2X", "HSE06", "wB97X-D"):
            assert needs_d3(method) is False

    def test_unknown_method_doesnt_need_d3(self):
        # Default: only methods explicitly in _NEEDS_D3 get the wrap.
        assert needs_d3("FUTURE-METHOD") is False


# =====================================================================
# maybe_apply_d3 — graceful degradation when dftd3 unavailable
# =====================================================================


class _FakeMf:
    """Stand-in for a PySCF mf object — just needs to be identity-comparable."""

    def __init__(self, label):
        self.label = label


class TestMaybeApplyD3:
    """AUDIT F04 — maybe_apply_d3 now returns (mf, dispersion_applied) so
    callers can record whether a D3-requiring result is actually missing
    its dispersion correction, instead of silently keeping the original
    method label on an uncorrected result."""

    def test_no_d3_method_returns_mf_unchanged_and_none_flag(self):
        mf = _FakeMf("B3LYP")
        result_mf, dispersion_applied = maybe_apply_d3(mf, "B3LYP")
        assert result_mf is mf
        assert dispersion_applied is None

    def test_wb97x_d_returns_mf_unchanged_and_none_flag(self):
        # AUDIT F03: wB97X-D's dispersion is already in the XC functional —
        # maybe_apply_d3 must be a no-op for it (never imports pyscf.dftd3),
        # and dispersion_applied is None (not applicable), not False.
        mf = _FakeMf("wB97X-D")
        result_mf, dispersion_applied = maybe_apply_d3(mf, "wB97X-D")
        assert result_mf is mf
        assert dispersion_applied is None

    def test_d3_method_without_backend_returns_mf_unchanged_and_false_flag(
        self, monkeypatch
    ):
        # No D3 backend installed (pyscf-dispersion missing, as on Windows
        # where PySCF isn't installable at all). The helper must return the
        # original mf, flagged dispersion_applied=False, without raising.
        monkeypatch.setattr("quantui.session_calc._d3_backend", lambda: None)
        mf = _FakeMf("PBE-D3")
        result_mf, dispersion_applied = maybe_apply_d3(mf, "PBE-D3")
        assert result_mf is mf
        assert dispersion_applied is False
        assert not hasattr(mf, "disp")

    def test_d3_warning_written_to_progress_stream(self, monkeypatch):
        import io

        monkeypatch.setattr("quantui.session_calc._d3_backend", lambda: None)
        stream = io.StringIO()
        maybe_apply_d3(_FakeMf("PBE-D3"), "PBE-D3", progress_stream=stream)
        out = stream.getvalue()
        # User must see the missing-dispersion warning.
        assert "without D3 correction" in out
        assert "PBE-D3" in out

    def test_d3_warning_logged_even_without_progress_stream(self, monkeypatch, caplog):
        # AUDIT F04: the optimizer path used to call maybe_apply_d3 with no
        # progress_stream at all, so a missing D3 backend gave NO warning
        # anywhere. It must now always be logged, stream or not.
        import logging

        monkeypatch.setattr("quantui.session_calc._d3_backend", lambda: None)
        with caplog.at_level(logging.WARNING, logger="quantui.session_calc"):
            maybe_apply_d3(_FakeMf("PBE-D3"), "PBE-D3")

        assert any(
            "without D3 correction" in rec.message for rec in caplog.records
        ), caplog.text

    def test_pyscf_dispersion_backend_sets_zero_damping_d3(self, monkeypatch):
        # "-D3" is Grimme's zero-damping D3 (Gaussian GD3), set through
        # PySCF's built-in mf.disp so gradients and Hessians include it.
        monkeypatch.setattr("quantui.session_calc._d3_backend", lambda: "dispersion")
        mf = _FakeMf("PBE-D3")
        result_mf, dispersion_applied = maybe_apply_d3(mf, "pbe-d3")
        assert result_mf is mf
        assert dispersion_applied is True
        assert mf.disp == "d3zero"


@pytest.mark.slow
class TestD3RealPySCF:
    """PBE-D3 actually lowers the energy by the D3 term (pyscf-dispersion)."""

    def test_pbe_d3_energy_includes_dispersion(self):
        pytest.importorskip("pyscf")
        pytest.importorskip("pyscf.dispersion")
        from pyscf import dft, gto
        from pyscf.scf import dispersion

        mol = gto.M(
            atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587;"
            "O 0 0 2.9; H 0 0.757 3.487; H 0 -0.757 3.487",
            basis="sto-3g",
            verbose=0,
        )
        plain = dft.RKS(mol)
        plain.xc = "pbe"
        e_plain = plain.kernel()

        mf = dft.RKS(mol)
        mf.xc = resolve_xc("PBE-D3")
        mf, applied = maybe_apply_d3(mf, "PBE-D3")
        e_d3 = mf.kernel()

        assert applied is True
        e_disp = dispersion.get_dispersion(mf)
        assert e_disp < 0
        assert e_d3 == pytest.approx(e_plain + e_disp, abs=1e-7)
        # Dispersion reaches the analytic gradient too (geometry opt path).
        assert mf.nuc_grad_method().kernel().shape == (6, 3)


# =====================================================================
# Coverage check — every DFT entry point uses the helpers
# =====================================================================


class TestEntryPointsUseHelpers:
    """The bug bit because freq_calc / tddft_calc / optimizer / nmr_calc
    bypassed the alias lookup. These source-level tests guard against
    a regression that re-introduces ``mf.xc = method`` directly.
    """

    def test_session_calc_uses_resolve_xc(self):
        # The real DFT branch lives in ``_run_session_calc_body`` (inner
        # function ``run_in_session`` calls), so grep the module source
        # rather than just the public wrapper.
        from quantui import session_calc

        src = inspect.getsource(session_calc)
        assert "resolve_xc(method)" in src
        assert "maybe_apply_d3(" in src
        assert "mf, method, progress_stream=progress_stream" in src

    def test_freq_calc_uses_resolve_xc(self):
        from quantui import freq_calc

        # The full module source — covers both the outer SCF setup and
        # any inner SCF helpers.
        src = inspect.getsource(freq_calc)
        assert "resolve_xc" in src
        # The inner displaced-SCF helper reads mf.xc directly (which by
        # then is already resolved), so maybe_apply_d3 only appears in
        # the outer setup. One usage is enough.

    def test_tddft_calc_uses_resolve_xc(self):
        from quantui import tddft_calc

        src = inspect.getsource(tddft_calc)
        assert "resolve_xc" in src
        assert "maybe_apply_d3" in src

    def test_optimizer_uses_resolve_xc(self):
        from quantui import optimizer

        src = inspect.getsource(optimizer)
        assert "resolve_xc" in src
        assert "maybe_apply_d3" in src

    def test_nmr_calc_uses_resolve_xc(self):
        from quantui import nmr_calc

        src = inspect.getsource(nmr_calc)
        assert "resolve_xc" in src
        assert "maybe_apply_d3" in src

    def test_script_template_embeds_alias_resolution(self):
        # The script-export template generates a standalone .py file
        # — can't depend on quantui imports — so the alias table is
        # inlined.
        from quantui.config import PYSCF_SCRIPT_TEMPLATE

        # The literal alias for wB97X-D in the template should be the true
        # functional's full LibXC name (AUDIT F03 fix), not bare wb97x.
        # Doubled-brace literals in the template appear as single braces
        # in the output.
        assert "'wB97X-D': 'hyb_gga_xc_wb97x_d'" in PYSCF_SCRIPT_TEMPLATE
        assert "_NEEDS_D3" in PYSCF_SCRIPT_TEMPLATE
        # Neither the black-listed short alias nor the wrong-functional
        # bare-wb97x substitution should appear.
        assert "'wB97X-D': 'wb97x-d'" not in PYSCF_SCRIPT_TEMPLATE
        assert "'wB97X-D': 'wb97x'" not in PYSCF_SCRIPT_TEMPLATE
        # wB97X-D must not be wrapped in external D3 (its dispersion is
        # already built into hyb_gga_xc_wb97x_d) — only PBE-D3 remains in
        # _NEEDS_D3. Doubled braces are this template's literal-brace escape.
        assert "_NEEDS_D3 = {{'PBE-D3'}}" in PYSCF_SCRIPT_TEMPLATE


# =====================================================================
# Sanity: aliases stay in sync with config.SUPPORTED_METHODS
# =====================================================================


class TestAliasTableConsistency:
    def test_every_d3_method_has_an_alias(self):
        # If a method is in _NEEDS_D3 it MUST also be in _XC_ALIAS
        # — otherwise resolve_xc passes the display name straight to
        # PySCF, which is exactly the bug.
        for method in _NEEDS_D3:
            assert method in _XC_ALIAS, (
                f"{method!r} is in _NEEDS_D3 but not in _XC_ALIAS — "
                "PySCF will receive the display name and likely error."
            )

    def test_all_aliased_methods_in_supported_list(self):
        # Sanity: every alias key is actually a method the UI exposes
        # — otherwise the alias is dead code that no calc path can hit.
        from quantui.config import SUPPORTED_METHODS

        for method in _XC_ALIAS:
            assert method in SUPPORTED_METHODS, (
                f"{method!r} is aliased in _XC_ALIAS but not in "
                f"config.SUPPORTED_METHODS — dead code or removed method."
            )
