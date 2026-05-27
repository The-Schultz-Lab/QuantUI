"""Tests for the session-55 xc-alias / D3-dispersion resolution helpers.

The user's tier-3 calibration output showed ``H₂O wB97X-D/6-31G*`` erroring
at 0.01 s — PySCF rejects ``mf.xc = "wb97x-d"`` because that composite
name is on the dftd3 black-list (pyscf/pyscf#2069). The fix:

- Alias ``wB97X-D`` to bare ``wb97x``.
- Add ``wB97X-D`` to ``_NEEDS_D3`` so dispersion is applied via
  ``pyscf.dftd3``, matching the UI label that already promises D3.
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
    def test_wb97x_d_resolves_to_bare_wb97x(self):
        # The session-55 bug: PySCF rejects "wb97x-d". Bare wb97x is
        # the right xc string; D3 dispersion is applied separately.
        assert resolve_xc("wB97X-D") == "wb97x"

    def test_wb97x_d_case_insensitive(self):
        # Users sometimes type "WB97X-D" or "wb97x-d" — all should resolve.
        for spelling in ("wB97X-D", "WB97X-D", "wb97x-d", "Wb97x-D"):
            assert resolve_xc(spelling) == "wb97x"

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
    def test_wb97x_d_needs_d3(self):
        # The session-55 fix: wB97X-D now needs external D3.
        assert needs_d3("wB97X-D") is True

    def test_pbe_d3_needs_d3(self):
        assert needs_d3("PBE-D3") is True

    def test_case_insensitive(self):
        assert needs_d3("WB97X-D") is True
        assert needs_d3("pbe-d3") is True

    def test_dispersion_free_methods_dont_need_d3(self):
        for method in ("RHF", "UHF", "B3LYP", "PBE0", "M06-2X", "HSE06"):
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
    def test_no_d3_method_returns_mf_unchanged(self):
        mf = _FakeMf("B3LYP")
        result = maybe_apply_d3(mf, "B3LYP")
        assert result is mf

    def test_d3_method_with_missing_pyscf_returns_mf_unchanged(self, monkeypatch):
        # Simulate pyscf.dftd3 being absent (typical on Windows where
        # PySCF isn't installable at all). The helper must return the
        # original mf without raising.
        import builtins

        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "pyscf.dftd3" or name.startswith("pyscf.dftd3"):
                raise ImportError("simulated")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        mf = _FakeMf("wB97X-D")
        # Without progress_stream — must not raise.
        result = maybe_apply_d3(mf, "wB97X-D")
        assert result is mf

    def test_d3_warning_written_to_progress_stream(self, monkeypatch):
        import builtins
        import io

        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "pyscf.dftd3" or name.startswith("pyscf.dftd3"):
                raise ImportError("simulated")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        stream = io.StringIO()
        maybe_apply_d3(_FakeMf("wB97X-D"), "wB97X-D", progress_stream=stream)
        out = stream.getvalue()
        # User must see the missing-dispersion warning.
        assert "dftd3 not available" in out
        assert "wB97X-D" in out


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
        assert "maybe_apply_d3(mf, method" in src

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

        # The literal alias for wB97X-D in the template should be the
        # bare functional (post-session-55 fix). Doubled-brace literals
        # in the template appear as single braces in the output.
        assert "'wB97X-D': 'wb97x'" in PYSCF_SCRIPT_TEMPLATE
        assert "_NEEDS_D3" in PYSCF_SCRIPT_TEMPLATE
        # The old (broken) "wb97x-d" string must NOT appear.
        assert "'wB97X-D': 'wb97x-d'" not in PYSCF_SCRIPT_TEMPLATE


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
