"""Tests for M-EST / EST.4 — four-tier calibration suite.

Covers:

- Each of the 4 tier constants is well-formed (non-empty, each entry
  has a valid 7- or 8-tuple shape).
- The 8-tuple format (with explicit ``calc_type``) is correctly
  normalized by ``_normalize_entry``.
- Tier 3 contains at least one entry of each non-SP calc-type.
- Tier 4 strict-contains tier 3 (and so on up the chain).
- ``_MODE_TO_SUITE`` resolves all the mode strings — both the new
  tier names and the legacy aliases.
- ``run_calibration(mode="bogus")`` falls back to tier 1 without
  crashing (graceful degradation).

All tests are platform-independent. The PySCF-gated execution of
``run_calibration`` itself lives in ``tests/test_benchmarks.py`` —
this file checks the suite *shape* without running PySCF.
"""

from __future__ import annotations

import pytest

from quantui import benchmarks
from quantui.benchmarks import (
    _MODE_TO_SUITE,
    BENCHMARK_SUITE,
    BENCHMARK_SUITE_LONG,
    BENCHMARK_SUITE_TIER1,
    BENCHMARK_SUITE_TIER2,
    BENCHMARK_SUITE_TIER3,
    BENCHMARK_SUITE_TIER4,
    _normalize_entry,
)

_SP = "single_point"
_OPT = "geometry_opt"
_FREQ = "frequency"


class TestTierSuites:
    def test_tier1_alias_matches_legacy_short(self):
        # Back-compat: BENCHMARK_SUITE_TIER1 is the same object as
        # BENCHMARK_SUITE (existing tests + app.py imports rely on this).
        assert BENCHMARK_SUITE_TIER1 is BENCHMARK_SUITE

    def test_tier2_alias_matches_legacy_long(self):
        assert BENCHMARK_SUITE_TIER2 is BENCHMARK_SUITE_LONG

    def test_tier2_extends_tier1(self):
        # Tier 2 contains every tier-1 entry plus more.
        assert len(BENCHMARK_SUITE_TIER2) > len(BENCHMARK_SUITE_TIER1)
        for entry in BENCHMARK_SUITE_TIER1:
            assert entry in BENCHMARK_SUITE_TIER2

    def test_tier3_extends_tier2(self):
        assert len(BENCHMARK_SUITE_TIER3) > len(BENCHMARK_SUITE_TIER2)
        for entry in BENCHMARK_SUITE_TIER2:
            assert entry in BENCHMARK_SUITE_TIER3

    def test_tier4_extends_tier3(self):
        assert len(BENCHMARK_SUITE_TIER4) > len(BENCHMARK_SUITE_TIER3)
        for entry in BENCHMARK_SUITE_TIER3:
            assert entry in BENCHMARK_SUITE_TIER4

    def test_tier1_and_tier2_are_sp_only(self):
        # Lower tiers stay 7-tuple (pure single-point) by design — the
        # user explicitly wanted tier 2 to remain SP-only.
        for entry in BENCHMARK_SUITE_TIER1:
            assert len(entry) == 7
        for entry in BENCHMARK_SUITE_TIER2:
            assert len(entry) == 7

    def test_tier3_introduces_geom_opt_and_freq(self):
        # Tier 3 must add at least one geom-opt AND at least one freq.
        calc_types = {_normalize_entry(e)["calc_type"] for e in BENCHMARK_SUITE_TIER3}
        assert _OPT in calc_types
        assert _FREQ in calc_types
        # And keep the SP majority.
        n_sp = sum(
            1 for e in BENCHMARK_SUITE_TIER3 if _normalize_entry(e)["calc_type"] == _SP
        )
        assert n_sp > len(BENCHMARK_SUITE_TIER3) // 2

    def test_tier4_has_post_hf_anchors(self):
        # Tier 4 must include MP2 + CCSD entries so the β=5.0 / β=6.0
        # scaling exponents in calc_log have calibration data.
        methods = {_normalize_entry(e)["method"] for e in BENCHMARK_SUITE_TIER4}
        assert "MP2" in methods
        assert "CCSD" in methods

    def test_tier4_includes_benzene_freq(self):
        # Benzene B3LYP/6-31G* frequency is the workhorse parallel-IR
        # anchor (12 atoms × 6 = 72 inner SCFs).
        labels = [_normalize_entry(e)["label"] for e in BENCHMARK_SUITE_TIER4]
        assert any("benzene" in lbl.lower() and "freq" in lbl.lower() for lbl in labels)


class TestNormalizeEntry:
    def test_seven_tuple_defaults_to_single_point(self):
        entry = (
            "H₂ RHF/STO-3G",
            ["H", "H"],
            [[0, 0, 0], [0, 0, 0.74]],
            0,
            1,
            "RHF",
            "STO-3G",
        )
        out = _normalize_entry(entry)
        assert out["calc_type"] == _SP
        assert out["method"] == "RHF"
        assert out["basis"] == "STO-3G"

    def test_eight_tuple_overrides_calc_type(self):
        entry = (
            "H₂O B3LYP/STO-3G [GeoOpt]",
            ["O", "H", "H"],
            [[0, 0, 0], [0.7, 0.6, 0], [-0.7, 0.6, 0]],
            0,
            1,
            "B3LYP",
            "STO-3G",
            "geometry_opt",
        )
        out = _normalize_entry(entry)
        assert out["calc_type"] == "geometry_opt"

    def test_invalid_length_raises_valueerror(self):
        with pytest.raises(ValueError, match="7 or 8 fields"):
            _normalize_entry(("label", ["H"]))  # only 2 fields

    def test_all_tier_entries_normalize_cleanly(self):
        # Every entry in every tier must normalize without raising.
        for tier in (
            BENCHMARK_SUITE_TIER1,
            BENCHMARK_SUITE_TIER2,
            BENCHMARK_SUITE_TIER3,
            BENCHMARK_SUITE_TIER4,
        ):
            for entry in tier:
                out = _normalize_entry(entry)
                assert out["calc_type"] in (_SP, _OPT, _FREQ)
                assert len(out["atoms"]) == len(out["coords"])


class TestModeToSuite:
    def test_new_tier_names_resolve(self):
        assert _MODE_TO_SUITE["tier1"] is BENCHMARK_SUITE_TIER1
        assert _MODE_TO_SUITE["tier2"] is BENCHMARK_SUITE_TIER2
        assert _MODE_TO_SUITE["tier3"] is BENCHMARK_SUITE_TIER3
        assert _MODE_TO_SUITE["tier4"] is BENCHMARK_SUITE_TIER4

    def test_legacy_short_long_aliases(self):
        # Back-compat: any pinned UI state or older callers using "short"
        # or "long" should still resolve.
        assert _MODE_TO_SUITE["short"] is BENCHMARK_SUITE_TIER1
        assert _MODE_TO_SUITE["long"] is BENCHMARK_SUITE_TIER2


class TestUnknownModeFallback:
    def test_unknown_mode_does_not_raise(self):
        # PySCF-gated: when PySCF is absent the per-step error path
        # already prevents any actual calculation, but we still want
        # run_calibration to *not crash* on a typo'd mode string.
        result = benchmarks.run_calibration(mode="bogus_mode")
        # Falls back to tier1 — verify by checking the mode field.
        assert result.mode == "tier1"


class TestCalibrationResult:
    def test_n_total_uses_active_mode(self):
        from quantui.benchmarks import CalibrationResult

        r1 = CalibrationResult(timestamp="t", mode="tier1")
        r2 = CalibrationResult(timestamp="t", mode="tier2")
        r3 = CalibrationResult(timestamp="t", mode="tier3")
        r4 = CalibrationResult(timestamp="t", mode="tier4")
        assert r1.n_total == len(BENCHMARK_SUITE_TIER1)
        assert r2.n_total == len(BENCHMARK_SUITE_TIER2)
        assert r3.n_total == len(BENCHMARK_SUITE_TIER3)
        assert r4.n_total == len(BENCHMARK_SUITE_TIER4)
        # Strict ordering by tier depth.
        assert r1.n_total < r2.n_total < r3.n_total < r4.n_total
