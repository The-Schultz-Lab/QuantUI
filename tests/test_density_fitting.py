"""Density fitting (RI) opt-in — M-DF (DF.1 + DF.4 + DF.7).

Covers the plumbing and the settings/telemetry gate, the parts a cloud session
can verify without PySCF:

* ``quantui.density_fitting.try_density_fit`` — the helper, mirroring
  ``gpu_offload.try_to_gpu``: never raises, off unless enabled, and applies
  ``mf.density_fit()`` when it is.
* the ``compute.density_fit`` user setting (additive, default off);
* the ``density_fit`` field on ``log_calculation`` and its partition in the
  time estimator, so fitted and unfitted runs of the same chemistry are never
  averaged into one bimodal pool;
* the ``SessionResult.density_fit`` field.

The measured speedup / accuracy numbers are a PySCF-gated concern (DF.2) and are
not asserted here. Platform-independent: no PySCF, no widgets front-end.
"""

from __future__ import annotations

import pytest

from quantui import calc_log, density_fitting
from quantui.user_settings import UserSettings

# ══ The helper ═══════════════════════════════════════════════════════════════


class _FakeFittedMF:
    """Sentinel returned by a successful ``.density_fit()``."""


class _FakeMF:
    def __init__(self) -> None:
        self.auxbasis_arg = "UNSET"

    def density_fit(self, auxbasis=None):
        self.auxbasis_arg = auxbasis
        return _FakeFittedMF()


class _BoomMF:
    def density_fit(self, auxbasis=None):
        raise RuntimeError("density_fit blew up")


class TestTryDensityFit:
    def test_disabled_returns_original_unchanged(self):
        mf = _FakeMF()
        out, used = density_fitting.try_density_fit(mf, enabled=False)
        assert out is mf
        assert used is False

    def test_enabled_applies_density_fit(self):
        mf = _FakeMF()
        out, used = density_fitting.try_density_fit(mf, enabled=True)
        assert isinstance(out, _FakeFittedMF)
        assert used is True

    def test_auxbasis_is_forwarded(self):
        mf = _FakeMF()
        density_fitting.try_density_fit(
            mf, enabled=True, auxbasis="def2-universal-jfit"
        )
        assert mf.auxbasis_arg == "def2-universal-jfit"

    def test_failure_falls_back_and_never_raises(self):
        mf = _BoomMF()
        out, used = density_fitting.try_density_fit(mf, enabled=True)
        assert out is mf
        assert used is False

    def test_enabled_none_reads_settings_gate(self, monkeypatch):
        mf = _FakeMF()
        monkeypatch.setattr(
            density_fitting, "_density_fit_enabled_in_settings", lambda: True
        )
        _, used = density_fitting.try_density_fit(mf)
        assert used is True

        mf2 = _FakeMF()
        monkeypatch.setattr(
            density_fitting, "_density_fit_enabled_in_settings", lambda: False
        )
        out2, used2 = density_fitting.try_density_fit(mf2)
        assert out2 is mf2
        assert used2 is False


# ══ The settings gate ════════════════════════════════════════════════════════


class TestSettingsGate:
    def test_reflects_saved_setting(self):
        s = UserSettings.load()
        s.compute.density_fit = True
        s.save()
        assert density_fitting._density_fit_enabled_in_settings() is True
        s.compute.density_fit = False
        s.save()
        assert density_fitting._density_fit_enabled_in_settings() is False

    def test_unreadable_settings_default_to_off(self, monkeypatch):
        """A settings glitch must never silently switch DF *on*.

        The opposite of the GPU gate, which fails open — here the safe default
        is exact integrals.
        """

        def boom(*_a, **_k):
            raise OSError("cannot read settings")

        monkeypatch.setattr(
            "quantui.user_settings.UserSettings.load", staticmethod(boom)
        )
        assert density_fitting._density_fit_enabled_in_settings() is False


# ══ The user setting ═════════════════════════════════════════════════════════


class TestComputeSettingDensityFit:
    def test_default_is_off(self):
        assert UserSettings().compute.density_fit is False

    def test_round_trips(self):
        s = UserSettings()
        s.compute.density_fit = True
        s.save()
        assert UserSettings.load().compute.density_fit is True

    def test_invalid_value_falls_back_to_default(self):
        parsed = UserSettings._from_dict(
            {"_schema_version": 1, "compute": {"density_fit": "yes please"}}
        )
        assert parsed.compute.density_fit is False

    def test_absent_key_is_off(self):
        parsed = UserSettings._from_dict({"_schema_version": 1, "compute": {}})
        assert parsed.compute.density_fit is False

    def test_gpu_and_df_are_independent(self):
        parsed = UserSettings._from_dict(
            {
                "_schema_version": 1,
                "compute": {"gpu_enabled": False, "density_fit": True},
            }
        )
        assert parsed.compute.gpu_enabled is False
        assert parsed.compute.density_fit is True


# ══ The perf-log field (DF.4) ════════════════════════════════════════════════


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    calc_log._READ_ALL_CACHE.clear()
    yield tmp_path
    calc_log._READ_ALL_CACHE.clear()


def _last_record():
    calc_log._READ_ALL_CACHE.clear()
    return calc_log._read_all(calc_log._perf_path())[-1]


class TestLogCalculationDensityFit:
    def _log(self, **overrides):
        params = dict(
            formula="H2O",
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="6-31G*",
            n_iterations=8,
            elapsed_s=12.0,
            converged=True,
        )
        params.update(overrides)
        calc_log.log_calculation(**params)
        return _last_record()

    def test_true_is_recorded(self, log_dir):
        assert self._log(density_fit=True)["density_fit"] is True

    def test_false_is_recorded(self, log_dir):
        assert self._log(density_fit=False)["density_fit"] is False

    def test_none_omits_the_key(self, log_dir):
        assert "density_fit" not in self._log(density_fit=None)


# ══ Estimator partitioning (DF.4) ════════════════════════════════════════════


def _record(**overrides) -> dict:
    base = {
        "timestamp": "2026-08-01T00:00:00+00:00",
        "formula": "H2O",
        "n_atoms": 3,
        "n_electrons": 10,
        "method": "RHF",
        "basis": "6-31G",
        "n_iterations": 8,
        "elapsed_s": 10.0,
        "converged": True,
        "n_basis": 13,
        "n_cores": 1,
        "calc_type": "single_point",
    }
    base.update(overrides)
    return base


def _estimate(records, **kwargs):
    params = {
        "n_atoms": 3,
        "n_electrons": 10,
        "method": "RHF",
        "basis": "6-31G",
        "n_basis": 13,
        "calc_type": "single_point",
    }
    params.update(kwargs)
    return calc_log.estimate_time_from_records(records, **params)


class TestEstimatorDensityFitPartition:
    def test_prefers_matching_df_pool(self):
        """A DF prediction ignores unfitted timings of the same chemistry."""
        records = [
            _record(density_fit=True, elapsed_s=10.0),
            _record(density_fit=True, elapsed_s=10.0),
            _record(density_fit=False, elapsed_s=1000.0),
            _record(density_fit=False, elapsed_s=1000.0),
        ]
        est = _estimate(records, density_fit=True)
        assert est is not None
        assert est["seconds"] == pytest.approx(10.0, rel=0.01)

    def test_legacy_records_count_as_unfitted(self):
        """Records predating M-DF were all unfitted — admit them to the DF=False
        pool, not the DF=True one."""
        records = [_record(elapsed_s=10.0) for _ in range(6)]  # no density_fit key
        est = _estimate(records, density_fit=False)
        assert est is not None
        # Used the pool (not forced to fall back on an empty DF=False set).
        assert est["seconds"] == pytest.approx(10.0, rel=0.01)

    def test_df_requested_but_history_unfitted_falls_back_and_downgrades(self):
        matched = _estimate(
            [_record(density_fit=False, elapsed_s=10.0) for _ in range(8)],
            density_fit=False,
        )
        fell_back = _estimate(
            [_record(density_fit=False, elapsed_s=10.0) for _ in range(8)],
            density_fit=True,
        )
        assert matched["confidence"] == "high"
        assert fell_back["confidence"] == "medium"

    def test_three_axis_downgrades_compose(self):
        """DF + device + provenance fall-backs should stack, not read as one."""
        records = [
            _record(
                source="calibration",
                gpu_used=False,
                density_fit=False,
                elapsed_s=10.0,
            )
            for _ in range(8)
        ]
        est = _estimate(records, source="app", gpu_used=True, density_fit=True)
        assert est is not None
        assert est["confidence"] == "low"


# ══ SessionResult field ══════════════════════════════════════════════════════


class TestSessionResultDensityFitField:
    def test_default_is_false(self):
        from quantui.session_calc import SessionResult

        r = SessionResult(
            energy_hartree=-1.0,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=8,
            method="B3LYP",
            basis="6-31G*",
            formula="H2O",
        )
        assert r.density_fit is False

    def test_can_store_true(self):
        from quantui.session_calc import SessionResult

        r = SessionResult(
            energy_hartree=-1.0,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=8,
            method="B3LYP",
            basis="6-31G*",
            formula="H2O",
            density_fit=True,
        )
        assert r.density_fit is True
