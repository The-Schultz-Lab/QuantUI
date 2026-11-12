"""Dispersion / SCF-rescue provenance on Frequency and NMR results (audit SCF-7).

``run_freq_calc`` and ``run_nmr_calc`` discarded ``maybe_apply_d3``'s
``dispersion_applied`` flag and the reference SCF's ``scf_rescue_stage``:
a PBE-D3 frequency/NMR run without a D3 backend still read "PBE-D3" with
only a log line, and a rescued SCF looked ordinary. Both are now carried on
FreqResult / NMRResult, saved in result.json and the batch payloads, and
shown on the live and History cards.
"""

from __future__ import annotations

import io
import json

import pytest

pytest.importorskip("pyscf")

from quantui.molecule import Molecule  # noqa: E402


def _h2() -> Molecule:
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


@pytest.fixture
def no_d3(monkeypatch):
    import quantui.session_calc as sc

    monkeypatch.setattr(sc, "_d3_backend", lambda: None)


@pytest.fixture
def forced_rescue(monkeypatch):
    """Make every run_scf_with_rescue report a level-shift rescue."""
    import quantui.scf_robust as sr

    real = sr.run_scf_with_rescue

    def fake(mf, *a, **kw):
        e = real(mf, *a, **kw)
        mf.scf_rescue_stage = sr.SCF_RESCUE_LEVEL_SHIFT
        return e

    monkeypatch.setattr(sr, "run_scf_with_rescue", fake)


def _freq(method: str):
    from quantui.freq_calc import run_freq_calc

    return run_freq_calc(_h2(), method, "STO-3G", progress_stream=io.StringIO())


def _nmr(method: str):
    pytest.importorskip("pyscf.prop.nmr")
    from quantui.nmr_calc import run_nmr_calc

    return run_nmr_calc(_h2(), method, "STO-3G", progress_stream=io.StringIO())


class TestResultFields:
    def test_freq_pbe_d3_without_backend_is_flagged(self, no_d3):
        r = _freq("PBE-D3")
        assert r.dispersion_applied is False
        assert r.scf_rescue_stage == "none"

    def test_freq_rhf_has_no_dispersion_flag(self):
        assert _freq("RHF").dispersion_applied is None

    def test_freq_rescue_stage_recorded(self, forced_rescue):
        assert _freq("RHF").scf_rescue_stage == "level_shift"

    def test_nmr_pbe_d3_without_backend_is_flagged(self, no_d3):
        r = _nmr("PBE-D3")
        assert r.dispersion_applied is False

    def test_nmr_rescue_stage_recorded(self, forced_rescue):
        assert _nmr("RHF").scf_rescue_stage == "level_shift"


class TestPersistenceAndCards:
    def test_saved_result_json_keeps_provenance(self, no_d3, tmp_path):
        from quantui.results_storage import save_result

        r = _freq("PBE-D3")
        r.scf_rescue_stage = "bootstrap"
        d = save_result(r, results_dir=tmp_path, calc_type="frequency")
        data = json.loads((d / "result.json").read_text())
        assert data["dispersion_applied"] is False
        assert data["scf_rescue_stage"] == "bootstrap"

    def test_batch_payloads_keep_provenance(self, no_d3):
        from quantui.backends.worker_payload import (
            freq_result_payload,
            nmr_result_payload,
        )

        f = _freq("PBE-D3")
        p = freq_result_payload(f, _h2())
        assert p["dispersion_applied"] is False
        assert p["scf_rescue_stage"] == "none"
        n = _nmr("PBE-D3")
        n.scf_rescue_stage = "failed"
        q = nmr_result_payload(n)
        assert q["dispersion_applied"] is False
        assert q["scf_rescue_stage"] == "failed"

    def test_live_cards_show_warnings(self, no_d3):
        from quantui.app_formatters import format_freq_result, format_nmr_result

        f = _freq("PBE-D3")
        f.scf_rescue_stage = "level_shift"
        html = format_freq_result(f)
        assert "NOT applied" in html
        assert "SCF rescue" in html
        n = _nmr("PBE-D3")
        n.scf_rescue_stage = "bootstrap"
        html = format_nmr_result(n)
        assert "NOT applied" in html
        assert "SCF rescue" in html

    def test_ordinary_run_has_no_warning_rows(self):
        from quantui.app_formatters import format_freq_result

        html = format_freq_result(_freq("RHF"))
        assert "NOT applied" not in html
        assert "SCF rescue" not in html

    @pytest.mark.parametrize("ct", ["frequency", "nmr"])
    def test_history_card_shows_warnings(self, ct):
        from quantui.app_formatters import format_past_result

        data = {
            "calc_type": ct,
            "formula": "H2",
            "method": "PBE-D3",
            "basis": "STO-3G",
            "energy_hartree": -1.1,
            "converged": True,
            "dispersion_applied": False,
            "scf_rescue_stage": "level_shift",
        }
        html = format_past_result(data)
        assert "NOT applied" in html
        assert "SCF rescue" in html
