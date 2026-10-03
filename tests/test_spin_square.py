"""⟨S²⟩ recorded for open-shell SCFs and reorganization-energy points (ISSUE.12).

A spin-contaminated UHF/UKS ion SCF gives a wrong energy and so a wrong λ;
without ⟨S²⟩ in the result there is no way to spot it afterwards.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from quantui.app_formatters import format_past_result, format_reorg_result
from quantui.molecule import Molecule
from quantui.reorganization_energy import (
    S2_KEYS,
    ReorganizationEnergyResult,
    ReorgChannelResult,
    ideal_s2,
    s2_points,
    spin_contaminated,
)
from quantui.results_storage import load_result, save_result

OH = Molecule(["O", "H"], [[0, 0, 0], [0, 0, 0.97]], multiplicity=2)
WATER = Molecule(
    ["O", "H", "H"], [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]]
)
H2 = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])


class TestIdealAndThreshold:
    @pytest.mark.parametrize("mult,ideal", [(1, 0.0), (2, 0.75), (3, 2.0), (4, 3.75)])
    def test_ideal_is_s_s_plus_1(self, mult, ideal):
        assert ideal_s2(mult) == pytest.approx(ideal)

    def test_doublet_threshold_is_ten_percent(self):
        assert not spin_contaminated(0.76, 2)
        assert not spin_contaminated(0.82, 2)
        assert spin_contaminated(0.83, 2)

    def test_singlet_uses_an_absolute_limit(self):
        assert not spin_contaminated(0.05, 1)
        assert spin_contaminated(0.2, 1)

    def test_missing_value_is_never_flagged(self):
        assert not spin_contaminated(None, 2)

    def test_points_carry_the_right_multiplicity(self):
        pts = {k: m for k, _, m in s2_points(2, 1)}
        assert pts == {
            "s2_neutral_at_neutral": 1,
            "s2_ion_at_ion": 2,
            "s2_ion_at_neutral": 2,
            "s2_neutral_at_ion": 1,
        }


class TestSessionResult:
    def test_open_shell_values(self):
        pytest.importorskip("pyscf")
        from quantui.session_calc import run_in_session

        uhf = run_in_session(OH, "UHF", "STO-3G", progress_stream=io.StringIO())
        assert uhf.spin_square == pytest.approx(0.753, abs=5e-3)
        assert uhf.multiplicity == 2
        # RHF on a doublet dispatches to ROHF, which is a pure spin state.
        rohf = run_in_session(OH, "RHF", "STO-3G", progress_stream=io.StringIO())
        assert rohf.scf_variant == "ROHF"
        assert rohf.spin_square == pytest.approx(0.75, abs=1e-8)

    def test_closed_shell_is_none(self):
        pytest.importorskip("pyscf")
        from quantui.session_calc import run_in_session

        r = run_in_session(WATER, "RHF", "STO-3G", progress_stream=io.StringIO())
        assert r.spin_square is None and r.multiplicity == 1


class TestReorgRun:
    def test_every_point_records_s2(self):
        pytest.importorskip("pyscf")
        pytest.importorskip("ase")
        from quantui.reorganization_energy import run_reorganization_energy

        stream = io.StringIO()
        r = run_reorganization_energy(
            H2, method="UHF", basis="STO-3G", mode="hole", progress_stream=stream
        )
        ch = r.channels[0]
        # H2+ has one electron: UHF is exactly a doublet.
        assert ch.s2_ion_at_ion == pytest.approx(0.75, abs=1e-8)
        assert ch.s2_ion_at_neutral == pytest.approx(0.75, abs=1e-8)
        # UHF singlet H2 near equilibrium stays closed-shell-like.
        assert ch.s2_neutral_at_neutral == pytest.approx(0.0, abs=1e-6)
        assert "⟨S²⟩ = 0.7500 (ideal 0.7500)" in stream.getvalue()
        assert "⟨S²⟩ ion @ R_ion = 0.7500" in r.summary()

    def test_rhf_closed_shell_neutral_records_none(self):
        pytest.importorskip("pyscf")
        pytest.importorskip("ase")
        from quantui.reorganization_energy import run_reorganization_energy

        r = run_reorganization_energy(
            H2, method="RHF", basis="STO-3G", mode="hole", progress_stream=io.StringIO()
        )
        ch = r.channels[0]
        assert ch.s2_neutral_at_neutral is None and ch.s2_neutral_at_ion is None
        assert ch.s2_ion_at_ion == pytest.approx(0.75, abs=1e-8)


def _reorg_result(**s2) -> ReorganizationEnergyResult:
    ch = ReorgChannelResult(
        kind="hole",
        ion_charge=1,
        ion_multiplicity=2,
        e_neutral_at_neutral=-76.4,
        e_ion_at_ion=-76.0,
        e_ion_at_neutral=-75.98,
        e_neutral_at_ion=-76.39,
        lambda1_hartree=0.02,
        lambda2_hartree=0.01,
        lambda_hartree=0.03,
        converged=True,
        **s2,
    )
    return ReorganizationEnergyResult(
        formula="H2O",
        method="B3LYP",
        basis="6-31G*",
        mode="hole",
        molecule=WATER,
        neutral_charge=0,
        neutral_multiplicity=1,
        neutral_energy_hartree=-76.4,
        channels=[ch],
    )


class TestReorgPersistence:
    def test_save_load_and_both_cards(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        res = _reorg_result(s2_ion_at_ion=0.7531, s2_ion_at_neutral=0.91)
        d = save_result(
            res,
            pyscf_log="",
            calc_type="reorganization_energy",
            spectra=res.to_spectra(),
        )
        data = load_result(d)
        ch = data["reorg_channels"][0]
        assert ch["s2_ion_at_ion"] == pytest.approx(0.7531)
        assert ch["s2_ion_at_neutral"] == pytest.approx(0.91)
        assert ch["s2_neutral_at_neutral"] is None
        spectra_ch = data["spectra"]["reorganization_energy"]["channels"][0]
        assert spectra_ch["s2_ion_at_neutral"] == pytest.approx(0.91)

        for card in (format_past_result(data, d), format_reorg_result(res)):
            assert "⟨S²⟩ ion @ R_ion" in card and "0.7531 (ideal 0.7500)" in card
            # 0.91 is 21 % above 0.75: flagged.
            assert card.count("spin contaminated") == 1

    def test_older_saves_render_without_s2_rows(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        d = save_result(
            _reorg_result(), pyscf_log="", calc_type="reorganization_energy"
        )
        data = load_result(d)
        for key in S2_KEYS:
            data["reorg_channels"][0].pop(key)
        card = format_past_result(data, d)
        assert "λ" in card and "⟨S²⟩" not in card

    def test_worker_payload_carries_s2(self):
        from quantui.backends.worker_payload import reorg_result_payload

        payload = reorg_result_payload(_reorg_result(s2_ion_at_ion=0.76))
        assert payload["channels"][0]["s2_ion_at_ion"] == pytest.approx(0.76)
        assert payload["channels"][0]["s2_neutral_at_ion"] is None

    @pytest.mark.parametrize("with_s2", [True, False])
    def test_slurm_ingest(self, tmp_path, monkeypatch, with_s2):
        from quantui.backends.slurm_ingest import ingest_staging_success
        from tests.slurm_ingest_helpers import (
            make_staging_record,
            patch_results_root,
            sample_payload,
        )

        patch_results_root(tmp_path, monkeypatch)
        payload = sample_payload("reorganization_energy")
        if with_s2:
            payload["channels"][0]["s2_ion_at_ion"] = 0.7522
        record, _ = make_staging_record(
            tmp_path, payload, calc_type="reorganization_energy"
        )
        data = load_result(ingest_staging_success(record))
        got = data["reorg_channels"][0]["s2_ion_at_ion"]
        assert got == (pytest.approx(0.7522) if with_s2 else None)


class TestSinglePointCard:
    def _data(self, s2, mult):
        return {
            "calc_type": "single_point",
            "formula": "HO",
            "method": "UHF",
            "basis": "STO-3G",
            "energy_hartree": -74.36,
            "energy_ev": -74.36 * 27.211386,
            "converged": True,
            "spin_square": s2,
            "geometry": {
                "atoms": ["O", "H"],
                "coordinates": [[0, 0, 0], [0, 0, 0.97]],
                "charge": 0,
                "multiplicity": mult,
            },
        }

    def test_open_shell_row(self):
        card = format_past_result(self._data(0.7533, 2))
        assert "⟨S²⟩" in card and "0.7533 (ideal 0.7500)" in card
        assert "spin contaminated" not in card

    def test_contaminated_row_warns(self):
        assert "spin contaminated" in format_past_result(self._data(1.05, 2))

    def test_closed_shell_has_no_row(self):
        assert "⟨S²⟩" not in format_past_result(self._data(None, 1))

    def test_live_card_uses_the_result_multiplicity(self):
        from quantui.app_formatters import _result_extra_rows

        r = SimpleNamespace(spin_square=0.80, multiplicity=2, energy_hartree=-1.0)
        rows = _result_extra_rows(lambda k, d=None: getattr(r, k, d))
        assert "0.8000 (ideal 0.7500)" in rows
