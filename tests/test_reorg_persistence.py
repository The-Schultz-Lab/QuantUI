"""Reorganization-energy results survive History (M-REORG REORG.1/.2/.4/.6).

Reported by a student, 2026-08-05: *"after reorganization energy calcs are
reloaded from history, the actual reorg energy outputs and stuff aren't
displayed in the results tab like they are prior to loading from history."*

It presented as a display bug and was not one. ``save_result`` wrote a fixed
schema of scalar fields with no ``channels`` key, so the λ breakdown was never
persisted — the live card rendered from an object still in memory, and the
History card had nothing behind it. Fixing the formatter alone could not have
worked.

The structural fix is that **both cards now render from the same plain-dict
payload**, so "renders live" implies "renders after reload" rather than
depending on two implementations agreeing. These tests exercise the actual
save → load → render path rather than inspecting either formatter, because
inspecting a formatter is exactly what would have missed this.

Platform-independent: no PySCF, no browser.
"""

from __future__ import annotations

import pytest

from quantui.app_formatters import format_past_result, format_reorg_result
from quantui.molecule import Molecule
from quantui.reorganization_energy import (
    ReorganizationEnergyResult,
    ReorgChannelResult,
    geometry_rmsd,
    max_atom_displacement,
)
from quantui.results_storage import load_result, save_result


@pytest.fixture
def neutral() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )


@pytest.fixture
def cation() -> Molecule:
    # Deliberately displaced from the neutral: the relaxation is the physical
    # content of λ, so a fixture where nothing moved would test nothing.
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [1.03, 0.0, 0.0], [-0.27, 1.00, 0.0]],
        charge=1,
        multiplicity=2,
    )


@pytest.fixture
def result(neutral, cation) -> ReorganizationEnergyResult:
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
        ion_molecule=cation,
    )
    return ReorganizationEnergyResult(
        formula="H2O",
        method="B3LYP",
        basis="6-31G*",
        mode="hole",
        molecule=neutral,
        neutral_charge=0,
        neutral_multiplicity=1,
        neutral_energy_hartree=-76.4,
        channels=[ch],
    )


@pytest.fixture
def saved(result, tmp_path, monkeypatch):
    """A genuinely round-tripped result: saved to disk, then loaded back."""
    monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
    d = save_result(result, pyscf_log="", calc_type="reorganization_energy")
    return d, load_result(d)


class TestTheChannelDataSurvivesHistory:
    """The reported bug, tested through the real path."""

    def test_channels_are_persisted_at_all(self, saved):
        _, data = saved
        assert data.get("reorg_channels"), (
            "the λ payload is not being written — this is the bug, and no "
            "amount of formatter work can compensate"
        )

    def test_every_energy_of_the_four_point_scheme_round_trips(self, saved, result):
        _, data = saved
        got = data["reorg_channels"][0]
        src = result.channels[0]
        for field in (
            "e_neutral_at_neutral",
            "e_ion_at_ion",
            "e_ion_at_neutral",
            "e_neutral_at_ion",
            "lambda1_hartree",
            "lambda2_hartree",
            "lambda_hartree",
        ):
            assert got[field] == pytest.approx(getattr(src, field)), field

    def test_the_history_card_shows_lambda(self, saved):
        d, data = saved
        html = format_past_result(data, d)
        assert "λ" in html
        assert "0.8163" in html or "eV" in html

    def test_the_history_card_does_not_claim_data_is_missing(self, saved):
        d, data = saved
        assert "Re-run this calculation" not in format_past_result(data, d)

    def test_both_cards_render_the_same_channel_content(self, saved, result):
        """The structural guarantee. Two renderers drifting IS the bug."""
        d, data = saved
        live = format_reorg_result(result)
        history = format_past_result(data, d)
        for marker in ("λ₁ ion relaxation", "λ₂ neutral relaxation", "Geometry RMSD"):
            assert marker in live, f"live card lost {marker}"
            assert marker in history, f"history card lost {marker}"


class TestGeometriesAreRetained:
    def test_the_ion_geometry_is_saved(self, saved, cation):
        _, data = saved
        geom = data["reorg_channels"][0]["ion_geometry"]
        assert geom["atoms"] == list(cation.atoms)
        assert geom["coordinates"][1] == pytest.approx(list(cation.coordinates[1]))

    def test_the_neutral_geometry_travels_with_the_payload(self, saved):
        # The top-level schema stores atom_symbols but not coordinates, so
        # without this the history card cannot reach R_neutral — and with no
        # R_neutral there is no relaxation to compute.
        _, data = saved
        assert data["reorg_channels"][0].get("neutral_geometry")


class TestRelaxationMetrics:
    def test_rmsd_is_positive_when_the_ion_relaxed(self, neutral, cation):
        assert geometry_rmsd(neutral, cation) > 0

    def test_rmsd_is_zero_for_identical_geometries(self, neutral):
        assert geometry_rmsd(neutral, neutral) == pytest.approx(0.0)

    def test_mismatched_geometries_return_none_rather_than_raising(self, neutral):
        other = Molecule(atoms=["H", "H"], coordinates=[[0, 0, 0], [0, 0, 0.74]])
        assert geometry_rmsd(neutral, other) is None
        assert max_atom_displacement(neutral, other) is None

    def test_the_largest_shift_identifies_the_right_atom(self, neutral, cation):
        # RMSD averages the relaxation away; a single atom moving far in an
        # otherwise rigid molecule is the interesting case.
        idx, dist = max_atom_displacement(neutral, cation)
        assert idx == 2  # the O-H that moved furthest in the fixture
        assert dist > geometry_rmsd(neutral, cation)

    def test_relaxation_appears_on_the_card(self, saved):
        d, data = saved
        html = format_past_result(data, d)
        assert "Geometry RMSD" in html
        assert "Largest atom shift" in html


class TestOldResultsSayWhatToDo:
    """Requested explicitly: results saved before λ persistence cannot be
    recovered — λ is two optimizations and four SCF energies — so the card must
    say so and name the remedy rather than rendering an empty section."""

    def test_a_result_without_channels_gets_an_explanation(self, saved):
        d, data = saved
        stripped = {k: v for k, v in data.items() if k != "reorg_channels"}
        html = format_past_result(stripped, d)
        assert "Re-run this calculation" in html
        assert "not saved" in html

    def test_detection_is_by_absence_not_by_version(self, saved):
        # A version or timestamp cutoff would misfire on anything re-saved or
        # imported from elsewhere. The payload's presence is the ground truth.
        d, data = saved
        stripped = dict(data)
        stripped["reorg_channels"] = None
        stripped["_schema_version"] = 999
        assert "Re-run this calculation" in format_past_result(stripped, d)

    def test_other_calc_types_never_see_the_notice(self, saved):
        d, data = saved
        other = dict(data)
        other["calc_type"] = "single_point"
        other.pop("reorg_channels", None)
        assert "Re-run this calculation" not in format_past_result(other, d)
