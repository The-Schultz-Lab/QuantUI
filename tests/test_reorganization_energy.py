"""Unit tests for the reorganization-energy (Marcus 4-point) module.

These cover the pure-Python pieces — the λ arithmetic, unit conversions,
spin/method bookkeeping helpers, and the serialisable payload — none of which
require PySCF/ASE, so they run everywhere.  The full SCF-backed
``run_reorganization_energy`` pipeline is exercised by the end-to-end run
smoke test in the calculation tests.
"""

import pytest

from quantui.molecule import Molecule
from quantui.reorganization_energy import (
    HARTREE_TO_KCAL,
    VALID_MODES,
    ReorganizationEnergyResult,
    ReorgChannelResult,
    _ion_multiplicity,
    _promote_method,
    run_reorganization_energy,
)


def _channel(**kw):
    """Build a ReorgChannelResult with sensible defaults for math checks."""
    defaults = dict(
        kind="hole",
        ion_charge=1,
        ion_multiplicity=2,
        e_neutral_at_neutral=-100.0,
        e_ion_at_ion=-99.5,
        e_ion_at_neutral=-99.48,
        e_neutral_at_ion=-99.97,
        lambda1_hartree=0.02,
        lambda2_hartree=0.03,
        lambda_hartree=0.05,
        converged=True,
    )
    defaults.update(kw)
    return ReorgChannelResult(**defaults)


class TestChannelUnitConversions:
    def test_lambda_ev_matches_hartree(self):
        ch = _channel(lambda_hartree=0.05)
        assert ch.lambda_ev == pytest.approx(0.05 * 27.211386245988)

    def test_lambda_mev_is_1000x_ev(self):
        ch = _channel(lambda_hartree=0.05)
        assert ch.lambda_mev == pytest.approx(ch.lambda_ev * 1000.0)

    def test_lambda_kcal_matches_hartree(self):
        ch = _channel(lambda_hartree=0.05)
        assert ch.lambda_kcal == pytest.approx(0.05 * HARTREE_TO_KCAL)

    def test_labels(self):
        assert _channel(kind="hole").label == "Hole (cation)"
        assert _channel(kind="electron").label == "Electron (anion)"


class TestPromoteMethod:
    def test_rhf_promoted_to_uhf_when_open_shell(self):
        assert _promote_method("RHF", 2) == "UHF"
        assert _promote_method("HF", 2) == "UHF"

    def test_rhf_kept_when_closed_shell(self):
        assert _promote_method("RHF", 1) == "RHF"

    def test_dft_left_untouched(self):
        # DFT is auto restricted/unrestricted from spin downstream.
        assert _promote_method("B3LYP", 2) == "B3LYP"
        assert _promote_method("B3LYP", 1) == "B3LYP"


class TestIonMultiplicity:
    def test_cation_of_closed_shell_is_doublet(self):
        # Neutral H2O (10 e-, closed shell) → cation (9 e-) is a doublet.
        h2o = Molecule(["O", "H", "H"], [[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        assert _ion_multiplicity(h2o, ion_charge=1) == 2

    def test_anion_of_closed_shell_is_doublet(self):
        h2o = Molecule(["O", "H", "H"], [[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        assert _ion_multiplicity(h2o, ion_charge=-1) == 2

    def test_ion_of_open_shell_is_singlet(self):
        # A doublet radical (odd e-) → removing/adding one e- gives even → mult 1.
        no = Molecule(["N", "O"], [[0, 0, 0], [0, 0, 1.15]], multiplicity=2)
        assert _ion_multiplicity(no, ion_charge=1) == 1
        assert _ion_multiplicity(no, ion_charge=-1) == 1


class TestResultAggregate:
    def _result(self, channels):
        return ReorganizationEnergyResult(
            formula="H3N",
            method="B3LYP",
            basis="6-31G*",
            mode="both",
            molecule=Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]]),
            neutral_charge=0,
            neutral_multiplicity=1,
            neutral_energy_hartree=-56.5,
            channels=channels,
            converged=True,
            n_total_opt_steps=12,
        )

    def test_energy_hartree_is_neutral_energy(self):
        r = self._result([_channel()])
        assert r.energy_hartree == -56.5
        assert r.energy_ev == pytest.approx(-56.5 * 27.211386245988)

    def test_channel_lookup(self):
        hole = _channel(kind="hole")
        elec = _channel(kind="electron", ion_charge=-1)
        r = self._result([hole, elec])
        assert r.channel("hole") is hole
        assert r.channel("electron") is elec
        assert r.channel("nope") is None

    def test_to_spectra_roundtrips_key_fields(self):
        r = self._result([_channel(lambda_hartree=0.05)])
        payload = r.to_spectra()
        assert "reorganization_energy" in payload
        block = payload["reorganization_energy"]
        assert block["mode"] == "both"
        assert block["channels"][0]["lambda_hartree"] == 0.05
        assert block["channels"][0]["lambda_ev"] == pytest.approx(
            0.05 * 27.211386245988
        )

    def test_summary_mentions_channels(self):
        r = self._result([_channel(kind="hole"), _channel(kind="electron")])
        text = r.summary()
        assert "Reorganization Energy" in text
        assert "Hole (cation)" in text
        assert "Electron (anion)" in text


class TestRunValidation:
    def test_invalid_mode_raises(self):
        h2 = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])
        with pytest.raises(ValueError):
            run_reorganization_energy(h2, mode="banana")

    def test_valid_modes_constant(self):
        assert set(VALID_MODES) == {"hole", "electron", "both"}
