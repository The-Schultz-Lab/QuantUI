"""
Direct unit tests for quantui.backends.worker_payload's *_result_payload()
functions.

No test imported this module directly before (only exercised indirectly
through worker.py end-to-end tests) — exactly the gap that let ISSUE.10
(session_result_payload silently dropping Mulliken charges + dipole moment)
go unnoticed: "a regression test asserting these keys are non-None for a
plain CPU single-point would have caught this immediately" (GOTCHAS.md).
"""

from __future__ import annotations

from types import SimpleNamespace

from quantui.backends.worker_payload import (
    freq_result_payload,
    nmr_result_payload,
    session_result_payload,
    tddft_result_payload,
)
from quantui.molecule import Molecule


def _session_result(**overrides) -> SimpleNamespace:
    defaults = dict(
        energy_hartree=-1608.701471,
        homo_lumo_gap_ev=3.2,
        converged=True,
        n_iterations=42,
        method="B3LYP",
        basis="def2-SVP",
        formula="Mn(H2O)6",
        mulliken_charges=[0.82, -0.41, -0.41, -0.41, -0.41, -0.41, -0.41],
        dipole_moment_debye=1.85,
        dipole_vector_debye=[0.1, -0.2, 1.8],
        atom_symbols=["Mn", "O", "O", "O", "O", "O", "O"],
        scf_rescue_stage="bootstrap",
        scf_variant="UKS",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestSessionResultPayload:
    """M-ISSUES ISSUE.10 — Mulliken charges + dipole moment must survive the
    batch result.json round trip; they used to be computed by
    session_calc.run_in_session() and then silently dropped here."""

    def test_mulliken_charges_present(self):
        payload = session_result_payload(_session_result())
        assert payload["mulliken_charges"] == [
            0.82,
            -0.41,
            -0.41,
            -0.41,
            -0.41,
            -0.41,
            -0.41,
        ]

    def test_dipole_moment_present(self):
        payload = session_result_payload(_session_result())
        assert payload["dipole_moment_debye"] == 1.85
        assert payload["dipole_vector_debye"] == [0.1, -0.2, 1.8]

    def test_atom_symbols_present_for_interpreting_charges(self):
        payload = session_result_payload(_session_result())
        assert payload["atom_symbols"][0] == "Mn"

    def test_none_when_gpu_offload_path_never_computed_them(self):
        """mf.mulliken_pop is NotImplemented on the GPU-offload path —
        session_calc.py leaves these None on SessionResult in that case;
        the payload must pass None through, not raise or substitute 0."""
        payload = session_result_payload(
            _session_result(
                mulliken_charges=None,
                dipole_moment_debye=None,
                dipole_vector_debye=None,
            )
        )
        assert payload["mulliken_charges"] is None
        assert payload["dipole_moment_debye"] is None
        assert payload["dipole_vector_debye"] is None

    def test_missing_from_an_older_sessionresult_defaults_to_none(self):
        """A SessionResult predating these fields (or ISSUE.10/SCFR/UXP2.10)
        must serialize cleanly via the getattr guard, not raise."""
        minimal = SimpleNamespace(
            energy_hartree=-76.0,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=8,
            method="RHF",
            basis="STO-3G",
            formula="H2O",
        )
        payload = session_result_payload(minimal)
        assert payload["mulliken_charges"] is None
        assert payload["dipole_moment_debye"] is None
        assert payload["dipole_vector_debye"] is None
        assert payload["atom_symbols"] is None
        assert payload["scf_rescue_stage"] == "none"
        assert payload["scf_variant"] is None

    def test_scf_rescue_stage_present(self):
        payload = session_result_payload(_session_result())
        assert payload["scf_rescue_stage"] == "bootstrap"

    def test_scf_variant_present(self):
        payload = session_result_payload(_session_result())
        assert payload["scf_variant"] == "UKS"

    def test_calc_type_and_core_fields_unchanged(self):
        payload = session_result_payload(_session_result())
        assert payload["calc_type"] == "single_point"
        assert payload["energy_hartree"] == -1608.701471
        assert payload["converged"] is True
        assert payload["formula"] == "Mn(H2O)6"


class TestFreqTddftNmrResultPayloadScfVariant:
    """M-UX2 UXP2.10 — the same provenance field promoted to the other
    calc types' batch payloads."""

    def test_freq_result_payload_carries_scf_variant(self):
        result = SimpleNamespace(
            energy_hartree=-1600.0,
            homo_lumo_gap_ev=None,
            converged=True,
            n_iterations=30,
            method="B3LYP",
            basis="def2-SVP",
            formula="Fe(H2O)6",
            displacements=None,
            frequencies_cm1=[],
            ir_intensities=[],
            raman_activities=[],
            zpve_hartree=0.0,
            thermo=None,
            scf_variant="UKS",
        )
        molecule = Molecule(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        payload = freq_result_payload(result, molecule)
        assert payload["scf_variant"] == "UKS"

    def test_tddft_result_payload_carries_scf_variant(self):
        result = SimpleNamespace(
            energy_hartree=-1600.0,
            homo_lumo_gap_ev=None,
            converged=True,
            n_iterations=20,
            method="B3LYP",
            basis="def2-SVP",
            formula="Co(H2O)6",
            excitation_energies_ev=[],
            oscillator_strengths=[],
            wavelengths_nm=lambda: [],
        )
        payload = tddft_result_payload(result)
        # tddft_calc.TDDFTResult sets scf_variant; a bare SimpleNamespace
        # without it must still serialize (None), not raise.
        assert payload["scf_variant"] is None

    def test_nmr_result_payload_carries_scf_variant(self):
        result = SimpleNamespace(
            converged=True,
            method="B3LYP",
            basis="6-31G*",
            formula="CH4",
            atom_symbols=["C", "H", "H", "H", "H"],
            shielding_iso_ppm=[],
            chemical_shifts_ppm={},
            reference_compound="TMS",
            reference_key="B3LYP/6-31G*",
            is_fallback_reference=False,
            scf_variant="RKS",
        )
        payload = nmr_result_payload(result)
        assert payload["scf_variant"] == "RKS"
