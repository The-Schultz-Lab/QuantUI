"""
M-BATCH2 B2.8 — chemistry constants with one source.

The electron count and the solvent-capable calc types used to be written out
in several modules, which drifted (the SLURM estimate once counted Mn/Co/Ni/Mo
as 0 electrons).
"""

from __future__ import annotations

import pytest

from quantui.config import electron_count


@pytest.mark.parametrize(
    "atoms,charge,expected",
    [
        (["O", "H", "H"], 0, 10),
        (["O", "H", "H"], 1, 9),
        (["Mn", "O", "O", "O", "O"], -1, 58),  # MnO4-
        (["cl", "CL", "Cl"], 0, 51),  # symbols are title-cased
        (["Xx"], 0, 0),  # unknown symbols count 0
    ],
)
def test_electron_count(atoms, charge, expected):
    assert electron_count(atoms, charge) == expected


def test_every_electron_count_agrees():
    from quantui.backends.base import CalculationRequest
    from quantui.backends.slurm_utils import estimate_slurm_resources
    from quantui.molecule import Molecule, suggest_multiplicity
    from quantui.xyz_input import electron_count as xyz_count

    atoms, charge = ["Mn", "O", "O", "O", "O"], -1
    mol = Molecule(
        atoms,
        [[0, 0, 0], [0, 0, 1.6], [0, 1.6, 0], [1.6, 0, 0], [-1, -1, -1]],
        charge=charge,
    )
    assert mol.get_electron_count() == xyz_count(atoms, charge) == 58
    assert suggest_multiplicity(atoms, charge) == 1
    # The estimate runs (it used to see 0 electrons for the metal).
    req = CalculationRequest(
        request_id="r",
        calc_type="single_point",
        method="B3LYP",
        basis="def2-SVP",
        charge=charge,
        multiplicity=1,
        molecule={"atoms": atoms, "coords": mol.coordinates},
    )
    assert estimate_slurm_resources(req)["memory_gb"] >= 1


def test_app_solvent_types_follow_the_batch_set():
    from quantui.app_runflow import _CALC_TYPE_KEYS, _SOLVENT_SUPPORTED_CALC_TYPES
    from quantui.backends.batch_input import SOLVENT_CALC_TYPES

    keys = {_CALC_TYPE_KEYS[label] for label in _SOLVENT_SUPPORTED_CALC_TYPES}
    assert keys == set(SOLVENT_CALC_TYPES) | {"transition_state"}
    assert "NMR Shielding" not in _SOLVENT_SUPPORTED_CALC_TYPES
    assert "PES Scan" not in _SOLVENT_SUPPORTED_CALC_TYPES
