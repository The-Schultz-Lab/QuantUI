"""Structure search keeps the resolver's charge and multiplicity (audit DATA-1).

The search handler used to rebuild the molecule from the XYZ text alone, so
ions and radicals loaded as neutral singlets (even electron counts) or failed
as "Not found" (odd electron counts).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("rdkit")

import requests  # noqa: E402

from quantui.app import QuantUIApp  # noqa: E402
from quantui.structure_providers import (  # noqa: E402
    ResolvedStructure,
    molecule_from_resolved,
    resolve_structure_with_details,
)


class _ImmediateLoop:
    def call_soon_threadsafe(self, fn, *args):
        fn(*args)


@pytest.fixture(autouse=True)
def _offline():
    with patch("requests.get", side_effect=requests.ConnectionError("offline")):
        yield


@pytest.mark.parametrize(
    "query, charge, mult",
    [
        ("[NH4+]", 1, 1),
        ("[OH-]", -1, 1),
        ("[O-]S(=O)(=O)[O-]", -2, 1),
        ("[CH3]", 0, 2),
        ("O2", 0, 3),  # bundled library entry: triplet oxygen
    ],
)
def test_search_loads_the_resolved_charge_and_multiplicity(query, charge, mult):
    app = QuantUIApp()
    app._resolve_and_apply(query, _ImmediateLoop())
    mol = app._molecule
    assert mol is not None, app.pubchem_msg.value
    assert (mol.charge, mol.multiplicity) == (charge, mult)
    assert (app.charge_si.value, app.mult_si.value) == (charge, mult)


def test_resolved_message_states_a_non_default_charge():
    result, message = resolve_structure_with_details("[NH4+]")
    assert result is not None and result.charge == 1
    assert "Charge +1" in message


def test_impossible_resolver_spin_falls_back_to_a_consistent_one():
    # A resolver that knows the charge but reports no radical: NH4 neutral
    # has 11 electrons, so multiplicity 1 is impossible -> doublet.
    xyz = (
        "5\n\nN 0 0 0\nH 0.6 0.6 0.6\nH -0.6 -0.6 0.6\nH -0.6 0.6 -0.6\nH 0.6 -0.6 -0.6"
    )
    mol = molecule_from_resolved(
        ResolvedStructure(xyz=xyz, source="pubchem", charge=0, multiplicity=1)
    )
    assert (mol.charge, mol.multiplicity) == (0, 2)
