"""Tests for online-search disambiguation (M-STRUCT STRUCT.5).

Platform-independent: network mocked, no PySCF. Exercises the candidate
backend + the app pick-list handlers (no event loop / no real threads).
"""

from unittest.mock import Mock, patch

from quantui import structure_providers
from quantui.app import QuantUIApp
from quantui.pubchem import search_pubchem_candidates

_XYLENE_CIDS = [7237, 7929, 7809]  # o-, m-, p-xylene
_XYLENE_PROPS = {
    "PropertyTable": {
        "Properties": [
            {
                "CID": 7237,
                "MolecularFormula": "C8H10",
                "MolecularWeight": "106.16",
                "Title": "o-Xylene",
            },
            {
                "CID": 7929,
                "MolecularFormula": "C8H10",
                "MolecularWeight": "106.16",
                "Title": "m-Xylene",
            },
            {
                "CID": 7809,
                "MolecularFormula": "C8H10",
                "MolecularWeight": "106.16",
                "Title": "p-Xylene",
            },
        ]
    }
}


# ============================================================================
# Backend: search_pubchem_candidates / search_candidates
# ============================================================================


class TestCandidateBackend:
    @patch("quantui.pubchem._http_get")
    @patch("quantui.pubchem.search_cids_by_name")
    def test_candidates_built_from_batch_props(self, mock_cids, mock_http):
        mock_cids.return_value = _XYLENE_CIDS
        mock_http.return_value = Mock(
            status_code=200,
            json=Mock(return_value=_XYLENE_PROPS),
            raise_for_status=Mock(),
        )
        cands = search_pubchem_candidates("xylene")
        assert [c["cid"] for c in cands] == _XYLENE_CIDS  # search order preserved
        assert cands[0]["title"] == "o-Xylene"
        assert cands[0]["formula"] == "C8H10"
        assert abs(cands[0]["mw"] - 106.16) < 0.01

    @patch("quantui.pubchem.search_cids_by_name")
    def test_candidates_empty_when_no_cids(self, mock_cids):
        mock_cids.return_value = []
        assert search_pubchem_candidates("zzznotreal") == []

    def test_search_candidates_skips_non_name_types(self):
        # SMILES / CID resolve to a single structure → no disambiguation.
        assert structure_providers.search_candidates("CC(=O)O") == []
        assert structure_providers.search_candidates("962") == []

    @patch("quantui.structure_providers.search_pubchem_candidates")
    def test_search_candidates_swallows_network_error(self, mock_cands):
        from quantui.pubchem import PubChemAPIError

        mock_cands.side_effect = PubChemAPIError("offline")
        # Falls back to [] so the caller uses the full single-result chain.
        assert structure_providers.search_candidates("xylene") == []

    @patch("quantui.structure_providers.search_pubchem_candidates")
    def test_search_candidates_returns_list_for_name(self, mock_cands):
        mock_cands.return_value = [
            {"cid": 1, "title": "x", "formula": "CH4", "mw": 16.0}
        ]
        out = structure_providers.search_candidates("methane")
        assert len(out) == 1


# ============================================================================
# App pick-list handlers
# ============================================================================


class TestPickListHandlers:
    def test_candidates_dropdown_hidden_initially(self):
        app = QuantUIApp()
        assert app.pubchem_candidates_dd.layout.display == "none"

    def test_show_candidates_populates_and_reveals(self):
        app = QuantUIApp()
        cands = [
            {"cid": 7237, "title": "o-Xylene", "formula": "C8H10", "mw": 106.16},
            {"cid": 7929, "title": "m-Xylene", "formula": "C8H10", "mw": 106.16},
        ]
        app._show_pubchem_candidates("xylene", cands)
        values = [v for _, v in app.pubchem_candidates_dd.options]
        assert values == ["", "7237", "7929"]  # placeholder + 2 cids
        assert app.pubchem_candidates_dd.layout.display == ""
        assert "2 matches" in app.pubchem_msg.value
        # Showing the list must not auto-load a molecule.
        assert app._molecule is None

    def test_hide_candidates_resets(self):
        app = QuantUIApp()
        app._show_pubchem_candidates(
            "xylene",
            [
                {"cid": 1, "title": "a", "formula": "C8H10", "mw": 1.0},
                {"cid": 2, "title": "b", "formula": "C8H10", "mw": 1.0},
            ],
        )
        app._hide_pubchem_candidates()
        assert app.pubchem_candidates_dd.layout.display == "none"
        assert app.pubchem_candidates_dd.value == ""

    def test_candidate_select_placeholder_is_noop(self):
        app = QuantUIApp()
        app._on_pubchem_candidate_selected({"new": ""})
        assert app._molecule is None

    def test_candidate_select_ignored_during_refresh(self):
        app = QuantUIApp()
        app._pubchem_cand_refreshing = True
        # Should return before touching the event loop / spawning a thread.
        app._on_pubchem_candidate_selected({"new": "7237"})
        assert app._molecule is None
