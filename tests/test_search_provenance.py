"""Search-provenance, activity-indicator, and backend-persistence tests.

Covers the offline-manual findings of 2026-06-15:

- **#1b** the loaded-molecule card is labeled by where the structure ACTUALLY
  came from (PubChem / CACTUS / SMILES / Library / Library-offline), not always
  "PubChem", and an offline FALLBACK surfaces a no-network note.
- **#2** the toolbar activity indicator lights during the search resolver chain
  and is balanced (ends at every terminal point).
- **#4a** a backend choice made via any toggle persists across sessions.

Platform-independent: the resolver is mocked; no network, no PySCF.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from quantui.molecule import Molecule


@pytest.fixture
def app(tmp_path, monkeypatch):
    # Isolate settings per test (critical under pytest-xdist — workers must not
    # collide on ~/.quantui/settings.json).
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    from quantui.app import QuantUIApp

    return QuantUIApp()


@pytest.fixture
def water():
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )


class TestSourceLabel:
    """#1b — label by actual provenance, warn on offline fallback."""

    def test_offline_fallback_warns_no_network(self, app, water):
        app._apply_pubchem_search_result(
            "aspirin", water, None, "library-offline-fallback"
        )
        assert "No network" in app.pubchem_msg.value
        assert "bundled library" in app.pubchem_msg.value

    def test_offline_fallback_molecule_label(self, app, water):
        with patch.object(app, "_set_molecule") as m:
            app._apply_pubchem_search_result(
                "aspirin", water, None, "library-offline-fallback"
            )
        assert m.call_args[0][1] == "Library (offline): aspirin"

    def test_pubchem_source_label_no_warning(self, app, water):
        with patch.object(app, "_set_molecule") as m:
            app._apply_pubchem_search_result("benzene", water, None, "pubchem")
        assert m.call_args[0][1] == "PubChem: benzene"
        assert "from PubChem" in app.pubchem_msg.value
        assert "No network" not in app.pubchem_msg.value

    def test_smiles_source_label(self, app, water):
        with patch.object(app, "_set_molecule") as m:
            app._apply_pubchem_search_result("CCO", water, None, "rdkit-smiles")
        assert m.call_args[0][1] == "SMILES: CCO"

    def test_cactus_source_label(self, app, water):
        with patch.object(app, "_set_molecule") as m:
            app._apply_pubchem_search_result("taxol", water, None, "cactus")
        assert m.call_args[0][1] == "NCI CACTUS: taxol"


class TestSearchActivityIndicator:
    """#2 — the activity light is balanced across the search lifecycle."""

    def test_resolve_terminal_ends_activity(self, app, water):
        app._activity_begin("searching", kind="ui")
        assert app._activity_count == 1
        app._apply_pubchem_search_result("benzene", water, None, "pubchem")
        assert app._activity_count == 0

    def test_error_terminal_ends_activity(self, app):
        app._activity_begin("searching", kind="ui")
        app._apply_pubchem_search_result("zzz", None, ValueError("not found"), None)
        assert app._activity_count == 0

    def test_candidate_picklist_terminal_ends_activity(self, app):
        app._activity_begin("searching", kind="ui")
        app._show_pubchem_candidates(
            "xylene",
            [
                {"title": "o-xylene", "formula": "C8H10", "mw": 106.16, "cid": 7237},
                {"title": "m-xylene", "formula": "C8H10", "mw": 106.16, "cid": 7929},
            ],
        )
        assert app._activity_count == 0


class TestResolveStructureWithMessage:
    """#1b backend — resolve_structure_with_message reports provenance."""

    def test_returns_source_and_offline_flag(self):
        from quantui import structure_providers as sp

        fake = sp.ResolvedStructure(
            xyz="1\n\nH 0 0 0",
            source="library-offline-fallback",
            formula="H2O",
            num_atoms=3,
            num_heavy_atoms=1,
        )
        with patch.object(sp, "resolve_structure", return_value=fake):
            xyz, msg, source, is_offline = sp.resolve_structure_with_message("aspirin")
        assert xyz is not None
        assert source == "library-offline-fallback"
        assert is_offline is True

    def test_failure_returns_none_source(self):
        from quantui import structure_providers as sp

        with patch.object(sp, "resolve_structure", side_effect=ValueError("nope")):
            xyz, msg, source, is_offline = sp.resolve_structure_with_message("zzqx")
        assert xyz is None
        assert source is None
        assert is_offline is False
        assert "Could not resolve" in msg

    def test_student_friendly_resolve_unchanged_two_tuple(self):
        """The public 2-tuple API is preserved (delegates to the new fn)."""
        from quantui import structure_providers as sp

        fake = sp.ResolvedStructure(xyz="1\n\nH 0 0 0", source="pubchem", formula="H2O")
        with patch.object(sp, "resolve_structure", return_value=fake):
            result = sp.student_friendly_resolve("water")
        assert isinstance(result, tuple) and len(result) == 2


class TestBackendPreferencePersists:
    """#4a — a backend choice via any toggle survives the session."""

    def _app_with_pref(self, tmp_path, monkeypatch, initial):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"_schema_version": 1, "viz": {"default_backend": initial}})
        )
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(path))
        from quantui.app import QuantUIApp

        return QuantUIApp(), path

    def test_calculate_tab_toggle_persists(self, tmp_path, monkeypatch):
        a, path = self._app_with_pref(tmp_path, monkeypatch, "auto")
        if not (a._viz_availability.py3dmol and a._viz_availability.plotlymol):
            pytest.skip("both backends needed")
        a._on_viz_backend_changed({"new": "plotlymol"})
        assert a._viz_backend_preference == "plotlymol"
        assert json.loads(path.read_text())["viz"]["default_backend"] == "plotlymol"

    def test_analysis_tab_toggle_persists(self, tmp_path, monkeypatch):
        a, path = self._app_with_pref(tmp_path, monkeypatch, "auto")
        if not (a._viz_availability.py3dmol and a._viz_availability.plotlymol):
            pytest.skip("both backends needed")
        a._on_viz_backend_changed_ana({"new": "py3dmol"})
        assert a._viz_backend_preference == "py3dmol"
        assert json.loads(path.read_text())["viz"]["default_backend"] == "py3dmol"
