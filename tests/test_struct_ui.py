"""Tests for the library browse/search UI (M-STRUCT STRUCT.9).

Instantiates QuantUIApp() headlessly (no .display()) and drives the library
widgets + handlers directly. Platform-independent; no PySCF.
"""

import ipywidgets as widgets

from quantui import molecule_library as ml
from quantui.app import QuantUIApp
from quantui.app_builders import library_result_options

# ============================================================================
# library_result_options helper
# ============================================================================


class TestLibraryResultOptions:
    def test_leads_with_placeholder(self):
        opts, _ = library_result_options()
        assert opts[0][1] == ""  # placeholder value is empty

    def test_category_filter(self):
        opts, _ = library_result_options(category="amino-acid")
        ids = [v for _, v in opts if v]
        assert len(ids) == 20

    def test_search_by_formula(self):
        opts, _ = library_result_options(query="C6H6")
        ids = [v for _, v in opts if v]
        assert "C6H6" in ids

    def test_note_reports_count(self):
        _, note = library_result_options(category="aromatic")
        assert "1 match" in note

    def test_note_reports_truncation(self):
        _, note = library_result_options()  # all categories → > 200
        assert "first 200" in note


# ============================================================================
# Widgets exist + are wired
# ============================================================================


class TestLibraryWidgets:
    def test_library_widgets_exist(self):
        app = QuantUIApp()
        assert isinstance(app.lib_category_dd, widgets.Dropdown)
        assert isinstance(app.lib_search_txt, widgets.Text)
        assert isinstance(app.lib_results_dd, widgets.Dropdown)
        assert isinstance(app.lib_count_lbl, widgets.HTML)

    def test_category_dropdown_lists_categories(self):
        app = QuantUIApp()
        values = [v for _, v in app.lib_category_dd.options]
        assert "" in values  # "All categories"
        for cat in ml.categories():
            assert cat in values

    def test_search_text_is_not_continuous(self):
        app = QuantUIApp()
        assert app.lib_search_txt.continuous_update is False


# ============================================================================
# Handlers
# ============================================================================


class TestLibraryHandlers:
    def test_select_loads_molecule(self):
        app = QuantUIApp()
        assert app._molecule is None
        app._on_lib_select({"new": "aspirin"})
        assert app._molecule is not None
        assert app._molecule.get_formula() == "C9H8O4"

    def test_select_placeholder_is_noop(self):
        app = QuantUIApp()
        app._on_lib_select({"new": ""})
        assert app._molecule is None

    def test_select_ignored_during_refresh(self):
        app = QuantUIApp()
        app._lib_refreshing = True
        app._on_lib_select({"new": "aspirin"})
        assert app._molecule is None

    def test_filter_change_repopulates_results(self):
        app = QuantUIApp()
        app.lib_category_dd.value = "amino-acid"
        app._on_lib_filter_changed({"new": "amino-acid"})
        ids = [v for _, v in app.lib_results_dd.options if v]
        assert len(ids) == 20
        assert app.lib_results_dd.value == ""  # reset to placeholder, no load
        assert app._molecule is None

    def test_search_then_select_bulk_entry(self):
        app = QuantUIApp()
        # Pick any bulk id from the store and confirm it loads via the handler.
        bulk = ml.search("", category="bulk-qm9", limit=1)
        if not bulk:
            return  # bulk not built in this checkout
        bulk_id = bulk[0]["id"]
        app._on_lib_select({"new": bulk_id})
        assert app._molecule is not None
        assert len(app._molecule.atoms) >= 1

    def test_refresh_updates_count_label(self):
        app = QuantUIApp()
        app.lib_category_dd.value = "aromatic"
        app._refresh_lib_results()
        assert "1 match" in app.lib_count_lbl.value
