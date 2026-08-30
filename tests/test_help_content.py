"""
Tests for quantui.help_content

Validates the help topic bank structure and the help_panel() widget builder.
"""

import pytest

from quantui.help_content import (
    HELP_TOPICS,
    VALID_TOPICS,
    help_panel,
)


class TestHelpTopics:
    """Validate the HELP_TOPICS dictionary structure."""

    def test_is_dict(self):
        assert isinstance(HELP_TOPICS, dict)

    def test_non_empty(self):
        assert len(HELP_TOPICS) >= 4

    def test_expected_keys_present(self):
        for key in ("charge", "multiplicity", "method", "basis_set"):
            assert key in HELP_TOPICS, f"Missing expected topic: {key}"

    def test_each_entry_has_title_and_body(self):
        for key, entry in HELP_TOPICS.items():
            assert "title" in entry, f"Topic '{key}' missing 'title'"
            assert "body" in entry, f"Topic '{key}' missing 'body'"

    def test_titles_are_nonempty_strings(self):
        for key, entry in HELP_TOPICS.items():
            assert (
                isinstance(entry["title"], str) and entry["title"]
            ), f"Topic '{key}': title must be non-empty string"

    def test_bodies_are_nonempty_strings(self):
        for key, entry in HELP_TOPICS.items():
            assert (
                isinstance(entry["body"], str) and entry["body"]
            ), f"Topic '{key}': body must be non-empty string"

    def test_valid_topics_matches_keys(self):
        assert VALID_TOPICS == frozenset(HELP_TOPICS.keys())


class TestHelpPanel:
    """Tests for the help_panel() widget builder."""

    def test_returns_html_widget(self):
        import ipywidgets as widgets

        panel = help_panel("charge")
        assert isinstance(panel, widgets.HTML)

    def test_contains_details_element(self):
        panel = help_panel("charge")
        assert "<details" in panel.value
        assert "</details>" in panel.value

    def test_contains_topic_title(self):
        panel = help_panel("charge")
        assert HELP_TOPICS["charge"]["title"] in panel.value

    def test_all_topics_produce_panels(self):
        for topic in VALID_TOPICS:
            panel = help_panel(topic)
            assert panel.value  # non-empty HTML

    def test_unknown_topic_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown help topic"):
            help_panel("nonexistent_topic")

    def test_method_panel_mentions_rhf_uhf(self):
        panel = help_panel("method")
        assert "RHF" in panel.value
        assert "UHF" in panel.value

    def test_basis_set_panel_mentions_sto3g(self):
        panel = help_panel("basis_set")
        assert "STO-3G" in panel.value

    def test_run_progress_panel_covers_status_and_heartbeat(self):
        panel = help_panel("run_progress")
        assert "status" in panel.value.lower()
        assert (
            "heartbeat" in panel.value.lower() or "still working" in panel.value.lower()
        )
        assert "TD-DFT" in panel.value
