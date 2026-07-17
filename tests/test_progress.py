"""
Tests for quantui.progress

Validates the StepProgress widget's state machine and HTML rendering.
"""

from quantui.progress import StepProgress


class TestStepProgressInit:
    """Tests for StepProgress construction."""

    def test_creates_with_labels(self):
        sp = StepProgress(["A", "B", "C"])
        assert sp.widget is not None

    def test_all_steps_start_pending(self):
        sp = StepProgress(["A", "B"])
        html = sp.widget.value
        # Both should show the pending icon (⬜)
        assert html.count("⬜") == 2

    def test_widget_is_html(self):
        import ipywidgets as widgets

        sp = StepProgress(["Step one"])
        assert isinstance(sp.widget, widgets.HTML)

    def test_labels_appear_in_html(self):
        sp = StepProgress(["Parse input", "Validate atoms"])
        html = sp.widget.value
        assert "Parse input" in html
        assert "Validate atoms" in html

    def test_step_numbers_sequential(self):
        sp = StepProgress(["A", "B", "C"])
        html = sp.widget.value
        assert "Step 1:" in html
        assert "Step 2:" in html
        assert "Step 3:" in html


class TestStepProgressTransitions:
    """Tests for state transitions."""

    def test_start_shows_active_icon(self):
        sp = StepProgress(["A", "B"])
        sp.start(0)
        html = sp.widget.value
        assert "⏳" in html
        # Step B still pending
        assert "⬜" in html

    def test_complete_shows_done_icon(self):
        sp = StepProgress(["A", "B"])
        sp.start(0)
        sp.complete(0)
        html = sp.widget.value
        assert "✅" in html

    def test_fail_shows_fail_icon(self):
        sp = StepProgress(["A", "B"])
        sp.start(0)
        sp.fail(0)
        html = sp.widget.value
        assert "❌" in html

    def test_complete_with_message(self):
        sp = StepProgress(["A"])
        sp.complete(0, "Found 3 atoms")
        html = sp.widget.value
        assert "Found 3 atoms" in html

    def test_fail_with_message(self):
        sp = StepProgress(["A"])
        sp.fail(0, "Invalid element")
        html = sp.widget.value
        assert "Invalid element" in html

    def test_start_clears_previous_message(self):
        sp = StepProgress(["A"])
        sp.fail(0, "old error")
        sp.start(0)
        html = sp.widget.value
        assert "old error" not in html

    def test_multi_step_workflow(self):
        sp = StepProgress(["Parse", "Validate", "Display"])
        sp.start(0)
        sp.complete(0)
        sp.start(1)
        sp.complete(1, "OK")
        sp.start(2)
        html = sp.widget.value
        assert html.count("✅") == 2
        assert html.count("⏳") == 1

    def test_reset_returns_all_to_pending(self):
        sp = StepProgress(["A", "B"])
        sp.start(0)
        sp.complete(0)
        sp.start(1)
        sp.reset()
        html = sp.widget.value
        assert html.count("⬜") == 2
        assert "✅" not in html
        assert "⏳" not in html


class TestStepProgressEdgeCases:
    """Edge case tests."""

    def test_single_step(self):
        sp = StepProgress(["Only step"])
        sp.start(0)
        sp.complete(0)
        assert "✅" in sp.widget.value

    def test_empty_message_ok(self):
        sp = StepProgress(["A"])
        sp.complete(0, "")
        # Empty message should not crash
        assert "✅" in sp.widget.value

    def test_html_special_chars_in_message(self):
        sp = StepProgress(["A"])
        sp.complete(0, "Found <3 atoms & 2 bonds")
        # L audit fix: labels/messages must be HTML-escaped, not embedded
        # as-is — a message containing "<"/">" (e.g. echoing a parse error
        # back from user input) used to break the widget's markup instead
        # of rendering as literal text.
        assert "Found &lt;3 atoms &amp; 2 bonds" in sp.widget.value
        assert "<3 atoms" not in sp.widget.value

    def test_html_escapes_script_tag_in_label_and_message(self):
        sp = StepProgress(["Parse <script>alert(1)</script>"])
        sp.fail(0, "<img src=x onerror=alert(2)>")
        value = sp.widget.value
        assert "<script>" not in value
        assert "<img" not in value
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in value
        assert "&lt;img src=x onerror=alert(2)&gt;" in value
