"""
Visual progress indicators for multi-step notebook operations.

Provides a lightweight ``StepProgress`` widget that displays numbered
steps with status icons, designed for showing students what QuantUI
is doing during operations like molecule validation, PubChem fetches,
and job submission.

Usage::

    from quantui.progress import StepProgress

    steps = StepProgress(["Parse coordinates", "Validate atoms", "Check spin"])
    display(steps.widget)

    steps.start(0)
    # ... do step 0 ...
    steps.complete(0)

    steps.start(1)
    # ... do step 1 ...
    steps.fail(1, "Invalid element symbol 'Xx'")
"""

from __future__ import annotations

import html
from typing import List, Optional

import ipywidgets as widgets

from quantui import theme as _theme


class StepProgress:
    """
    A numbered step-by-step progress indicator using HTML.

    Each step shows an icon reflecting its state:

    - ⬜ not started
    - ⏳ in progress
    - ✅ completed
    - ❌ failed

    Args:
        step_labels: Human-readable labels for each step.
    """

    _ICONS = {
        "pending": "⬜",
        "active": "⏳",
        "done": "✅",
        "fail": "❌",
    }

    def __init__(self, step_labels: List[str]) -> None:
        self._labels = list(step_labels)
        self._states: List[str] = ["pending"] * len(self._labels)
        self._messages: List[Optional[str]] = [None] * len(self._labels)
        self._html = widgets.HTML()
        self._render()

    @property
    def widget(self) -> widgets.HTML:
        """The displayable widget."""
        return self._html

    def start(self, index: int) -> None:
        """Mark step *index* as in-progress."""
        self._states[index] = "active"
        self._messages[index] = None
        self._render()

    def complete(self, index: int, message: Optional[str] = None) -> None:
        """Mark step *index* as successfully completed."""
        self._states[index] = "done"
        self._messages[index] = message
        self._render()

    def fail(self, index: int, message: Optional[str] = None) -> None:
        """Mark step *index* as failed."""
        self._states[index] = "fail"
        self._messages[index] = message
        self._render()

    def reset(self) -> None:
        """Reset all steps to pending."""
        self._states = ["pending"] * len(self._labels)
        self._messages = [None] * len(self._labels)
        self._render()

    def _render(self) -> None:
        lines = []
        for i, (label, state) in enumerate(zip(self._labels, self._states)):
            icon = self._ICONS[state]
            weight = "bold" if state == "active" else "normal"
            color = _theme.css.ACCENT_ERROR if state == "fail" else _theme.css.TEXT_BODY
            line = (
                f'<div style="font-size:13px; padding:2px 0; '
                f'font-weight:{weight}; color:{color};">'
                f"{icon} <b>Step {i + 1}:</b> {html.escape(label)}"
            )
            message = self._messages[i]
            if message:
                line += f" — <i>{html.escape(message)}</i>"
            line += "</div>"
            lines.append(line)

        self._html.value = (
            f'<div style="border:1px solid {_theme.css.BORDER}; border-radius:6px; '
            f"padding:8px 12px; margin:6px 0; background:{_theme.css.BG_PANEL}; "
            'max-width:600px;">' + "\n".join(lines) + "</div>"
        )
