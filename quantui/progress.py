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

    def __init__(self, step_labels: List[str], *, horizontal: bool = False) -> None:
        self._horizontal = horizontal
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
        if self._horizontal:
            self._render_horizontal()
            return
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

    _NUMBERS = "①②③④⑤⑥⑦⑧⑨"

    def _render_horizontal(self) -> None:
        """One line: ✓ Molecule › ② Method › ③ Run › ④ Results.

        The four-row box took ~150 px above every Calculate view.
        """
        parts = []
        for i, (label, state) in enumerate(zip(self._labels, self._states)):
            mark = (
                "✓"
                if state == "done"
                else "✗" if state == "fail" else self._NUMBERS[i % len(self._NUMBERS)]
            )
            if state == "active":
                style = f"color:{_theme.css.TEXT_STRONG};font-weight:600"
            elif state == "done":
                style = f"color:{_theme.css.ACCENT_SUCCESS}"
            elif state == "fail":
                style = f"color:{_theme.css.ACCENT_ERROR}"
            else:
                style = f"color:{_theme.css.TEXT_MUTED}"
            text = f"{mark} {html.escape(label)}"
            message = self._messages[i]
            if message and state == "fail":
                text += f" — {html.escape(message)}"
            parts.append(f'<span style="{style}">{text}</span>')
        sep = f'<span style="color:{_theme.css.TEXT_MUTED};margin:0 8px">›</span>'
        self._html.value = (
            '<div style="font-size:14px;margin:4px 0 8px">' + sep.join(parts) + "</div>"
        )
