"""A scrolling list of checkboxes with the ``SelectMultiple`` interface.

Compare used a ``SelectMultiple`` list box, which needs Ctrl/Shift-click to
pick more than one row — not obvious to students (M-UIPOLISH L12). This keeps
the attributes the app reads and writes (``options`` as ``(label, value)``
pairs, ``value`` as a tuple of the checked values) so callers are unchanged.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import ipywidgets as widgets


class CheckList(widgets.VBox):
    """Checkboxes for ``(label, value)`` options; empty values are notes."""

    def __init__(self, options: Sequence[Tuple[str, Any]] = (), **kwargs: Any) -> None:
        kwargs.setdefault(
            "layout",
            widgets.Layout(max_height="260px", overflow_y="auto", width="100%"),
        )
        super().__init__(**kwargs)
        self._options: List[Tuple[str, Any]] = []
        self._boxes: List[Tuple[widgets.Checkbox, Any]] = []
        self.options = list(options)

    @property
    def options(self) -> List[Tuple[str, Any]]:
        return list(self._options)

    @options.setter
    def options(self, options: Sequence[Tuple[str, Any]]) -> None:
        self._options = [(str(lbl), val) for lbl, val in options]
        rows: List[Any] = []
        self._boxes = []
        for label, value in self._options:
            if value in ("", None):
                rows.append(widgets.HTML(f"<i>{label}</i>"))
                continue
            box = widgets.Checkbox(
                value=False,
                description=label,
                indent=False,
                layout=widgets.Layout(width="auto", margin="0"),
            )
            self._boxes.append((box, value))
            rows.append(box)
        self.children = tuple(rows)

    @property
    def value(self) -> Tuple[Any, ...]:
        return tuple(v for box, v in self._boxes if box.value)

    @value.setter
    def value(self, values: Sequence[Any]) -> None:
        wanted = set(values or ())
        for box, v in self._boxes:
            box.value = v in wanted
