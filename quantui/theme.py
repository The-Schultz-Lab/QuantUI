"""Theme colour tokens (M-THEME).

Why this module exists
----------------------
QuantUI's dark mode is **not a palette swap** — it is a whole-page CSS filter::

    html { filter: invert(1) hue-rotate(180deg) !important; }
    canvas, img, iframe, video { filter: invert(1) hue-rotate(180deg) !important; }

(see ``app.QuantUIApp._theme_css``). Every colour in the UI is written once, for
light mode, and dark mode is that same colour mathematically inverted. The
counter-filter on ``canvas/img/iframe/video`` double-inverts embedded renderers
(3-D viewers, plots) back to their intended appearance.

**The consequence that matters, and the reason this module exists:** you cannot
tune light and dark independently. There is exactly one source value per colour,
and dark mode is whatever that value inverts to.

The mid-tone rule
-----------------
That constraint has a non-obvious implication for anything whose job is
*separation* rather than *legibility*:

- A **light** grey border (``#e2e8f0``) looks correct on a white page, but
  inverts to a near-black border (``#121820``) on a near-black panel — invisible.
- A **dark** border has the mirror-image problem: fine in dark mode, invisible
  in light.
- A **mid-tone** border stays mid-tone under inversion, so it is visible in
  *both*. Mid-tones are the only values that survive.

Measured, not assumed (2026-07-31)
----------------------------------
Contrast ratios computed through the actual filter chain (invert, then the W3C
``hue-rotate`` matrix), against the panel background ``#f8fafc``:

===========  ==============  =============  ==================================
candidate    light vs panel  dark vs panel  verdict
===========  ==============  =============  ==================================
``#e2e8f0``  1.18:1          1.14:1         the old value — fails both
``#c0ccd8``  1.56:1          1.65:1         fails both
``#94a3b8``  2.45:1          3.12:1         passes dark only
``#7d8ea3``  3.20:1          4.30:1         **passes both** — chosen
``#64748b``  4.55:1          6.17:1         passes both, but heavy in light
===========  ==============  =============  ==================================

``BORDER`` is the *lightest* value clearing WCAG 1.4.11's **3:1** bar for
non-text UI components in both modes. ``#64748b`` clears it more comfortably but
reads as a heavy rule in light mode, which is a real cost for a subtle panel
divider.

What was NOT the problem
------------------------
Text contrast was measured too, and it was never broken — body text on a panel
is 7.24:1 in light and *improves* to 8.91:1 inverted. The user-reported "fix
dark mode contrasts" was, on measurement, entirely about **structural
separation**: panels and their borders were invisible against the page, which is
why the same request also asked to "add borders". Text colours are deliberately
left alone here.

Scope of this module today
--------------------------
Originally just the THEME.5 border fix. As of 2026-08-21, also a text-tier
greyscale set and a status-accent set (see below) extracted from the 9
widget-building "chrome" modules (``app.py``, ``app_builders.py``,
``app_formatters.py``, ``app_runflow.py``, and others) — 379 of that file
set's ~436 hardcoded-hex occurrences, covering the 22 highest-frequency
distinct values. Still not a full-palette migration: the plotting/3-D-viewer
modules (``analytics.py``, ``orbital_visualization.py``,
``app_visualization.py``, ``visualization_py3dmol.py``, ``ir_plot.py``) are
untouched — a wrong substitution there risks an actual rendering regression
a cloud session with no browser can't catch — and a long tail of ~70
low-frequency chrome values remains too. Most of what's *left* is still
*semantic* accents (error red, success green, link blue) whose hue survives
``hue-rotate(180)`` and looks correct in both modes already; the text/accent
tokens added here are a maintainability move (one name instead of ~380
copy-pasted literals), not a correctness fix — unlike ``BORDER``/
``BORDER_STRONG`` above, which changed values and needed real contrast
measurement.

When THEME.6 (customisable palettes) lands, the invert filter has to go — a
palette system cannot work when dark mode is a derived inversion rather than an
independent set of values. At that point these tokens become the seam: they
grow light/dark variants and the ``_theme_css`` filter is replaced. Keeping them
named here means that change edits this file plus the CSS, not hundreds of call
sites again.
"""

from __future__ import annotations

# ── Structural / chrome ──────────────────────────────────────────────────────

#: Panel, card, and viewer borders. Mid-tone so it survives inversion — see the
#: module docstring's table. Replaced ``#e2e8f0`` / ``#e5e7eb`` / ``#cbd5e1`` /
#: ``#c0ccd8``, all of which were invisible in dark mode (1.14-1.65:1).
BORDER = "#7d8ea3"

#: Emphasised border for elements that should read as framed even at a glance
#: (the 3-D viewer frame, which sits on its own rather than in a card stack).
BORDER_STRONG = "#64748b"

#: Light legacy border/rule colour (tables, dividers). Kept distinct from
#: ``BORDER`` rather than merged into it — this module's own docstring warns
#: that a light border like this one is exactly the shape of value that
#: disappears under the dark-mode invert filter; flagged here, not yet
#: re-measured or replaced, so a later contrast pass has one name to fix
#: instead of the original scattered literal.
BORDER_LEGACY = "#ccc"

#: Background used for panel/card chrome (result cards, descriptor cards).
#: Also the reference panel background this module's own WCAG measurements
#: (see the docstring table) were computed against.
BG_PANEL = "#f8fafc"

# ── Text (greyscale tiers) ────────────────────────────────────────────────────
# Extracted 2026-08-21 (M-THEME Execution Sequence step 1) from ~300 scattered
# literal occurrences across app_formatters.py, app_builders.py, app.py,
# app_runflow.py, descriptor_cards.py, and other widget-building modules —
# named so they're one greppable set instead of loose hex strings, and so a
# future *harmonization* pass (there are more of these than there should be;
# see below) only has to touch this file plus whatever it introduces, not 300
# call sites again.
#
# Deliberately NOT harmonized to fewer distinct shades in this pass: each
# token keeps its call sites' exact original value, so migrating call sites to
# reference these is a pure extract-to-constant refactor with zero rendered-
# pixel change — safe to do without a browser, unlike the border fix (THEME.5)
# above, which needed real contrast measurement because it *changed* values.
# This module's own docstring already argues these greys were not the
# reported defect ("text contrast was never broken") — the value here is
# maintainability (one name per shade, no more copy-pasted hex) and getting
# migrated code ready for THEME.6, not a correctness fix. A later, visually
# verified pass can still collapse TEXT_MUTED/_MUTED_LIGHT/_FAINT into fewer
# WCAG-measured values the way BORDER/BORDER_STRONG already were.
TEXT_HEADING = "#000"
TEXT_LABEL = "#444"
TEXT_SECONDARY = "#555"
TEXT_MUTED = "#666"
TEXT_MUTED_LIGHT = "#777"
TEXT_FAINT = "#888"
TEXT_SUBTLE = "#94a3b8"
TEXT_BODY = "#334155"
TEXT_STRONG = "#1e293b"
#: Same numeric value as ``BORDER_STRONG`` today, coincidentally — kept as a
#: separate name because the ~25 call sites using it are text colour, not
#: borders. Decoupled on purpose: a later border-only or text-only retune
#: must not silently move the other.
TEXT_SLATE = "#64748b"
TEXT_SLATE_DARK = "#475569"

# ── Status accents ────────────────────────────────────────────────────────────
# This module's docstring notes semantic accents (error/success/warning hues)
# were not the reported defect — they survive ``hue-rotate(180)`` and read
# correctly in both modes already. Named here anyway for the same
# maintainability reason as the text tier: one call site can't drift from
# another when they share a name instead of a copy-pasted literal. The "_ALT"
# / "_LIGHT" siblings are distinct pre-existing values (a second red, a
# lighter amber, …), not renamed — see the text-tier note above on why this
# pass doesn't consolidate them.
ACCENT_ERROR = "#b91c1c"
ACCENT_ERROR_ALT = "#c00"
ACCENT_SUCCESS = "#16a34a"
ACCENT_SUCCESS_BG = "#f0fff0"
ACCENT_SUCCESS_ALT = "#4caf50"
ACCENT_WARNING = "#b45309"
ACCENT_WARNING_LIGHT = "#f59e0b"
ACCENT_INFO = "#2563eb"
ACCENT_PURPLE = "#7c3aed"
ACCENT_TEAL = "#0d9488"


def frame_viewer_html(view_html: str, *, width: int, controls: str = "") -> str:
    """Wrap a 3-D viewer fragment in the standard frame, sized to the viewer.

    Every 3-D viewer in the app — static molecule, optimization trajectory,
    vibrational mode, classical pre-opt preview — goes through here so they all
    frame identically and a token change lands everywhere at once.

    ``width`` must be the viewer's own pixel width. The frame is built at that
    width **here**, in the code that knows it, rather than as a CSS class on the
    hosting ``widgets.Output``. Measured in the browser 2026-08-03: an Output's
    children are Lumino widgets that JupyterLab sizes with explicit pixel widths
    tracking the window, so a class-borne ``fit-content`` always resolves to the
    full page width. The border then enclosed a wide strip of dead space beside
    the plot, which is actively misleading — it implies the whole strip is
    interactive and hides where the page can be scrolled without dragging the
    3-D view.

    ``controls`` is optional stepper/player markup rendered below the viewer.
    It is inset so the buttons don't sit flush against the frame; the viewer
    itself stays flush, since its canvas is its own edge.

    Note for callers: the hosting Output must not pin a fixed ``height``, or the
    frame's bottom edge is clipped off by ``overflow: hidden``. Use
    ``min_height`` — see ``app_builders.build_shared_widgets``.
    """
    if controls:
        controls = f'<div style="padding:0 8px 6px">{controls}</div>'
    return (
        f'<div style="width:{width}px;max-width:100%;'
        f"border:1px solid {BORDER_STRONG};border-radius:6px;"
        f'overflow:hidden">{view_html}{controls}</div>'
    )


__all__ = [
    "BORDER",
    "BORDER_STRONG",
    "BORDER_LEGACY",
    "BG_PANEL",
    "TEXT_HEADING",
    "TEXT_LABEL",
    "TEXT_SECONDARY",
    "TEXT_MUTED",
    "TEXT_MUTED_LIGHT",
    "TEXT_FAINT",
    "TEXT_SUBTLE",
    "TEXT_BODY",
    "TEXT_STRONG",
    "TEXT_SLATE",
    "TEXT_SLATE_DARK",
    "ACCENT_ERROR",
    "ACCENT_ERROR_ALT",
    "ACCENT_SUCCESS",
    "ACCENT_SUCCESS_BG",
    "ACCENT_SUCCESS_ALT",
    "ACCENT_WARNING",
    "ACCENT_WARNING_LIGHT",
    "ACCENT_INFO",
    "ACCENT_PURPLE",
    "ACCENT_TEAL",
    "frame_viewer_html",
]
