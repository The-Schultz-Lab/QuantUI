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
Only the tokens needed for the THEME.5 fix. It is intentionally not a
full-palette migration: the codebase has ~390 hardcoded hex literals across 17
files, most of them *semantic* accents (error red, success green, link blue)
whose hue survives ``hue-rotate(180)`` and which look correct in both modes
already. Migrating those wholesale would be a large, visually-unverifiable
change for no user-visible gain.

When THEME.6 (customisable palettes) lands, the invert filter has to go — a
palette system cannot work when dark mode is a derived inversion rather than an
independent set of values. At that point these tokens become the seam: they
grow light/dark variants and the ``_theme_css`` filter is replaced. Keeping them
named here means that change edits this file plus the CSS, not 19 call sites
again.
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


__all__ = ["BORDER", "BORDER_STRONG", "frame_viewer_html"]
