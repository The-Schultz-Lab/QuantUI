"""Theme colour contrast guarantees (M-THEME THEME.5).

User report (2026-07-29): *"Fix dark mode contrasts and add borders (current
approach does not work)."*

Measurement, not assumption, found the actual defect. Dark mode is a whole-page
CSS ``filter: invert(1) hue-rotate(180deg)``, so every colour is written once
for light mode and dark mode is its mathematical inversion. Under that filter:

- **Text was never the problem** — body text on a panel measures 7.24:1 in light
  and *improves* to 8.91:1 inverted.
- **Structural separation was** — panel-vs-page was 1.03:1 and border-vs-panel
  1.14:1 in dark mode. Panels and their borders were invisible, which is why the
  same request asked for contrast *and* borders in one breath.

The fix is the mid-tone rule (see ``quantui/theme.py``): light borders vanish
when inverted, dark borders vanish in light mode, and only mid-tones survive
both. These tests re-derive the filter maths and assert the tokens still clear
WCAG 1.4.11's 3:1 bar for non-text UI components in **both** modes — so a future
"let's soften those borders" tweak fails here rather than silently making dark
mode unusable again.

Platform-independent: pure colour maths, no widgets.
"""

from __future__ import annotations

import math

import pytest

from quantui import theme

# Backgrounds these tokens are drawn against, in light-mode source values.
PANEL_BG = "#f8fafc"
PAGE_BG = "#ffffff"

# WCAG 2.1 SC 1.4.11 (Non-text Contrast): UI components and graphical objects
# need 3:1. Borders are UI components, not text — 4.5:1 is the wrong bar here
# and would force a heavier border than the design wants in light mode.
UI_COMPONENT_MIN = 3.0


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _hue_rotate(c: tuple[int, int, int], deg: float) -> tuple[int, int, int]:
    """The W3C filter ``hue-rotate`` matrix — a linear approximation, NOT a true
    HSL rotation. Using the real matrix matters: an HSL model would predict
    different results than the browser actually produces."""
    a, b = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = (
        (0.213 + a * 0.787 - b * 0.213, 0.715 - a * 0.715 - b * 0.715, 0.072 - a * 0.072 + b * 0.928),
        (0.213 - a * 0.213 + b * 0.143, 0.715 + a * 0.285 + b * 0.140, 0.072 - a * 0.072 - b * 0.283),
        (0.213 - a * 0.213 - b * 0.787, 0.715 - a * 0.715 + b * 0.715, 0.072 + a * 0.928 + b * 0.072),
    )
    return tuple(  # type: ignore[return-value]
        max(0, min(255, round(sum(m[i][j] * c[j] for j in range(3))))) for i in range(3)
    )


def _dark(hex_colour: str) -> tuple[int, int, int]:
    """What the dark-mode filter chain turns *hex_colour* into."""
    inverted = tuple(255 - x for x in _hex_rgb(hex_colour))
    return _hue_rotate(inverted, 180)  # type: ignore[arg-type]


def _relative_luminance(c: tuple[int, int, int]) -> float:
    def channel(v: int) -> float:
        s = v / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(x) for x in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


class TestContrastMathIsCorrect:
    """Guard the measuring instrument before trusting its measurements."""

    def test_white_on_black_is_maximal(self):
        assert _contrast((255, 255, 255), (0, 0, 0)) == pytest.approx(21.0, abs=0.01)

    def test_identical_colours_have_no_contrast(self):
        assert _contrast((120, 120, 120), (120, 120, 120)) == pytest.approx(1.0)

    def test_the_filter_inverts_lightness(self):
        # White must become black under invert; this is the property the whole
        # mid-tone argument rests on.
        assert _relative_luminance(_dark("#ffffff")) < 0.01
        assert _relative_luminance(_dark("#000000")) > 0.9


class TestBordersSurviveBothModes:
    @pytest.mark.parametrize("bg", [PANEL_BG, PAGE_BG], ids=["on-panel", "on-page"])
    def test_border_clears_the_ui_bar_in_light_mode(self, bg):
        assert _contrast(_hex_rgb(theme.BORDER), _hex_rgb(bg)) >= UI_COMPONENT_MIN

    @pytest.mark.parametrize("bg", [PANEL_BG, PAGE_BG], ids=["on-panel", "on-page"])
    def test_border_clears_the_ui_bar_in_dark_mode(self, bg):
        # The regression that prompted THEME.5: the old #e2e8f0 scored 1.14:1
        # here, so panels had no visible edge at all against the dark page.
        assert _contrast(_dark(theme.BORDER), _dark(bg)) >= UI_COMPONENT_MIN

    @pytest.mark.parametrize("bg", [PANEL_BG, PAGE_BG], ids=["on-panel", "on-page"])
    def test_strong_border_clears_the_bar_in_both_modes(self, bg):
        assert _contrast(_hex_rgb(theme.BORDER_STRONG), _hex_rgb(bg)) >= UI_COMPONENT_MIN
        assert _contrast(_dark(theme.BORDER_STRONG), _dark(bg)) >= UI_COMPONENT_MIN

    def test_strong_is_actually_stronger_than_the_default(self):
        # If these ever converge, BORDER_STRONG has quietly stopped meaning
        # anything and the viewer frame loses its intended emphasis.
        assert _contrast(_hex_rgb(theme.BORDER_STRONG), _hex_rgb(PANEL_BG)) > _contrast(
            _hex_rgb(theme.BORDER), _hex_rgb(PANEL_BG)
        )

    def test_the_old_values_would_fail_this_suite(self):
        # Documents *why* the tokens exist. If someone reverts a border to one
        # of these, the tests above catch it — this asserts they'd be right to.
        for old in ("#e2e8f0", "#e5e7eb", "#cbd5e1", "#c0ccd8"):
            assert _contrast(_dark(old), _dark(PANEL_BG)) < UI_COMPONENT_MIN


class TestTokensAreActuallyUsed:
    def test_app_css_has_no_unreplaced_sentinels(self):
        # _APP_CSS substitutes tokens via .replace() rather than an f-string
        # (the block is dense with CSS braces). A typo'd sentinel would ship a
        # literal "__Q_BORDER__" as a CSS colour and silently drop the border.
        from quantui.app import _APP_CSS

        assert "__Q_BORDER__" not in _APP_CSS
        assert "__Q_BORDER_STRONG__" not in _APP_CSS
        # Order-of-replacement bug: replacing the short sentinel first leaves
        # a dangling "_STRONG__" behind.
        assert "_STRONG__" not in _APP_CSS

    def test_app_css_carries_the_tokens(self):
        from quantui.app import _APP_CSS

        assert theme.BORDER in _APP_CSS
        assert theme.BORDER_STRONG in _APP_CSS

    def test_retired_border_greys_are_gone_from_in_app_chrome(self):
        # analytics.py is deliberately excluded: it writes a STANDALONE
        # dashboard HTML opened directly in a browser, so the app's invert
        # filter never applies and its light borders are correct as they are.
        import pathlib
        import re

        pkg = pathlib.Path(theme.__file__).parent
        retired = re.compile(
            r"border[a-z_-]*\s*[:=]\s*f?\"?[^;\"]*#(e2e8f0|e5e7eb|cbd5e1|c0ccd8)",
            re.IGNORECASE,
        )
        offenders = [
            f.name
            for f in pkg.glob("*.py")
            if f.name != "analytics.py" and retired.search(f.read_text(encoding="utf-8"))
        ]
        assert offenders == [], f"retired border greys still used in: {offenders}"
