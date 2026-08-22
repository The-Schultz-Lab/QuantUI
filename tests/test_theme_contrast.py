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
        (
            0.213 + a * 0.787 - b * 0.213,
            0.715 - a * 0.715 - b * 0.715,
            0.072 - a * 0.072 + b * 0.928,
        ),
        (
            0.213 - a * 0.213 + b * 0.143,
            0.715 + a * 0.285 + b * 0.140,
            0.072 - a * 0.072 - b * 0.283,
        ),
        (
            0.213 - a * 0.213 - b * 0.787,
            0.715 - a * 0.715 + b * 0.715,
            0.072 + a * 0.928 + b * 0.072,
        ),
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
        assert (
            _contrast(_hex_rgb(theme.BORDER_STRONG), _hex_rgb(bg)) >= UI_COMPONENT_MIN
        )
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
        # M-THEME text-tier tokens added 2026-08-21 (h1/h3 colour, spinner
        # border-top-color) — same class of bug, same guard.
        assert "__Q_" not in _APP_CSS

    def test_app_css_carries_the_border_token(self):
        from quantui.app import _APP_CSS

        assert theme.BORDER in _APP_CSS

    def test_app_css_carries_the_text_and_accent_tokens(self):
        from quantui.app import _APP_CSS

        assert theme.TEXT_STRONG in _APP_CSS
        assert theme.TEXT_SLATE in _APP_CSS
        assert theme.ACCENT_INFO in _APP_CSS

    def test_no_viewer_border_is_drawn_from_a_css_class(self):
        # Measured in the browser 2026-08-03: an Output widget cannot
        # shrink-wrap. Its children are Lumino widgets that JupyterLab's layout
        # engine pins to explicit full-window pixel widths, so `fit-content`
        # (which is min(max-content, …)) can only ever resolve to full width —
        # a class-borne border spans the page no matter what. Three attempts
        # were made before that measurement. The border therefore lives on the
        # rendered fragment, where the pixel width is known; a reintroduced
        # .quantui-viewer-frame would draw a second, full-width box around it.
        from quantui.app import _APP_CSS

        assert "quantui-viewer-frame" not in _APP_CSS

    @pytest.mark.parametrize("width", [600, 420])
    def test_rendered_fragment_carries_its_own_sized_border(self, width):
        # The border lives on the fragment, not on the hosting Output widget's
        # CSS class, because render_molecule_html is the only place that knows
        # the viewer's pixel width — so the frame cannot drift out of sync with
        # whatever a caller passes, and it hugs the plot instead of spanning
        # the page.
        pytest.importorskip("py3Dmol")
        from quantui.molecule import Molecule
        from quantui.visualization_py3dmol import render_molecule_html

        mol = Molecule(atoms=["N", "N"], coordinates=[[0, 0, 0], [0, 0, 1.10]])
        html = render_molecule_html(mol, width=width)

        assert html.startswith(f'<div style="width:{width}px;max-width:100%;')
        assert f"border:1px solid {theme.BORDER_STRONG}" in html
        # The info box must be INSIDE the frame — it was what forced the old
        # full-width sizing, and it should align with the canvas.
        assert "Molecule Information" in html

    def test_no_viewer_output_carries_a_frame_class(self):
        # Every 3D viewer now renders through render_molecule_html, so each
        # already has a fitted border. Adding the class back to any of them
        # would draw a second, full-width box around the tight one.
        import pathlib

        builders = (pathlib.Path(theme.__file__).parent / "app_builders.py").read_text(
            encoding="utf-8"
        )
        assert 'add_class("quantui-viewer-frame")' not in builders

    def test_the_calculate_viewer_reserves_room_for_the_whole_fragment(self):
        # Regression: viz_output was pinned to height="510px" — sized for the
        # 500px canvas alone — with overflow hidden, which sliced the bottom
        # border off once the info box moved inside the bordered fragment. A
        # minimum reserves the same layout space without ever clipping.
        import pathlib
        import re

        builders = (pathlib.Path(theme.__file__).parent / "app_builders.py").read_text(
            encoding="utf-8"
        )
        block = re.search(
            r"app\.viz_output = widgets\.Output\((.*?)\n    \)", builders, re.S
        )
        assert block is not None, "viz_output construction not found"
        body = block.group(1)
        assert "min_height=" in body, "a fixed height clips the fragment's border"
        assert re.search(r"[^_]height=", body) is None

    def test_the_analysis_backend_toggle_keeps_the_border(self):
        # The re-render path a backend toggle takes must be the same one the
        # first draw takes, or switching py3Dmol <-> plotlymol3d silently drops
        # the fragment (and its border) for a bare display() call.
        import pathlib

        app_src = (pathlib.Path(theme.__file__).parent / "app.py").read_text(
            encoding="utf-8"
        )
        rerender = app_src[app_src.index("def _rerender_3d_views") :]
        rerender = rerender[: rerender.index("def _update_analysis_backend_label")]
        assert "_render_molecule_html(" in rerender
        assert "_display_molecule(" not in rerender


class TestEveryViewerIsFramed:
    """Every 3-D viewer frames identically, and nothing clips the frame.

    The border lives on the rendered fragment (see theme.frame_viewer_html),
    which means two things can silently break it: a render path that forgets to
    call the helper, and a hosting Output pinned to a fixed height, whose
    overflow:hidden slices the bottom edge off. Both happened during THEME.5,
    so both are asserted here.
    """

    @staticmethod
    def _src(name: str) -> str:
        import pathlib

        return (pathlib.Path(theme.__file__).parent / name).read_text(encoding="utf-8")

    @classmethod
    def _body(cls, module: str, func: str) -> str:
        """Source of one top-level function. Black separates top-level
        definitions with two blank lines, which is a more reliable end marker
        than scanning for the next ``def`` — the module-level JS constants
        between these builders would otherwise be swallowed into the body."""
        src = cls._src(module)
        body = src[src.index(f"def {func}(") :]
        end = body.find("\n\n\n")
        return body if end == -1 else body[:end]

    def test_the_helper_frames_at_the_viewer_width(self):
        html = theme.frame_viewer_html("<canvas></canvas>", width=460)
        assert html.startswith('<div style="width:460px;max-width:100%;')
        assert f"border:1px solid {theme.BORDER_STRONG}" in html
        assert html.endswith("</div>")

    def test_controls_are_inset_but_the_viewer_is_flush(self):
        # The canvas is its own edge, so it should touch the frame; buttons
        # sitting flush against a visible border read as clipped.
        html = theme.frame_viewer_html("<canvas></canvas>", width=460, controls="<b/>")
        assert '<canvas></canvas><div style="padding:0 8px 6px"><b/></div>' in html

    def test_no_controls_means_no_empty_padding_div(self):
        assert "padding" not in theme.frame_viewer_html("<x/>", width=100)

    @pytest.mark.parametrize(
        "builder",
        [
            "build_preopt_preview_html",
            "build_trajectory_viewer_html",
            "build_vib_viewer_html",
        ],
    )
    def test_every_animation_builder_returns_a_framed_fragment(self, builder):
        # Each of these has early-return paths (single frame, missing viewer
        # id) that originally returned bare viewer HTML. Every exit must frame,
        # or an animation loses its border in exactly the cases hardest to
        # reproduce by hand — which is why this counts returns rather than
        # checking that the helper is called somewhere in the body.
        import re

        body = self._body("app_visualization.py", builder)
        # Anchored to the line start so the `"return s;"` inside these
        # builders' embedded JS label snippets isn't counted as a Python exit.
        n_returns = len(re.findall(r"^ +return ", body, re.M))
        assert n_returns, f"no returns found in {builder}"
        assert body.count("frame_viewer_html") == n_returns, (
            f"{builder} has {n_returns} return paths but frames only "
            f"{body.count('frame_viewer_html')} of them"
        )

    def test_both_vibration_backends_frame_their_animation(self):
        # A user who prefers plotlymol (or lacks py3Dmol) hits a different
        # renderer for the same viewer; it must look the same.
        for fn in ("_render_vib_mode_py3dmol", "_render_vib_mode_plotlymol"):
            body = self._body("app_visualization.py", fn)
            assert "frame_viewer_html" in body, f"{fn} emits an unframed animation"

    def test_the_standalone_export_is_left_unframed(self):
        # build_vib_export_html writes a file opened OUTSIDE the app, where the
        # invert filter never applies — same carve-out as analytics.py.
        body = self._body("app_visualization.py", "build_vib_export_html")
        assert "frame_viewer_html" not in body

    @pytest.mark.parametrize(
        "module,widget",
        [
            ("app_builders.py", "app.viz_output"),
            ("app_builders.py", "app.preopt_preview_output"),
            ("app_builders.py", "app.vib_output"),
            ("app_visualization.py", "viewer_output"),
        ],
    )
    def test_no_viewer_host_pins_a_fixed_height(self, module, widget):
        # The failure this catches is subtle: everything renders, the border is
        # there, and only the BOTTOM edge is missing — which reads as a CSS bug
        # rather than a sizing one. Calculate-tab regression, 2026-08-03.
        import re

        src = self._src(module)
        block = src[src.index(f"{widget} = widgets.Output(") :]
        block = block[: block.index("\n    )") + 6]
        assert re.search(r"[^_]height=", block) is None, (
            f"{widget} pins a fixed height; overflow:hidden will clip the "
            "frame's bottom border"
        )

    def test_the_preopt_host_no_longer_draws_its_own_border(self):
        # It used to carry a widget-level border at max-width 480px. With the
        # fragment framed, that would draw a second box around the first.
        src = self._src("app_builders.py")
        block = src[src.index("app.preopt_preview_output = widgets.Output(") :]
        block = block[: block.index("\n    )") + 6]
        assert "border=" not in block


class TestRetiredValues:
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
            if f.name != "analytics.py"
            and retired.search(f.read_text(encoding="utf-8"))
        ]
        assert offenders == [], f"retired border greys still used in: {offenders}"


class TestNoRawHexReintroducedInMigratedChrome:
    """M-THEME Execution Sequence step 1 (2026-08-21): the text/border/accent
    tokens in ``theme.py`` were extracted from raw hex literals scattered
    across these widget-building modules. This guards the migration from
    quietly eroding — a future edit pasting ``color:#555`` back in instead of
    ``{_theme.TEXT_SECONDARY}`` should fail here, not resurface as a silent
    duplicate literal for the next audit to rediscover.

    Excludes ``theme.py`` itself (the source of truth) and the plotting/3-D
    modules (``analytics.py``, ``orbital_visualization.py``,
    ``app_visualization.py``, ``visualization_py3dmol.py``, ``ir_plot.py``) —
    those were deliberately left out of this pass (M-THEME roadmap 14) since a
    wrong substitution there risks a real rendering regression this suite
    can't see; migrating them is future work, not yet a regression to catch.
    ``results_storage.py`` is also excluded — its calc-type badge colours are
    already one small, well-factored dict, not the scattered-literal problem
    this migration targets.
    """

    MIGRATED_FILES = (
        "app_formatters.py",
        "app_builders.py",
        "app.py",
        "app_runflow.py",
        "descriptor_cards.py",
        "help_content.py",
        "app_analysis.py",
        "app_history.py",
        "calc_log.py",
    )

    #: theme.py attributes whose values were extracted from these files.
    #: Kept as an explicit list (not "every theme.py string attribute") so a
    #: token added later for a *new* use doesn't retroactively demand every
    #: historical file be clean of a value it never used to begin with.
    MIGRATED_TOKENS = (
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
        "BG_PANEL",
        "BORDER_LEGACY",
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
    )

    def test_no_migrated_value_appears_as_a_raw_literal(self):
        import pathlib
        import re

        pkg = pathlib.Path(theme.__file__).parent
        values = [getattr(theme, name) for name in self.MIGRATED_TOKENS]
        # Longest first so e.g. "#94a3b8" isn't shadowed by a shorter value
        # that happens to prefix-match earlier in an unordered scan.
        values.sort(key=len, reverse=True)
        alternation = "|".join(re.escape(v) for v in values)
        pattern = re.compile(f"(?:{alternation})\\b", re.IGNORECASE)
        entity_pattern = re.compile(r"&#[0-9a-fA-F]{3,8};")

        offenders = {}
        for fname in self.MIGRATED_FILES:
            text = (pkg / fname).read_text(encoding="utf-8")
            text = entity_pattern.sub("", text)  # HTML entities, not colours
            found = sorted(set(pattern.findall(text)))
            if found:
                offenders[fname] = found
        assert (
            offenders == {}
        ), f"raw hex reintroduced where a theme.py token exists: {offenders}"
