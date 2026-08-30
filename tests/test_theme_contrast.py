"""Theme colour contrast guarantees (M-THEME THEME.5 / THEME.6).

User report (2026-07-29): *"Fix dark mode contrasts and add borders (current
approach does not work)."*

THEME.5 fixed structural separation with mid-tone ``BORDER`` tokens. THEME.6
retired the whole-page CSS invert filter in favour of preset palettes with
explicit light/dark (and tinted) values — these tests assert each palette's
border tokens still clear WCAG 1.4.11's 3:1 bar against its own panel/page
backgrounds.

Platform-independent: pure colour maths, no widgets.
"""

from __future__ import annotations

import pytest

from quantui import theme

# WCAG 2.1 SC 1.4.11 (Non-text Contrast): UI components need 3:1 against adjacent
# backgrounds. Borders are UI components, not body text.
UI_COMPONENT_MIN = 3.0


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


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


class TestPaletteRegistry:
    def test_four_presets_ship(self):
        assert set(theme.PALETTE_IDS) == {
            "Light",
            "Dark",
            "Dark Blue",
            "Dark Maroon",
        }

    def test_theme_css_block_sets_page_background_var(self):
        block = theme.theme_css_block("Dark")
        assert "--q-page-bg:" in block
        assert "var(--q-page-bg)" in block

    @pytest.mark.parametrize("palette_id", theme.PALETTE_IDS)
    def test_plotly_colours_track_palette(self, palette_id):
        colours = theme.plotly_colors(palette_id)
        palette = theme.get_palette(palette_id)
        assert colours["plot_bgcolor"] == palette.bg_panel
        assert colours["paper_bgcolor"] == palette.page_bg


class TestBordersMeetContrastBar:
    @pytest.mark.parametrize("palette_id", theme.PALETTE_IDS)
    def test_border_clears_ui_bar_on_panel_and_page(self, palette_id):
        palette = theme.get_palette(palette_id)
        for bg in (palette.bg_panel, palette.page_bg):
            assert (
                _contrast(_hex_rgb(palette.border), _hex_rgb(bg)) >= UI_COMPONENT_MIN
            ), palette_id

    @pytest.mark.parametrize("palette_id", theme.PALETTE_IDS)
    def test_strong_border_clears_ui_bar_on_panel_and_page(self, palette_id):
        palette = theme.get_palette(palette_id)
        for bg in (palette.bg_panel, palette.page_bg):
            assert (
                _contrast(_hex_rgb(palette.border_strong), _hex_rgb(bg))
                >= UI_COMPONENT_MIN
            ), palette_id

    def test_strong_is_actually_stronger_than_default_on_light(self):
        palette = theme.get_palette("Light")
        assert _contrast(
            _hex_rgb(palette.border_strong), _hex_rgb(palette.bg_panel)
        ) > _contrast(_hex_rgb(palette.border), _hex_rgb(palette.bg_panel))

    def test_the_old_values_would_fail_on_inverted_dark(self):
        """Documents why mid-tone borders replaced #e2e8f0 under the old filter."""
        inverted_panel = tuple(255 - x for x in _hex_rgb(theme.BG_PANEL))
        for old in ("#e2e8f0", "#e5e7eb", "#cbd5e1", "#c0ccd8"):
            old_inv = tuple(255 - x for x in _hex_rgb(old))
            assert _contrast(old_inv, inverted_panel) < UI_COMPONENT_MIN


class TestTokensAreActuallyUsed:
    def test_app_css_has_no_unreplaced_sentinels(self):
        from quantui.app import _APP_CSS

        assert "__Q_" not in _APP_CSS

    def test_app_css_uses_css_variables(self):
        from quantui.app import _APP_CSS

        assert "var(--q-border)" in _APP_CSS
        assert "var(--q-text-strong)" in _APP_CSS
        assert "var(--q-text-slate)" in _APP_CSS
        assert "var(--q-accent-info)" in _APP_CSS

    def test_no_viewer_border_is_drawn_from_a_css_class(self):
        from quantui.app import _APP_CSS

        assert "quantui-viewer-frame" not in _APP_CSS

    @pytest.mark.parametrize("width", [600, 420])
    def test_rendered_fragment_carries_its_own_sized_border(self, width):
        pytest.importorskip("py3Dmol")
        from quantui.molecule import Molecule
        from quantui.visualization_py3dmol import render_molecule_html

        mol = Molecule(atoms=["N", "N"], coordinates=[[0, 0, 0], [0, 0, 1.10]])
        html = render_molecule_html(mol, width=width)

        assert html.startswith(f'<div style="width:{width}px;max-width:100%;')
        assert f"border:1px solid {theme.css.BORDER_STRONG}" in html
        assert "Molecule Information" in html

    def test_no_viewer_output_carries_a_frame_class(self):
        import pathlib

        builders = (pathlib.Path(theme.__file__).parent / "app_builders.py").read_text(
            encoding="utf-8"
        )
        assert 'add_class("quantui-viewer-frame")' not in builders

    def test_the_calculate_viewer_reserves_room_for_the_whole_fragment(self):
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
        import pathlib

        app_src = (pathlib.Path(theme.__file__).parent / "app.py").read_text(
            encoding="utf-8"
        )
        rerender = app_src[app_src.index("def _rerender_3d_views") :]
        rerender = rerender[: rerender.index("def _update_analysis_backend_label")]
        assert "_render_molecule_html(" in rerender
        assert "_display_molecule(" not in rerender


class TestEveryViewerIsFramed:
    """Every 3-D viewer frames identically, and nothing clips the frame."""

    @staticmethod
    def _src(name: str) -> str:
        import pathlib

        return (pathlib.Path(theme.__file__).parent / name).read_text(encoding="utf-8")

    @classmethod
    def _body(cls, module: str, func: str) -> str:
        src = cls._src(module)
        body = src[src.index(f"def {func}(") :]
        end = body.find("\n\n\n")
        return body if end == -1 else body[:end]

    @pytest.mark.parametrize(
        "builder",
        [
            "build_preopt_preview_html",
            "build_trajectory_viewer_html",
            "build_vib_viewer_html",
        ],
    )
    def test_every_animation_builder_returns_a_framed_fragment(self, builder):
        import re

        body = self._body("app_visualization.py", builder)
        n_returns = len(re.findall(r"^ +return ", body, re.M))
        assert n_returns, f"no returns found in {builder}"
        assert body.count("frame_viewer_html") == n_returns, (
            f"{builder} has {n_returns} return paths but frames only "
            f"{body.count('frame_viewer_html')} of them"
        )

    def test_both_vibration_backends_frame_their_animation(self):
        for fn in ("_render_vib_mode_py3dmol", "_render_vib_mode_plotlymol"):
            body = self._body("app_visualization.py", fn)
            assert "frame_viewer_html" in body, f"{fn} emits an unframed animation"

    def test_the_standalone_export_is_left_unframed(self):
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
        import re

        src = self._src(module)
        block = src[src.index(f"{widget} = widgets.Output(") :]
        block = block[: block.index("\n    )") + 6]
        assert re.search(r"[^_]height=", block) is None, (
            f"{widget} pins a fixed height; overflow:hidden will clip the "
            "frame's bottom border"
        )

    def test_the_preopt_host_no_longer_draws_its_own_border(self):
        src = self._src("app_builders.py")
        block = src[src.index("app.preopt_preview_output = widgets.Output(") :]
        block = block[: block.index("\n    )") + 6]
        assert "border=" not in block


class TestRetiredValues:
    def test_retired_border_greys_are_gone_from_in_app_chrome(self):
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
    """M-THEME Execution Sequence step 1: migrated chrome must keep using tokens."""

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
        offenders: list[str] = []
        for fname in self.MIGRATED_FILES:
            path = pkg / fname
            text = path.read_text(encoding="utf-8")
            for token in self.MIGRATED_TOKENS:
                value = getattr(theme, token)
                if not isinstance(value, str) or not value.startswith("#"):
                    continue
                pattern = re.compile(re.escape(value) + r"(?![0-9a-fA-F])")
                if pattern.search(text):
                    offenders.append(f"{fname}: {value} ({token})")
        assert offenders == [], (
            "raw hex reintroduced where a theme.py token exists: " f"{offenders}"
        )
