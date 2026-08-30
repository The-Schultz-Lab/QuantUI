"""Theme colour tokens and preset palettes (M-THEME).

QuantUI themes are **preset palettes** backed by CSS custom properties. Each
palette defines independent light/dark (or tinted) values — the whole-page
``filter: invert(1) hue-rotate(180deg)`` approach was retired in THEME.6.

Inline HTML and app chrome reference tokens via ``theme.css.BORDER`` etc.
(``var(--q-border)``), so a theme switch updates every surface that uses those
vars without re-rendering widget HTML.

Plot/3-D modules still carry some hardcoded plot colours; those are routed
through ``plotly_colors()`` where the app layer applies themes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# ── Default (Light) token values ─────────────────────────────────────────────
# Kept as module-level names for tests and callers that need a concrete hex.

BORDER = "#7d8ea3"
BORDER_STRONG = "#64748b"
BORDER_LEGACY = "#ccc"
BG_PANEL = "#f8fafc"
PAGE_BG = "#ffffff"

TEXT_HEADING = "#000"
TEXT_LABEL = "#444"
TEXT_SECONDARY = "#555"
TEXT_MUTED = "#666"
TEXT_MUTED_LIGHT = "#777"
TEXT_FAINT = "#888"
TEXT_SUBTLE = "#94a3b8"
TEXT_BODY = "#334155"
TEXT_STRONG = "#1e293b"
TEXT_SLATE = "#64748b"
TEXT_SLATE_DARK = "#475569"

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

_TOKEN_FIELDS: Tuple[str, ...] = (
    "page_bg",
    "border",
    "border_strong",
    "border_legacy",
    "bg_panel",
    "text_heading",
    "text_label",
    "text_secondary",
    "text_muted",
    "text_muted_light",
    "text_faint",
    "text_subtle",
    "text_body",
    "text_strong",
    "text_slate",
    "text_slate_dark",
    "accent_error",
    "accent_error_alt",
    "accent_success",
    "accent_success_bg",
    "accent_success_alt",
    "accent_warning",
    "accent_warning_light",
    "accent_info",
    "accent_purple",
    "accent_teal",
)


@dataclass(frozen=True)
class ThemePalette:
    """One named preset palette — all chrome/plot base colours."""

    palette_id: str
    display_name: str
    is_dark: bool
    page_bg: str
    border: str
    border_strong: str
    border_legacy: str
    bg_panel: str
    text_heading: str
    text_label: str
    text_secondary: str
    text_muted: str
    text_muted_light: str
    text_faint: str
    text_subtle: str
    text_body: str
    text_strong: str
    text_slate: str
    text_slate_dark: str
    accent_error: str
    accent_error_alt: str
    accent_success: str
    accent_success_bg: str
    accent_success_alt: str
    accent_warning: str
    accent_warning_light: str
    accent_info: str
    accent_purple: str
    accent_teal: str

    def css_variables(self) -> Dict[str, str]:
        """Map ``--q-*`` custom-property names to hex values."""
        out: Dict[str, str] = {}
        for field in _TOKEN_FIELDS:
            key = field.replace("_", "-")
            out[f"--q-{key}"] = getattr(self, field)
        return out


def _light_palette() -> ThemePalette:
    return ThemePalette(
        palette_id="Light",
        display_name="Light",
        is_dark=False,
        page_bg=PAGE_BG,
        border=BORDER,
        border_strong=BORDER_STRONG,
        border_legacy=BORDER_LEGACY,
        bg_panel=BG_PANEL,
        text_heading=TEXT_HEADING,
        text_label=TEXT_LABEL,
        text_secondary=TEXT_SECONDARY,
        text_muted=TEXT_MUTED,
        text_muted_light=TEXT_MUTED_LIGHT,
        text_faint=TEXT_FAINT,
        text_subtle=TEXT_SUBTLE,
        text_body=TEXT_BODY,
        text_strong=TEXT_STRONG,
        text_slate=TEXT_SLATE,
        text_slate_dark=TEXT_SLATE_DARK,
        accent_error=ACCENT_ERROR,
        accent_error_alt=ACCENT_ERROR_ALT,
        accent_success=ACCENT_SUCCESS,
        accent_success_bg=ACCENT_SUCCESS_BG,
        accent_success_alt=ACCENT_SUCCESS_ALT,
        accent_warning=ACCENT_WARNING,
        accent_warning_light=ACCENT_WARNING_LIGHT,
        accent_info=ACCENT_INFO,
        accent_purple=ACCENT_PURPLE,
        accent_teal=ACCENT_TEAL,
    )


def _dark_palette() -> ThemePalette:
    return ThemePalette(
        palette_id="Dark",
        display_name="Dark",
        is_dark=True,
        page_bg="#0f172a",
        border="#64748b",
        border_strong="#94a3b8",
        border_legacy="#475569",
        bg_panel="#1e293b",
        text_heading="#f8fafc",
        text_label="#e2e8f0",
        text_secondary="#cbd5e1",
        text_muted="#94a3b8",
        text_muted_light="#94a3b8",
        text_faint="#64748b",
        text_subtle="#64748b",
        text_body="#e2e8f0",
        text_strong="#f1f5f9",
        text_slate="#94a3b8",
        text_slate_dark="#cbd5e1",
        accent_error="#f87171",
        accent_error_alt="#ef4444",
        accent_success="#4ade80",
        accent_success_bg="#14532d",
        accent_success_alt="#22c55e",
        accent_warning="#fbbf24",
        accent_warning_light="#fcd34d",
        accent_info="#60a5fa",
        accent_purple="#a78bfa",
        accent_teal="#2dd4bf",
    )


def _dark_blue_palette() -> ThemePalette:
    return ThemePalette(
        palette_id="Dark Blue",
        display_name="Dark Blue",
        is_dark=True,
        page_bg="#0a1628",
        border="#7a9fcc",
        border_strong="#8fb0d9",
        border_legacy="#3d5a80",
        bg_panel="#0f2847",
        text_heading="#e8f0ff",
        text_label="#c5d9f2",
        text_secondary="#a8c4e8",
        text_muted="#8fb0d9",
        text_muted_light="#7a9fcc",
        text_faint="#6b8fbf",
        text_subtle="#6b8fbf",
        text_body="#d6e6ff",
        text_strong="#f0f6ff",
        text_slate="#8fb0d9",
        text_slate_dark="#a8c4e8",
        accent_error="#f87171",
        accent_error_alt="#ef4444",
        accent_success="#4ade80",
        accent_success_bg="#0d3320",
        accent_success_alt="#22c55e",
        accent_warning="#fbbf24",
        accent_warning_light="#fcd34d",
        accent_info="#7eb6ff",
        accent_purple="#a78bfa",
        accent_teal="#5eead4",
    )


def _dark_maroon_palette() -> ThemePalette:
    return ThemePalette(
        palette_id="Dark Maroon",
        display_name="Dark Maroon",
        is_dark=True,
        page_bg="#1a0a0f",
        border="#8b5a6b",
        border_strong="#a06b7d",
        border_legacy="#6b4554",
        bg_panel="#2d1219",
        text_heading="#fce8ef",
        text_label="#e8c4d0",
        text_secondary="#d4a8b8",
        text_muted="#b88898",
        text_muted_light="#a87888",
        text_faint="#986878",
        text_subtle="#986878",
        text_body="#f0d8e0",
        text_strong="#fff5f8",
        text_slate="#b88898",
        text_slate_dark="#d4a8b8",
        accent_error="#f87171",
        accent_error_alt="#ef4444",
        accent_success="#4ade80",
        accent_success_bg="#1a3320",
        accent_success_alt="#22c55e",
        accent_warning="#fbbf24",
        accent_warning_light="#fcd34d",
        accent_info="#93c5fd",
        accent_purple="#c4b5fd",
        accent_teal="#5eead4",
    )


PALETTES: Dict[str, ThemePalette] = {
    p.palette_id: p
    for p in (
        _light_palette(),
        _dark_palette(),
        _dark_blue_palette(),
        _dark_maroon_palette(),
    )
}

PALETTE_IDS: Tuple[str, ...] = tuple(PALETTES.keys())
DEFAULT_PALETTE_ID = "Dark"


def get_palette(palette_id: str) -> ThemePalette:
    """Return a palette by id, falling back to the default."""
    return PALETTES.get(palette_id, PALETTES[DEFAULT_PALETTE_ID])


def theme_css_block(palette_id: str) -> str:
    """Inject CSS custom properties for *palette_id* (THEME.6)."""
    palette = get_palette(palette_id)
    lines = [f"  {k}: {v};" for k, v in palette.css_variables().items()]
    vars_block = "\n".join(lines)
    return (
        "<style>"
        ":root {\n"
        f"{vars_block}\n"
        "}\n"
        "html, body, .jp-OutputArea-output, .widget-html-content "
        "{ background-color: var(--q-page-bg) !important; "
        "color: var(--q-text-body) !important; }\n"
        "</style>"
    )


def plotly_colors(palette_id: str) -> dict:
    """Plotly layout colours for the selected palette."""
    palette = get_palette(palette_id)
    return {
        "plot_bgcolor": palette.bg_panel,
        "paper_bgcolor": palette.page_bg,
        "font_color": palette.text_strong,
        "grid_color": palette.border,
        "scene_bgcolor": "#000000" if palette.is_dark else "#ffffff",
    }


class _CssVarRefs:
    """CSS ``var(--q-*)`` references for inline HTML — theme-switchable."""

    BORDER = "var(--q-border)"
    BORDER_STRONG = "var(--q-border-strong)"
    BORDER_LEGACY = "var(--q-border-legacy)"
    BG_PANEL = "var(--q-bg-panel)"
    PAGE_BG = "var(--q-page-bg)"
    TEXT_HEADING = "var(--q-text-heading)"
    TEXT_LABEL = "var(--q-text-label)"
    TEXT_SECONDARY = "var(--q-text-secondary)"
    TEXT_MUTED = "var(--q-text-muted)"
    TEXT_MUTED_LIGHT = "var(--q-text-muted-light)"
    TEXT_FAINT = "var(--q-text-faint)"
    TEXT_SUBTLE = "var(--q-text-subtle)"
    TEXT_BODY = "var(--q-text-body)"
    TEXT_STRONG = "var(--q-text-strong)"
    TEXT_SLATE = "var(--q-text-slate)"
    TEXT_SLATE_DARK = "var(--q-text-slate-dark)"
    ACCENT_ERROR = "var(--q-accent-error)"
    ACCENT_ERROR_ALT = "var(--q-accent-error-alt)"
    ACCENT_SUCCESS = "var(--q-accent-success)"
    ACCENT_SUCCESS_BG = "var(--q-accent-success-bg)"
    ACCENT_SUCCESS_ALT = "var(--q-accent-success-alt)"
    ACCENT_WARNING = "var(--q-accent-warning)"
    ACCENT_WARNING_LIGHT = "var(--q-accent-warning-light)"
    ACCENT_INFO = "var(--q-accent-info)"
    ACCENT_PURPLE = "var(--q-accent-purple)"
    ACCENT_TEAL = "var(--q-accent-teal)"


css = _CssVarRefs()


def frame_viewer_html(view_html: str, *, width: int, controls: str = "") -> str:
    """Wrap a 3-D viewer fragment in the standard frame, sized to the viewer."""
    if controls:
        controls = f'<div style="padding:0 8px 6px">{controls}</div>'
    return (
        f'<div style="width:{width}px;max-width:100%;'
        f"border:1px solid {css.BORDER_STRONG};border-radius:6px;"
        f'overflow:hidden">{view_html}{controls}</div>'
    )


__all__ = [
    "ACCENT_ERROR",
    "ACCENT_ERROR_ALT",
    "ACCENT_INFO",
    "ACCENT_PURPLE",
    "ACCENT_SUCCESS",
    "ACCENT_SUCCESS_ALT",
    "ACCENT_SUCCESS_BG",
    "ACCENT_TEAL",
    "ACCENT_WARNING",
    "ACCENT_WARNING_LIGHT",
    "BG_PANEL",
    "BORDER",
    "BORDER_LEGACY",
    "BORDER_STRONG",
    "DEFAULT_PALETTE_ID",
    "PAGE_BG",
    "PALETTE_IDS",
    "PALETTES",
    "TEXT_BODY",
    "TEXT_FAINT",
    "TEXT_HEADING",
    "TEXT_LABEL",
    "TEXT_MUTED",
    "TEXT_MUTED_LIGHT",
    "TEXT_SECONDARY",
    "TEXT_SLATE",
    "TEXT_SLATE_DARK",
    "TEXT_STRONG",
    "TEXT_SUBTLE",
    "ThemePalette",
    "css",
    "frame_viewer_html",
    "get_palette",
    "plotly_colors",
    "theme_css_block",
]
