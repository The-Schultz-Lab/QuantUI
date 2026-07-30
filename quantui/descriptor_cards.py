"""Method / basis descriptor cards.

Replaces the inline multi-paragraph educational-notes block next to the
method / basis dropdowns with two compact "descriptor cards" — an icon, a
one-line title, and a single distilled sentence — styled like the History-tab
result cards (light background + coloured left border).

Pure string builders (no widgets / no PySCF import) so the card content is
unit-testable in isolation. Icons are inline SVG (offline-safe — never a CDN
asset) drawn with ``stroke="currentColor"`` so they take the family accent
colour from the wrapping span and invert with the global dark-mode filter like
the rest of the UI (inline SVG is not one of the ``canvas/img/iframe/video``
tags the filter excludes).

Method family comes from ``config.METHOD_INFO[method]["type"]``
(``hf`` / ``dft`` / ``wavefunction``); the one-line body reuses that entry's
``use_for`` so the card can never drift from the help text. Basis family is
derived from the basis-set name.
"""

from __future__ import annotations

from . import config

# ── Icons (24×24 inline SVG, currentColor) ───────────────────────────────────

_SVG_OPEN = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round">'
)

# Hartree-Fock — paired spins (↑↓).
_ICON_HF = (
    _SVG_OPEN + '<line x1="8" y1="4" x2="8" y2="20"/>'
    '<polyline points="5 7 8 4 11 7"/>'
    '<line x1="16" y1="20" x2="16" y2="4"/>'
    '<polyline points="13 17 16 20 19 17"/></svg>'
)

# DFT — electron-density cloud.
_ICON_DFT = (
    _SVG_OPEN + '<path d="M7 18a4 4 0 0 1 0-8 5 5 0 0 1 9.6-1.5A3.5 3.5 0 0 1 '
    '17 18Z" fill="currentColor" fill-opacity="0.15"/></svg>'
)

# Post-HF wavefunction — correlated wave.
_ICON_WAVE = _SVG_OPEN + '<path d="M3 12 q3 -7 6 0 t6 0 t6 0"/></svg>'

# Minimal basis — a single tight function.
_ICON_BASIS_MINIMAL = (
    _SVG_OPEN + '<circle cx="12" cy="12" r="4" fill="currentColor"/></svg>'
)

# Split-valence (Pople) — a core function plus a split valence shell.
_ICON_BASIS_POPLE = (
    _SVG_OPEN + '<circle cx="12" cy="12" r="3.5" fill="currentColor"/>'
    '<circle cx="12" cy="12" r="9"/></svg>'
)

# Correlation-consistent — systematically nested shells.
_ICON_BASIS_CC = (
    _SVG_OPEN + '<circle cx="12" cy="12" r="3"/>'
    '<circle cx="12" cy="12" r="6.5"/>'
    '<circle cx="12" cy="12" r="10"/></svg>'
)

# def2 (Karlsruhe) — a function with polarisation lobes.
_ICON_BASIS_DEF2 = (
    _SVG_OPEN + '<circle cx="12" cy="12" r="5"/>'
    '<line x1="12" y1="1.5" x2="12" y2="4.5"/>'
    '<line x1="12" y1="19.5" x2="12" y2="22.5"/>'
    '<line x1="1.5" y1="12" x2="4.5" y2="12"/>'
    '<line x1="19.5" y1="12" x2="22.5" y2="12"/></svg>'
)

# ── Family → (accent fg, light bg, icon) ─────────────────────────────────────
# Palette mirrors the calc-type badge colours in app_formatters so the whole
# app reads as one system.

_METHOD_FAMILY_STYLE = {
    "hf": ("#2563eb", "#eff6ff", _ICON_HF),
    "dft": ("#7c3aed", "#f5f3ff", _ICON_DFT),
    "wavefunction": ("#b45309", "#fffbeb", _ICON_WAVE),
}
_METHOD_FALLBACK_STYLE = ("#475569", "#f8fafc", _ICON_HF)

_BASIS_FAMILY_STYLE = {
    "minimal": ("#64748b", "#f8fafc", _ICON_BASIS_MINIMAL),
    "pople": ("#0d9488", "#f0fdfa", _ICON_BASIS_POPLE),
    "cc": ("#15803d", "#f0fdf4", _ICON_BASIS_CC),
    "def2": ("#c2410c", "#fff7ed", _ICON_BASIS_DEF2),
}

# ── Basis family classification + one-line copy ──────────────────────────────

_BASIS_COPY = {
    "minimal": (
        "Minimal",
        "Fastest, lowest accuracy — great for learning, not research.",
    ),
    "pople": (
        "Split-valence (Pople)",
        "Balanced speed/accuracy; * and ** add polarisation for bonds "
        "and lone pairs.",
    ),
    "cc": (
        "Correlation-consistent",
        "Systematic convergence; best paired with correlated methods " "(MP2 / CCSD).",
    ),
    "def2": (
        "Karlsruhe (def2)",
        "Optimised for DFT; def2-SVP a solid default, def2-TZVP near "
        "complete-basis accuracy.",
    ),
}


def basis_family(basis: str) -> str:
    """Classify a basis-set name into an icon/copy family key."""
    if basis == "STO-3G":
        return "minimal"
    if "cc-pV" in basis:
        return "cc"
    if "def2" in basis:
        return "def2"
    # 3-21G and the whole 6-31G family are Pople split-valence sets.
    if basis.startswith("6-31") or basis == "3-21G" or basis.startswith("6-311"):
        return "pople"
    return "pople"


# ── Card HTML ────────────────────────────────────────────────────────────────


def _card_html(*, fg: str, bg: str, icon: str, title: str, body: str) -> str:
    """Assemble one descriptor card (icon + title + one-line body)."""
    return (
        f'<div style="display:flex;gap:10px;align-items:flex-start;'
        f"background:{bg};border-left:4px solid {fg};padding:8px 11px;"
        f'border-radius:4px;margin:6px 0;max-width:250px">'
        f'<span style="color:{fg};flex:0 0 auto;line-height:0;margin-top:1px">'
        f"{icon}</span>"
        f'<div style="min-width:0">'
        f'<div style="font-weight:700;font-size:12px;color:{fg};'
        f'letter-spacing:0.01em">{title}</div>'
        f'<div style="font-size:11.5px;color:#475569;line-height:1.35;'
        f'margin-top:2px">{body}</div>'
        f"</div></div>"
    )


def method_card_html(method: str) -> str:
    """Return the descriptor-card HTML for *method*."""
    info = config.METHOD_INFO.get(method)
    if not info:
        fg, bg, icon = _METHOD_FALLBACK_STYLE
        return _card_html(fg=fg, bg=bg, icon=icon, title=method, body="")
    fg, bg, icon = _METHOD_FAMILY_STYLE.get(
        info.get("type", ""), _METHOD_FALLBACK_STYLE
    )
    # The label is "NAME — Descriptor…"; the part after the em dash is a tidy
    # family descriptor ("DFT Hybrid Functional", "Restricted Hartree-Fock").
    label = info.get("label", method)
    suffix = label.split("—", 1)[1].strip() if "—" in label else ""
    title = f"{method} · {suffix}" if suffix else method
    body = info.get("use_for", info.get("description", ""))
    return _card_html(fg=fg, bg=bg, icon=icon, title=title, body=body)


def basis_card_html(basis: str) -> str:
    """Return the descriptor-card HTML for *basis*."""
    fam = basis_family(basis)
    fg, bg, icon = _BASIS_FAMILY_STYLE[fam]
    fam_label, body = _BASIS_COPY[fam]
    title = f"{basis} · {fam_label}"
    # Starred Pople sets have an equivalent parenthesis spelling that textbooks
    # use interchangeably (6-31G* == 6-31G(d)). Show it so the dropdown entry is
    # recognisable to someone who only knows the other form. Deliberately ONE
    # short line: these cards exist because the previous inline notes were "a lot
    # of word clutter" (FR-DESCRIPTOR-CARDS), so the full notation table lives in
    # the basis-set help topic instead.
    alias = config.pople_notation_alias(basis)
    if alias:
        body += f' <span style="color:#64748b">Also written <b>{alias}</b>.</span>'
    return _card_html(fg=fg, bg=bg, icon=icon, title=title, body=body)
