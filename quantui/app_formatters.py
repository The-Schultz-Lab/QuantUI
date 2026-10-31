"""Result-card HTML formatters used by QuantUIApp."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Optional

from quantui import theme as _theme


def _converged_color(converged: bool) -> str:
    return _theme.css.ACCENT_SUCCESS if converged else _theme.css.ACCENT_ERROR_ALT


def signed(x: Any, ndigits: int) -> str:
    """``{x:+.nf}`` without a negative zero ("−0.0000" for −1e-6)."""
    return f"{round(float(x), ndigits) + 0.0:+.{ndigits}f}"


# Elements written before H in a binary hydride: NH3, PH3, SiH4, BH3 (Hill
# order gives H3N, H3P, H4Si). Groups 16/17 keep H first: H2O, H2S, HCl.
_HYDRIDE_ELEMENT_FIRST = frozenset(
    {"B", "Al", "Ga", "C", "Si", "Ge", "Sn", "N", "P", "As", "Sb", "Bi"}
)


def display_formula(formula: Any) -> str:
    """Hill formula → the way chemists write it (``"H3N"`` → ``"NH3"``).

    Only binary hydrides of groups 13–15 are reordered; everything else keeps
    Hill order, which is already conventional (CH4, H2O, C6H6).
    """
    text = str(formula or "")
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", text)
    if "".join(e + n for e, n in parts) != text or len(parts) != 2:
        return text
    (e1, n1), (e2, n2) = parts
    if e1 == "H" and e2 in _HYDRIDE_ELEMENT_FIRST:
        return f"{e2}{n2}H{n1}"
    return text


def friendly_timestamp(ts: Any) -> str:
    """``"2026-10-25_04-31-58-600504"`` → ``"2026-10-25 04:31"``.

    Result folders are named by a microsecond timestamp; History, Compare and
    the Analysis heading showed it raw. Anything that doesn't
    parse is returned unchanged.
    """
    from datetime import datetime

    text = str(ts or "")
    for fmt in ("%Y-%m-%d_%H-%M-%S-%f", "%Y-%m-%d_%H-%M-%S"):
        try:
            return datetime.strptime(text[:26], fmt).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return text


def _method_basis_label(method: str, basis: str, scf_variant: str | None) -> str:
    """``"{method}/{basis}"``, with an explicit "(RKS)"/"(UKS)" suffix when
    the dispatched SCF variant differs from the plain method name.

    The restricted/unrestricted choice for a DFT functional (e.g. B3LYP) is
    fully automatic from multiplicity — correctly so, never a user toggle —
    but nothing in the results panel used to say which one actually ran. A
    student running Lab 2's six-metal series (5/6 open-shell) asked the
    instructor directly whether QuantUI even supported UKS.
    No suffix for "RHF"/"UHF" methods, where the method
    name already says it, or for a calc type/older saved result that never
    set ``scf_variant``.
    """
    label = f"{method}/{basis}"
    if scf_variant and scf_variant.upper() != method.upper():
        label += f" ({scf_variant})"
    return label


# Open-shell SCF variants whose orbitals split into separate alpha/beta
# channels (UHF/UKS: two independent sets; ROHF: a single spatial-orbital
# set but singly-occupied orbitals that still only have a well-defined
# alpha-channel HOMO/LUMO in the usual sense).
_OPEN_SHELL_SCF_VARIANTS = frozenset({"UHF", "UKS", "ROHF", "ROKS"})


def _homo_lumo_gap_label(scf_variant: str | None) -> str:
    """ "HOMO-LUMO gap", qualified "(α)" for an open-shell reference.

    session_calc.py's gap extraction always
    reads spin channel 0 (alpha) for a 2-D ``mo_energy`` array (its own
    comment: "UHF: ... use alpha spin for the gap estimate"). The result
    card never said so, presenting a single-channel number as if it were
    an unqualified property — silently dropping the beta-channel gap,
    which can differ meaningfully for an open-shell system.
    """
    if scf_variant and scf_variant.upper() in _OPEN_SHELL_SCF_VARIANTS:
        return "HOMO-LUMO gap (α)"
    return "HOMO-LUMO gap"


def _result_card_open(*, accent: str | None = None, extra_style: str = "") -> str:
    border = accent or _theme.css.ACCENT_SUCCESS_ALT
    style = (
        f"background:{_theme.css.BG_PANEL};border:1px solid {_theme.css.BORDER};"
        f"border-left:4px solid {border};padding:10px 14px;border-radius:4px;"
        f"margin:6px 0"
    )
    if extra_style:
        style = f"{style};{extra_style}"
    return f'<div class="quantui-result-card" style="{style}">'


_RESULT_CARD_CLOSE = "</div>"


def _result_card_table_open() -> str:
    return (
        '<table class="quantui-result-card-table" '
        'style="margin-top:8px;font-size:14px;border-collapse:collapse;width:auto">'
    )


def _s2_html(s2: Any, multiplicity: Any) -> str:
    """'0.7533 (ideal 0.7500)' plus a warning when spin-contaminated."""
    from quantui.reorganization_energy import ideal_s2, spin_contaminated

    text = f"{float(s2):.4f}"
    if not multiplicity:
        return text
    mult = int(multiplicity)
    text += f" (ideal {ideal_s2(mult):.4f})"
    if spin_contaminated(s2, mult):
        text += (
            f' <span style="color:{_theme.css.ACCENT_WARNING}">'
            "⚠ spin contaminated: energies may be unreliable</span>"
        )
    return text


def _result_extra_rows(get: Any) -> str:
    """Build the shared 'extra' result-card rows from an accessor.

    ``get(key, default=None)`` reads a field from either a result object
    (``getattr``) or a saved ``result.json`` dict (``dict.get``). Used by BOTH
    :func:`format_result` (live) and :func:`format_past_result` (history) so the
    two cards can never drift again — a past regression had the compute
    device / dipole / Mulliken rows existed only on the live card. Rows:
    post-HF correlation breakdown (MP2 / CCSD / (T)), solvent, compute device
    (always shown), dipole moment, Mulliken charges.
    """

    def _num(label: str, value: str) -> str:
        return (
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">{label}</td>'
            f'<td style="color:{_theme.css.TEXT_HEADING}">{value}</td></tr>'
        )

    rows = ""
    energy = get("energy_hartree", 0.0)

    _engine_id = str(get("engine_id", "pyscf") or "pyscf")
    rows += _num("Quantum engine", "PyFock" if _engine_id == "pyfock" else "PySCF")

    # Post-HF correlation breakdown — energy_hartree already includes every
    # contribution, so the HF reference is the total minus those.
    _mp2 = get("mp2_correlation_hartree")
    if _mp2 is not None:
        rows += _num("HF reference", f"{energy - _mp2:.8f} Ha")
        rows += _num("MP2 correlation", f"{_mp2:.8f} Ha")
    _ccsd = get("ccsd_correlation_hartree")
    _ccsd_t = get("ccsd_t_correction_hartree")
    if _ccsd is not None:
        rows += _num("HF reference", f"{energy - _ccsd - (_ccsd_t or 0.0):.8f} Ha")
        rows += _num("CCSD correlation", f"{_ccsd:.8f} Ha")
        if _ccsd_t is not None:
            rows += _num("(T) triples correction", f"{_ccsd_t:.8f} Ha")

    _solvent = get("solvent")
    if _solvent is not None:
        rows += _num("Solvent (PCM)", str(_solvent))

    _s2 = get("spin_square")
    if _s2 is not None:
        _mult = get("multiplicity") or (get("geometry") or {}).get("multiplicity")
        rows += _num("⟨S²⟩", _s2_html(_s2, _mult))

    _pg = _point_group_html(get)
    if _pg:
        rows += _num("Point group", _pg)

    # Compute device — always shown; old saved results lack the
    # field and safely read "CPU".
    if bool(get("gpu_used", False)):
        _name = get("gpu_name")
        _device = (
            f'<span style="color:{_theme.css.ACCENT_SUCCESS}">🚀 GPU</span>'
            f' &mdash; <span style="font-family:monospace">{_name}</span>'
            if _name
            else f'<span style="color:{_theme.css.ACCENT_SUCCESS}">🚀 GPU</span>'
        )
    else:
        _device = f'<span style="color:{_theme.css.TEXT_SECONDARY}">CPU</span>'
    rows += (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Compute device</td>'
        f"<td>{_device}</td></tr>"
    )

    # Density fitting — shown only when it was applied (the non-default case
    # worth flagging to the student); exact-integral runs stay uncluttered, and
    # result types without the field safely omit it.
    if bool(get("density_fit", False)):
        rows += (
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Density fitting</td>'
            '<td><span style="color:#0369a1">⚡ RI</span> '
            f'<span style="color:{_theme.css.TEXT_MUTED_LIGHT};font-size:12px">'
            "(approximate 2-electron integrals)</span></td></tr>"
        )

    # session_calc.py extracts both properties
    # from ``mf`` (the HF/DFT reference) even for MP2/CCSD/CCSD(T), which
    # never builds a correlated density here. That is a real, potentially
    # intentional teaching simplification (a correlated dipole/population
    # needs a relaxed/unrelaxed density from the post-HF method itself,
    # which this app does not compute) — but the card must say so instead
    # of presenting an HF-level dipole/population as if it came from the
    # requested correlated method.
    _is_post_hf = _mp2 is not None or _ccsd is not None
    _post_hf_note = (
        f' <span style="color:{_theme.css.TEXT_MUTED_LIGHT};font-size:12px">'
        "(HF reference — not a correlated MP2/CCSD property)</span>"
        if _is_post_hf
        else ""
    )

    _dip = get("dipole_moment_debye")
    if _dip is not None:
        _vec = get("dipole_vector_debye")
        if _vec is not None and len(_vec) >= 3:
            _dip_str = (
                f"{_dip:.4f} D"
                f' <span style="color:{_theme.css.TEXT_MUTED_LIGHT};font-size:12px">'
                f"(μ = [{signed(_vec[0], 3)}, {signed(_vec[1], 3)}, "
                f"{signed(_vec[2], 3)}] D)</span>"
            )
        else:
            _dip_str = (
                f"{_dip:.4f} D"
                f' <span style="color:{_theme.css.TEXT_MUTED_LIGHT};font-size:12px">'
                "(magnitude only — μ components not saved)</span>"
            )
        rows += _num("Dipole moment", _dip_str + _post_hf_note)

    _chg = get("mulliken_charges")
    _syms = get("atom_symbols")
    if _chg is not None and _syms is not None:
        _charge_str = "  ".join(f"{sym}:{c:+.3f}" for sym, c in zip(_syms, _chg))
        rows += (
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL};vertical-align:top">'
            f"Mulliken charges</td>"
            f'<td style="color:{_theme.css.TEXT_HEADING};font-family:monospace;font-size:12px;'
            f'word-break:break-all">{_charge_str}{_post_hf_note}</td></tr>'
        )
    return rows


def format_result(r: Any) -> str:
    """Format a single-point-style result card."""
    # For CCSD/CCSD(T), r.converged also folds in the CC
    # amplitude solve's own convergence, so a bare "SCF converged" label
    # would misleadingly blame the reference SCF for a CC-only failure.
    _conv_label = (
        "Converged" if getattr(r, "cc_converged", None) is not None else "SCF converged"
    )
    _conv = "Yes" if r.converged else "No (treat results with caution)"
    _cc = _converged_color(r.converged)
    _gap = f"{r.homo_lumo_gap_ev:.4f} eV" if r.homo_lumo_gap_ev is not None else "N/A"
    _rows = "".join(
        f"<tr>"
        f'<td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">{k}</td>'
        f'<td style="color:{vc}">{v}</td>'
        f"</tr>"
        for k, v, vc in [
            (
                "Total energy",
                f"{r.energy_hartree:.8f} Ha &ensp;({r.energy_ev:.4f} eV)",
                _theme.css.TEXT_HEADING,
            ),
            (
                _homo_lumo_gap_label(getattr(r, "scf_variant", None)),
                _gap,
                _theme.css.TEXT_HEADING,
            ),
            (_conv_label, _conv, _cc),
            (
                "SCF iterations",
                (
                    "—"
                    if getattr(r, "n_iterations", None) in (None, -1)
                    else str(r.n_iterations)
                ),
                _theme.css.TEXT_HEADING,
            ),
        ]
    )
    _extra = _result_extra_rows(lambda k, d=None: getattr(r, k, d))
    return (
        _result_card_open() + f"<b>{display_formula(r.formula)} &mdash; "
        f"{_method_basis_label(r.method, r.basis, getattr(r, 'scf_variant', None))}</b>"
        + _result_card_table_open()
        + f"{_rows}{_extra}</table>"
        + _RESULT_CARD_CLOSE
    )


def format_opt_result(r: Any) -> str:
    """Format a geometry-optimization result card."""
    _conv = "Yes" if r.converged else "No (max steps reached)"
    _cc = _converged_color(r.converged)
    _rows = "".join(
        f"<tr>"
        f'<td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">{k}</td>'
        f'<td style="color:{vc}">{v}</td>'
        f"</tr>"
        for k, v, vc in [
            ("Final energy", f"{r.energy_hartree:.8f} Ha", _theme.css.TEXT_HEADING),
            (
                "Energy change",
                f"{r.energy_change_hartree:+.6f} Ha",
                _theme.css.TEXT_HEADING,
            ),
            ("Opt converged", _conv, _cc),
            ("Steps taken", str(r.n_steps), _theme.css.TEXT_HEADING),
            ("Geometry RMSD", f"{r.rmsd_angstrom:.4f} Å", _theme.css.TEXT_HEADING),
        ]
    )
    return (
        _result_card_open()
        + f"<b>Geometry Optimisation &mdash; {display_formula(r.formula)} ({r.method}/{r.basis})</b>"
        + _result_card_table_open()
        + f"{_rows}"
        + _solvent_row(getattr(r, "solvent", None), "optimized in solvent")
        + _frozen_row(getattr(r, "frozen_atoms", None))
        + _point_group_row(_opt_point_group_html(r))
        + "</table>"
        + _RESULT_CARD_CLOSE
    )


def _point_group_html(get: Any) -> str:
    """Point group of the result's geometry as HTML, or ''.

    Reads ``pyscf_mol_atom`` (live results) or the saved frequency
    ``spectra.molecule`` block; other saved results carry no geometry here.
    """
    try:
        from quantui.symmetry import detect_point_group, point_group_of_atom_list

        mol_atom = get("pyscf_mol_atom")
        if mol_atom:
            pg = point_group_of_atom_list(mol_atom)
        else:
            spectra = get("spectra") or {}
            mol = spectra.get("molecule") if isinstance(spectra, dict) else None
            if not mol or not mol.get("atoms"):
                return ""
            pg = detect_point_group(mol["atoms"], mol.get("coords") or [])
        return pg.summary_html() if pg is not None else ""
    except Exception:  # noqa: BLE001 — informational row only
        return ""


def _frozen_row(frozen: Any) -> str:
    if not frozen:
        return ""
    nums = ", ".join(str(int(i) + 1) for i in frozen)
    return (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Frozen atoms</td>'
        f'<td style="color:{_theme.css.TEXT_HEADING}">{nums}'
        f'<span style="color:{_theme.css.TEXT_MUTED}"> &mdash; held fixed '
        "(constrained optimization)</span></td></tr>"
    )


def _opt_point_group_html(r: Any) -> str:
    """Point group of an optimization's final geometry ('' if unknown)."""
    try:
        from quantui.symmetry import point_group_of_molecule

        mol = getattr(r, "molecule", None)
        pg = point_group_of_molecule(mol) if mol is not None else None
        return pg.summary_html() if pg is not None else ""
    except Exception:  # noqa: BLE001 — informational row only
        return ""


def _point_group_row(pg_html: str) -> str:
    if not pg_html:
        return ""
    return (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Point group</td>'
        f'<td style="color:{_theme.css.TEXT_HEADING}">{pg_html}</td></tr>'
    )


def _solvent_row(solvent: Any, note: str) -> str:
    """'Solvent (PCM)' card row for a solvated result; '' for gas phase."""
    if not solvent:
        return ""
    return (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Solvent (PCM)</td>'
        f'<td style="color:{_theme.css.TEXT_HEADING}">{solvent}'
        f'<span style="color:{_theme.css.TEXT_MUTED}"> &mdash; {note}</span></td></tr>'
    )


def saddle_retry_rows(block: Optional[dict]) -> str:
    """Result-card row for a saddle-point retry, live or from History.

    *block* is ``FreqResult.imaginary_mode_retry`` /
    ``result.json`` ``spectra.ir.imaginary_mode_retry``.
    """
    if not block:
        return ""
    followed = ", ".join(
        f"{abs(float(f)):.1f}i" for f in block.get("followed_cm1") or []
    )
    remaining = block.get("remaining_imaginary_cm1") or []
    outcome = (
        "still imaginary: "
        + ", ".join(f"{abs(float(f)):.1f}i" for f in remaining)
        + " cm⁻¹"
        if remaining
        else "none left"
    )
    return (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Saddle-point retry</td>'
        f'<td style="color:{_theme.css.TEXT_HEADING}">first pass had {followed} cm⁻¹; '
        f"displaced, re-optimized ({block.get('reopt_steps', '?')} steps) "
        f"and recomputed &mdash; {outcome}</td></tr>"
    )


def _thermo_card_rows(
    *,
    temperature_k: float,
    pressure_atm: float,
    zpve_hartree: Optional[float],
    h_hartree: float,
    s_jmol: float,
    g_hartree: float,
    energy_hartree: Optional[float],
) -> str:
    """Thermochemistry rows for the Frequency card, live and History.

    H and G are shown in hartree with their thermal correction relative to the
    electronic energy in kJ/mol. The old absolute G in kJ/mol (e.g. −200595
    kJ/mol) is not a meaningful number on its own.
    """
    _kj = 2625.4996  # kJ/mol per hartree
    label = f'style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}"'
    value = f'style="color:{_theme.css.TEXT_HEADING}"'

    def _corr(x: float) -> str:
        if energy_hartree is None:
            return ""
        return (
            f'<span style="color:{_theme.css.TEXT_MUTED}"> &nbsp;'
            f"({(x - float(energy_hartree)) * _kj:+.2f} kJ/mol vs E)</span>"
        )

    rows = (
        f'<tr><td colspan="2" style="padding:6px 0 2px 0;color:{_theme.css.TEXT_MUTED};'
        f'font-size:12px;font-style:italic">'
        f"&#8212; Thermochemistry at {float(temperature_k):.0f} K / "
        f"{float(pressure_atm):g} atm &#8212;</td></tr>"
    )
    if zpve_hartree is not None:
        rows += f"<tr><td {label}>ZPVE</td><td {value}>{float(zpve_hartree):.6f} Ha</td></tr>"
    rows += (
        f"<tr><td {label}>H</td><td {value}>{float(h_hartree):.6f} Ha"
        f"{_corr(float(h_hartree))}</td></tr>"
        f"<tr><td {label}>S</td><td {value}>{float(s_jmol):.2f} J/(mol&middot;K)</td></tr>"
        f"<tr><td {label}>G</td><td {value}>{float(g_hartree):.6f} Ha"
        f"{_corr(float(g_hartree))}</td></tr>"
    )
    return rows


def stationary_point_rows(max_force: Optional[float]) -> str:
    """Result-card rows for the input-geometry force check, live or History.

    Harmonic frequencies and thermochemistry assume a stationary point (zero
    gradient). Nothing is shown for results saved before the check existed.
    """
    from quantui.freq_calc import STATIONARY_FMAX_EV_PER_ANG, is_stationary

    ok = is_stationary(max_force)
    if ok is None:
        return ""
    _color = _theme.css.TEXT_HEADING if ok else _theme.css.ACCENT_ERROR_ALT
    rows = (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Max force</td>'
        f'<td style="color:{_color}">{max_force:.4f} eV/Å'
        f'<span style="color:{_theme.css.TEXT_MUTED}"> &mdash; '
        f"{'stationary point' if ok else 'not a stationary point'}"
        f" (threshold {STATIONARY_FMAX_EV_PER_ANG:g})</span></td></tr>"
    )
    if not ok:
        rows += (
            f'<tr><td colspan="2" style="padding:4px 0;color:{_theme.css.ACCENT_WARNING}">'
            "&#9888; Not a minimum at this level of theory, so the values below "
            "aren't meaningful. Tick <b>Geometry optimization before "
            "calculation</b> and re-run.</td></tr>"
        )
    return rows


def _freq_section_rows(get: Any) -> str:
    """Real/imaginary mode counts and the frequency list (live and History).

    ``get`` reads ``frequencies_cm1`` / ``displacements`` / ``pyscf_mol_atom``
    from a live result or from a saved result's ``spectra`` blocks.
    """
    from types import SimpleNamespace

    freqs = [float(f) for f in (get("frequencies_cm1") or [])]
    if not freqs:
        return ""
    n_real = sum(1 for f in freqs if f > 0)
    n_imag = sum(1 for f in freqs if f < 0)
    lbl = f'style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}"'
    val = f'style="color:{_theme.css.TEXT_HEADING}"'
    rows = f"<tr><td {lbl}>Real modes</td><td {val}>{n_real}</td></tr>"
    if n_imag:
        rows += (
            f"<tr><td {lbl}>Imaginary modes</td>"
            f'<td style="color:{_theme.css.ACCENT_ERROR_ALT}">{n_imag} — geometry may '
            "not be a minimum</td></tr>"
        )
    listing = _frequency_list_html(
        SimpleNamespace(
            frequencies_cm1=freqs,
            displacements=get("displacements"),
            pyscf_mol_atom=get("pyscf_mol_atom"),
        )
    )
    if listing:
        rows += f"<tr><td {lbl}>Frequencies (cm⁻¹)</td><td {val}>{listing}</td></tr>"
    return rows


def _saved_freq_accessor(data: dict[str, Any]) -> Any:
    """``get`` for :func:`_freq_section_rows` over a saved result.json."""
    spectra = data.get("spectra") or {}
    ir = spectra.get("ir") or {}
    mol = spectra.get("molecule") or {}
    mol_atom = (
        list(zip(mol.get("atoms") or [], mol.get("coords") or []))
        if mol.get("atoms")
        else None
    )
    values = {
        "frequencies_cm1": ir.get("frequencies_cm1"),
        "displacements": ir.get("displacements"),
        "pyscf_mol_atom": mol_atom,
    }
    return lambda k, d=None: values.get(k, d)


def _basic_rows(get: Any, conv_label: str, conv_text: str, conv_color: str) -> str:
    """Energy / gap / convergence / iterations — first rows of every card."""
    energy = get("energy_hartree")
    gap = get("homo_lumo_gap_ev")
    n_iter = get("n_iterations")
    rows = [
        (
            "Total energy",
            f"{energy:.8f} Ha &ensp;({energy * 27.211386245988:.4f} eV)",
            _theme.css.TEXT_HEADING,
        ),
        (
            _homo_lumo_gap_label(get("scf_variant")),
            f"{gap:.4f} eV" if gap is not None else "N/A",
            _theme.css.TEXT_HEADING,
        ),
        (conv_label, conv_text, conv_color),
        (
            "SCF iterations",
            "—" if n_iter in (None, -1) else str(n_iter),
            _theme.css.TEXT_HEADING,
        ),
    ]
    return "".join(
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">{k}</td>'
        f'<td style="color:{vc}">{v}</td></tr>'
        for k, v, vc in rows
    )


def _frequency_list_html(r: Any, *, limit: int = 9) -> str:
    """``1797 (a<sub>1</sub>), 4349 (a<sub>1</sub>), 4673 (b<sub>2</sub>)``.

    Real modes in ascending order, each with its symmetry label
    when the displacements and geometry are on the result.
    """
    freqs = [float(f) for f in (getattr(r, "frequencies_cm1", None) or [])]
    labels: list = [None] * len(freqs)
    disps = getattr(r, "displacements", None)
    mol_atom = getattr(r, "pyscf_mol_atom", None)
    if disps is not None and mol_atom:
        try:
            from quantui.symmetry import label_mode_irreps

            irreps = label_mode_irreps(mol_atom, freqs, disps)
            if irreps is not None:
                labels = list(irreps.labels)
        except Exception:  # noqa: BLE001 — labels are optional
            pass
    real = sorted((f, lbl) for f, lbl in zip(freqs, labels) if f > 0)
    if not real:
        return ""
    from quantui.symmetry import html_irrep

    # A degenerate set is listed once: "2020 (e) ×2".
    groups: list = []
    for f, lbl in real:
        if groups and lbl and groups[-1][1] == lbl and abs(groups[-1][0] - f) < 1.0:
            groups[-1][2] += 1
        else:
            groups.append([f, lbl, 1])
    items = [
        f"{f:.0f}"
        + (f" ({html_irrep(lbl)})" if lbl else "")
        + (f" &times;{n}" if n > 1 else "")
        for f, lbl, n in groups[:limit]
    ]
    more = sum(n for _, _, n in groups[limit:])
    tail = (
        f' <span style="color:{_theme.css.TEXT_MUTED}">+{more} more in the mode table</span>'
        if more > 0
        else ""
    )
    return ", ".join(items) + tail


def format_freq_result(r: Any) -> str:
    """Format a frequency-analysis result card.

    Same rows, in the same order, as the History card for the result
    (:func:`format_past_result`) — they used to show different fields.
    """
    # r.converged now also requires the Hessian/harmonic-
    # analysis step to have completed, not just the reference SCF, so the
    # row is labeled/colored on overall status rather than "SCF converged".
    _conv = "Yes" if r.converged else "No (treat with caution)"
    _cc = _converged_color(r.converged)

    def get(k: str, d: Any = None) -> Any:
        return getattr(r, k, d)

    _rows = (
        stationary_point_rows(get("max_force_ev_per_ang"))
        + _basic_rows(get, "Converged", _conv, _cc)
        + _freq_section_rows(get)
        + saddle_retry_rows(get("imaginary_mode_retry"))
        + _result_extra_rows(get)
    )
    _thermo_rows = ""
    _thermo = getattr(r, "thermo", None)
    if _thermo is None:
        _thermo_rows = (
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">ZPVE</td>'
            f'<td style="color:{_theme.css.TEXT_HEADING}">{r.zpve_hartree:.6f} Ha</td></tr>'
        )
    else:
        _thermo_rows = _thermo_card_rows(
            temperature_k=_thermo.temperature_k,
            pressure_atm=getattr(_thermo, "pressure_atm", 1.0),
            zpve_hartree=r.zpve_hartree,
            h_hartree=_thermo.H_hartree,
            s_jmol=_thermo.S_jmol,
            g_hartree=_thermo.G_hartree,
            energy_hartree=r.energy_hartree,
        )
    return (
        _result_card_open()
        + f"<b>Frequency Analysis &mdash; {display_formula(r.formula)} &middot; "
        f"{_method_basis_label(r.method, r.basis, getattr(r, 'scf_variant', None))}</b>"
        + _result_card_table_open()
        + f"{_rows}{_thermo_rows}</table>"
        + _RESULT_CARD_CLOSE
    )


def ts_verdict_rows(block: Optional[dict]) -> str:
    """Result-card rows for a transition-state search.

    *block* is the ``transition_state`` spectra entry
    (:meth:`quantui.ts_search.TSResult.to_spectra`), so the live and History
    cards render the same thing.
    """
    if not block:
        return ""
    n = block.get("n_imaginary")
    ok = n == 1 and bool(block.get("search_converged"))
    color = _theme.css.ACCENT_SUCCESS if ok else _theme.css.ACCENT_WARNING
    icon = "✓" if ok else "⚠"
    imag = block.get("imaginary_cm1") or []
    imag_txt = (
        ", ".join(f"{abs(float(f)):.0f}i" for f in imag) + " cm⁻¹" if imag else "none"
    )
    steps = block.get("n_steps")
    search = (
        f"{steps} steps, "
        f"{'converged' if block.get('search_converged') else 'NOT converged'}"
    )

    def _row(label: str, value: str, vc: str) -> str:
        return (
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL};'
            f'vertical-align:top">{label}</td>'
            f'<td style="color:{vc}">{value}</td></tr>'
        )

    return (
        _row("Verdict", f"{icon} {html.escape(str(block.get('verdict', '')))}", color)
        + _row("Imaginary frequencies", imag_txt, _theme.css.TEXT_HEADING)
        + _row("Saddle-point search", search, _theme.css.TEXT_HEADING)
    )


def format_ts_result(r: Any) -> str:
    """Format a transition-state search result card."""
    _rows = f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Energy</td>' f'<td style="color:{_theme.css.TEXT_HEADING}">{r.energy_hartree:.8f} Ha</td></tr>' + ts_verdict_rows(
        r.to_spectra()
    ) + _solvent_row(
        getattr(r, "solvent", None), "search and frequencies in solvent"
    ) + _point_group_row(
        _point_group_html(lambda k, d=None: getattr(r, k, d))
    )
    return (
        _result_card_open()
        + f"<b>Transition State &mdash; {display_formula(r.formula)} "
        f"({_method_basis_label(r.method, r.basis, getattr(r, 'scf_variant', None))})</b>"
        + _result_card_table_open()
        + f"{_rows}</table>"
        + _RESULT_CARD_CLOSE
    )


def irc_rows(block: Optional[dict]) -> str:
    """Result-card rows for an IRC (the ``irc`` spectra block, live or saved)."""
    if not block:
        return ""
    fwd, rev = block.get("forward") or {}, block.get("reverse") or {}
    ok = bool(fwd.get("converged")) and bool(rev.get("converged"))
    b_start = block.get("barrier_from_start_kj")
    b_end = block.get("barrier_from_end_kj")
    downhill = all(isinstance(b, (int, float)) and b > 0.5 for b in (b_start, b_end))
    color = _theme.css.ACCENT_SUCCESS if ok and downhill else _theme.css.ACCENT_WARNING
    icon = "✓" if ok and downhill else "⚠"

    def _row(label: str, value: str, vc: str) -> str:
        return (
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL};'
            f'vertical-align:top">{label}</td>'
            f'<td style="color:{vc}">{value}</td></tr>'
        )

    def _kj(v: Any) -> str:
        return f"{float(v):.1f} kJ/mol" if isinstance(v, (int, float)) else "n/a"

    def _side(d: dict) -> str:
        n = d.get("n_steps", "?")
        tail = (
            f" ({d['irc_steps']} on the path, then minimized)"
            if d.get("irc_steps") is not None
            else ""
        )
        return f"{n} steps{tail}, " + (
            "reached a minimum" if d.get("converged") else "NOT converged"
        )

    return (
        _row("Verdict", f"{icon} {html.escape(str(block.get('verdict', '')))}", color)
        + _row(
            "Barrier",
            f"{_kj(b_start)} from the start of the path, {_kj(b_end)} from its end",
            _theme.css.TEXT_HEADING,
        )
        + _row("Path, reverse side", _side(rev), _theme.css.TEXT_HEADING)
        + _row("Path, forward side", _side(fwd), _theme.css.TEXT_HEADING)
    )


def format_irc_result(r: Any) -> str:
    """Format an IRC result card."""
    _rows = (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">'
        "Transition-state energy</td>"
        f'<td style="color:{_theme.css.TEXT_HEADING}">{r.energy_hartree:.8f} Ha</td></tr>'
        + irc_rows(r.to_spectra())
        + _solvent_row(getattr(r, "solvent", None), "every step in solvent")
    )
    return (
        _result_card_open() + f"<b>IRC &mdash; {display_formula(r.formula)} "
        f"({_method_basis_label(r.method, r.basis, None)})</b>"
        + _result_card_table_open()
        + f"{_rows}</table>"
        + _RESULT_CARD_CLOSE
    )


def format_tddft_result(r: Any) -> str:
    """Format a TD-DFT / UV-Vis result card."""
    # r.converged now folds in per-root TD convergence, so the
    # row is labeled/colored on overall status, not just the ground SCF.
    _conv = "Yes" if r.converged else "No (treat with caution)"
    _cc = _converged_color(r.converged)
    _n_converged = getattr(r, "n_converged_states", None)
    _states_detail = str(len(r.excitation_energies_ev))
    if _n_converged is not None and _n_converged != len(r.excitation_energies_ev):
        _states_detail += f" ({_n_converged} converged)"
    header_rows = (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Ground-state energy</td>'
        f'<td style="color:{_theme.css.TEXT_HEADING}">{r.energy_hartree:.8f} Ha</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">States computed</td>'
        f'<td style="color:{_theme.css.TEXT_HEADING}">{_states_detail}</td></tr>'
        + _solvent_row(getattr(r, "solvent", None), "non-equilibrium excitations")
    )
    exc_table = ""
    if r.excitation_energies_ev:
        wl = r.wavelengths_nm()
        exc_rows = []
        for i, (e_ev, f_osc) in enumerate(
            zip(r.excitation_energies_ev[:8], r.oscillator_strengths[:8]), 1
        ):
            bold = "font-weight:bold" if f_osc > 0.05 else ""
            exc_rows.append(
                f'<tr style="{bold}">'
                f'<td style="padding:2px 12px 2px 0;color:{_theme.css.TEXT_SECONDARY}">S{i}</td>'
                f'<td style="padding:2px 12px 2px 0;color:{_theme.css.TEXT_HEADING}">{e_ev:.3f} eV</td>'
                f'<td style="padding:2px 12px 2px 0;color:{_theme.css.TEXT_HEADING}">{wl[i - 1]:.1f} nm</td>'
                f'<td style="padding:2px 4px 2px 0;color:{_theme.css.TEXT_HEADING}">f = {f_osc:.4f}</td>'
                f"</tr>"
            )
        if len(r.excitation_energies_ev) > 8:
            exc_rows.append(
                f'<tr><td colspan="4" style="color:{_theme.css.TEXT_FAINT};font-size:12px">… '
                f"and {len(r.excitation_energies_ev) - 8} more states</td></tr>"
            )
        exc_table = (
            f'<tr><td colspan="2" style="padding:8px 0 2px;color:{_theme.css.TEXT_LABEL};font-weight:bold">'
            "Vertical excitations:</td></tr>"
            "<tr>"
            f'<th style="text-align:left;color:{_theme.css.TEXT_SECONDARY};font-size:12px;padding:2px 12px 2px 0">State</th>'
            f'<th style="text-align:left;color:{_theme.css.TEXT_SECONDARY};font-size:12px;padding:2px 12px 2px 0">Energy</th>'
            f'<th style="text-align:left;color:{_theme.css.TEXT_SECONDARY};font-size:12px;padding:2px 12px 2px 0">λ</th>'
            f'<th style="text-align:left;color:{_theme.css.TEXT_SECONDARY};font-size:12px">Osc. str.</th></tr>'
            + "".join(exc_rows)
        )
    return (
        _result_card_open()
        + f"<b>TD-DFT / UV-Vis &mdash; {display_formula(r.formula)} "
        f"({_method_basis_label(r.method, r.basis, getattr(r, 'scf_variant', None))})</b>"
        + _result_card_table_open()
        + f"{header_rows}{exc_table}</table>"
        + _RESULT_CARD_CLOSE
    )


def format_nmr_result(r: Any) -> str:
    """Format an NMR shielding result card."""
    _conv = "Yes" if r.converged else "No (treat with caution)"
    _cc = _converged_color(r.converged)
    header_rows = (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">SCF converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Reference</td>'
        f'<td style="color:{_theme.css.TEXT_HEADING}">{r.reference_compound} ({r.method}/{r.basis})</td></tr>'
    )
    # Surface a fallback-reference substitution explicitly
    # rather than letting the row above imply an exact-match reference.
    if getattr(r, "is_fallback_reference", False):
        _ref_key = getattr(r, "reference_key", "") or "a different level of theory"
        header_rows += (
            f'<tr><td colspan="2" style="padding:3px 0 0">'
            f'<span style="color:{_theme.css.ACCENT_WARNING};font-size:12px">'
            f"⚠ No reference at {r.method}/{r.basis} — shifts use {_ref_key} "
            "constants instead.</span></td></tr>"
        )

    def _nmr_table(label: str, shifts: list, sym: str) -> str:
        if not shifts:
            return ""
        rows = "".join(
            f"<tr>"
            f'<td style="padding:2px 14px 2px 0;color:{_theme.css.TEXT_SECONDARY}">{sym}-{n}</td>'
            f'<td style="color:{_theme.css.TEXT_HEADING}">{d:.2f} ppm</td>'
            f"</tr>"
            for n, (_i, d) in enumerate(shifts, 1)
        )
        return (
            f'<tr><td colspan="2" style="padding:8px 0 2px;color:{_theme.css.TEXT_LABEL};font-weight:bold">'
            f"{label} shifts (vs. TMS):</td></tr>"
            f"<tr>"
            f'<th style="text-align:left;color:{_theme.css.TEXT_SECONDARY};font-size:12px;padding:2px 14px 2px 0">Atom</th>'
            f'<th style="text-align:left;color:{_theme.css.TEXT_SECONDARY};font-size:12px">δ (ppm)</th></tr>'
            + rows
        )

    h_table = _nmr_table("¹H", r.h_shifts(), "H")
    c_table = _nmr_table("¹³C", r.c_shifts(), "C")

    _basis_warn = ""
    if r.basis.upper() in ("STO-3G", "3-21G"):
        _basis_warn = (
            '<tr><td colspan="2" style="padding:6px 0 0">'
            f'<span style="color:{_theme.css.ACCENT_WARNING};font-size:12px">'
            f"⚠ {r.basis} gives qualitative NMR only — use 6-31G* or better.</span>"
            "</td></tr>"
        )

    # The reference shielding constants table only
    # covers a handful of method/basis combinations; any other combination
    # silently substitutes the B3LYP/6-31G* constants, which can shift the
    # reported ppm values by several ppm relative to a properly calibrated
    # reference. Surface that substitution rather than let it pass silently.
    _ref_warn = ""
    if getattr(r, "is_fallback_reference", False):
        _ref_warn = (
            '<tr><td colspan="2" style="padding:6px 0 0">'
            f'<span style="color:{_theme.css.ACCENT_WARNING};font-size:12px">'
            f"⚠ No calibrated TMS reference for {r.method}/{r.basis} — using "
            f"{getattr(r, 'reference_key', 'B3LYP/6-31G*')} constants instead. "
            "Shifts may be off by a few ppm.</span>"
            "</td></tr>"
        )

    _empty = ""
    if not r.h_shifts() and not r.c_shifts():
        _empty = (
            f'<tr><td colspan="2" style="color:{_theme.css.TEXT_FAINT};font-size:12px">'
            "No ¹H or ¹³C atoms found in this molecule.</td></tr>"
        )

    return (
        _result_card_open() + f"<b>NMR Shielding &mdash; {display_formula(r.formula)} "
        f"({_method_basis_label(r.method, r.basis, getattr(r, 'scf_variant', None))})</b>"
        + _result_card_table_open()
        + f"{header_rows}{h_table}{c_table}{_empty}{_basis_warn}{_ref_warn}</table>"
        + _RESULT_CARD_CLOSE
    )


def format_pes_scan_result(r: Any) -> str:
    """Format a PESScanResult as an HTML result card."""
    import math

    from quantui.measurement import atom_label

    n_fail = sum(1 for e in r.energies_hartree if not math.isfinite(e))
    _conv = (
        "Yes"
        if r.converged_all
        else f"No ({n_fail} failed point{'s' if n_fail != 1 else ''})"
    )
    _cc = _converged_color(r.converged_all)

    finite = [e for e in r.energies_hartree if math.isfinite(e)]
    _e_row = ""
    if finite:
        e_min = min(finite)
        e_max = max(finite)
        barrier_kcal = (e_max - e_min) * 627.509474
        min_i = r.energies_hartree.index(e_min)
        min_coord = r.scan_parameter_values[min_i]
        _e_row = (
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Min energy</td>'
            f'<td style="color:{_theme.css.TEXT_HEADING}">{e_min:.8f} Ha '
            f"(point {min_i + 1}, {min_coord:.3f} {r.scan_unit})</td></tr>"
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Energy range</td>'
            f'<td style="color:{_theme.css.TEXT_HEADING}">{barrier_kcal:.2f} kcal/mol</td></tr>'
        )

    mol = r.coordinates_list[0] if getattr(r, "coordinates_list", None) else None
    if mol is not None:
        _idx_str = " – ".join(atom_label(mol, i) for i in r.atom_indices)
    else:
        _idx_str = " – ".join(str(i + 1) for i in r.atom_indices)

    return (
        _result_card_open()
        + f"<b>PES Scan &mdash; {display_formula(r.formula)} ({r.method}/{r.basis})</b>"
        + _result_card_table_open()
        + f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Scan type</td>'
        f'<td style="color:{_theme.css.TEXT_HEADING}">{r.scan_type.capitalize()} ({_idx_str})</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Range</td>'
        f'<td style="color:{_theme.css.TEXT_HEADING}">{r.scan_parameter_values[0]:.3f} → '
        f"{r.scan_parameter_values[-1]:.3f} {r.scan_unit} "
        f"({r.n_steps} points)</td></tr>"
        f"{_e_row}"
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">All converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f"</table>" + _RESULT_CARD_CLOSE
    )


_HARTREE_TO_EV = 27.211386245988
_HARTREE_TO_KCAL = 627.509474


def _attach_relaxation_from_saved(payload: list[dict], data: dict) -> None:
    """RMSD/displacement for a result loaded from disk.

    The neutral geometry comes from the saved molecule rather than a live
    result object; everything downstream is the same computation the live card
    does, so the two cards report identical numbers.
    """
    from quantui.molecule import Molecule

    # The neutral geometry travels inside the channel payload — see
    # _reorg_channels_payload for why it is not read from the top level.
    geom = next(
        (c.get("neutral_geometry") for c in payload if c.get("neutral_geometry")), None
    )
    if not geom:
        return
    try:
        neutral = Molecule(
            atoms=list(geom["atoms"]),
            coordinates=[list(c) for c in geom["coordinates"]],
            charge=geom.get("charge", 0),
            multiplicity=geom.get("multiplicity", 1),
        )
    except Exception:  # noqa: BLE001 — never break a history card
        return

    class _Holder:
        molecule = neutral

    _attach_relaxation(payload, _Holder())


def _attach_relaxation(payload: list[dict], result: Any = None) -> None:
    """Add RMSD / largest-atom-shift to each channel, in place.

    Computed at DISPLAY time from geometries that are already stored, rather
    than persisted as another number that could drift out of step with the
    coordinates it describes.
    """
    from quantui.molecule import Molecule
    from quantui.reorganization_energy import geometry_rmsd, max_atom_displacement

    neutral = getattr(result, "molecule", None) if result is not None else None
    for entry in payload:
        geom = entry.get("ion_geometry")
        if neutral is None or not geom:
            continue
        try:
            ion = Molecule(
                atoms=list(geom["atoms"]),
                coordinates=[list(c) for c in geom["coordinates"]],
                charge=geom.get("charge", 0),
                multiplicity=geom.get("multiplicity", 1),
            )
        except Exception:  # noqa: BLE001 — a readout must not break a card
            continue
        rmsd = geometry_rmsd(neutral, ion)
        if rmsd is None:
            continue
        entry["relaxation"] = {
            "rmsd": rmsd,
            "max_atom": max_atom_displacement(neutral, ion),
        }


def reorg_comparison_html(entries: list[tuple[str, dict]]) -> str:
    """Side-by-side λ table for several saved reorganization-energy results.

    ``entries`` is ``[(label, loaded_result_dict), ...]``.

    This is the workflow reorganization energy exists for: screening candidate
    molecules by how much they reorganize. One λ in isolation is hard to judge —
    it is only meaningful against other candidates — so a comparison view is
    arguably the point of the calculation rather than a nicety.

    Rendered as its own table rather than as extra columns on the general
    comparison: λ is per-CHANNEL (hole and electron), so it does not fit a
    one-row-per-result grid without either duplicating rows or inventing a
    combined number that has no physical meaning.

    Results with no channel payload are listed with a note rather than skipped —
    silently omitting them would look like they were never selected.
    """
    if not entries:
        return ""

    kinds: list[str] = []
    for _, data in entries:
        for ch in data.get("reorg_channels") or []:
            if ch.get("kind") and ch["kind"] not in kinds:
                kinds.append(ch["kind"])
    if not kinds:
        return ""

    # theme.BORDER, not a hand-picked grey. Dark mode is a whole-page colour
    # inversion, so a light grey rule inverts to near-black on a near-black page
    # and vanishes — a defect this table once reintroduced, until
    # test_retired_border_greys_are_gone_from_in_app_chrome caught it.
    th = f"text-align:left;padding:6px 12px;border-bottom:2px solid {_theme.css.BORDER}"
    td = f"padding:5px 12px;border-bottom:1px solid {_theme.css.BORDER}"
    head = (
        f'<th style="{th}">Result</th><th style="{th}">Method / basis</th>'
        + "".join(f'<th style="{th}">λ {k} (eV)</th>' for k in kinds)
        + f'<th style="{th}">Relaxation (Å)</th>'
    )

    rows = []
    for label, data in entries:
        channels = data.get("reorg_channels") or []
        if not channels:
            rows.append(
                f'<tr><td style="{td}">{label}</td>'
                f'<td style="{td}" colspan="{len(kinds) + 2}">'
                '<span style="color:#92400e">λ not saved — re-run to compare'
                "</span></td></tr>"
            )
            continue
        _attach_relaxation_from_saved(channels, data)
        by_kind = {c.get("kind"): c for c in channels}
        cells = []
        for k in kinds:
            ch = by_kind.get(k)
            lam = ch.get("lambda_hartree") if ch else None
            cells.append(
                f'<td style="{td};font-family:monospace">'
                + ("—" if lam is None else f"{lam * _HARTREE_TO_EV:.4f}")
                + "</td>"
            )
        rmsds = [c["relaxation"]["rmsd"] for c in channels if c.get("relaxation")]
        relax = f"{max(rmsds):.4f}" if rmsds else "—"
        rows.append(
            f'<tr><td style="{td}">{label}</td>'
            f'<td style="{td}">{data.get("method", "?")}/{data.get("basis", "?")}</td>'
            + "".join(cells)
            + f'<td style="{td};font-family:monospace">{relax}</td></tr>'
        )

    return (
        '<div style="margin-top:14px">'
        '<h4 style="margin:0 0 4px">Reorganization energy</h4>'
        f'<p style="color:{_theme.css.TEXT_SECONDARY};font-size:12px;margin:0 0 6px">'
        "Lower λ means less geometric reorganization on charging — generally "
        "favourable for charge transport. Relaxation is the largest per-channel "
        "RMSD between the neutral and ion geometries.</p>"
        '<table style="border-collapse:collapse;font-size:13px">'
        f"<tr>{head}</tr>{''.join(rows)}</table></div>"
    )


def reorg_channels_html(channels: list[dict]) -> str:
    """Render reorganization-energy channels from PLAIN DATA.

    Both the live card and the History card go through here. They used to have
    separate implementations — the live one reading attributes off a result
    object, the history one having no channels at all — which is precisely how
    the reported bug arose. A single renderer over plain dicts is what stops
    them drifting again: the live path converts its objects to the same shape
    the saved file holds, so anything that renders live renders after reload.
    """
    if not channels:
        return ""

    def _label(kind: str) -> str:
        return {
            "hole": "Hole transfer (cation)",
            "electron": "Electron transfer (anion)",
        }.get(kind, str(kind).title())

    blocks = []
    for ch in channels:
        lam = ch.get("lambda_hartree")
        rows = [
            (
                "λ",
                (
                    "—"
                    if lam is None
                    else f"{lam * _HARTREE_TO_EV:.4f} eV "
                    f"({lam * _HARTREE_TO_KCAL:.2f} kcal/mol)"
                ),
            ),
            ("λ₁ ion relaxation", _ev(ch.get("lambda1_hartree"))),
            ("λ₂ neutral relaxation", _ev(ch.get("lambda2_hartree"))),
            (
                "Ion state",
                f"charge {ch.get('ion_charge', 0):+d}, "
                f"mult {ch.get('ion_multiplicity', '?')}",
            ),
        ]
        # Geometry relaxation: what λ physically measures. Only
        # present once the ion geometry is saved, so older results simply omit
        # these rows rather than showing blanks.
        # ⟨S²⟩ per open-shell point; absent on older saves.
        from quantui.reorganization_energy import s2_points

        _neutral_mult = (ch.get("neutral_geometry") or {}).get("multiplicity", 1)
        for key, label, mult in s2_points(
            ch.get("ion_multiplicity") or 1, _neutral_mult
        ):
            if ch.get(key) is not None:
                rows.append((f"⟨S²⟩ {label}", _s2_html(ch[key], mult)))
        relax = ch.get("relaxation")
        if relax:
            rows.append(("Geometry RMSD", f"{relax['rmsd']:.4f} Å"))
            if relax.get("max_atom") is not None:
                idx, dist = relax["max_atom"]
                rows.append(("Largest atom shift", f"{dist:.4f} Å (atom {idx + 1})"))
        body = "".join(
            f'<tr><td style="padding:2px 18px 2px 0;color:{_theme.css.TEXT_LABEL}">{k}</td>'
            f'<td style="color:{_theme.css.TEXT_HEADING};font-family:monospace">{v}</td></tr>'
            for k, v in rows
        )
        blocks.append(
            f'<div style="margin-top:8px">'
            f'<b style="font-size:13px;color:#166534">{_label(ch.get("kind", ""))}</b>'
            f'<table style="margin-top:2px;font-size:13px;border-collapse:collapse">'
            f"{body}</table></div>"
        )
    # "Ion state" above reports a multiplicity
    # chosen by electron-count parity/minimal spin (see
    # reorganization_energy._ion_multiplicity's own docstring), which is a
    # convenient default, NOT a ground-state determination — most relevant
    # for a transition-metal ion, where the true ground state can be
    # higher-spin. Any PCM solvent selected applies only to the four
    # single-point energies above, evaluated at gas-phase-optimized
    # geometries (the relaxations themselves are not solvent-optimized).
    # Both scope notes are stated here, once, rather than left implicit.
    blocks.append(
        f'<div style="margin-top:6px;font-size:12px;color:{_theme.css.TEXT_MUTED_LIGHT}">'
        "Ion multiplicity is the minimal-spin default from electron-count "
        "parity, not a ground-state determination (relevant for transition-"
        "metal ions). Solvent (if selected) applies only to the single-"
        "point energies above; geometries are optimized in the gas phase."
        "</div>"
    )
    return "".join(blocks)


def _ev(hartree: Any) -> str:
    return "—" if hartree is None else f"{hartree * _HARTREE_TO_EV:.4f} eV"


def reorg_missing_data_notice() -> str:
    """Shown on a reorganization-energy result saved without its channel data.

    Those runs never wrote their channel data, and it cannot be recovered — λ
    is two geometry optimizations and four SCF energies. So the card says so
    and names the remedy, rather than rendering an empty section that reads
    like a rendering failure.

    Detected by ABSENCE of the payload, not by version or timestamp: a result
    re-saved or imported from elsewhere would defeat a version cutoff.
    """
    return (
        '<div style="margin-top:8px;padding:8px 10px;border-radius:6px;'
        f"background:{_theme.css.SURFACE_WARNING_BG};border:1px solid {_theme.css.ACCENT_WARNING_LIGHT};"
        f'font-size:13px;color:{_theme.css.TEXT_BODY}">'
        "<b>⚠ Reorganization-energy details were not saved for this result.</b><br>"
        "Results produced before QuantUI gained λ persistence did not store the "
        "per-channel energies or geometries, and they cannot be recovered from "
        "what was saved. <b>Re-run this calculation</b> to see the λ breakdown, "
        "the four-point energies and the geometry comparison."
        "</div>"
    )


def format_reorg_result(r: Any) -> str:
    """Format a reorganization-energy (Marcus 4-point) result card."""
    _conv = "Yes" if r.converged else "No (some steps did not converge)"
    _cc = _converged_color(r.converged)

    # Same renderer, same shape as the saved payload — see reorg_channels_html.
    from quantui.results_storage import _reorg_channels_payload

    _payload = _reorg_channels_payload(r) or []
    _attach_relaxation(_payload, r)
    _channels_html = reorg_channels_html(_payload)
    return (
        _result_card_open() + f"<b>Reorganization Energy (Marcus 4-point) &mdash; "
        f"{display_formula(r.formula)} ({r.method}/{r.basis})</b>"
        + _result_card_table_open()
        + f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Neutral energy</td>'
        f'<td style="color:{_theme.css.TEXT_HEADING}">{r.neutral_energy_hartree:.8f} Ha</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">Total opt steps</td>'
        f'<td style="color:{_theme.css.TEXT_HEADING}">{r.n_total_opt_steps}</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL}">All converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f"</table>{_channels_html}" + _RESULT_CARD_CLOSE
    )


def slurm_history_marker(data: dict[str, Any]) -> str:
    """Short History-list marker for a SLURM result, e.g. ``🖥 SLURM 812345·a2 ``.

    Empty for local results and for results saved before SLURM provenance
    was recorded.
    """
    if data.get("execution_backend") != "slurm":
        return ""
    info = data.get("slurm") or {}
    job_id = info.get("job_id")
    attempt = info.get("attempt")
    text = "🖥 SLURM"
    if job_id:
        text += f" {job_id}"
        if attempt:
            text += f"·a{attempt}"
    return text + " "


def _slurm_provenance_row(data: dict[str, Any]) -> str:
    """History-card row naming the SLURM job, attempt and job folder."""
    if data.get("execution_backend") != "slurm":
        return ""
    info = data.get("slurm") or {}
    parts = ["SLURM batch"]
    if info.get("job_id"):
        parts.append(f"job {html.escape(str(info['job_id']))}")
    if info.get("attempt"):
        parts.append(f"attempt {int(info['attempt'])}")
    value = " &middot; ".join(parts)
    folder = info.get("attempt_dir") or info.get("job_dir")
    if folder:
        value += (
            f'<br><span style="font-family:monospace;font-size:12px;'
            f'color:{_theme.css.TEXT_MUTED_LIGHT}">{html.escape(str(folder))}</span>'
        )
    return (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.css.TEXT_LABEL};'
        f'vertical-align:top">Ran on</td>'
        f'<td style="color:{_theme.css.TEXT_HEADING}">{value}</td></tr>'
    )


def format_past_result(data: dict[str, Any], result_dir: Optional[Path] = None) -> str:
    """Format a saved result.json payload as an HTML result card."""

    _ct_labels = {
        "single_point": (
            "Single Point",
            _theme.css.ACCENT_INFO,
            _theme.css.SURFACE_INFO_BG,
        ),
        "geometry_opt": (
            "Geometry Optimization",
            _theme.css.ACCENT_PURPLE,
            _theme.css.SURFACE_PURPLE_BG,
        ),
        "frequency": (
            "Frequency Analysis",
            _theme.css.ACCENT_SUCCESS,
            _theme.css.SURFACE_GREEN_BG,
        ),
        "tddft": ("TD-DFT", _theme.css.ACCENT_WARNING, _theme.css.SURFACE_WARNING_BG),
        "nmr": ("NMR", _theme.css.ACCENT_TEAL, _theme.css.SURFACE_TEAL_BG),
        "pes_scan": (
            "PES Scan",
            _theme.css.ACCENT_WARNING,
            _theme.css.SURFACE_ORANGE_BG,
        ),
        "transition_state": (
            "Transition State",
            _theme.css.ACCENT_PURPLE,
            _theme.css.SURFACE_PURPLE_BG,
        ),
        "irc": ("IRC", _theme.css.ACCENT_PURPLE, _theme.css.SURFACE_PURPLE_BG),
    }
    ct = data.get("calc_type", "")
    _ct_label, _ct_fg, _ct_bg = _ct_labels.get(
        ct,
        (
            ct.replace("_", " ").title(),
            _theme.css.TEXT_SECONDARY,
            _theme.css.SURFACE_MUTED_BG,
        ),
    )
    _ct_badge = (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f"background:{_ct_bg};color:{_ct_fg};font-size:12px;font-weight:700;"
        f'letter-spacing:0.03em;margin-bottom:6px">{_ct_label}</span>'
    )
    _conv = "Yes" if data.get("converged") else "No (treat results with caution)"
    _cc = _converged_color(bool(data.get("converged")))
    # data["converged"] already
    # folds in CC-amplitude (post-HF), TD-root (tddft), or Hessian
    # (frequency) convergence on top of the reference SCF's own status,
    # exactly like the live cards (format_result/format_freq_result/
    # format_tddft_result). A bare "SCF converged" label here would blame
    # the SCF for a CC/TD/Hessian-only failure whose reference SCF was
    # fine — mirror each live formatter's label choice.
    if ct in ("frequency", "transition_state", "irc"):
        _conv_label = "Converged"
    elif ct == "tddft":
        _conv_label = "Converged"
    elif data.get("cc_converged") is not None:
        _conv_label = "Converged"
    else:
        _conv_label = "SCF converged"
    _rows = _basic_rows(lambda k, d=None: data.get(k, d), _conv_label, _conv, _cc)
    ts = data.get("timestamp", "")

    # Shared 'extra' rows (correlation breakdown / solvent / device / dipole /
    # Mulliken) — same builder as the live card so the two never drift.
    _extra = _result_extra_rows(lambda k, d=None: data.get(k, d))
    _extra += _slurm_provenance_row(data)

    # The saved thumbnail (a light, tiny copy of this card) was unreadable,
    # and glaring in Dark theme; the card itself carries the same data.
    _thumb_html = ""

    # Restore persisted frequency thermochemistry into the
    # History card. Older saved results (or a Hessian-only run with no
    # thermo block) have no "thermo" key at all; that's a silent no-op,
    # not an error.
    _thermo_html = ""
    if ct == "frequency":
        # Same order as the live card (format_freq_result): warning, basics,
        # modes, extras, thermochemistry.
        _ir_block = (data.get("spectra") or {}).get("ir") or {}
        _rows = (
            stationary_point_rows(_ir_block.get("max_force_ev_per_ang"))
            + _rows
            + _freq_section_rows(_saved_freq_accessor(data))
            + saddle_retry_rows(_ir_block.get("imaginary_mode_retry"))
        )
        _thermo = ((data.get("spectra") or {}).get("ir") or {}).get("thermo")
        if _thermo:
            _thermo_html += _thermo_card_rows(
                temperature_k=_thermo.get("temperature_k", 298.15),
                pressure_atm=_thermo.get("pressure_atm", 1.0),
                zpve_hartree=_thermo.get("zpve_hartree"),
                h_hartree=_thermo["H_hartree"],
                s_jmol=_thermo["S_jmol"],
                g_hartree=_thermo["G_hartree"],
                energy_hartree=data.get("energy_hartree"),
            )

    # Reorganization-energy channels. This is the reported bug: the
    # card came back without the numbers the calculation exists to produce.
    # Keyed on the calc type AND the payload, so a reorg result saved before λ
    # persistence gets an explanation instead of a silently incomplete card.
    _ts_html = ""
    if ct == "transition_state":
        _ts_html = ts_verdict_rows((data.get("spectra") or {}).get("transition_state"))
    elif ct == "irc":
        _ts_html = irc_rows((data.get("spectra") or {}).get("irc"))

    _reorg_html = ""
    if ct == "reorganization_energy":
        _channels = data.get("reorg_channels")
        if _channels:
            _attach_relaxation_from_saved(_channels, data)
            _reorg_html = reorg_channels_html(_channels)
        else:
            _reorg_html = reorg_missing_data_notice()

    return (
        _result_card_open(extra_style="overflow:hidden")
        + f"{_thumb_html}"
        + f"{_ct_badge}<br>"
        + f'<b>{display_formula(data["formula"])} &mdash; '
        f'{_method_basis_label(data["method"], data["basis"], data.get("scf_variant"))}</b>'
        f'&ensp;<small style="color:{_theme.css.TEXT_MUTED_LIGHT}">{ts}</small>'
        + _result_card_table_open()
        + f"{_rows}{_ts_html}{_extra}{_thermo_html}</table>{_reorg_html}"
        + _RESULT_CARD_CLOSE
    )
