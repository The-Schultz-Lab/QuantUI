"""Result-card HTML formatters used by QuantUIApp."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from quantui import theme as _theme


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
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">{label}</td>'
            f'<td style="color:{_theme.TEXT_HEADING}">{value}</td></tr>'
        )

    rows = ""
    energy = get("energy_hartree", 0.0)

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

    # Compute device — always shown; old saved results lack the
    # field and safely read "CPU".
    if bool(get("gpu_used", False)):
        _name = get("gpu_name")
        _device = (
            f'<span style="color:{_theme.ACCENT_SUCCESS}">🚀 GPU</span>'
            f' &mdash; <span style="font-family:monospace">{_name}</span>'
            if _name
            else f'<span style="color:{_theme.ACCENT_SUCCESS}">🚀 GPU</span>'
        )
    else:
        _device = f'<span style="color:{_theme.TEXT_SECONDARY}">CPU</span>'
    rows += (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Compute device</td>'
        f"<td>{_device}</td></tr>"
    )

    # Density fitting — shown only when it was applied (the non-default case
    # worth flagging to the student); exact-integral runs stay uncluttered, and
    # result types without the field safely omit it.
    if bool(get("density_fit", False)):
        rows += (
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Density fitting</td>'
            '<td><span style="color:#0369a1">⚡ RI</span> '
            f'<span style="color:{_theme.TEXT_MUTED_LIGHT};font-size:12px">'
            "(approximate 2-electron integrals)</span></td></tr>"
        )

    _dip = get("dipole_moment_debye")
    if _dip is not None:
        _vec = get("dipole_vector_debye")
        if _vec is not None and len(_vec) >= 3:
            _dip_str = (
                f"{_dip:.4f} D"
                f' <span style="color:{_theme.TEXT_MUTED_LIGHT};font-size:12px">'
                f"(μ = [{float(_vec[0]):+.3f}, {float(_vec[1]):+.3f}, "
                f"{float(_vec[2]):+.3f}] D)</span>"
            )
        else:
            _dip_str = (
                f"{_dip:.4f} D"
                f' <span style="color:{_theme.TEXT_MUTED_LIGHT};font-size:12px">'
                "(magnitude only — μ components not saved)</span>"
            )
        rows += _num("Dipole moment", _dip_str)

    _chg = get("mulliken_charges")
    _syms = get("atom_symbols")
    if _chg is not None and _syms is not None:
        _charge_str = "  ".join(f"{sym}:{c:+.3f}" for sym, c in zip(_syms, _chg))
        rows += (
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL};vertical-align:top">'
            f"Mulliken charges</td>"
            f'<td style="color:{_theme.TEXT_HEADING};font-family:monospace;font-size:12px;'
            f'word-break:break-all">{_charge_str}</td></tr>'
        )
    return rows


def format_result(r: Any) -> str:
    """Format a single-point-style result card."""
    _conv = "Yes" if r.converged else "No (treat results with caution)"
    _cc = "green" if r.converged else _theme.ACCENT_ERROR_ALT
    _gap = f"{r.homo_lumo_gap_ev:.4f} eV" if r.homo_lumo_gap_ev is not None else "N/A"
    _rows = "".join(
        f"<tr>"
        f'<td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">{k}</td>'
        f'<td style="color:{vc}">{v}</td>'
        f"</tr>"
        for k, v, vc in [
            (
                "Total energy",
                f"{r.energy_hartree:.8f} Ha &ensp;({r.energy_ev:.4f} eV)",
                _theme.TEXT_HEADING,
            ),
            ("HOMO-LUMO gap", _gap, _theme.TEXT_HEADING),
            ("SCF converged", _conv, _cc),
            (
                "SCF iterations",
                (
                    "—"
                    if getattr(r, "n_iterations", None) in (None, -1)
                    else str(r.n_iterations)
                ),
                _theme.TEXT_HEADING,
            ),
        ]
    )
    _extra = _result_extra_rows(lambda k, d=None: getattr(r, k, d))
    return (
        f'<div style="background:{_theme.ACCENT_SUCCESS_BG};border-left:4px solid {_theme.ACCENT_SUCCESS_ALT};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>{r.formula} &mdash; {r.method}/{r.basis}</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f"{_rows}{_extra}</table></div>"
    )


def format_opt_result(r: Any) -> str:
    """Format a geometry-optimization result card."""
    _conv = "Yes" if r.converged else "No (max steps reached)"
    _cc = "green" if r.converged else _theme.ACCENT_ERROR_ALT
    _rows = "".join(
        f"<tr>"
        f'<td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">{k}</td>'
        f'<td style="color:{vc}">{v}</td>'
        f"</tr>"
        for k, v, vc in [
            ("Final energy", f"{r.energy_hartree:.8f} Ha", _theme.TEXT_HEADING),
            (
                "Energy change",
                f"{r.energy_change_hartree:+.6f} Ha",
                _theme.TEXT_HEADING,
            ),
            ("Opt converged", _conv, _cc),
            ("Steps taken", str(r.n_steps), _theme.TEXT_HEADING),
            ("Geometry RMSD", f"{r.rmsd_angstrom:.4f} Å", _theme.TEXT_HEADING),
        ]
    )
    return (
        f'<div style="background:{_theme.ACCENT_SUCCESS_BG};border-left:4px solid {_theme.ACCENT_SUCCESS_ALT};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>Geometry Optimisation &mdash; {r.formula} ({r.method}/{r.basis})</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f"{_rows}</table></div>"
    )


def format_freq_result(r: Any) -> str:
    """Format a frequency-analysis result card."""
    _conv = "Yes" if r.converged else "No (treat with caution)"
    _cc = "green" if r.converged else _theme.ACCENT_ERROR_ALT
    n_real = r.n_real_modes()
    n_imag = r.n_imaginary_modes()
    real_freqs = sorted(f for f in r.frequencies_cm1 if f > 0)[:6]
    freq_str = "  ".join(f"{f:.1f}" for f in real_freqs)
    if len([f for f in r.frequencies_cm1 if f > 0]) > 6:
        freq_str += " …"
    imag_note = ""
    if n_imag > 0:
        imag_note = (
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Imaginary modes</td>'
            f'<td style="color:{_theme.ACCENT_ERROR_ALT}">{n_imag} — geometry may not be a minimum</td></tr>'
        )
    _rows = (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">SCF energy</td>'
        f'<td style="color:{_theme.TEXT_HEADING}">{r.energy_hartree:.8f} Ha</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">SCF converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Real modes</td>'
        f'<td style="color:{_theme.TEXT_HEADING}">{n_real}</td></tr>'
        + imag_note
        + (
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Frequencies (cm⁻¹)</td>'
            f'<td style="color:{_theme.TEXT_HEADING};font-family:monospace">{freq_str or "none"}</td></tr>'
            if real_freqs
            else ""
        )
        + f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">ZPVE</td>'
        f'<td style="color:{_theme.TEXT_HEADING}">{r.zpve_hartree:.6f} Ha '
        f"({r.zpve_hartree * 27.211386245988:.4f} eV)</td></tr>"
    )
    _thermo_rows = ""
    _thermo = getattr(r, "thermo", None)
    if _thermo is not None:
        _kj = 2625.5  # kJ/mol per Hartree
        _thermo_rows = (
            f'<tr><td colspan="2" style="padding:6px 0 2px 0;color:{_theme.TEXT_MUTED};'
            f'font-size:12px;font-style:italic">'
            f"&#8212; Thermochemistry at {_thermo.temperature_k:.0f} K / 1 atm &#8212;"
            f"</td></tr>"
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">H (298 K)</td>'
            f'<td style="color:{_theme.TEXT_HEADING}">{_thermo.H_hartree:.6f} Ha</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">S (298 K)</td>'
            f'<td style="color:{_theme.TEXT_HEADING}">{_thermo.S_jmol:.2f} J/(mol·K)</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">G (298 K)</td>'
            f'<td style="color:{_theme.TEXT_HEADING}">{_thermo.G_hartree:.6f} Ha'
            f" ({_thermo.G_hartree * _kj:.2f} kJ/mol)</td></tr>"
        )
    return (
        f'<div style="background:{_theme.ACCENT_SUCCESS_BG};border-left:4px solid {_theme.ACCENT_SUCCESS_ALT};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>Frequency Analysis &mdash; {r.formula} ({r.method}/{r.basis})</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f"{_rows}{_thermo_rows}</table></div>"
    )


def format_tddft_result(r: Any) -> str:
    """Format a TD-DFT / UV-Vis result card."""
    _conv = "Yes" if r.converged else "No (treat with caution)"
    _cc = "green" if r.converged else _theme.ACCENT_ERROR_ALT
    header_rows = (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Ground-state energy</td>'
        f'<td style="color:{_theme.TEXT_HEADING}">{r.energy_hartree:.8f} Ha</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">SCF converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">States computed</td>'
        f'<td style="color:{_theme.TEXT_HEADING}">{len(r.excitation_energies_ev)}</td></tr>'
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
                f'<td style="padding:2px 12px 2px 0;color:{_theme.TEXT_SECONDARY}">S{i}</td>'
                f'<td style="padding:2px 12px 2px 0;color:{_theme.TEXT_HEADING}">{e_ev:.3f} eV</td>'
                f'<td style="padding:2px 12px 2px 0;color:{_theme.TEXT_HEADING}">{wl[i - 1]:.1f} nm</td>'
                f'<td style="padding:2px 4px 2px 0;color:{_theme.TEXT_HEADING}">f = {f_osc:.4f}</td>'
                f"</tr>"
            )
        if len(r.excitation_energies_ev) > 8:
            exc_rows.append(
                f'<tr><td colspan="4" style="color:{_theme.TEXT_FAINT};font-size:12px">… '
                f"and {len(r.excitation_energies_ev) - 8} more states</td></tr>"
            )
        exc_table = (
            f'<tr><td colspan="2" style="padding:8px 0 2px;color:{_theme.TEXT_LABEL};font-weight:bold">'
            "Vertical excitations:</td></tr>"
            "<tr>"
            f'<th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px;padding:2px 12px 2px 0">State</th>'
            f'<th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px;padding:2px 12px 2px 0">Energy</th>'
            f'<th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px;padding:2px 12px 2px 0">λ</th>'
            f'<th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px">Osc. str.</th></tr>'
            + "".join(exc_rows)
        )
    return (
        f'<div style="background:{_theme.ACCENT_SUCCESS_BG};border-left:4px solid {_theme.ACCENT_SUCCESS_ALT};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>TD-DFT / UV-Vis &mdash; {r.formula} ({r.method}/{r.basis})</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f"{header_rows}{exc_table}</table></div>"
    )


def format_nmr_result(r: Any) -> str:
    """Format an NMR shielding result card."""
    _conv = "Yes" if r.converged else "No (treat with caution)"
    _cc = "green" if r.converged else _theme.ACCENT_ERROR_ALT
    header_rows = (
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">SCF converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Reference</td>'
        f'<td style="color:{_theme.TEXT_HEADING}">{r.reference_compound} ({r.method}/{r.basis})</td></tr>'
    )

    def _nmr_table(label: str, shifts: list, sym: str) -> str:
        if not shifts:
            return ""
        rows = "".join(
            f"<tr>"
            f'<td style="padding:2px 14px 2px 0;color:{_theme.TEXT_SECONDARY}">{sym}-{n}</td>'
            f'<td style="color:{_theme.TEXT_HEADING}">{d:.2f} ppm</td>'
            f"</tr>"
            for n, (_i, d) in enumerate(shifts, 1)
        )
        return (
            f'<tr><td colspan="2" style="padding:8px 0 2px;color:{_theme.TEXT_LABEL};font-weight:bold">'
            f"{label} shifts (vs. TMS):</td></tr>"
            f"<tr>"
            f'<th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px;padding:2px 14px 2px 0">Atom</th>'
            f'<th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px">δ (ppm)</th></tr>'
            + rows
        )

    h_table = _nmr_table("¹H", r.h_shifts(), "H")
    c_table = _nmr_table("¹³C", r.c_shifts(), "C")

    _basis_warn = ""
    if r.basis.upper() in ("STO-3G", "3-21G"):
        _basis_warn = (
            '<tr><td colspan="2" style="padding:6px 0 0">'
            f'<span style="color:{_theme.ACCENT_WARNING};font-size:12px">'
            f"⚠ {r.basis} gives qualitative NMR only — use 6-31G* or better.</span>"
            "</td></tr>"
        )

    # M4 audit fix (2026-07-14): the reference shielding constants table only
    # covers a handful of method/basis combinations; any other combination
    # silently substitutes the B3LYP/6-31G* constants, which can shift the
    # reported ppm values by several ppm relative to a properly calibrated
    # reference. Surface that substitution rather than let it pass silently.
    _ref_warn = ""
    if getattr(r, "is_fallback_reference", False):
        _ref_warn = (
            '<tr><td colspan="2" style="padding:6px 0 0">'
            f'<span style="color:{_theme.ACCENT_WARNING};font-size:12px">'
            f"⚠ No calibrated TMS reference for {r.method}/{r.basis} — using "
            f"{getattr(r, 'reference_key', 'B3LYP/6-31G*')} constants instead. "
            "Shifts may be off by a few ppm.</span>"
            "</td></tr>"
        )

    _empty = ""
    if not r.h_shifts() and not r.c_shifts():
        _empty = (
            f'<tr><td colspan="2" style="color:{_theme.TEXT_FAINT};font-size:12px">'
            "No ¹H or ¹³C atoms found in this molecule.</td></tr>"
        )

    return (
        f'<div style="background:{_theme.ACCENT_SUCCESS_BG};border-left:4px solid {_theme.ACCENT_SUCCESS_ALT};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>NMR Shielding &mdash; {r.formula} ({r.method}/{r.basis})</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f"{header_rows}{h_table}{c_table}{_empty}{_basis_warn}{_ref_warn}</table></div>"
    )


def format_pes_scan_result(r: Any) -> str:
    """Format a PESScanResult as an HTML result card."""
    _conv = "Yes" if r.converged_all else "No (some points did not converge)"
    _cc = "green" if r.converged_all else _theme.ACCENT_ERROR_ALT
    if r.energies_hartree:
        e_min = min(r.energies_hartree)
        e_max = max(r.energies_hartree)
        barrier_kcal = (e_max - e_min) * 627.509474
        _e_row = (
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Min energy</td>'
            f'<td style="color:{_theme.TEXT_HEADING}">{e_min:.8f} Ha</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Energy range</td>'
            f'<td style="color:{_theme.TEXT_HEADING}">{barrier_kcal:.2f} kcal/mol</td></tr>'
        )
    else:
        _e_row = ""
    _idx_str = "–".join(str(i + 1) for i in r.atom_indices)
    return (
        f'<div style="background:{_theme.ACCENT_SUCCESS_BG};border-left:4px solid {_theme.ACCENT_SUCCESS_ALT};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>PES Scan &mdash; {r.formula} ({r.method}/{r.basis})</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Scan type</td>'
        f'<td style="color:{_theme.TEXT_HEADING}">{r.scan_type.capitalize()} ({_idx_str})</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Range</td>'
        f'<td style="color:{_theme.TEXT_HEADING}">{r.scan_parameter_values[0]:.3f} → '
        f"{r.scan_parameter_values[-1]:.3f} {r.scan_unit} "
        f"({r.n_steps} points)</td></tr>"
        f"{_e_row}"
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">All converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f"</table></div>"
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
    """Add RMSD / largest-atom-shift to each channel, in place (REORG.4).

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
    # and vanishes — the defect THEME.5 fixed, and which this table reintroduced
    # until test_retired_border_greys_are_gone_from_in_app_chrome caught it.
    th = f"text-align:left;padding:6px 12px;border-bottom:2px solid {_theme.BORDER}"
    td = f"padding:5px 12px;border-bottom:1px solid {_theme.BORDER}"
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
        f'<p style="color:{_theme.TEXT_SECONDARY};font-size:12px;margin:0 0 6px">'
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
        # Geometry relaxation (REORG.4): what λ physically measures. Only
        # present once the ion geometry is saved, so older results simply omit
        # these rows rather than showing blanks.
        relax = ch.get("relaxation")
        if relax:
            rows.append(("Geometry RMSD", f"{relax['rmsd']:.4f} Å"))
            if relax.get("max_atom") is not None:
                idx, dist = relax["max_atom"]
                rows.append(("Largest atom shift", f"{dist:.4f} Å (atom {idx + 1})"))
        body = "".join(
            f'<tr><td style="padding:2px 18px 2px 0;color:{_theme.TEXT_LABEL}">{k}</td>'
            f'<td style="color:{_theme.TEXT_HEADING};font-family:monospace">{v}</td></tr>'
            for k, v in rows
        )
        blocks.append(
            f'<div style="margin-top:8px">'
            f'<b style="font-size:13px;color:#166534">{_label(ch.get("kind", ""))}</b>'
            f'<table style="margin-top:2px;font-size:13px;border-collapse:collapse">'
            f"{body}</table></div>"
        )
    return "".join(blocks)


def _ev(hartree: Any) -> str:
    return "—" if hartree is None else f"{hartree * _HARTREE_TO_EV:.4f} eV"


def reorg_missing_data_notice() -> str:
    """Shown on a reorganization-energy result saved before REORG.1.

    Those runs never wrote their channel data, and it cannot be recovered — λ
    is two geometry optimizations and four SCF energies. So the card says so
    and names the remedy, rather than rendering an empty section that reads
    like a rendering failure.

    Detected by ABSENCE of the payload, not by version or timestamp: a result
    re-saved or imported from elsewhere would defeat a version cutoff.
    """
    return (
        '<div style="margin-top:8px;padding:8px 10px;border-radius:6px;'
        f'background:#fef3c7;border:1px solid {_theme.ACCENT_WARNING_LIGHT};font-size:13px;color:#78350f">'
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
    _cc = "green" if r.converged else _theme.ACCENT_ERROR_ALT

    # Same renderer, same shape as the saved payload — see reorg_channels_html.
    from quantui.results_storage import _reorg_channels_payload

    _payload = _reorg_channels_payload(r) or []
    _attach_relaxation(_payload, r)
    _channels_html = reorg_channels_html(_payload)
    return (
        f'<div style="background:{_theme.ACCENT_SUCCESS_BG};border-left:4px solid {_theme.ACCENT_SUCCESS_ALT};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>Reorganization Energy (Marcus 4-point) &mdash; "
        f"{r.formula} ({r.method}/{r.basis})</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Neutral energy</td>'
        f'<td style="color:{_theme.TEXT_HEADING}">{r.neutral_energy_hartree:.8f} Ha</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">Total opt steps</td>'
        f'<td style="color:{_theme.TEXT_HEADING}">{r.n_total_opt_steps}</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">All converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f"</table>{_channels_html}</div>"
    )


def format_past_result(data: dict[str, Any], result_dir: Optional[Path] = None) -> str:
    """Format a saved result.json payload as an HTML result card."""
    import base64 as _b64

    _ct_labels = {
        "single_point": ("Single Point", _theme.ACCENT_INFO, "#dbeafe"),
        "geometry_opt": ("Geometry Optimization", _theme.ACCENT_PURPLE, "#ede9fe"),
        "frequency": ("Frequency Analysis", "#15803d", "#dcfce7"),
        "tddft": ("TD-DFT", _theme.ACCENT_WARNING, "#fef3c7"),
        "nmr": ("NMR", _theme.ACCENT_TEAL, "#ccfbf1"),
        "pes_scan": ("PES Scan", "#c2410c", "#ffedd5"),
    }
    ct = data.get("calc_type", "")
    _ct_label, _ct_fg, _ct_bg = _ct_labels.get(
        ct, (ct.replace("_", " ").title(), _theme.TEXT_SECONDARY, "#f3f4f6")
    )
    _ct_badge = (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f"background:{_ct_bg};color:{_ct_fg};font-size:12px;font-weight:700;"
        f'letter-spacing:0.03em;margin-bottom:6px">{_ct_label}</span>'
    )
    _conv = "Yes" if data.get("converged") else "No (treat results with caution)"
    _cc = "green" if data.get("converged") else _theme.ACCENT_ERROR_ALT
    _gap = (
        f"{data['homo_lumo_gap_ev']:.4f} eV"
        if data.get("homo_lumo_gap_ev") is not None
        else "N/A"
    )
    _rows = "".join(
        f"<tr>"
        f'<td style="padding:3px 18px 3px 0;color:{_theme.TEXT_LABEL}">{k}</td>'
        f'<td style="color:{vc}">{v}</td>'
        f"</tr>"
        for k, v, vc in [
            (
                "Total energy",
                f"{data['energy_hartree']:.8f} Ha &ensp;({data['energy_ev']:.4f} eV)",
                _theme.TEXT_HEADING,
            ),
            ("HOMO-LUMO gap", _gap, _theme.TEXT_HEADING),
            ("SCF converged", _conv, _cc),
            (
                "SCF iterations",
                (
                    "—"
                    if data.get("n_iterations") in (None, -1)
                    else str(data.get("n_iterations"))
                ),
                _theme.TEXT_HEADING,
            ),
        ]
    )
    ts = data.get("timestamp", "")

    # Shared 'extra' rows (correlation breakdown / solvent / device / dipole /
    # Mulliken) — same builder as the live card so the two never drift.
    _extra = _result_extra_rows(lambda k, d=None: data.get(k, d))

    # Embed thumbnail if saved
    _thumb_html = ""
    if result_dir is not None:
        _thumb_path = Path(result_dir) / "thumbnail.png"
        if _thumb_path.exists():
            _img_b64 = _b64.b64encode(_thumb_path.read_bytes()).decode()
            _thumb_html = (
                f'<img src="data:image/png;base64,{_img_b64}" '
                f'style="float:right;margin:0 0 6px 14px;border-radius:4px;'
                f'border:1px solid {_theme.BORDER}" width="173" height="108" />'
            )

    # Reorganization-energy channels (REORG.1). This is the reported bug: the
    # card came back without the numbers the calculation exists to produce.
    # Keyed on the calc type AND the payload, so a reorg result saved before λ
    # persistence gets an explanation instead of a silently incomplete card.
    _reorg_html = ""
    if ct == "reorganization_energy":
        _channels = data.get("reorg_channels")
        if _channels:
            _attach_relaxation_from_saved(_channels, data)
            _reorg_html = reorg_channels_html(_channels)
        else:
            _reorg_html = reorg_missing_data_notice()

    return (
        f'<div style="background:{_theme.ACCENT_SUCCESS_BG};border-left:4px solid {_theme.ACCENT_SUCCESS_ALT};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;overflow:hidden">'
        f"{_thumb_html}"
        f"{_ct_badge}<br>"
        f'<b>{data["formula"]} &mdash; {data["method"]}/{data["basis"]}</b>'
        f'&ensp;<small style="color:{_theme.TEXT_MUTED_LIGHT}">{ts}</small>'
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f"{_rows}{_extra}</table>{_reorg_html}</div>"
    )
