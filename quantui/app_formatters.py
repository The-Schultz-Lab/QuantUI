"""Result-card HTML formatters used by QuantUIApp."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def format_result(r: Any) -> str:
    """Format a single-point-style result card."""
    _conv = "Yes" if r.converged else "No (treat results with caution)"
    _cc = "green" if r.converged else "#c00"
    _gap = f"{r.homo_lumo_gap_ev:.4f} eV" if r.homo_lumo_gap_ev is not None else "N/A"
    _rows = "".join(
        f"<tr>"
        f'<td style="padding:3px 18px 3px 0;color:#444">{k}</td>'
        f'<td style="color:{vc}">{v}</td>'
        f"</tr>"
        for k, v, vc in [
            (
                "Total energy",
                f"{r.energy_hartree:.8f} Ha &ensp;({r.energy_ev:.4f} eV)",
                "#000",
            ),
            ("HOMO-LUMO gap", _gap, "#000"),
            ("SCF converged", _conv, _cc),
            (
                "SCF iterations",
                (
                    "—"
                    if getattr(r, "n_iterations", None) in (None, -1)
                    else str(r.n_iterations)
                ),
                "#000",
            ),
        ]
    )
    _extra = ""
    # MP2: show HF reference energy separately
    _mp2_corr = getattr(r, "mp2_correlation_hartree", None)
    if _mp2_corr is not None:
        _hf_e = r.energy_hartree - _mp2_corr
        _extra += (
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">HF reference</td>'
            f'<td style="color:#000">{_hf_e:.8f} Ha</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">MP2 correlation</td>'
            f'<td style="color:#000">{_mp2_corr:.8f} Ha</td></tr>'
        )
    _solvent = getattr(r, "solvent", None)
    if _solvent is not None:
        _extra += (
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">Solvent (PCM)</td>'
            f'<td style="color:#000">{_solvent}</td></tr>'
        )
    _dip = getattr(r, "dipole_moment_debye", None)
    if _dip is not None:
        _extra += (
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">Dipole moment</td>'
            f'<td style="color:#000">{_dip:.4f} D</td></tr>'
        )
    _chg = getattr(r, "mulliken_charges", None)
    _syms = getattr(r, "atom_symbols", None)
    if _chg is not None and _syms is not None:
        _charge_str = "  ".join(f"{sym}:{c:+.3f}" for sym, c in zip(_syms, _chg))
        _extra += (
            f'<tr><td style="padding:3px 18px 3px 0;color:#444;vertical-align:top">'
            f"Mulliken charges</td>"
            f'<td style="color:#000;font-family:monospace;font-size:12px;'
            f'word-break:break-all">{_charge_str}</td></tr>'
        )
    return (
        f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>{r.formula} &mdash; {r.method}/{r.basis}</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f"{_rows}{_extra}</table></div>"
    )


def format_opt_result(r: Any) -> str:
    """Format a geometry-optimization result card."""
    _conv = "Yes" if r.converged else "No (max steps reached)"
    _cc = "green" if r.converged else "#c00"
    _rows = "".join(
        f"<tr>"
        f'<td style="padding:3px 18px 3px 0;color:#444">{k}</td>'
        f'<td style="color:{vc}">{v}</td>'
        f"</tr>"
        for k, v, vc in [
            ("Final energy", f"{r.energy_hartree:.8f} Ha", "#000"),
            ("Energy change", f"{r.energy_change_hartree:+.6f} Ha", "#000"),
            ("Opt converged", _conv, _cc),
            ("Steps taken", str(r.n_steps), "#000"),
            ("Geometry RMSD", f"{r.rmsd_angstrom:.4f} Å", "#000"),
        ]
    )
    return (
        f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>Geometry Optimisation &mdash; {r.formula} ({r.method}/{r.basis})</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f"{_rows}</table></div>"
    )


def format_freq_result(r: Any) -> str:
    """Format a frequency-analysis result card."""
    _conv = "Yes" if r.converged else "No (treat with caution)"
    _cc = "green" if r.converged else "#c00"
    n_real = r.n_real_modes()
    n_imag = r.n_imaginary_modes()
    real_freqs = sorted(f for f in r.frequencies_cm1 if f > 0)[:6]
    freq_str = "  ".join(f"{f:.1f}" for f in real_freqs)
    if len([f for f in r.frequencies_cm1 if f > 0]) > 6:
        freq_str += " …"
    imag_note = ""
    if n_imag > 0:
        imag_note = (
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">Imaginary modes</td>'
            f'<td style="color:#c00">{n_imag} — geometry may not be a minimum</td></tr>'
        )
    _rows = (
        f'<tr><td style="padding:3px 18px 3px 0;color:#444">SCF energy</td>'
        f'<td style="color:#000">{r.energy_hartree:.8f} Ha</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:#444">SCF converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:#444">Real modes</td>'
        f'<td style="color:#000">{n_real}</td></tr>'
        + imag_note
        + (
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">Frequencies (cm⁻¹)</td>'
            f'<td style="color:#000;font-family:monospace">{freq_str or "none"}</td></tr>'
            if real_freqs
            else ""
        )
        + f'<tr><td style="padding:3px 18px 3px 0;color:#444">ZPVE</td>'
        f'<td style="color:#000">{r.zpve_hartree:.6f} Ha '
        f"({r.zpve_hartree * 27.211386245988:.4f} eV)</td></tr>"
    )
    _thermo_rows = ""
    _thermo = getattr(r, "thermo", None)
    if _thermo is not None:
        _kj = 2625.5  # kJ/mol per Hartree
        _thermo_rows = (
            f'<tr><td colspan="2" style="padding:6px 0 2px 0;color:#666;'
            f'font-size:12px;font-style:italic">'
            f"&#8212; Thermochemistry at {_thermo.temperature_k:.0f} K / 1 atm &#8212;"
            f"</td></tr>"
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">H (298 K)</td>'
            f'<td style="color:#000">{_thermo.H_hartree:.6f} Ha</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">S (298 K)</td>'
            f'<td style="color:#000">{_thermo.S_jmol:.2f} J/(mol·K)</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">G (298 K)</td>'
            f'<td style="color:#000">{_thermo.G_hartree:.6f} Ha'
            f" ({_thermo.G_hartree * _kj:.2f} kJ/mol)</td></tr>"
        )
    return (
        f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>Frequency Analysis &mdash; {r.formula} ({r.method}/{r.basis})</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f"{_rows}{_thermo_rows}</table></div>"
    )


def format_tddft_result(r: Any) -> str:
    """Format a TD-DFT / UV-Vis result card."""
    _conv = "Yes" if r.converged else "No (treat with caution)"
    _cc = "green" if r.converged else "#c00"
    header_rows = (
        f'<tr><td style="padding:3px 18px 3px 0;color:#444">Ground-state energy</td>'
        f'<td style="color:#000">{r.energy_hartree:.8f} Ha</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:#444">SCF converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:#444">States computed</td>'
        f'<td style="color:#000">{len(r.excitation_energies_ev)}</td></tr>'
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
                f'<td style="padding:2px 12px 2px 0;color:#555">S{i}</td>'
                f'<td style="padding:2px 12px 2px 0;color:#000">{e_ev:.3f} eV</td>'
                f'<td style="padding:2px 12px 2px 0;color:#000">{wl[i - 1]:.1f} nm</td>'
                f'<td style="padding:2px 4px 2px 0;color:#000">f = {f_osc:.4f}</td>'
                f"</tr>"
            )
        if len(r.excitation_energies_ev) > 8:
            exc_rows.append(
                f'<tr><td colspan="4" style="color:#888;font-size:12px">… '
                f"and {len(r.excitation_energies_ev) - 8} more states</td></tr>"
            )
        exc_table = (
            '<tr><td colspan="2" style="padding:8px 0 2px;color:#444;font-weight:bold">'
            "Vertical excitations:</td></tr>"
            "<tr>"
            '<th style="text-align:left;color:#555;font-size:12px;padding:2px 12px 2px 0">State</th>'
            '<th style="text-align:left;color:#555;font-size:12px;padding:2px 12px 2px 0">Energy</th>'
            '<th style="text-align:left;color:#555;font-size:12px;padding:2px 12px 2px 0">λ</th>'
            '<th style="text-align:left;color:#555;font-size:12px">Osc. str.</th></tr>'
            + "".join(exc_rows)
        )
    return (
        f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>TD-DFT / UV-Vis &mdash; {r.formula} ({r.method}/{r.basis})</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f"{header_rows}{exc_table}</table></div>"
    )


def format_nmr_result(r: Any) -> str:
    """Format an NMR shielding result card."""
    _conv = "Yes" if r.converged else "No (treat with caution)"
    _cc = "green" if r.converged else "#c00"
    header_rows = (
        f'<tr><td style="padding:3px 18px 3px 0;color:#444">SCF converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:#444">Reference</td>'
        f'<td style="color:#000">{r.reference_compound} ({r.method}/{r.basis})</td></tr>'
    )

    def _nmr_table(label: str, shifts: list, sym: str) -> str:
        if not shifts:
            return ""
        rows = "".join(
            f"<tr>"
            f'<td style="padding:2px 14px 2px 0;color:#555">{sym}-{n}</td>'
            f'<td style="color:#000">{d:.2f} ppm</td>'
            f"</tr>"
            for n, (_i, d) in enumerate(shifts, 1)
        )
        return (
            f'<tr><td colspan="2" style="padding:8px 0 2px;color:#444;font-weight:bold">'
            f"{label} shifts (vs. TMS):</td></tr>"
            f"<tr>"
            f'<th style="text-align:left;color:#555;font-size:12px;padding:2px 14px 2px 0">Atom</th>'
            f'<th style="text-align:left;color:#555;font-size:12px">δ (ppm)</th></tr>'
            + rows
        )

    h_table = _nmr_table("¹H", r.h_shifts(), "H")
    c_table = _nmr_table("¹³C", r.c_shifts(), "C")

    _basis_warn = ""
    if r.basis.upper() in ("STO-3G", "3-21G"):
        _basis_warn = (
            '<tr><td colspan="2" style="padding:6px 0 0">'
            '<span style="color:#b45309;font-size:12px">'
            f"⚠ {r.basis} gives qualitative NMR only — use 6-31G* or better.</span>"
            "</td></tr>"
        )

    _empty = ""
    if not r.h_shifts() and not r.c_shifts():
        _empty = (
            '<tr><td colspan="2" style="color:#888;font-size:12px">'
            "No ¹H or ¹³C atoms found in this molecule.</td></tr>"
        )

    return (
        f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>NMR Shielding &mdash; {r.formula} ({r.method}/{r.basis})</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f"{header_rows}{h_table}{c_table}{_empty}{_basis_warn}</table></div>"
    )


def format_pes_scan_result(r: Any) -> str:
    """Format a PESScanResult as an HTML result card."""
    _conv = "Yes" if r.converged_all else "No (some points did not converge)"
    _cc = "green" if r.converged_all else "#c00"
    if r.energies_hartree:
        e_min = min(r.energies_hartree)
        e_max = max(r.energies_hartree)
        barrier_kcal = (e_max - e_min) * 627.509474
        _e_row = (
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">Min energy</td>'
            f'<td style="color:#000">{e_min:.8f} Ha</td></tr>'
            f'<tr><td style="padding:3px 18px 3px 0;color:#444">Energy range</td>'
            f'<td style="color:#000">{barrier_kcal:.2f} kcal/mol</td></tr>'
        )
    else:
        _e_row = ""
    _idx_str = "–".join(str(i + 1) for i in r.atom_indices)
    return (
        f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
        f'padding:10px 14px;border-radius:4px;margin:6px 0">'
        f"<b>PES Scan &mdash; {r.formula} ({r.method}/{r.basis})</b>"
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f'<tr><td style="padding:3px 18px 3px 0;color:#444">Scan type</td>'
        f'<td style="color:#000">{r.scan_type.capitalize()} ({_idx_str})</td></tr>'
        f'<tr><td style="padding:3px 18px 3px 0;color:#444">Range</td>'
        f'<td style="color:#000">{r.scan_parameter_values[0]:.3f} → '
        f"{r.scan_parameter_values[-1]:.3f} {r.scan_unit} "
        f"({r.n_steps} points)</td></tr>"
        f"{_e_row}"
        f'<tr><td style="padding:3px 18px 3px 0;color:#444">All converged</td>'
        f'<td style="color:{_cc}">{_conv}</td></tr>'
        f"</table></div>"
    )


def format_past_result(data: dict[str, Any], result_dir: Optional[Path] = None) -> str:
    """Format a saved result.json payload as an HTML result card."""
    import base64 as _b64

    _ct_labels = {
        "single_point": ("Single Point", "#2563eb", "#dbeafe"),
        "geometry_opt": ("Geometry Optimization", "#7c3aed", "#ede9fe"),
        "frequency": ("Frequency Analysis", "#15803d", "#dcfce7"),
        "tddft": ("TD-DFT", "#b45309", "#fef3c7"),
        "nmr": ("NMR", "#0d9488", "#ccfbf1"),
        "pes_scan": ("PES Scan", "#c2410c", "#ffedd5"),
    }
    ct = data.get("calc_type", "")
    _ct_label, _ct_fg, _ct_bg = _ct_labels.get(
        ct, (ct.replace("_", " ").title(), "#555", "#f3f4f6")
    )
    _ct_badge = (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f"background:{_ct_bg};color:{_ct_fg};font-size:12px;font-weight:700;"
        f'letter-spacing:0.03em;margin-bottom:6px">{_ct_label}</span>'
    )
    _conv = "Yes" if data.get("converged") else "No (treat results with caution)"
    _cc = "green" if data.get("converged") else "#c00"
    _gap = (
        f"{data['homo_lumo_gap_ev']:.4f} eV"
        if data.get("homo_lumo_gap_ev") is not None
        else "N/A"
    )
    _rows = "".join(
        f"<tr>"
        f'<td style="padding:3px 18px 3px 0;color:#444">{k}</td>'
        f'<td style="color:{vc}">{v}</td>'
        f"</tr>"
        for k, v, vc in [
            (
                "Total energy",
                f"{data['energy_hartree']:.8f} Ha &ensp;({data['energy_ev']:.4f} eV)",
                "#000",
            ),
            ("HOMO-LUMO gap", _gap, "#000"),
            ("SCF converged", _conv, _cc),
            (
                "SCF iterations",
                (
                    "—"
                    if data.get("n_iterations") in (None, -1)
                    else str(data.get("n_iterations"))
                ),
                "#000",
            ),
        ]
    )
    ts = data.get("timestamp", "")

    # Embed thumbnail if saved
    _thumb_html = ""
    if result_dir is not None:
        _thumb_path = Path(result_dir) / "thumbnail.png"
        if _thumb_path.exists():
            _img_b64 = _b64.b64encode(_thumb_path.read_bytes()).decode()
            _thumb_html = (
                f'<img src="data:image/png;base64,{_img_b64}" '
                f'style="float:right;margin:0 0 6px 14px;border-radius:4px;'
                f'border:1px solid #e2e8f0" width="173" height="108" />'
            )

    return (
        f'<div style="background:#f0fff0;border-left:4px solid #4CAF50;'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;overflow:hidden">'
        f"{_thumb_html}"
        f"{_ct_badge}<br>"
        f'<b>{data["formula"]} &mdash; {data["method"]}/{data["basis"]}</b>'
        f'&ensp;<small style="color:#777">{ts}</small>'
        f'<table style="margin-top:8px;font-size:14px;border-collapse:collapse">'
        f"{_rows}</table></div>"
    )
