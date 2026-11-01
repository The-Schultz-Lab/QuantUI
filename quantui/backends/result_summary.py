"""
Plain-text summary of a batch job's result.json.

Used by ``quantui results JOB_DIR`` inside the image. The ``quantui-batch``
launcher on the login node keeps a standard-library port of ``summarize``
(it cannot import QuantUI); ``tests/test_batch_submit.py`` holds the two
outputs equal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

HARTREE_TO_KJ = 2625.4996


def _num(value: Any, fmt: str) -> str:
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return "n/a"


def summarize(result: Dict[str, Any], job_dir: Path, attempt: Path) -> List[str]:
    """Plain-text summary lines for one result.json."""
    calc = result.get("calc_type", "?")
    lines = [
        f"{job_dir.name}  ({attempt.name})",
        f"  {calc}  {result.get('method', '?')}/{result.get('basis', '?')}"
        + (f"  {result['scf_variant']}" if result.get("scf_variant") else "")
        + (f"  {result['formula']}" if result.get("formula") else ""),
    ]
    converged = result.get("converged")
    lines.append(
        "  converged:        "
        + ("yes" if converged else "NO (treat the numbers with care)")
    )
    if calc != "nmr":
        lines.append(
            f"  energy:           {_num(result.get('energy_hartree'), '.8f')} hartree"
        )
    if result.get("homo_lumo_gap_ev") is not None:
        lines.append(
            f"  HOMO-LUMO gap:    {_num(result['homo_lumo_gap_ev'], '.3f')} eV"
        )
    if result.get("dipole_moment_debye") is not None:
        lines.append(
            f"  dipole moment:    {_num(result['dipole_moment_debye'], '.3f')} D"
        )
    if calc == "geometry_opt":
        lines.append(f"  optimizer steps:  {result.get('n_steps', '?')}")
        lines.append(
            f"  next step:        quantui-batch submit --from {job_dir.name} --calc frequency"
        )
    spectra = result.get("spectra") or {}
    if calc == "transition_state":
        ts = spectra.get("transition_state") or {}
        lines.append(f"  search steps:     {result.get('n_steps', '?')}")
        if ts.get("verdict"):
            lines.append(
                "  verdict:          " + str(ts["verdict"]).replace("cm⁻¹", "cm-1")
            )
    if calc == "irc":
        irc = spectra.get("irc") or {}
        lines.append(f"  path steps:       {result.get('n_steps', '?')}")
        lines.append(
            f"  barrier:          {_num(irc.get('barrier_from_start_kj'), '.1f')} "
            "kJ/mol from the start of the path, "
            f"{_num(irc.get('barrier_from_end_kj'), '.1f')} kJ/mol from its end"
        )
        if irc.get("verdict"):
            lines.append("  verdict:          " + str(irc["verdict"]))
    ir = spectra.get("ir") or {}
    if ir.get("frequencies_cm1"):
        freqs = [float(f) for f in ir["frequencies_cm1"]]
        ints = [float(i) for i in (ir.get("ir_intensities") or [0.0] * len(freqs))]
        imag = [f for f in freqs if f < 0]
        lines.append(f"  vibrational modes: {len(freqs)}")
        if calc == "transition_state":
            # The verdict's own count, which leaves out numerical noise.
            counted = (spectra.get("transition_state") or {}).get("imaginary_cm1")
            counted = imag if counted is None else [float(f) for f in counted]
            shown = ", ".join(f"{abs(f):.1f}i" for f in counted)
            lines.append(
                f"  imaginary modes:  {len(counted)}"
                + (f" ({shown} cm-1)" if counted else "")
                + "; a transition state has exactly one"
            )
        elif imag:
            shown = ", ".join(f"{abs(f):.1f}i" for f in imag)
            lines.append(
                f"  IMAGINARY modes:  {len(imag)} ({shown} cm-1): not a true minimum"
            )
        else:
            lines.append("  imaginary modes:  none (a true minimum)")
        retry = ir.get("imaginary_mode_retry") or {}
        if retry.get("followed_cm1"):
            first = ", ".join(f"{abs(float(f)):.1f}i" for f in retry["followed_cm1"])
            lines.append(
                f"  saddle-point retry: first pass had {first} cm-1; displaced "
                f"and re-optimized ({retry.get('reopt_steps', '?')} steps)"
            )
        strongest = sorted(
            (pair for pair in zip(freqs, ints) if pair[0] > 0),
            key=lambda p: p[1],
            reverse=True,
        )[:5]
        if strongest:
            lines.append("  strongest IR bands (cm-1, unscaled; km/mol):")
            for f, i in sorted(strongest):
                lines.append(f"    {f:9.1f}   {i:8.1f}")
        thermo = ir.get("thermo") or {}
        if thermo.get("G_hartree") is not None:
            lines.append(
                f"  Gibbs energy:     {_num(thermo['G_hartree'], '.8f')} hartree "
                f"at {_num(thermo.get('temperature_k'), '.2f')} K"
            )
    uv = spectra.get("uv_vis") or {}
    if uv.get("excitation_energies_ev"):
        energies = uv["excitation_energies_ev"]
        waves = uv.get("wavelengths_nm") or [None] * len(energies)
        strengths = uv.get("oscillator_strengths") or [None] * len(energies)
        lines.append("  excited states (eV, nm, oscillator strength f):")
        for n, (e, w, f) in enumerate(zip(energies, waves, strengths), 1):
            if n > 8:
                lines.append(f"    ... {len(energies) - 8} more")
                break
            lines.append(
                f"    {n:2d}  {_num(e, '7.3f')}  {_num(w, '7.1f')}  {_num(f, '.4f')}"
            )
    nmr = spectra.get("nmr") or {}
    if nmr.get("chemical_shifts_ppm"):
        symbols = nmr.get("atom_symbols") or []
        lines.append(
            f"  chemical shifts (ppm, vs {nmr.get('reference_compound', 'TMS')}):"
        )
        for idx, shift in sorted(
            nmr["chemical_shifts_ppm"].items(), key=lambda kv: int(kv[0])
        ):
            sym = symbols[int(idx)] if int(idx) < len(symbols) else "?"
            lines.append(f"    {sym}{int(idx) + 1:<4d} {_num(shift, '8.2f')}")
    files = sorted(p.name for p in attempt.iterdir() if p.is_file())
    lines.append(f"  folder:           {attempt}")
    lines.append(f"  files:            {', '.join(files)}")
    return lines
