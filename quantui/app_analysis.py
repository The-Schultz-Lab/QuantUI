"""Analysis panel state and population helpers used by QuantUIApp."""

from __future__ import annotations

import html as _html_mod
import types as _types_mod
from typing import Any, Optional

import ipywidgets as widgets

from . import theme as _theme

_PANEL_UNAVAILABLE_STYLE = (
    "padding:12px 16px;color:#6b7280;font-size:13px;font-style:italic"
)

_MULLIKEN_DIPOLE_TOOLTIP = "Green arrow along μ through the molecular centre of mass"
_MULLIKEN_DIPOLE_TOOLTIP_DISABLED = (
    "Unavailable: this result has no saved μx, μy, μz — re-run to enable the 3D arrow"
)

_CALC_TYPE_LABELS = {
    "single_point": "Single Point",
    "geometry_opt": "Geometry Opt",
    "frequency": "Frequency",
    "tddft": "UV-Vis (TD-DFT)",
    "nmr": "NMR Shielding",
    "pes_scan": "PES Scan",
}

_CALC_TYPE_BADGES = {
    "single_point": "Single Point",
    "geometry_opt": "Geometry Optimization",
    "frequency": "Frequency Analysis",
    "tddft": "UV-Vis (TD-DFT)",
    "nmr": "NMR Shielding",
    "pes_scan": "PES Scan",
}


def _panel_unavailable_html(message: str) -> str:
    return f'<div style="{_PANEL_UNAVAILABLE_STYLE}">{_html_mod.escape(message)}</div>'


def _set_panel_unavailable_message(app: Any, panel_name: str, message: str) -> None:
    panel = app._ana_unavail_msgs.get(panel_name)
    if panel is not None:
        panel.value = _panel_unavailable_html(message)


def _reset_unavailable_messages_for_context(app: Any, ctx: Any) -> None:
    expected_panels = {
        panel_name
        for panel_name, _method_name, _auto in app._PANEL_REGISTRY.get(
            ctx.calc_type, []
        )
    }
    calc_label = _CALC_TYPE_LABELS.get(
        ctx.calc_type,
        str(ctx.calc_type).replace("_", " ").title(),
    )
    for panel_name in app._ana_panel_names:
        if panel_name in expected_panels:
            _set_panel_unavailable_message(
                app,
                panel_name,
                (
                    f"Not available for this {calc_label} result: "
                    "required data is missing or could not be loaded."
                ),
            )
            continue
        when = app._ana_when_map.get(panel_name, "relevant")
        _set_panel_unavailable_message(
            app,
            panel_name,
            f"Not available - run a {when} calculation first.",
        )


def _analysis_heading_label(ctx: Any) -> str:
    """Return the analysis heading text aligned with history dropdown labels."""
    badge = _CALC_TYPE_BADGES.get(ctx.calc_type, str(ctx.calc_type or "Unknown"))
    core = f"[{badge}] {ctx.label}"
    ts = str(getattr(ctx, "timestamp", "") or "").strip()
    return f"{ts}  ·  {core}" if ts else core


def build_ana_switcher(app: Any, *, layout_fn: Any) -> None:
    """Initialise analysis panel state and wire accordion re-render observers."""
    panel_meta = [
        (name, getattr(app, attr), when) for name, attr, when in app._PANEL_META
    ]
    app._ana_when_map = {name: when for name, _acc, when in panel_meta}
    app._ana_panel_names = [m[0] for m in panel_meta]
    app._ana_accordions = [m[1] for m in panel_meta]
    app._ana_available = set()
    app._ana_active = ""
    app._ana_unavail_html = widgets.HTML(
        value="",
        layout=layout_fn(display="none", margin="4px 0 8px"),
    )

    # Wrap each accordion child with both an unavailable message and real content.
    app._ana_unavail_msgs = {}
    app._ana_content_boxes = {}
    for name, acc, when in panel_meta:
        unavail = widgets.HTML(
            value=_panel_unavailable_html(
                f"Not available - run a {when} calculation first."
            ),
            layout=layout_fn(display=""),
        )
        content = acc.children[0]
        app._ana_unavail_msgs[name] = unavail
        app._ana_content_boxes[name] = content
        content.layout.display = "none"
        acc.children = (widgets.VBox([unavail, content]),)
        acc.layout.display = ""  # always in the DOM
        acc.selected_index = None  # collapsed until activated

    # Re-render Plotly charts when their accordion is expanded by header click.
    app._ir_accordion.observe(
        app._safe_cb(app._on_ir_accordion_show), names=["selected_index"]
    )
    app._tddft_accordion.observe(
        app._safe_cb(app._on_tddft_accordion_show), names=["selected_index"]
    )
    app._orb_accordion.observe(
        app._safe_cb(app._on_orb_accordion_show), names=["selected_index"]
    )
    app._mulliken_accordion.observe(
        app._safe_cb(app._on_mulliken_accordion_show), names=["selected_index"]
    )


def select_ana_panel(app: Any, name: str) -> None:
    """Expand the named panel and collapse all others."""
    app._ana_active = name
    app._ana_unavail_html.layout.display = "none"
    for panel_name, acc in zip(app._ana_panel_names, app._ana_accordions):
        acc.selected_index = 0 if panel_name == name else None


def activate_ana_panel(app: Any, name: str, auto_select: bool = True) -> None:
    """Mark a panel as available and reveal its content."""
    app._ana_available.add(name)
    if name in app._ana_unavail_msgs:
        app._ana_unavail_msgs[name].layout.display = "none"
        app._ana_content_boxes[name].layout.display = ""
    if auto_select:
        app._select_ana_panel(name)


def deactivate_all_ana_panels(app: Any) -> None:
    """Reset all panels to collapsed/unavailable for a new run/context."""
    app._ana_available.clear()
    app._ana_active = ""
    app._ana_unavail_html.layout.display = "none"
    for name, acc in zip(app._ana_panel_names, app._ana_accordions):
        if name in app._ana_unavail_msgs:
            app._ana_unavail_msgs[name].layout.display = ""
            app._ana_content_boxes[name].layout.display = "none"
        acc.selected_index = None


def apply_analysis_context(app: Any, ctx: Any) -> None:
    """Populate Analysis panels from context and activate panels with data."""
    app._deactivate_all_ana_panels()
    _reset_unavailable_messages_for_context(app, ctx)
    app._pending_traj_result = None
    # Safety-net cache so on_traj_expand can recover if the initial render's
    # outputs are missing from traj_output by the time the user views the
    # accordion. Cleared here at context reset; re-set by each
    # _pop_*_trajectory populate method.
    app._last_traj_result = None
    app._traj_render_token = int(getattr(app, "_traj_render_token", 0)) + 1
    app._iso_render_token = int(getattr(app, "_iso_render_token", 0)) + 1
    # Orbital state is consumed by pop_isosurface (and ana_pop_iso_generate
    # when the user clicks Generate). Reset here so a context that doesn't
    # populate these fields (history result without orbitals.npz)
    # cannot leak the prior calc's orbital arrays into the Isosurface panel
    # of an unrelated molecule. Each populate method that wants the panel
    # to activate re-sets these in show_orbital_diagram.
    app._last_orb_info = None
    app._last_orb_mo_coeff = None
    app._last_orb_mo_occ = None
    # Mulliken state consumed by the Populations panel — reset so a context
    # without charges cannot leak the prior calc's chart into this one.
    app._last_mulliken_symbols = None
    app._last_mulliken_charges = None
    app._last_mulliken_dipole = None
    app._last_mulliken_dipole_vector = None
    app._last_mulliken_fig = None
    dip_cb = getattr(app, "_mulliken_dipole_cb", None)
    if dip_cb is not None:
        dip_cb.disabled = False
        dip_cb.value = True
        dip_cb.tooltip = _MULLIKEN_DIPOLE_TOOLTIP
    color_cb = getattr(app, "_mulliken_color_cb", None)
    if color_cb is not None:
        color_cb.value = True
    note = getattr(app, "_mulliken_overlay_note", None)
    if note is not None:
        note.value = ""
    app._last_orb_mol_atom = None
    app._last_orb_mol_basis = None
    app.traj_accordion.set_title(0, "Trajectory Viewer")
    # traj_output is a VBox (see app_builders.py traj_output construction);
    # clear children instead of clear_output.
    app.traj_output.children = ()
    app._orb_iso_output.clear_output()

    first_auto_selected = False
    expected_panels = {
        panel_name
        for panel_name, _method_name, _want_auto in app._PANEL_REGISTRY.get(
            ctx.calc_type, []
        )
    }
    for panel_name, method_name, want_auto in app._PANEL_REGISTRY.get(
        ctx.calc_type, []
    ):
        try:
            ok = bool(getattr(app, method_name)(ctx))
        except Exception as panel_exc:
            ok = False
            try:
                from quantui import calc_log as _clog

                _clog.log_event(
                    "ana_panel_error",
                    f"{method_name}: {type(panel_exc).__name__}: {panel_exc}"[:300],
                )
            except Exception:
                pass
        if ok:
            do_auto = want_auto and not first_auto_selected
            app._activate_ana_panel(panel_name, auto_select=do_auto)
            if do_auto:
                first_auto_selected = True

    missing_expected = sorted(expected_panels - app._ana_available)
    if missing_expected:
        try:
            from quantui import calc_log as _clog

            _clog.log_event(
                "ana_expected_panel_missing",
                f"{ctx.calc_type}: {', '.join(missing_expected)}"[:300],
                calc_type=ctx.calc_type,
                source=ctx.source,
                missing_panels=missing_expected,
            )
        except Exception:
            pass

    source_suffix = " (from History)" if ctx.source == "history" else ""
    heading = _analysis_heading_label(ctx)
    app._analysis_context_lbl.value = (
        f'<p style="color:{_theme.TEXT_SECONDARY};font-size:13px;margin:4px 0 12px">'
        f"Analysing: {_html_mod.escape(heading)}{source_suffix}</p>"
    )
    has_any = bool(app._ana_available)
    app._to_analysis_btn.layout.display = "" if has_any else "none"
    app._analysis_empty_html.layout.display = "none" if has_any else ""


def pop_energies(app: Any, ctx: Any) -> bool:
    """Populate Energies panel from live result or history orbitals."""
    result = ctx.live_result
    if result is None and ctx.result_dir is not None:
        try:
            from quantui.results_storage import load_orbitals

            orb = load_orbitals(ctx.result_dir)
            orb.formula = ctx.formula
            result = orb
        except Exception:
            return False
    return bool(app._show_orbital_diagram(result))


def pop_reorg_geometries(app: Any, ctx: Any) -> bool:
    """Populate the Geometries panel for a reorganization-energy result.

    Works from a live result OR a saved one by reading the same channel payload
    both now carry — the shape REORG.1 introduced. That is deliberate: the
    original bug was two paths reading different things, so this one never had
    the chance to grow a second reader.

    Returns False when the payload is absent, which is exactly the pre-REORG.1
    case; the panel then stays unavailable rather than showing an empty viewer,
    and the results card explains why.
    """
    channels, neutral = _reorg_payload(app, ctx)
    if not channels or not neutral:
        _set_panel_unavailable_message(
            app,
            "Geometries",
            (
                "Not available for this result: the per-channel geometries were "
                "not saved. Re-run the calculation to enable the comparison."
            ),
        )
        return False

    from quantui.reorganization_energy import reorg_geometries

    geoms = reorg_geometries(channels, neutral)
    if len(geoms) < 2:
        return False
    app._reorg_geometries = geoms
    app._reorg_overlay_pair.options = [
        (f"{geoms[0]['label'].split(' — ')[0]} vs {g['label'].split(' — ')[0]}", i)
        for i in range(1, len(geoms))
        for g in [geoms[i]]
    ]
    if app._reorg_overlay_pair.options:
        app._reorg_overlay_pair.value = app._reorg_overlay_pair.options[0][1]
    render_reorg_geometries(app)
    return True


def _reorg_payload(app: Any, ctx: Any) -> tuple[list, dict]:
    """Channel list + neutral geometry, from a live result or a saved one."""
    live = getattr(ctx, "live_result", None)
    if live is not None and getattr(live, "channels", None):
        from quantui.results_storage import _reorg_channels_payload

        payload = _reorg_channels_payload(live) or []
        neutral = payload[0].get("neutral_geometry") if payload else None
        return payload, neutral or {}

    result_dir = getattr(ctx, "result_dir", None)
    if result_dir is not None:
        try:
            from quantui import load_result

            data = load_result(result_dir)
            payload = data.get("reorg_channels") or []
            neutral = payload[0].get("neutral_geometry") if payload else None
            return payload, neutral or {}
        except Exception:  # noqa: BLE001 — a missing panel, never a crash
            return [], {}
    return [], {}


def render_reorg_geometries(app: Any) -> None:
    """Draw the current view (stepper or overlay) into the Geometries panel."""
    geoms = getattr(app, "_reorg_geometries", None)
    if not geoms:
        return
    from quantui.app_visualization import (
        build_reorg_geometry_viewer_html,
        build_reorg_overlay_html,
    )

    bg = app._plotly_theme_colors()["scene_bgcolor"]
    try:
        if app._reorg_view_toggle.value == "overlay":
            idx = int(app._reorg_overlay_pair.value or 1)
            html = build_reorg_overlay_html(
                geoms[0],
                geoms[idx],
                bgcolor=bg,
                exaggerate=float(getattr(app._reorg_exaggerate, "value", 1.0)),
            )
        else:
            from quantui.app_builders import _REORG_PNG_INBOX_CLASS

            capture_class = (
                _REORG_PNG_INBOX_CLASS
                if getattr(app, "_reorg_png_inbox", None) is not None
                else ""
            )
            html = build_reorg_geometry_viewer_html(
                geoms, bgcolor=bg, capture_class=capture_class
            )
        app._set_html_output(app._reorg_geom_output, html)
    except Exception as exc:  # noqa: BLE001
        app._set_html_output(
            app._reorg_geom_output,
            f'<p style="color:{_theme.ACCENT_ERROR};padding:8px">Geometry view failed: {exc}</p>',
        )


def on_reorg_view_changed(app: Any, change: Any = None) -> None:
    """Toggle between stepper and overlay; the pair picker only applies to one."""
    is_overlay = app._reorg_view_toggle.value == "overlay"
    app._reorg_overlay_pair.layout.display = "" if is_overlay else "none"
    # Only meaningful for the overlay — the stepper shows true positions.
    app._reorg_exaggerate.layout.display = "" if is_overlay else "none"
    render_reorg_geometries(app)


def pop_isosurface(app: Any, ctx: Any) -> bool:
    """Populate Isosurface availability from orbital state.

    Uses ``getattr(..., None)`` for the orbital state fields rather than
    direct attribute access. The attributes are initialized in
    ``QuantUIApp.__init__`` and reset in ``apply_analysis_context`` so they
    are always present in practice, but the defensive read mirrors the
    pattern used by ``render_orbital_isosurface`` and keeps this populator
    robust against future refactors that might call it before the context
    reset has run.
    """
    return (
        getattr(app, "_last_orb_mo_coeff", None) is not None
        and getattr(app, "_last_orb_mol_atom", None) is not None
        and getattr(app, "_last_orb_mol_basis", None) is not None
    )


def pop_geo_trajectory(app: Any, ctx: Any) -> bool:
    """Populate Trajectory panel for geometry optimization contexts."""
    traj = None
    energies: list = []
    if ctx.live_result is not None:
        traj = getattr(ctx.live_result, "trajectory", None)
        energies = list(getattr(ctx.live_result, "energies_hartree", []))
    elif ctx.result_dir is not None:
        traj_file = ctx.result_dir / "trajectory.json"
        if not traj_file.exists():
            _set_panel_unavailable_message(
                app,
                "Trajectory",
                (
                    "Not available for this Geometry Opt history result: "
                    "trajectory.json is missing."
                ),
            )
            return False
        try:
            from quantui.results_storage import load_trajectory

            traj, energies = load_trajectory(ctx.result_dir)
        except Exception as exc:
            _set_panel_unavailable_message(
                app,
                "Trajectory",
                (
                    "Not available for this Geometry Opt history result: "
                    f"failed to load trajectory data ({type(exc).__name__})."
                ),
            )
            return False
    if not traj or len(traj) < 2:
        _set_panel_unavailable_message(
            app,
            "Trajectory",
            (
                "Not available for this Geometry Opt result: "
                "trajectory data has fewer than 2 frames."
            ),
        )
        return False
    stub = _types_mod.SimpleNamespace(
        trajectory=traj,
        energies_hartree=energies,
        formula=ctx.formula,
    )
    app._pending_traj_result = stub
    app._last_traj_result = stub
    return True


def pop_preopt_trajectory(app: Any, ctx: Any) -> bool:
    """Populate Trajectory panel for the frequency-time DFT geometry
    optimization trajectory.

    The wrapped operation is a full DFT geom-opt
    at the user's method/basis, not the classical LJ pre-opt that lives
    in ``quantui/preopt.py``. The function name + ``preopt_trajectory.json``
    filename stay (renaming the saved file would break history replay of
    older results) but user-facing strings now say "geometry optimization".
    """
    if ctx.source == "live":
        pre = ctx.preopt_result
        if pre is None:
            return False
        traj = getattr(pre, "trajectory", None)
        energies = list(getattr(pre, "energies_hartree", []))
    else:
        if ctx.result_dir is None:
            return False
        preopt_path = ctx.result_dir / "preopt_trajectory.json"
        if not preopt_path.exists():
            _set_panel_unavailable_message(
                app,
                "Trajectory",
                (
                    "Not available for this Frequency history result: "
                    "preopt_trajectory.json is missing (geometry "
                    "optimization may have been disabled)."
                ),
            )
            return False
        try:
            from quantui.results_storage import load_trajectory

            traj, energies = load_trajectory(
                ctx.result_dir, filename="preopt_trajectory.json"
            )
        except Exception as exc:
            from quantui import calc_log as _clog

            _clog.log_event(
                "pop_preopt_trajectory_error",
                f"{type(exc).__name__}: {exc}"[:300],
            )
            _set_panel_unavailable_message(
                app,
                "Trajectory",
                (
                    "Not available for this Frequency history result: "
                    f"failed to load geometry-optimization trajectory "
                    f"({type(exc).__name__})."
                ),
            )
            return False
    if not traj or len(traj) < 2:
        _set_panel_unavailable_message(
            app,
            "Trajectory",
            (
                "Not available for this Frequency result: "
                "geometry-optimization trajectory has fewer than 2 frames."
            ),
        )
        return False
    stub = _types_mod.SimpleNamespace(
        trajectory=traj,
        energies_hartree=energies,
        formula=ctx.formula,
    )
    app._pending_traj_result = stub
    app._last_traj_result = stub
    app.traj_accordion.set_title(0, "Geometry Optimization Trajectory")
    return True


def pop_vibrational(app: Any, ctx: Any) -> bool:
    """Populate Vibrational panel from live or history frequency data."""
    if ctx.live_result is not None:
        freq_stub = ctx.live_result
        mol = ctx.molecule
    else:
        ir = ctx.spectra_data.get("ir", {})
        mol_data = ctx.spectra_data.get("molecule", {})
        freqs = ir.get("frequencies_cm1")
        ints = ir.get("ir_intensities")
        disps = ir.get("displacements")
        if not freqs:
            _set_panel_unavailable_message(
                app,
                "Vibrational",
                (
                    "Not available for this Frequency history result: "
                    "no frequency data was saved (`frequencies_cm1` empty "
                    "or missing). Re-run the Frequency calculation to "
                    "populate this panel."
                ),
            )
            return False
        if not mol_data.get("atoms"):
            _set_panel_unavailable_message(
                app,
                "Vibrational",
                (
                    "Not available for this Frequency history result: "
                    "no molecule geometry was saved with the result. "
                    "Re-run the Frequency calculation to populate this panel."
                ),
            )
            return False
        if not disps:
            _set_panel_unavailable_message(
                app,
                "Vibrational",
                (
                    "Not available for this Frequency history result: "
                    "per-mode atomic displacements were not persisted with "
                    "this calculation (a known limitation for older saved "
                    "results — displacements began being written to disk in "
                    "a later QuantUI version). Re-run the Frequency "
                    "calculation to enable the animation. "
                    "The IR Spectrum panel still works for this result."
                ),
            )
            return False
        from quantui.molecule import Molecule as _Mol

        mol = _Mol(
            atoms=mol_data["atoms"],
            coordinates=mol_data["coords"],
            charge=mol_data.get("charge", 0),
            multiplicity=mol_data.get("multiplicity", 1),
        )
        freq_stub = _types_mod.SimpleNamespace(
            frequencies_cm1=freqs,
            ir_intensities=ints,
            displacements=disps,
        )
    return bool(app._show_vib_animation(freq_stub, mol))


def pop_ir_spectrum(app: Any, ctx: Any) -> bool:
    """Populate IR panel from live or history frequency data."""
    if ctx.live_result is not None:
        freq_stub = ctx.live_result
    else:
        ir = ctx.spectra_data.get("ir", {})
        freqs = ir.get("frequencies_cm1")
        if not freqs:
            return False
        freq_stub = _types_mod.SimpleNamespace(
            frequencies_cm1=freqs,
            ir_intensities=ir.get("ir_intensities") or [],
        )
    return bool(app._show_ir_spectrum(freq_stub))


def pop_uv_vis(app: Any, ctx: Any) -> bool:
    """Populate UV-Vis panel from live or history TDDFT data."""
    if ctx.live_result is not None:
        energies_ev = list(getattr(ctx.live_result, "excitation_energies_ev", []))
        osc = list(getattr(ctx.live_result, "oscillator_strengths", []))
        try:
            wl = list(ctx.live_result.wavelengths_nm())
        except Exception:
            wl = [1240.0 / e for e in energies_ev if e > 0]
    else:
        uv = ctx.spectra_data.get("uv_vis", {})
        energies_ev = list(uv.get("excitation_energies_ev") or [])
        osc = list(uv.get("oscillator_strengths") or [])
        wl = list(uv.get("wavelengths_nm") or [])
    if not energies_ev or not osc:
        return False
    return bool(app._show_uv_vis_spectrum(energies_ev, osc, wl))


def pop_nmr_shielding(app: Any, ctx: Any) -> bool:
    """Populate NMR panel from live or history shielding data."""
    if ctx.live_result is not None:
        result = ctx.live_result
        atom_symbols = list(getattr(result, "atom_symbols", []))
        shielding = list(getattr(result, "shielding_iso_ppm", []))
        try:
            h_shifts = result.h_shifts()
            c_shifts = result.c_shifts()
        except Exception:
            h_shifts, c_shifts = [], []
        ref = getattr(result, "reference_compound", "TMS")
    else:
        nmr = ctx.spectra_data.get("nmr", {})
        atom_symbols = nmr.get("atom_symbols", [])
        shielding = nmr.get("shielding_iso_ppm", [])
        chem = nmr.get("chemical_shifts_ppm", {})
        ref = nmr.get("reference_compound", "TMS")
        h_shifts = [
            (int(i), d)
            for i, d in chem.items()
            if int(i) < len(atom_symbols) and atom_symbols[int(i)] == "H"
        ]
        c_shifts = [
            (int(i), d)
            for i, d in chem.items()
            if int(i) < len(atom_symbols) and atom_symbols[int(i)] == "C"
        ]
    if not atom_symbols:
        return False

    def _shift_table(label: str, shifts: list, sym: str) -> str:
        if not shifts:
            return ""
        rows = "".join(
            f'<tr><td style="padding:2px 14px 2px 0;color:{_theme.TEXT_SECONDARY}">{sym}-{n}</td>'
            f'<td style="color:{_theme.TEXT_HEADING}">{d:.2f} ppm</td></tr>'
            for n, (_i, d) in enumerate(sorted(shifts, key=lambda x: x[0]), 1)
        )
        return (
            f'<tr><td colspan="2" style="padding:8px 0 2px;font-weight:600">'
            f"{label} shifts (vs. {ref}):</td></tr>"
            f'<tr><th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px;padding:2px 14px 2px 0">Atom</th>'
            f'<th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px">δ (ppm)</th></tr>'
            + rows
        )

    shielding_rows = "".join(
        f'<tr><td style="padding:2px 10px 2px 0;color:{_theme.TEXT_SECONDARY}">{sym}{i + 1}</td>'
        f'<td style="color:{_theme.TEXT_HEADING}">{s:.2f}</td></tr>'
        for i, (sym, s) in enumerate(zip(atom_symbols, shielding))
    )
    html = (
        f'<div style="font-size:13px">'
        f'<table style="border-collapse:collapse;margin-bottom:8px">'
        f'<tr><th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px;padding:2px 10px 2px 0">Atom</th>'
        f'<th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px">σ (ppm)</th></tr>'
        f"{shielding_rows}</table>"
        f'<table style="border-collapse:collapse">'
        f"{_shift_table('¹H', h_shifts, 'H')}"
        f"{_shift_table('¹³C', c_shifts, 'C')}"
        f"</table></div>"
    )
    app._nmr_output.value = html
    return True


def _mulliken_payload(
    ctx: Any,
) -> tuple[list[str], list[float], Optional[float], Optional[list[float]]]:
    """Atom symbols, Mulliken charges, ‖μ‖, and μ vector from live or history.

    History loads top-level ``result.json`` fields (already persisted by
    ``save_result``) rather than ``spectra`` — charges are not spectra data.
    Older results may lack ``dipole_vector_debye``; magnitude still loads.
    """
    if ctx.live_result is not None:
        result = ctx.live_result
        symbols = list(getattr(result, "atom_symbols", None) or [])
        charges = list(getattr(result, "mulliken_charges", None) or [])
        dip = getattr(result, "dipole_moment_debye", None)
        dip_f = float(dip) if dip is not None else None
        vec_raw = getattr(result, "dipole_vector_debye", None)
        vec = [float(x) for x in vec_raw] if vec_raw is not None else None
        if vec is not None and len(vec) < 3:
            vec = None
        return symbols, charges, dip_f, vec

    result_dir = getattr(ctx, "result_dir", None)
    if result_dir is None:
        return [], [], None, None
    try:
        from quantui import load_result

        data = load_result(result_dir)
    except Exception:  # noqa: BLE001 — missing panel, never a crash
        return [], [], None, None
    symbols = list(data.get("atom_symbols") or [])
    charges_raw = data.get("mulliken_charges")
    if charges_raw is None:
        return symbols, [], None, None
    charges = [float(c) for c in charges_raw]
    dip = data.get("dipole_moment_debye")
    dip_f = float(dip) if dip is not None else None
    vec_raw = data.get("dipole_vector_debye")
    vec = [float(x) for x in vec_raw] if vec_raw is not None else None
    if vec is not None and len(vec) < 3:
        vec = None
    return symbols, charges, dip_f, vec


def pop_mulliken(app: Any, ctx: Any) -> bool:
    """Populate the Mulliken Populations panel (table + bar chart + overlays).

    Returns False when charges are missing so the accordion stays on its
    placeholder (reflections/07 Rules 2–4). Both live and history paths share
    this code via :func:`_mulliken_payload`.
    """
    try:
        symbols, charges, dipole, dip_vec = _mulliken_payload(ctx)
        if not symbols or not charges or len(symbols) != len(charges):
            if getattr(ctx, "source", "") == "history":
                msg = (
                    "Mulliken charges were not saved with this History result. "
                    "Re-run Single Point or Geometry Opt to populate this panel "
                    "(table, chart, charge colours, and dipole arrow)."
                )
            else:
                msg = (
                    "Mulliken charges are not available for this result "
                    "(the run may have stopped before population analysis)."
                )
            _set_panel_unavailable_message(app, "Populations", msg)
            return False
        return bool(
            show_mulliken_populations(
                app, symbols, charges, dipole, dipole_vector=dip_vec
            )
        )
    except Exception as exc:
        try:
            from quantui import calc_log as _clog

            _clog.log_event(
                "mulliken_populate_error",
                f"{type(exc).__name__}: {exc}"[:300],
                calc_type=getattr(ctx, "calc_type", ""),
            )
        except Exception:  # noqa: BLE001 — telemetry self-guard
            pass
        _set_panel_unavailable_message(
            app,
            "Populations",
            (
                "Could not load Mulliken data for this result "
                f"({type(exc).__name__})."
            ),
        )
        return False


def show_mulliken_populations(
    app: Any,
    atom_symbols: list,
    charges: list,
    dipole_debye: Optional[float] = None,
    *,
    dipole_vector: Optional[list] = None,
) -> bool:
    """Write Mulliken table + Plotly chart into the Populations accordion."""
    if not atom_symbols or not charges or len(atom_symbols) != len(charges):
        return False

    app._last_mulliken_symbols = list(atom_symbols)
    app._last_mulliken_charges = [float(c) for c in charges]
    app._last_mulliken_dipole = (
        float(dipole_debye) if dipole_debye is not None else None
    )
    if dipole_vector is not None and len(dipole_vector) >= 3:
        app._last_mulliken_dipole_vector = [float(x) for x in dipole_vector[:3]]
        if app._last_mulliken_dipole is None:
            import math

            app._last_mulliken_dipole = math.sqrt(
                sum(v * v for v in app._last_mulliken_dipole_vector)
            )
    else:
        app._last_mulliken_dipole_vector = None

    # Enable/disable the dipole toggle when no vector is available (old History).
    has_vec = app._last_mulliken_dipole_vector is not None
    dip_cb = getattr(app, "_mulliken_dipole_cb", None)
    if dip_cb is not None:
        dip_cb.disabled = not has_vec
        if not has_vec:
            dip_cb.value = False
        dip_cb.tooltip = (
            _MULLIKEN_DIPOLE_TOOLTIP if has_vec else _MULLIKEN_DIPOLE_TOOLTIP_DISABLED
        )

    note = getattr(app, "_mulliken_overlay_note", None)
    if note is not None:
        if has_vec:
            note.value = ""
        else:
            note.value = (
                f'<p style="font-size:12px;color:{_theme.TEXT_MUTED_LIGHT};'
                'margin:0 0 8px;font-style:italic">'
                "Dipole arrow needs saved μ<sub>x</sub>, μ<sub>y</sub>, "
                "μ<sub>z</sub> — re-run this calculation to enable.</p>"
            )

    q_sum = sum(app._last_mulliken_charges)
    summary_bits = [
        f"Sum of Mulliken charges: <b>{q_sum:+.4f} e</b>"
        f' <span style="color:{_theme.TEXT_MUTED_LIGHT};font-size:12px">'
        "(should match the molecular charge)</span>"
    ]
    dip_vec = app._last_mulliken_dipole_vector
    if app._last_mulliken_dipole is not None:
        if dip_vec is not None:
            vx, vy, vz = dip_vec
            summary_bits.append(
                f"Dipole moment: <b>{app._last_mulliken_dipole:.4f} D</b>"
                f' <span style="color:{_theme.TEXT_MUTED_LIGHT};font-size:12px">'
                f"(μ = [{vx:+.3f}, {vy:+.3f}, {vz:+.3f}] D — SCF density)</span>"
            )
        else:
            summary_bits.append(
                f"Dipole moment: <b>{app._last_mulliken_dipole:.4f} D</b>"
                f' <span style="color:{_theme.TEXT_MUTED_LIGHT};font-size:12px">'
                "(magnitude only — re-run to enable the 3D arrow)</span>"
            )
    app._mulliken_summary.value = (
        f'<p style="font-size:13px;color:{_theme.TEXT_HEADING};margin:0 0 8px">'
        + "<br>".join(summary_bits)
        + "</p>"
    )

    rows = "".join(
        f'<tr><td style="padding:2px 14px 2px 0;color:{_theme.TEXT_SECONDARY}">'
        f"{_html_mod.escape(sym)}{i + 1}</td>"
        f'<td style="color:{_theme.TEXT_HEADING};font-family:monospace">'
        f"{chg:+.4f}</td></tr>"
        for i, (sym, chg) in enumerate(
            zip(app._last_mulliken_symbols, app._last_mulliken_charges)
        )
    )
    app._mulliken_table.value = (
        f'<table style="border-collapse:collapse;margin:0 0 10px;font-size:13px">'
        f'<tr><th style="text-align:left;color:{_theme.TEXT_SECONDARY};'
        f'font-size:12px;padding:2px 14px 2px 0">Atom</th>'
        f'<th style="text-align:left;color:{_theme.TEXT_SECONDARY};'
        f'font-size:12px">Charge (e)</th></tr>'
        f"{rows}</table>"
    )

    update_mulliken_figure(app)
    # Push 3D overlays after the Analysis viewer has had a tick to finish
    # attaching (same retry spirit as measure's inject JS).
    try:
        from quantui.populations_overlay import push_populations_overlay

        push_populations_overlay(app)
    except Exception:  # noqa: BLE001 — overlay must never block the panel
        pass
    return True


def update_mulliken_figure(app: Any) -> None:
    """Re-render the Mulliken bar chart into ``app._mulliken_fig``."""
    symbols = getattr(app, "_last_mulliken_symbols", None)
    charges = getattr(app, "_last_mulliken_charges", None)
    if not symbols or not charges:
        return
    try:
        import plotly.io as _pio

        from quantui.mulliken_plot import plot_mulliken_charges

        fig = plot_mulliken_charges(symbols, charges)
        app._apply_plotly_theme(fig)
        app._last_mulliken_fig = fig
        app._set_html_output(
            app._mulliken_fig,
            _pio.to_html(
                fig,
                include_plotlyjs="require",
                full_html=False,
                config={"responsive": True},
            ),
        )
    except Exception as exc:
        app._last_mulliken_fig = None
        try:
            from quantui import calc_log as _clog

            _clog.log_event("mulliken_fig_error", f"{type(exc).__name__}: {exc}"[:300])
        except Exception:  # noqa: BLE001 — telemetry self-guard
            pass


def pop_pes_plot(app: Any, ctx: Any) -> bool:
    """Populate PES plot panel from live or history scan data."""
    result = ctx.live_result
    if result is None:
        scan = ctx.spectra_data.get("pes_scan", {})
        if not scan or not scan.get("energies_hartree"):
            return False
        energies_ha = scan["energies_hartree"]
        atom_indices = scan.get("atom_indices", [])
        scan_type = scan.get("scan_type", "bond")
        x_vals = scan.get("scan_parameter_values", [])
        e_min = min(energies_ha)
        hartree_to_kcal = 627.5094740631
        e_rel = [(e - e_min) * hartree_to_kcal for e in energies_ha]
        idx = [i + 1 for i in atom_indices]
        if scan_type == "bond":
            label = f"Bond {idx[0]}–{idx[1]} / Å" if len(idx) >= 2 else "Bond / Å"
        elif scan_type == "angle":
            label = (
                f"Angle {idx[0]}–{idx[1]}–{idx[2]} / °"
                if len(idx) >= 3
                else "Angle / °"
            )
        else:
            label = (
                f"Dihedral {idx[0]}–{idx[1]}–{idx[2]}–{idx[3]} / °"
                if len(idx) >= 4
                else "Dihedral / °"
            )
        result = _types_mod.SimpleNamespace(
            scan_type=scan_type,
            atom_indices=atom_indices,
            scan_parameter_values=x_vals,
            energies_hartree=energies_ha,
            energies_relative_kcal=e_rel,
            scan_coordinate_label=label,
            converged_all=True,
        )
    return bool(app._show_pes_scan_result(result))


def pop_pes_trajectory(app: Any, ctx: Any) -> bool:
    """Populate Trajectory panel from live or history PES scan data."""
    traj: list = []
    energies: list = []
    if ctx.live_result is not None:
        traj = list(getattr(ctx.live_result, "coordinates_list", []))
        energies = list(getattr(ctx.live_result, "energies_hartree", []))
    elif ctx.result_dir is not None:
        traj_file = ctx.result_dir / "trajectory.json"
        if not traj_file.exists():
            _set_panel_unavailable_message(
                app,
                "Trajectory",
                (
                    "Not available for this PES Scan history result: "
                    "trajectory.json is missing."
                ),
            )
            return False
        try:
            from quantui.results_storage import load_trajectory

            traj, energies = load_trajectory(ctx.result_dir)
        except Exception as exc:
            _set_panel_unavailable_message(
                app,
                "Trajectory",
                (
                    "Not available for this PES Scan history result: "
                    f"failed to load trajectory data ({type(exc).__name__})."
                ),
            )
            return False
    if not traj or len(traj) < 2:
        _set_panel_unavailable_message(
            app,
            "Trajectory",
            (
                "Not available for this PES Scan result: "
                "trajectory data has fewer than 2 frames."
            ),
        )
        return False
    stub = _types_mod.SimpleNamespace(
        coordinates_list=traj,
        energies_hartree=energies,
        trajectory=None,
        formula=ctx.formula,
    )
    app._pending_traj_result = stub
    app._last_traj_result = stub
    app.traj_accordion.set_title(0, "Geometry at Each Scan Point")
    return True
