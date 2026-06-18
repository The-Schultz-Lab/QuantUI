"""UI builder helpers used by QuantUIApp."""

from __future__ import annotations

import os
import sys
from typing import Any

import ipywidgets as widgets
from IPython.display import HTML, display

import quantui
from quantui import molecule_library as _ml
from quantui.help_content import HELP_TOPICS

# Friendlier labels for the library category filter.
_CATEGORY_LABELS = {
    "diatomic": "Diatomics",
    "triatomic": "Triatomics",
    "small-organic": "Small organics",
    "aromatic": "Aromatics",
    "amino-acid": "Amino acids",
    "nucleobase": "Nucleobases",
    "biomolecule": "Biomolecules",
    "solvent": "Solvents",
    "functional-group": "Functional groups",
    "hydrocarbon": "Hydrocarbons",
    "drug": "Drugs",
    "inorganic": "Inorganics",
    "ion": "Ions",
    "bulk-qm9": "Bulk (QM9)",
}


def _category_label(cat: str) -> str:
    return _CATEGORY_LABELS.get(cat, cat.replace("-", " ").title())


def library_result_options(query: str = "", category: str | None = None):
    """Build (label, id) options for the library results dropdown + a count note.

    Returns ``(options, note)``. ``options`` always leads with a placeholder so
    the dropdown's value can be reset without triggering a load.
    """
    limit = 200
    rows = _ml.search(query, category=category, limit=limit)
    opts = [("— select a molecule —", "")]
    for r in rows:
        opts.append((f"{r['name']}  ·  {r['formula']}", r["id"]))
    note = f"{len(rows)} match" + ("" if len(rows) == 1 else "es")
    if len(rows) >= limit:
        note = f"showing first {limit} — narrow with search/category"
    return opts, note


def build_status_panel(
    app: Any,
    *,
    layout_fn: Any,
    get_session_resources_fn: Any,
    load_last_calibration_label_fn: Any,
    pyscf_available: bool,
    ase_available: bool,
    pubchem_available: bool,
    visualization_available: bool,
    viz_default_backend: str = "auto",
    vib_framerate_fps: int = 10,
) -> None:
    """Build the Status tab panel."""
    cores, mem_gb = get_session_resources_fn()
    mem = f"{mem_gb} GB" if mem_gb is not None else "unknown"
    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    env = os.environ.get("CONDA_DEFAULT_ENV", "") or os.path.basename(
        os.environ.get("VIRTUAL_ENV", "")
    )
    cal_label = load_last_calibration_label_fn()

    def _ok(flag: bool, extra: str = "") -> str:
        tick = '<span style="color:#22c55e">&#10003;</span>'
        cross = '<span style="color:#ef4444">&#10007;</span>'
        return (tick if flag else cross) + (" " + extra if extra else "")

    # GPU offload indicator (M-GPU / GPU.2). Reuses the runtime detection
    # helper so this status line tracks the EXACT same logic the dispatcher
    # uses — no risk of drift between "what the user sees in Status" and
    # "what actually happens when they click Run".
    from .gpu_offload import is_gpu_available as _gpu_available_fn

    _gpu_avail, _gpu_name = _gpu_available_fn()
    if _gpu_avail:
        _gpu_msg = f"&mdash; <code>{_gpu_name}</code>"
        _gpu_flag = True
    else:
        _gpu_msg = "&mdash; <code>gpu4pyscf</code> not installed or no CUDA device"
        _gpu_flag = False

    items = [
        (
            "PySCF (calculations)",
            _ok(
                pyscf_available,
                "" if pyscf_available else "&mdash; Linux / macOS / WSL required",
            ),
        ),
        ("ASE (structure I/O, opt.)", _ok(ase_available)),
        ("PubChem search", _ok(pubchem_available)),
        ("3D viewer (py3Dmol)", _ok(visualization_available)),
        ("GPU offload (gpu4pyscf)", _ok(_gpu_flag, _gpu_msg)),
        ("CPU cores / Memory", f"<b>{cores}</b> cores / <b>{mem}</b>"),
    ]
    rows = "".join(
        f'<tr><td style="padding:3px 16px 3px 0;color:#64748b;font-size:13px">{k}</td>'
        f'<td style="font-size:13px">{v}</td></tr>'
        for k, v in items
    )

    env_badge = (
        f'&nbsp;&nbsp;<code style="font-size:11px;background:#e0e7ef;'
        f'padding:1px 5px;border-radius:3px;color:#334155">{env}</code>'
        if env and env not in ("base", "")
        else ""
    )
    cal_line = (
        f'<div style="margin-top:6px;font-size:12px;color:#94a3b8">'
        f"Timing calibration: {cal_label}</div>"
        if cal_label
        else '<div style="margin-top:6px;font-size:12px;color:#94a3b8">'
        "Timing calibration: not yet run &mdash; use the Calibrate panel in History</div>"
    )

    app._status_html = widgets.HTML(
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid #3b82f6;'
        f'padding:12px 16px;border-radius:6px;margin:4px 0 8px">'
        f'<div style="font-weight:600;font-size:14px;color:#1e293b">'
        f"QuantUI {quantui.__version__}"
        f'<span style="font-weight:400;font-size:12px;color:#94a3b8;margin-left:10px">'
        f"Python {py_ver}{env_badge}</span></div>"
        f'<table style="margin-top:10px;border-collapse:collapse">{rows}</table>'
        f"{cal_line}"
        f"</div>"
    )

    # ── Settings section ──────────────────────────────────────────────────
    # "Default 3D backend" — user preference persisted via UserSettings.
    # Drives viz_backend_router resolution. Distinct from the Calculate-tab
    # `viz_backend_toggle` (which selects the current effective backend for
    # interactive use).
    app.viz_default_backend_dd = widgets.ToggleButtons(
        options=[
            ("Auto", "auto"),
            ("py3Dmol", "py3dmol"),
            ("plotlymol3d", "plotlymol"),
        ],
        value=viz_default_backend,
        style={"button_width": "110px"},
        tooltips=[
            "Use the recommended backend per task (py3Dmol-first where supported).",
            "Always prefer py3Dmol when available.",
            "Always prefer plotlymol3d when available.",
        ],
    )
    settings_html = widgets.HTML(
        '<div style="background:#f8fafc;border:1px solid #e2e8f0;'
        "border-left:4px solid #94a3b8;padding:12px 16px;border-radius:6px;"
        'margin:8px 0 4px">'
        '<div style="font-weight:600;font-size:14px;color:#1e293b">Settings</div>'
        '<div style="font-size:12px;color:#475569;margin-top:8px;margin-bottom:4px">'
        "Default 3D backend "
        '<span style="color:#94a3b8;font-size:11px">'
        "(persists across launches)</span></div>"
        "</div>"
    )
    # Vibrational animation framerate (persists across launches).
    app.vib_framerate_si = widgets.IntSlider(
        value=vib_framerate_fps,
        min=5,
        max=60,
        step=1,
        description="Vib fps:",
        style={"description_width": "60px"},
        layout=layout_fn(width="320px", margin="6px 0 0 0"),
        readout=True,
        readout_format="d",
        tooltip=(
            "Frames per second for the py3Dmol vibrational animation. "
            "Higher = smoother + faster oscillation. Cached HTML is "
            "invalidated when this changes."
        ),
    )
    vib_fps_label = widgets.HTML(
        '<div style="font-size:12px;color:#475569;margin-top:10px;'
        'margin-bottom:0px">Vibrational animation framerate '
        '<span style="color:#94a3b8;font-size:11px">'
        "(persists across launches)</span></div>"
    )

    settings_box = widgets.VBox(
        [
            settings_html,
            app.viz_default_backend_dd,
            vib_fps_label,
            app.vib_framerate_si,
        ],
        layout=layout_fn(margin="0 0 8px 0"),
    )

    app._status_tab_panel = widgets.VBox(
        [app._status_html, settings_box],
        layout=layout_fn(padding="8px 0"),
    )


def build_history_section(
    app: Any,
    *,
    layout_fn: Any,
    pyscf_available: bool,
    benchmark_suite: list[Any],
    benchmark_suite_long: list[Any],
    load_last_calibration_label_fn: Any,
) -> None:
    """Build the History tab panel including calibration and perf widgets."""
    app.past_dd = widgets.Dropdown(
        description="Load:",
        options=[("(no saved results)", "")],
        style={"description_width": "50px"},
        layout=layout_fn(width="500px"),
    )
    app.past_refresh_btn = widgets.Button(
        description="Refresh",
        button_style="",
        icon="refresh",
        layout=layout_fn(width="100px"),
        tooltip="Rescan the results directory",
    )
    app.copy_path_btn = widgets.Button(
        description="Copy path",
        button_style="",
        icon="clipboard",
        layout=layout_fn(width="120px"),
        tooltip="Copy the results directory path to clipboard",
    )
    app.results_path_lbl = widgets.HTML()
    app.past_output = widgets.Output()
    app.view_log_btn = widgets.Button(
        description="View log",
        button_style="",
        icon="file-text-o",
        layout=layout_fn(width="110px"),
        tooltip="Open the full PySCF output log in the Output tab",
    )

    # M-EST / EST.4: 4-tier calibration selector. ToggleButtons works for
    # 4 options; switch to a Dropdown if a 5th tier is ever added. Tier 3
    # / tier 4 require PySCF (the geom-opt + freq dispatch); tier 1 / 2
    # are SP-only and gated separately by the run button.
    app._cal_mode_toggle = widgets.ToggleButtons(
        options=[
            ("Tier 1 — Quick (~15 s)", "tier1"),
            ("Tier 2 — Standard (~3–5 min)", "tier2"),
            ("Tier 3 — Mixed (~10–15 min)", "tier3"),
            ("Tier 4 — Deep (~30 min)", "tier4"),
        ],
        value="tier1",
        description="",
        button_style="",
        style={"description_width": "0px", "button_width": "200px"},
        layout=layout_fn(margin="0 0 8px"),
    )
    app._cal_run_btn = widgets.Button(
        description="Run Calibration",
        button_style="primary",
        icon="play",
        disabled=not pyscf_available,
        tooltip=(
            "Run the benchmark suite to calibrate time estimates"
            if pyscf_available
            else "PySCF required (Linux / macOS / WSL)"
        ),
        layout=layout_fn(width="180px"),
    )
    app._cal_stop_btn = widgets.Button(
        description="Stop",
        button_style="warning",
        icon="stop",
        tooltip="Abandon the rest of the calibration (current step is also killed).",
        layout=layout_fn(width="90px", display="none"),
    )
    # session 55 user request: replaced the hard 1800 s per-step timeout
    # with a Skip button so the user can abandon ONE step that's running
    # too long without losing the whole run. Distinct from Stop (which
    # abandons everything remaining).
    app._cal_skip_btn = widgets.Button(
        description="Skip step",
        button_style="info",
        icon="step-forward",
        tooltip=(
            "Abandon the current step and move on to the next. Other "
            "completed steps stay; the calibration continues."
        ),
        layout=layout_fn(width="120px", display="none"),
    )
    app._cal_progress = widgets.IntProgress(
        min=0,
        max=len(benchmark_suite),
        value=0,
        description="",
        bar_style="info",
        layout=layout_fn(width="300px", display="none"),
    )
    app._cal_step_label = widgets.HTML(
        value="",
        layout=layout_fn(display="none"),
    )
    app._cal_results_html = widgets.HTML(value="")

    app._perf_stats_html = widgets.HTML()
    app._perf_events_html = widgets.HTML()
    app._reset_btn = widgets.Button(
        description="Reset performance database",
        button_style="danger",
        icon="trash",
        layout=layout_fn(width="230px"),
    )
    app._reset_confirm_html = widgets.HTML(
        '<span style="color:#dc2626;font-size:13px">'
        "<b>Warning:</b> This will permanently delete all performance records. "
        "Time estimates will reset to &ldquo;no data&rdquo;.</span>"
    )
    app._reset_confirm_yes = widgets.Button(
        description="Yes, delete all records",
        button_style="danger",
        icon="check",
        layout=layout_fn(width="190px"),
    )
    app._reset_confirm_no = widgets.Button(
        description="Cancel",
        button_style="",
        icon="times",
        layout=layout_fn(width="90px"),
    )
    app._reset_confirm_box = widgets.VBox(
        [
            app._reset_confirm_html,
            widgets.HBox(
                [app._reset_confirm_yes, app._reset_confirm_no],
                layout=layout_fn(gap="8px", margin="4px 0 0"),
            ),
        ],
        layout=layout_fn(
            display="none",
            border="1px solid #fca5a5",
            padding="8px 10px",
            margin="6px 0 0",
        ),
    )

    perf_stats_panel = widgets.VBox(
        [
            app._perf_stats_html,
            widgets.HTML(
                '<p style="margin:10px 0 4px;color:#475569;font-size:13px;font-weight:600">'
                "Recent events (last 20)</p>"
            ),
            app._perf_events_html,
            widgets.HBox(
                [app._reset_btn],
                layout=layout_fn(margin="14px 0 4px"),
            ),
            app._reset_confirm_box,
        ]
    )
    app._perf_accordion = widgets.Accordion(
        children=[perf_stats_panel], selected_index=None
    )
    app._perf_accordion.set_title(0, "Performance stats")

    cal_last = load_last_calibration_label_fn()
    cal_note = (
        f'<p style="color:#64748b;font-size:12px;margin:0 0 6px">'
        f"Last run: {cal_last}</p>"
        if cal_last
        else ""
    )
    # M-EST / EST.4: import tier sizes lazily so we can refer to all four
    # in the panel blurb. ``benchmark_suite`` / ``benchmark_suite_long``
    # are kept as positional args for back-compat but new code prefers
    # the four named tiers.
    from quantui.benchmarks import (
        BENCHMARK_SUITE_TIER3 as _T3,
    )
    from quantui.benchmarks import (
        BENCHMARK_SUITE_TIER4 as _T4,
    )

    cal_panel = widgets.VBox(
        [
            widgets.HTML(
                f'<p style="color:#555;font-size:13px;margin:0 0 6px">'
                f"Benchmark this machine so the time estimator uses basis-function "
                f"scaling (N<sup>β</sup>) rather than generic defaults. "
                f"Tier 1 ({len(benchmark_suite)} calcs, ~15&nbsp;s) is a quick "
                f"SP-only smoke test; tier 2 ({len(benchmark_suite_long)} calcs, "
                f"~3–5&nbsp;min) expands the SP grid; "
                f"tier 3 ({len(_T3)} calcs, ~10–15&nbsp;min) adds small geometry "
                f"optimizations + frequency calcs; "
                f"tier 4 ({len(_T4)} calcs, up to ~30&nbsp;min) anchors every "
                f"calc-type × device combo for the most accurate predictions.</p>"
                + cal_note
            ),
            app._cal_mode_toggle,
            widgets.HBox(
                [app._cal_run_btn, app._cal_skip_btn, app._cal_stop_btn],
                layout=layout_fn(gap="6px", align_items="center"),
            ),
            app._cal_progress,
            app._cal_step_label,
            app._cal_results_html,
        ],
        layout=layout_fn(padding="4px 0"),
    )
    app._cal_accordion = widgets.Accordion(children=[cal_panel], selected_index=None)
    app._cal_accordion.set_title(0, "Calibrate time estimates")

    # POLISH.3 (M-POLISH, 2026-05-25): the History tab is now purely
    # the result-browser. Performance stats + Calibrate accordions
    # moved to the System Settings tab — see below — so the user finds
    # benchmarking + system state in one logical place.
    app.history_panel = widgets.VBox(
        [
            widgets.HTML(
                '<p style="color:#555;font-size:13px;margin:0 0 8px">'
                "Calculations are saved automatically. Select one below to view its results.</p>"
            ),
            widgets.HBox(
                [
                    app.past_dd,
                    app.past_refresh_btn,
                    app.copy_path_btn,
                    app.view_log_btn,
                ],
                layout=layout_fn(align_items="center", gap="8px"),
            ),
            app.results_path_lbl,
            app.past_output,
        ]
    )

    # POLISH.3: now that the calibration + performance accordions exist
    # (created above in this function), append them to the System
    # Settings tab. ``_status_tab_panel`` was built earlier in
    # ``build_status_panel`` without these — extend its children tuple.
    app._status_tab_panel.children = (
        *app._status_tab_panel.children,
        app._cal_accordion,
        app._perf_accordion,
    )

    app._refresh_results_browser()
    app._refresh_perf_stats()


def build_shared_widgets(
    app: Any,
    *,
    layout_fn: Any,
    step_progress_cls: Any,
    supported_methods: list[Any],
    supported_basis_sets: list[Any],
    default_method: str,
    default_basis: str,
    default_charge: int,
    default_multiplicity: int,
    default_fmax: float,
    default_opt_steps: int,
    preopt_available: bool,
    visualization_available: bool,
    both_viz_available: bool,
    default_viz_backend: Any,
    default_viz_style: str,
    default_lighting: str,
    viz_style_options: list[Any],
    plotlymol_viz: bool,
    lighting_options: list[Any],
    rdkit_available: bool,
) -> None:
    """Build shared widgets used across tabs and callbacks."""
    app.mol_info_html = widgets.HTML(
        value='<i style="color:#888">No molecule loaded yet.</i>'
    )
    app.mol_summary_compact = widgets.HTML(value="")
    # Fixed heights reserve space so swapping content (backend/palette toggle)
    # or streaming output never resizes the container — which would reflow the
    # page and jump the scrollbar (BUG-SCROLL). The molecule viewer renders at
    # render_molecule_html's default 500px; the run log scrolls internally.
    # overflow hidden (not auto): the 3D viewer is a fixed-size canvas, so it
    # needs no scrollbar — clipping a few px of margin avoids an internal
    # scrollbar that resets to the top on every backend/palette swap.
    app.viz_output = widgets.Output(layout=layout_fn(height="510px", overflow="hidden"))
    app.run_output = widgets.Output(
        layout=layout_fn(
            border="1px solid #c0ccd8",
            height="300px",
            padding="8px",
            overflow_y="auto",
        )
    )
    app.run_output.add_class("quantui-run-output")
    with app.run_output:
        display(
            HTML(
                '<p style="color:#999;font-style:italic;font-size:13px;margin:2px 0">'
                "No calculation run yet. PySCF output and any errors will appear here."
                "</p>"
            )
        )
    app.result_output = widgets.Output()
    app.result_viz_output = widgets.Output()
    app.comparison_output = widgets.Output()
    app._last_result_dir = None

    app._viz_backend = default_viz_backend
    if both_viz_available:
        app.viz_backend_toggle = widgets.ToggleButtons(
            options=[("PlotlyMol", "plotlymol"), ("py3Dmol", "py3dmol")],
            value=default_viz_backend,
            tooltips=["Plotly-based interactive viewer", "WebGL viewer (py3Dmol)"],
            style={"button_width": "90px"},
            layout=layout_fn(margin="2px 0 0 0"),
        )
    else:
        app.viz_backend_toggle = None  # type: ignore[assignment]

    app._viz_style = default_viz_style
    app._viz_lighting = default_lighting
    app.viz_style_dd = widgets.Dropdown(
        options=viz_style_options,
        value=default_viz_style,
        description="Style:",
        style={"description_width": "40px"},
        layout=layout_fn(width="180px"),
        disabled=not visualization_available,
    )
    lighting_available = visualization_available and plotlymol_viz
    app.viz_lighting_dd = widgets.Dropdown(
        options=lighting_options,
        value=default_lighting,
        description="Lighting:",
        style={"description_width": "58px"},
        layout=layout_fn(width="170px"),
        disabled=not lighting_available,
    )
    if not lighting_available:
        app.viz_lighting_dd.layout.visibility = "hidden"
    app.viz_controls_box = widgets.HBox(
        [app.viz_style_dd, app.viz_lighting_dd],
        layout=layout_fn(gap="8px", margin="2px 0 0 0", align_items="center"),
    )
    app.notes_output = widgets.Output()
    app.perf_estimate_html = widgets.HTML()

    app.step_progress = step_progress_cls(
        ["Choose molecule", "Set method", "Run", "Results"]
    )
    app.step_progress.start(0)

    app.method_dd = widgets.Dropdown(
        options=supported_methods,
        value=default_method,
        description="Method:",
        style={"description_width": "100px"},
        layout=layout_fn(width="260px"),
    )
    app.basis_dd = widgets.Dropdown(
        options=supported_basis_sets,
        value=default_basis,
        description="Basis Set:",
        style={"description_width": "100px"},
        layout=layout_fn(width="260px"),
    )
    app.charge_si = widgets.BoundedIntText(
        value=default_charge,
        min=-10,
        max=10,
        description="Charge:",
        style={"description_width": "100px"},
        layout=layout_fn(width="190px"),
    )
    app.mult_si = widgets.BoundedIntText(
        value=default_multiplicity,
        min=1,
        max=10,
        description="Multiplicity:",
        style={"description_width": "100px"},
        layout=layout_fn(width="190px"),
    )
    # POLISH.10 (M-POLISH, 2026-05-25): ``style={"description_width":
    # "initial"}`` removes the default left-side description gutter that
    # ipywidgets reserves on Checkbox, which was producing both the
    # indent the user noticed AND the horizontal scrollbar (description
    # gutter + ``width="100%"`` exceeded the container width). Letting
    # the checkbox size to its content also drops the scrollbar.
    app.preopt_cb = widgets.Checkbox(
        value=False,
        description="Classical pre-optimize geometry (fast, crude starting point)",
        disabled=not preopt_available,
        style={"description_width": "initial"},
        indent=False,
    )

    # Interactive pre-opt (M-PREOPT PREOPT.2/.3): run the bonded-FF pre-opt on
    # demand, watch it relax in-place, then keep or revert — instead of it being
    # a silent step buried inside the run.
    app.preopt_preview_btn = widgets.Button(
        description="Preview",
        icon="eye",
        button_style="",
        disabled=not preopt_available,
        layout=layout_fn(width="110px", height="28px"),
        tooltip="Watch the classical pre-optimization relax this geometry, "
        "then keep or revert it",
    )
    app.preopt_accept_btn = widgets.Button(
        description="Keep this geometry",
        icon="check",
        button_style="success",
        layout=layout_fn(width="190px", height="30px"),
        tooltip="Make the relaxed geometry the active molecule",
    )
    app.preopt_reset_btn = widgets.Button(
        description="Revert",
        icon="undo",
        button_style="warning",
        layout=layout_fn(width="110px", height="30px"),
        tooltip="Discard the preview and keep your original geometry",
    )
    app.preopt_preview_status = widgets.HTML("")
    app.preopt_preview_output = widgets.Output(
        layout=layout_fn(
            # 290px viewer + stepper controls (slider / play / compare) below.
            height="360px",
            width="100%",
            max_width="480px",
            border="1px solid #e2e8f0",
            overflow="hidden",
        )
    )
    # Whole preview block hidden until the user clicks Preview.
    app.preopt_preview_box = widgets.VBox(
        [
            app.preopt_preview_status,
            app.preopt_preview_output,
            widgets.HBox(
                [app.preopt_accept_btn, app.preopt_reset_btn],
                layout=layout_fn(gap="8px", margin="6px 0 0"),
            ),
        ],
        layout=layout_fn(display="none", margin="6px 0 4px", max_width="480px"),
    )

    from quantui.config import SOLVENT_OPTIONS as _SOLVENT_OPTS

    # POLISH.10: same fix as preopt_cb above — drop the gutter +
    # explicit width that produced the indent + scrollbar.
    app.solvent_cb = widgets.Checkbox(
        value=False,
        description="Implicit solvent (PCM)",
        style={"description_width": "initial"},
        indent=False,
    )
    app.solvent_dd = widgets.Dropdown(
        options=list(_SOLVENT_OPTS.keys()),
        value="Water",
        description="Solvent:",
        style={"description_width": "70px"},
        layout=layout_fn(width="200px", display="none"),
    )

    app.calc_type_dd = widgets.Dropdown(
        options=[
            "Single Point",
            "Geometry Opt",
            "Frequency",
            "UV-Vis (TD-DFT)",
            "NMR Shielding",
            "PES Scan",
        ],
        value="Single Point",
        description="Calc. Type:",
        style={"description_width": "100px"},
        layout=layout_fn(width="310px"),
    )
    app.fmax_fi = widgets.BoundedFloatText(
        value=default_fmax,
        min=0.001,
        max=1.0,
        step=0.005,
        description="Force thr. (eV/Å):",
        style={"description_width": "130px"},
        layout=layout_fn(width="270px"),
    )
    app.max_steps_si = widgets.BoundedIntText(
        value=default_opt_steps,
        min=10,
        max=1000,
        description="Max steps:",
        style={"description_width": "100px"},
        layout=layout_fn(width="200px"),
    )
    app.nstates_si = widgets.BoundedIntText(
        value=10,
        min=1,
        max=50,
        description="# states:",
        style={"description_width": "100px"},
        layout=layout_fn(width="180px"),
    )

    app._freq_seed_dd = widgets.Dropdown(
        options=[("(use current molecule)", "")],
        description="Seed geometry:",
        style={"description_width": "110px"},
        layout=layout_fn(width="auto", flex="1 1 auto", min_width="260px"),
        tooltip="Optionally load the final optimised geometry from a previous Geo Opt result",
    )
    app._freq_seed_refresh_btn = widgets.Button(
        description="",
        icon="refresh",
        layout=layout_fn(width="32px", height="32px"),
        tooltip="Refresh the list of saved geometry optimisations",
    )
    app._freq_preopt_cb = widgets.Checkbox(
        value=False,
        description="Geometry optimization before calculation (QM, slower)",
        style={"description_width": "initial"},
        indent=False,
    )
    app._freq_seed_note = widgets.HTML("")

    # UV-Vis (TD-DFT) seed-geometry parity with Frequency: lets the user run
    # the excited-state calculation on a previously optimised geometry rather
    # than the current input molecule. Same formula-filtered dropdown pattern
    # as the Frequency seed widgets above; refresh button + status note also
    # mirrored.
    app._tddft_seed_dd = widgets.Dropdown(
        options=[("(use current molecule)", "")],
        description="Seed geometry:",
        style={"description_width": "110px"},
        layout=layout_fn(width="auto", flex="1 1 auto", min_width="260px"),
        tooltip="Optionally load the final optimised geometry from a previous Geo Opt result",
    )
    app._tddft_seed_refresh_btn = widgets.Button(
        description="",
        icon="refresh",
        layout=layout_fn(width="32px", height="32px"),
        tooltip="Refresh the list of saved geometry optimisations",
    )
    app._tddft_seed_note = widgets.HTML("")

    app._scan_type_dd = widgets.Dropdown(
        options=["Bond", "Angle", "Dihedral"],
        value="Bond",
        description="Scan type:",
        style={"description_width": "80px"},
        layout=layout_fn(width="220px"),
    )
    atom_idx_layout = layout_fn(width="95px")
    atom_idx_style = {"description_width": "50px"}
    app._scan_atom1 = widgets.BoundedIntText(
        value=1,
        min=1,
        max=999,
        description="Atom 1:",
        style=atom_idx_style,
        layout=atom_idx_layout,
    )
    app._scan_atom2 = widgets.BoundedIntText(
        value=2,
        min=1,
        max=999,
        description="Atom 2:",
        style=atom_idx_style,
        layout=atom_idx_layout,
    )
    app._scan_atom3 = widgets.BoundedIntText(
        value=3,
        min=1,
        max=999,
        description="Atom 3:",
        style=atom_idx_style,
        layout=atom_idx_layout,
    )
    app._scan_atom4 = widgets.BoundedIntText(
        value=4,
        min=1,
        max=999,
        description="Atom 4:",
        style=atom_idx_style,
        layout=atom_idx_layout,
    )
    app._scan_atom34_box = widgets.HBox(
        [app._scan_atom3, app._scan_atom4],
        layout=layout_fn(gap="4px"),
    )
    app._scan_start = widgets.BoundedFloatText(
        value=0.5,
        min=0.01,
        max=1000.0,
        step=0.1,
        description="Start:",
        style={"description_width": "40px"},
        layout=layout_fn(width="140px"),
    )
    app._scan_stop = widgets.BoundedFloatText(
        value=2.0,
        min=0.01,
        max=1000.0,
        step=0.1,
        description="Stop:",
        style={"description_width": "40px"},
        layout=layout_fn(width="140px"),
    )
    app._scan_steps = widgets.BoundedIntText(
        value=10,
        min=2,
        max=100,
        description="Points:",
        style={"description_width": "50px"},
        layout=layout_fn(width="120px"),
    )
    app._scan_unit_lbl = widgets.HTML(
        '<span style="font-size:12px;color:#555">Å</span>'
    )

    app.calc_extra_opts = widgets.VBox([])

    app.method_help_btn = widgets.Button(
        description="?",
        button_style="",
        layout=layout_fn(width="28px", height="28px"),
        tooltip="RHF vs UHF — opens Help tab",
    )
    app.basis_help_btn = widgets.Button(
        description="?",
        button_style="",
        layout=layout_fn(width="28px", height="28px"),
        tooltip="Choosing a basis set — opens Help tab",
    )

    app.run_btn = widgets.Button(
        description="Run Calculation",
        button_style="success",
        icon="play",
        disabled=True,
        layout=layout_fn(width="200px", height="36px"),
    )
    app.run_status = widgets.Label()

    # Gracefully stops a running calculation at the next SCF cycle / opt step.
    # Disabled unless a calc is in flight (toggled by _do_run).
    app.cancel_btn = widgets.Button(
        description="Cancel",
        button_style="danger",
        icon="stop",
        disabled=True,
        layout=layout_fn(width="110px", height="36px"),
        tooltip="Stop the running calculation (at the next step)",
    )

    app.log_clear_btn = widgets.Button(
        description="Clear",
        button_style="",
        icon="times",
        layout=layout_fn(width="90px", height="26px"),
        tooltip="Clear calculation output (disabled while a calc is running)",
    )

    app.accumulate_btn = widgets.Button(
        description="Add to Comparison",
        button_style="info",
        icon="plus",
        disabled=True,
        layout=layout_fn(width="190px"),
    )
    app.clear_btn = widgets.Button(
        description="Clear",
        button_style="warning",
        icon="trash",
        layout=layout_fn(width="100px"),
    )
    app.export_btn = widgets.Button(
        description="Export Script",
        button_style="",
        icon="download",
        disabled=True,
        layout=layout_fn(width="160px"),
    )
    app.export_status = widgets.Label()
    rdkit_tip = (
        "" if rdkit_available else "Requires RDKit (conda install -c conda-forge rdkit)"
    )
    app.export_xyz_btn = widgets.Button(
        description="Export XYZ",
        icon="download",
        disabled=True,
        layout=layout_fn(width="130px"),
    )
    app.export_mol_btn = widgets.Button(
        description="Export MOL",
        icon="download",
        disabled=True,
        tooltip=rdkit_tip,
        layout=layout_fn(width="130px"),
    )
    app.export_pdb_btn = widgets.Button(
        description="Export PDB",
        icon="download",
        disabled=True,
        tooltip=rdkit_tip,
        layout=layout_fn(width="130px"),
    )
    app.struct_export_status = widgets.Label()
    # M-EXPORT / EXPORT.5: zip the entire result folder for emailing /
    # attaching to a writeup. Disabled until ``_last_result_dir`` is set.
    app._export_bundle_btn = widgets.Button(
        description="Export bundle (.zip)",
        icon="file-archive-o",
        disabled=True,
        tooltip=(
            "Zip the entire result folder (geometry, log, orbitals, cubes, "
            "spectra) for sharing."
        ),
        layout=layout_fn(width="180px"),
    )
    app._export_bundle_status = widgets.Label()


def build_theme_selector(app: Any, *, layout_fn: Any) -> None:
    """Build the theme selector widgets and apply default theme CSS."""
    app._theme_style = widgets.Output(
        layout=layout_fn(height="0px", overflow="hidden", margin="0", padding="0")
    )
    app._activity_btn = widgets.Button(
        description="Idle",
        icon="circle-o",
        tooltip="No active operations.",
        button_style="success",
        layout=layout_fn(width="118px", margin="0 8px 0 0"),
    )
    app.theme_btn = widgets.ToggleButtons(
        options=["Light", "Dark"],
        value="Dark",
        description="Theme:",
        style={"description_width": "48px", "button_width": "90px"},
        layout=layout_fn(margin="0"),
    )
    with app._theme_style:
        display(HTML(app._theme_css("Dark")))


def build_welcome_header(app: Any, *, layout_fn: Any = None) -> None:
    """Build the QuantUI welcome banner.

    POLISH.1 third iteration (M-POLISH, 2026-05-25): the
    ``<img src="data:image/svg+xml;base64,...">`` approach failed too
    — Voilà's HTML sanitizer (stricter than JupyterLab's) strips
    ``data:`` URIs from ``<img src>`` attributes. The third iteration
    uses ``widgets.Image(value=svg_bytes, format="svg+xml")`` which
    routes the SVG through Jupyter's binary widget channel, bypassing
    the HTML sanitizer entirely. CSS animations inside the SVG still
    run because the front-end serves it as an external SVG document.

    The original ``_welcome_html`` widget remains (the exit handler at
    ``app_runflow.on_exit_clicked`` rewrites its ``.value`` with the
    shutdown message; an HBox-based wrapper would break that path).
    A new ``_welcome_header`` HBox combines the logo widget + text
    widget for the display() entry point.
    """
    # Full SVG. Includes the orbital animations ported from
    # ``docs/logo.svg`` — three rings spinning at 9 s / 13 s reverse /
    # 17 s with prefers-reduced-motion respected.
    _logo_svg_raw = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 280 280">'
        "<defs>"
        "<style>"
        ".qring{transform-origin:140px 140px}"
        ".qring--1{animation:qspin1 9s linear infinite}"
        ".qring--2{animation:qspin2 13s linear infinite reverse;"
        "transform:rotate(60deg)}"
        ".qring--3{animation:qspin3 17s linear infinite;"
        "transform:rotate(120deg)}"
        "@keyframes qspin1{to{transform:rotate(360deg)}}"
        "@keyframes qspin2{to{transform:rotate(-300deg)}}"
        "@keyframes qspin3{to{transform:rotate(480deg)}}"
        "@media (prefers-reduced-motion:reduce){"
        ".qring{animation-play-state:paused}}"
        "</style>"
        '<filter id="q-glow" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="7" result="blur"/>'
        "<feMerge>"
        '<feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/>'
        "</feMerge></filter>"
        '<filter id="q-halo" x="-80%" y="-80%" width="260%" height="260%">'
        '<feGaussianBlur stdDeviation="22" result="blur"/>'
        "<feMerge>"
        '<feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/>'
        "</feMerge></filter>"
        "</defs>"
        '<circle cx="140" cy="140" r="48"'
        ' fill="rgba(37,99,235,0.20)" filter="url(#q-halo)"/>'
        '<g class="qring qring--1">'
        '<ellipse cx="140" cy="140" rx="115" ry="33" fill="none"'
        ' stroke="#0891b2" stroke-width="1.4" opacity="0.70"/>'
        '<circle cx="255" cy="140" r="5.5" fill="#67e8f9"/>'
        "</g>"
        '<g class="qring qring--2">'
        '<ellipse cx="140" cy="140" rx="115" ry="33" fill="none"'
        ' stroke="#0891b2" stroke-width="1.4" opacity="0.55"/>'
        '<circle cx="255" cy="140" r="4.5" fill="#93c5fd"/>'
        "</g>"
        '<g class="qring qring--3">'
        '<ellipse cx="140" cy="140" rx="115" ry="33" fill="none"'
        ' stroke="#3b82f6" stroke-width="1.4" opacity="0.42"/>'
        '<circle cx="255" cy="140" r="4" fill="#60a5fa"/>'
        "</g>"
        '<circle cx="140" cy="140" r="20"'
        ' fill="rgba(37,99,235,0.25)" filter="url(#q-glow)"/>'
        '<circle cx="140" cy="140" r="14"'
        ' fill="#2563eb" filter="url(#q-glow)"/>'
        '<circle cx="140" cy="140" r="8" fill="#60a5fa"/>'
        '<circle cx="137" cy="137" r="3" fill="rgba(255,255,255,0.45)"/>'
        "</svg>"
    )
    # widgets.Image takes the raw SVG bytes and serves them as
    # ``format="svg+xml"`` over Jupyter's BINARY widget channel — no
    # HTML sanitizer touches the bytes, no ``data:`` URI restriction.
    # The browser renders the SVG natively as an image (CSS animations
    # inside the SVG still play).
    app._welcome_logo = widgets.Image(
        value=_logo_svg_raw.encode("utf-8"),
        format="svg+xml",
        width=120,
        height=120,
    )

    # Text-only HTML. ``_welcome_html`` is kept as a pure HTML widget so
    # ``app_runflow.on_exit_clicked`` can still ``.value = ...`` it with
    # the shutdown message.
    text_html = (
        "<div>"
        '<div style="font-size:44px;font-weight:700;letter-spacing:-0.8px;'
        'color:#0f172a;line-height:1.05">QuantUI</div>'
        '<div style="font-size:20px;color:#475569;margin-top:7px">'
        "Free, open, and interactive quantum chemistry</div>"
        '<div style="font-size:13px;color:#94a3b8;margin-top:5px">'
        f"v{quantui.__version__} &nbsp;&middot;&nbsp; "
        "<b>Help</b> tab for instructions &nbsp;&middot;&nbsp; "
        "<b>System Settings</b> tab for environment + calibration</div>"
        "</div>"
    )
    app._welcome_html = widgets.HTML(value=text_html)

    # Container that combines logo + text. ``display()`` mounts this
    # instead of ``_welcome_html`` directly (see app.py:1001).
    _layout = (
        layout_fn if layout_fn is not None else (lambda **kw: widgets.Layout(**kw))
    )
    app._welcome_header = widgets.HBox(
        [app._welcome_logo, app._welcome_html],
        layout=_layout(
            align_items="center",
            justify_content="flex-start",
            padding="22px 4px 18px",
            margin="0 0 4px",
            border_bottom="1px solid #e2e8f0",
        ),
    )


def build_molecule_section(
    app: Any,
    *,
    layout_fn: Any,
    pubchem_available: bool,
    visualization_available: bool,
) -> None:
    """Build molecule input widgets and collapsed summary container."""
    # ── Library browse/search — category filter + search + results.
    cat_opts = [("All categories", "")] + [
        (_category_label(c), c) for c in _ml.categories()
    ]
    app.lib_category_dd = widgets.Dropdown(
        options=cat_opts,
        value="",
        description="Category:",
        style={"description_width": "70px"},
        layout=layout_fn(width="260px"),
    )
    app.lib_search_txt = widgets.Text(
        placeholder="search name or formula (e.g. aspirin, C6H6)",
        continuous_update=False,  # fire on Enter/blur, not per keystroke
        layout=layout_fn(width="300px"),
    )
    init_opts, init_note = library_result_options()
    app.lib_results_dd = widgets.Dropdown(
        options=init_opts,
        value="",
        description="Molecule:",
        style={"description_width": "70px"},
        layout=layout_fn(width="420px"),
    )
    app.lib_count_lbl = widgets.HTML(
        f'<span style="color:#888;font-size:12px">{init_note}</span>'
    )

    app.xyz_area = widgets.Textarea(
        placeholder=(
            "Paste XYZ coordinates (symbol  x  y  z):\n"
            "O  0.000  0.000  0.000\n"
            "H  0.757  0.587  0.000\n"
            "H -0.757  0.587  0.000"
        ),
        layout=layout_fn(width="440px", height="130px"),
    )
    app.xyz_btn = widgets.Button(
        description="Load XYZ", button_style="info", icon="upload"
    )
    app.xyz_msg = widgets.Label()

    app.pubchem_txt = widgets.Text(
        placeholder="name, SMILES, CID, or InChI  (e.g. aspirin, CC(=O)O, 2244)",
        layout=layout_fn(width="380px"),
    )
    app.pubchem_btn = widgets.Button(
        description="Search",
        button_style="info",
        icon="search",
        disabled=not pubchem_available,
        layout=layout_fn(width="100px"),
    )
    app.pubchem_msg = widgets.Label(
        value=(
            ""
            if pubchem_available
            else "PubChem unavailable — check internet connection"
        )
    )
    # Disambiguation pick-list — hidden until a query has >1 match.
    app.pubchem_candidates_dd = widgets.Dropdown(
        options=[("— pick a match —", "")],
        value="",
        description="Matches:",
        style={"description_width": "70px"},
        layout=layout_fn(width="460px"),
    )
    app.pubchem_candidates_dd.layout.display = "none"

    hint = '<p style="margin:4px 0 8px;color:#666;font-size:13px">'
    tab_preset = widgets.VBox(
        [
            widgets.HTML(
                hint + "Browse the bundled library — filter by category or "
                "search by name/formula. Includes thousands of small molecules "
                "for offline use.</p>"
            ),
            widgets.HBox([app.lib_category_dd, app.lib_search_txt]),
            app.lib_results_dd,
            app.lib_count_lbl,
        ]
    )
    tab_xyz = widgets.VBox(
        [
            widgets.HTML(
                hint + "Paste XYZ coordinates (element x y z, one atom per line).</p>"
            ),
            app.xyz_area,
            widgets.HBox([app.xyz_btn, app.xyz_msg]),
        ]
    )
    tab_pubchem = widgets.VBox(
        [
            widgets.HTML(
                hint + "Search by name, SMILES, CID, or InChI. Tries PubChem "
                "then NCI CACTUS, and falls back to the bundled library offline."
                "</p>"
            ),
            widgets.HBox([app.pubchem_txt, app.pubchem_btn]),
            app.pubchem_msg,
            app.pubchem_candidates_dd,
        ]
    )
    input_tab = widgets.Tab(children=[tab_preset, tab_xyz, tab_pubchem])
    for i, title in enumerate(["Library", "XYZ Input", "Online Search"]):
        input_tab.set_title(i, title)

    app.mol_input_expanded = widgets.VBox(
        [
            widgets.HTML('<h3 style="margin:8px 0 6px">Molecule Input</h3>'),
            input_tab,
        ]
    )
    app.change_mol_btn = widgets.Button(
        description="Change",
        button_style="",
        icon="pencil",
        layout=layout_fn(width="100px", height="32px"),
        tooltip="Re-expand the molecule input panel",
    )
    app.mol_input_collapsed = widgets.HBox(
        [app.mol_summary_compact, app.change_mol_btn],
        layout=layout_fn(align_items="center", gap="12px", padding="6px 0"),
    )
    mol_container_children = [
        app.mol_input_expanded,
        app.mol_info_html,
        app.viz_output,
    ]
    if app.viz_backend_toggle is not None:
        mol_container_children.append(app.viz_backend_toggle)
    if visualization_available:
        mol_container_children.append(app.viz_controls_box)
    app.mol_input_container = widgets.VBox(
        mol_container_children,
        layout=layout_fn(margin="0 0 4px 0"),
    )


def build_calc_setup(app: Any, *, layout_fn: Any) -> None:
    """Build the calculation setup panel."""
    app.calc_setup_panel = widgets.VBox(
        [
            widgets.HTML('<h3 style="margin:14px 0 6px">Calculation Setup</h3>'),
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            widgets.HBox(
                                [app.method_dd, app.method_help_btn],
                                layout=layout_fn(align_items="center", gap="4px"),
                            ),
                            widgets.HBox(
                                [app.basis_dd, app.basis_help_btn],
                                layout=layout_fn(align_items="center", gap="4px"),
                            ),
                        ]
                    ),
                    widgets.HTML("&ensp;&ensp;"),
                    widgets.VBox([app.charge_si, app.mult_si]),
                ]
            ),
            app.calc_type_dd,
            app.calc_extra_opts,
            widgets.HBox(
                [app.preopt_cb, app.preopt_preview_btn],
                layout=layout_fn(align_items="center", gap="10px"),
            ),
            app.preopt_preview_box,
            app._freq_preopt_cb,
            widgets.HBox(
                [app.solvent_cb, app.solvent_dd],
                layout=layout_fn(align_items="center", gap="4px"),
            ),
            app.notes_output,
        ]
    )


def build_run_section(app: Any, *, layout_fn: Any) -> None:
    """Build the run panel shown in the Calculate tab."""
    app.run_panel = widgets.VBox(
        [
            widgets.HTML(
                '<h3 style="margin:14px 0 6px">Run Calculation</h3>'
                '<p style="color:#555;font-size:13px;margin:0 0 8px">PySCF runs in this '
                "kernel. Output appears live below. Large molecules or high-accuracy basis "
                "sets may take several minutes on a laptop.</p>"
            ),
            app.perf_estimate_html,
            widgets.HBox([app.run_btn, app.cancel_btn, app.run_status]),
            widgets.HBox(
                [
                    widgets.HTML(
                        '<span style="font-size:13px;font-weight:600;color:#444">'
                        "Calculation Output</span>"
                    ),
                    app.log_clear_btn,
                ],
                layout=layout_fn(
                    align_items="center",
                    justify_content="space-between",
                    margin="10px 0 4px",
                    max_width="460px",
                ),
            ),
            app.run_output,
        ]
    )


def build_results_section(app: Any, *, layout_fn: Any) -> None:
    """Build results and analysis tab panels/widgets."""

    def _plot_export_row(prefix: str) -> widgets.HBox:
        fmt_dd = widgets.Dropdown(
            options=[("HTML", "html"), ("PNG", "png")],
            value="html",
            description="Export:",
            style={"description_width": "55px"},
            layout=layout_fn(width="170px"),
        )
        btn = widgets.Button(
            description="Save Plot",
            icon="download",
            layout=layout_fn(width="130px"),
            tooltip="Export the current plot as HTML or PNG",
        )
        # M-EXPORT / EXPORT.4: per-panel "Copy data" button that exports
        # the underlying numerical data to CSV (saved to result_dir) and
        # also attempts to copy to the system clipboard via the browser
        # API. Status widget below is shared with the Save Plot path —
        # whichever action runs last updates the visible status string.
        copy_btn = widgets.Button(
            description="Copy data",
            icon="clipboard",
            layout=layout_fn(width="120px"),
            tooltip=(
                "Save the plot's underlying (x, y) data to CSV in the "
                "result folder and copy it to the system clipboard"
            ),
        )
        status = widgets.HTML(value="", layout=layout_fn(margin="0 0 0 8px"))
        setattr(app, f"_{prefix}_export_fmt_dd", fmt_dd)
        setattr(app, f"_{prefix}_export_btn", btn)
        setattr(app, f"_{prefix}_copy_data_btn", copy_btn)
        setattr(app, f"_{prefix}_export_status", status)
        return widgets.HBox(
            [fmt_dd, btn, copy_btn, status],
            layout=layout_fn(align_items="center", margin="0 0 6px 0", gap="6px"),
        )

    pes_export_row = _plot_export_row("pes")
    app._pes_plot_html = widgets.Output(layout=layout_fn(width="100%"))
    app._pes_scan_accordion = widgets.Accordion(
        children=[
            widgets.VBox(
                [pes_export_row, app._pes_plot_html],
                layout=layout_fn(padding="8px"),
            )
        ],
        layout=layout_fn(display="none", margin="8px 0"),
    )
    app._pes_scan_accordion.set_title(0, "PES Energy Profile")
    app._pes_scan_accordion.selected_index = None

    # traj_output is a VBox container (NOT widgets.Output) so widget content
    # can be added as direct children via `traj_output.children = (...)`.
    # Using `widgets.Output` here previously caused widget references inside
    # `with output: display(widget)` to be deferred/asynchronous, leaving the
    # accordion visibly empty even after _show_opt_trajectory logged success.
    # See BUG-FRESH-TRAJ root-cause analysis in session 48.
    app.traj_output = widgets.VBox(layout=layout_fn(margin="0"))
    app.traj_accordion = widgets.Accordion(
        children=[app.traj_output],
        layout=layout_fn(display="none", margin="8px 0"),
    )
    app.traj_accordion.set_title(0, "Trajectory Viewer")
    app.traj_accordion.selected_index = None
    app.traj_accordion.observe(
        app._safe_cb(app._on_traj_expand), names=["selected_index"]
    )

    app.vib_mode_dd = widgets.Dropdown(
        description="Mode:",
        options=[],
        style={"description_width": "50px"},
        layout=layout_fn(width="360px"),
    )
    # Prev/next arrow buttons for one-step navigation through modes. Click
    # handlers step ``vib_mode_dd.value`` to the adjacent option; the
    # existing dropdown observer then drives the re-render. Mirrors the
    # trajectory-viewer prev/next pattern.
    app.vib_prev_btn = widgets.Button(
        icon="arrow-left",
        tooltip="Previous mode",
        layout=layout_fn(width="40px", margin="0 4px 0 0"),
        disabled=True,
    )
    app.vib_next_btn = widgets.Button(
        icon="arrow-right",
        tooltip="Next mode",
        layout=layout_fn(width="40px", margin="0 8px 0 4px"),
        disabled=True,
    )
    vib_mode_row = widgets.HBox(
        [app.vib_prev_btn, app.vib_mode_dd, app.vib_next_btn],
        layout=layout_fn(align_items="center", margin="0 0 4px 0"),
    )
    # Fixed-dimension Output container so the box never resizes between
    # content swaps (placeholder ↔ 3Dmol HTML). Without this, the empty
    # state between atomic outputs assignments briefly collapses the
    # container, the page reflows up, then reflows back when the new
    # content arrives — visible as a scroll-jump flicker on every mode
    # switch. Matches the trajectory frame_out fix pattern. 460+20=480
    # accommodates the py3Dmol view (460px) plus a small horizontal pad;
    # 420+20=440 likewise for the 420px view height.
    app.vib_output = widgets.Output(layout=layout_fn(height="440px", width="480px"))

    # Vibration animation export: writes the current mode as a self-contained
    # HTML file. Backend selection is independent of the user's default — see
    # _on_vib_export_animation: plotlymol3d is preferred for export quality,
    # py3Dmol is the fallback when plotlymol3d isn't installed.
    app._vib_export_btn = widgets.Button(
        description="⬇ Export Animation",
        tooltip=(
            "Save the current vibrational mode as a self-contained HTML "
            "file (plotlymol3d preferred for export quality; py3Dmol "
            "fallback if plotlymol3d is not installed)"
        ),
        layout=layout_fn(width="180px"),
    )
    app._vib_export_status = widgets.HTML(
        value="",
        layout=layout_fn(flex="1 1 auto", margin="0 0 0 8px"),
    )
    vib_export_row = widgets.HBox(
        [app._vib_export_btn, app._vib_export_status],
        layout=layout_fn(align_items="center", margin="6px 0 0 0"),
    )

    # Hidden sink for Python→JS calls that switch the single-viewer's mode
    # client-side (window.__quantuiVibSetMode). Kept in the DOM (not display:none)
    # so the injected Javascript executes; empty/cleared between calls so it
    # takes no visible space. See app_visualization._vib_bridge_set_mode.
    app._vib_js_bridge = widgets.Output(layout=layout_fn(margin="0", padding="0"))

    app.vib_accordion = widgets.Accordion(
        children=[
            widgets.VBox(
                [vib_mode_row, app.vib_output, vib_export_row, app._vib_js_bridge],
                layout=layout_fn(padding="8px"),
            )
        ],
        layout=layout_fn(display="none", margin="8px 0"),
    )
    app.vib_accordion.set_title(0, "Vibrational Mode Viewer")
    app.vib_accordion.selected_index = None

    app._ir_mode_toggle = widgets.ToggleButtons(
        options=["Stick", "Broadened"],
        value="Stick",
        style={"button_width": "80px"},
        layout=layout_fn(margin="0 8px 0 0"),
    )
    app._ir_fwhm_slider = widgets.FloatSlider(
        value=20.0,
        min=5.0,
        max=100.0,
        step=5.0,
        description="Line width (cm⁻¹):",
        readout_format=".0f",
        style={"description_width": "120px"},
        layout=layout_fn(width="300px", display="none"),
        # continuous_update=False so dragging the slider only fires on
        # release, not 30-60 times per second during the drag (BUG.9 fix).
        # Combined with the atomic outputs swap in _set_html_output this
        # eliminates the IR re-render storm that caused visible flicker.
        continuous_update=False,
    )
    # min_height matches the Plotly IR figure's intrinsic height (300px in
    # ir_plot.plot_ir_spectrum) so the Output container does not collapse
    # to 0px between renders. Pairs with the atomic outputs swap in
    # _set_html_output to keep mode toggle / slider changes flicker-free.
    app._ir_fig = widgets.Output(
        layout=layout_fn(width="100%", min_height="300px"),
    )
    ir_export_row = _plot_export_row("ir")

    ir_controls = widgets.HBox(
        [app._ir_mode_toggle, app._ir_fwhm_slider],
        layout=layout_fn(align_items="center", margin="0 0 6px 0"),
    )
    ir_body_children = [ir_controls, ir_export_row, app._ir_fig]
    app._ir_accordion = widgets.Accordion(
        children=[
            widgets.VBox(
                ir_body_children,
                layout=layout_fn(padding="8px"),
            )
        ],
        layout=layout_fn(display="none", margin="8px 0"),
    )
    app._ir_accordion.set_title(0, "IR Spectrum")
    app._ir_accordion.selected_index = None

    app._orb_ymin_input = widgets.BoundedFloatText(
        value=-30.0,
        min=-500.0,
        max=200.0,
        step=1.0,
        description="Y min:",
        layout=layout_fn(width="140px"),
        style={"description_width": "45px"},
    )
    app._orb_ymax_input = widgets.BoundedFloatText(
        value=5.0,
        min=-500.0,
        max=500.0,
        step=1.0,
        description="Y max:",
        layout=layout_fn(width="140px"),
        style={"description_width": "45px"},
    )
    app._orb_n_orb_input = widgets.BoundedIntText(
        value=20,
        min=4,
        max=200,
        step=2,
        description="Show N:",
        layout=layout_fn(width="120px"),
        style={"description_width": "50px"},
    )
    orb_controls_row = widgets.HBox(
        [
            widgets.HTML(
                '<span style="font-size:11px;color:#555;font-weight:600">Y range:</span>'
            ),
            app._orb_ymin_input,
            app._orb_ymax_input,
            widgets.HTML(
                '<span style="font-size:11px;color:#555;font-weight:600;margin-left:8px">'
                "Levels shown:</span>"
            ),
            app._orb_n_orb_input,
        ],
        layout=layout_fn(
            align_items="center",
            flex_wrap="wrap",
            gap="4px",
            margin="0 0 6px 0",
        ),
    )
    app._orb_diagram_html = widgets.Output(layout=layout_fn(width="100%"))
    orb_export_row = _plot_export_row("orb")
    orb_diagram_content: list[Any] = [
        orb_controls_row,
        orb_export_row,
        app._orb_diagram_html,
    ]
    app._orb_diagram_box = widgets.VBox(
        orb_diagram_content,
        layout=layout_fn(width="100%"),
    )
    app._orb_toggle = widgets.ToggleButtons(
        options=["HOMO-1", "HOMO", "LUMO", "LUMO+1"],
        value="HOMO",
        style={"button_width": "70px"},
        layout=layout_fn(margin="8px 0 4px 0"),
    )
    app._orb_iso_output = widgets.Output()
    app._orb_iso_controls = widgets.VBox(
        [
            widgets.HTML(
                '<span style="font-size:12px;color:#555;font-weight:bold">'
                "Orbital isosurface:</span>"
            ),
            app._orb_toggle,
            app._orb_iso_output,
        ],
        layout=layout_fn(display="none", margin="8px 0 0 0"),
    )
    app._orb_accordion = widgets.Accordion(
        children=[
            widgets.VBox(
                [app._orb_diagram_box],
                layout=layout_fn(padding="8px"),
            )
        ],
        layout=layout_fn(display="none", margin="8px 0"),
    )
    app._orb_accordion.set_title(0, "Energy-level Diagram")
    app._orb_accordion.selected_index = None

    app._iso_generate_btn = widgets.Button(
        description="Generate Isosurface",
        button_style="primary",
        icon="flask",
        disabled=True,
        tooltip=(
            "Generate a 3D orbital isosurface. "
            "Available after running or loading a Single Point or Geometry Optimization."
        ),
        layout=layout_fn(width="200px", margin="8px 0 4px 0"),
    )
    # M-EXPORT / EXPORT.5: copy the last-generated cube to the top-level
    # result dir under a friendly name (HOMO.cube / LUMO.cube / etc.).
    # Disabled until the first isosurface generation populates
    # ``app._last_cube_path``.
    app._iso_export_cube_btn = widgets.Button(
        description="Export cube",
        icon="download",
        disabled=True,
        tooltip=(
            "Copy the last-generated cube file to the result folder under a "
            "friendly name (e.g. HOMO.cube) for use in Avogadro / VMD / Multiwfn."
        ),
        layout=layout_fn(width="160px", margin="8px 0 4px 8px"),
    )
    app._iso_export_status = widgets.HTML(
        value="", layout=layout_fn(margin="0 0 0 8px")
    )
    iso_body = widgets.VBox(
        [
            widgets.HTML(
                '<p style="color:#555;font-size:12px;margin:0 0 8px">'
                "Visualise a molecular orbital as a 3D isosurface (Linux / WSL only — "
                "requires PySCF and RDKit). Run or load a Single Point or Geometry "
                "Optimization first, then click <b>Generate</b>.</p>"
            ),
            app._orb_iso_controls,
            widgets.HBox(
                [
                    app._iso_generate_btn,
                    app._iso_export_cube_btn,
                    app._iso_export_status,
                ],
                layout=layout_fn(align_items="center", gap="6px"),
            ),
        ],
        layout=layout_fn(padding="8px"),
    )
    app._iso_accordion = widgets.Accordion(
        children=[iso_body],
        layout=layout_fn(display="none", margin="8px 0"),
    )
    app._iso_accordion.set_title(0, "Orbital Isosurface")
    app._iso_accordion.selected_index = None

    app._uv_mode_toggle = widgets.ToggleButtons(
        options=["Stick", "Broadened"],
        value="Stick",
        style={"button_width": "80px"},
        layout=layout_fn(margin="0 8px 0 0"),
    )
    app._uv_fwhm_slider = widgets.FloatSlider(
        value=20.0,
        min=5.0,
        max=100.0,
        step=5.0,
        description="Line width (nm):",
        readout_format=".0f",
        style={"description_width": "110px"},
        layout=layout_fn(width="290px", display="none"),
        # Fire only on slider release — avoids a re-render storm during drag
        # that, combined with the full HTML output swap, causes the page
        # to scroll back to the top mid-drag.
        continuous_update=False,
    )
    # min_height matches the Plotly UV-Vis figure height (320px) so the
    # Output container does not briefly collapse to 0px during the atomic
    # outputs swap on mode/slider changes — same fix as the IR Output above.
    app._tddft_fig = widgets.Output(
        layout=layout_fn(width="100%", min_height="320px"),
    )
    uv_export_row = _plot_export_row("uv")
    uv_controls = widgets.HBox(
        [app._uv_mode_toggle, app._uv_fwhm_slider],
        layout=layout_fn(align_items="center", margin="0 0 6px 0"),
    )
    app._tddft_accordion = widgets.Accordion(
        children=[
            widgets.VBox(
                [uv_controls, uv_export_row, app._tddft_fig],
                layout=layout_fn(padding="8px"),
            )
        ],
        layout=layout_fn(display="none", margin="8px 0"),
    )
    app._tddft_accordion.set_title(0, "UV-Vis Absorption Spectrum")
    app._tddft_accordion.selected_index = None

    app._nmr_output = widgets.HTML(value="", layout=layout_fn(width="100%"))
    app._nmr_accordion = widgets.Accordion(
        children=[
            widgets.VBox(
                [app._nmr_output],
                layout=layout_fn(padding="8px"),
            )
        ],
        layout=layout_fn(display="none", margin="8px 0"),
    )
    app._nmr_accordion.set_title(0, "NMR Chemical Shifts")
    app._nmr_accordion.selected_index = None

    app._result_dir_label = widgets.HTML(
        value="",
        layout=layout_fn(display="none", margin="4px 0 0 0"),
    )

    app._result_log_output = widgets.Output()
    app._result_log_accordion = widgets.Accordion(
        children=[app._result_log_output],
        layout=layout_fn(display="none", margin="8px 0 0 0"),
    )
    app._result_log_accordion.set_title(0, "Full output log (pyscf.log)")
    app._result_log_accordion.selected_index = None

    app._go_results_btn = widgets.Button(
        description="→ View Results",
        button_style="success",
        layout=layout_fn(width="130px"),
    )
    app._go_analysis_btn = widgets.Button(
        description="→ View Analysis",
        button_style="info",
        layout=layout_fn(width="140px"),
    )
    app._completion_mol_lbl = widgets.HTML(value="")
    app._completion_banner = widgets.HBox(
        [
            widgets.HTML(
                '<span style="color:#22c55e;font-weight:600;font-size:13px">'
                "✓ Calculation complete — </span>"
            ),
            app._completion_mol_lbl,
            app._go_results_btn,
            app._go_analysis_btn,
        ],
        layout=layout_fn(
            display="none",
            align_items="center",
            gap="8px",
            padding="10px 12px",
            border="1px solid #bbf7d0",
            border_radius="6px",
            background_color="#f0fdf4",
            margin="8px 0",
        ),
    )

    app._to_analysis_btn = widgets.Button(
        description="→ View Analysis",
        button_style="",
        icon="bar-chart",
        layout=layout_fn(display="none", width="160px", margin="8px 0 0 0"),
    )
    app._viz_label = widgets.HTML(
        value="",
        layout=layout_fn(display="none"),
    )
    app.results_tab_panel = widgets.VBox(
        [
            widgets.HTML('<h3 style="margin:14px 0 6px">Results</h3>'),
            app.result_output,
            app._viz_label,
            app.result_viz_output,
            app._result_dir_label,
            app._to_analysis_btn,
        ],
        layout=layout_fn(padding="8px 0"),
    )
    app.results_panel = app.results_tab_panel

    app._analysis_mol_output = widgets.Output()

    # Analysis-tab backend toggle — mirrors the Calculate-tab `viz_backend_toggle`.
    # Created only when both backends are available (matches Calculate-tab
    # convention). Synchronized with the Calculate-tab toggle via
    # `_set_viz_preference` + `_viz_sync_in_progress` flag in app.py.
    if app.viz_backend_toggle is not None:
        app.viz_backend_toggle_ana = widgets.ToggleButtons(
            options=[("PlotlyMol", "plotlymol"), ("py3Dmol", "py3dmol")],
            value=app._viz_backend,
            tooltips=["Plotly-based interactive viewer", "WebGL viewer (py3Dmol)"],
            style={"button_width": "90px"},
            layout=layout_fn(margin="2px 0 4px 0"),
        )
        # Small "Rendering with: X" label — updated by _update_analysis_backend_label
        # after each render so the user can see what's actually rendering even
        # when preference is "auto" (per-task routing may select different
        # backends than the toggle suggests).
        app.viz_backend_label_ana = widgets.HTML(
            value=(
                '<span style="font-size:11px;color:#94a3b8;font-style:italic">'
                "Rendering with: —</span>"
            ),
            layout=layout_fn(margin="0 0 8px 0"),
        )
        ana_backend_row = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.HTML(
                            '<span style="font-size:11px;color:#94a3b8;'
                            'margin-right:6px;align-self:center">Backend:</span>'
                        ),
                        app.viz_backend_toggle_ana,
                    ],
                    layout=layout_fn(align_items="center"),
                ),
                app.viz_backend_label_ana,
            ],
        )
    else:
        app.viz_backend_toggle_ana = None  # type: ignore[assignment]
        app.viz_backend_label_ana = None  # type: ignore[assignment]
        ana_backend_row = None

    app._analysis_context_lbl = widgets.HTML(
        value=(
            '<p style="color:#555;font-size:13px;margin:4px 0 12px">'
            "No result loaded yet. Run a calculation or load one from History.</p>"
        )
    )
    app._analysis_empty_html = widgets.HTML(
        value=(
            '<p style="color:#888;font-size:13px;font-style:italic;margin:8px 0">'
            "No interactive analysis is available for this calculation type.<br>"
            "Run a Single Point, Geo Opt, or Frequency calculation to see "
            "energy-level diagrams, trajectory animations, and spectra here.</p>"
        ),
        layout=layout_fn(display="none"),
    )
    app._ana_unavail_html = widgets.HTML(value="", layout=layout_fn(display="none"))
    app._build_ana_switcher()

    ana_children = [
        app._analysis_context_lbl,
        app._analysis_mol_output,
    ]
    if ana_backend_row is not None:
        ana_children.append(ana_backend_row)
    ana_children.extend(
        [
            app._analysis_empty_html,
            app._ana_unavail_html,
            app._orb_accordion,
            app._pes_scan_accordion,
            app.traj_accordion,
            app.vib_accordion,
            app._ir_accordion,
            app._iso_accordion,
            app._tddft_accordion,
            app._nmr_accordion,
        ]
    )
    app.analysis_tab_panel = widgets.VBox(
        ana_children,
        layout=layout_fn(padding="8px 0"),
    )
    app.post_calc_panel = app.analysis_tab_panel


def build_compare_section(app: Any, *, layout_fn: Any, rdkit_available: bool) -> None:
    """Build compare tab widgets and export accordion."""
    app.compare_select = widgets.SelectMultiple(
        options=[("(no saved results)", "")],
        rows=8,
        description="",
        layout=layout_fn(width="100%"),
    )
    app.compare_refresh_btn = widgets.Button(
        description="Refresh",
        button_style="",
        icon="refresh",
        layout=layout_fn(width="100px"),
    )
    app.compare_btn = widgets.Button(
        description="Compare selected",
        button_style="primary",
        icon="bar-chart",
        disabled=True,
        layout=layout_fn(width="180px"),
    )
    app.compare_clear_btn = widgets.Button(
        description="Clear",
        button_style="warning",
        icon="times",
        layout=layout_fn(width="90px"),
    )
    app.compare_output = widgets.Output()

    app.compare_panel = widgets.VBox(
        [
            widgets.HTML(
                '<h3 style="margin:14px 0 6px">Compare Calculations</h3>'
                '<p style="color:#555;font-size:13px;margin:0 0 8px">'
                "Select two or more saved calculations to compare side-by-side. "
                "Hold Ctrl (or ⌘) to select multiple entries.</p>"
            ),
            widgets.HBox([app.compare_refresh_btn]),
            app.compare_select,
            widgets.HBox(
                [app.compare_btn, app.compare_clear_btn],
                layout=layout_fn(gap="8px", margin="6px 0"),
            ),
            app.compare_output,
        ],
        layout=layout_fn(padding="8px 0"),
    )

    rdkit_note = (
        ""
        if rdkit_available
        else '<p style="color:#888;font-size:12px;margin:4px 0 0">MOL/PDB export requires RDKit '
        "(<code>conda install -c conda-forge rdkit</code>).</p>"
    )
    export_content = widgets.VBox(
        [
            widgets.HTML(
                '<p style="color:#555;font-size:13px;margin:0 0 8px">'
                "Download a self-contained PySCF script you can study or run outside the notebook.</p>"
            ),
            widgets.HBox([app.export_btn, app.export_status]),
            widgets.HTML('<hr style="margin:10px 0 8px">'),
            widgets.HTML(
                '<p style="color:#555;font-size:13px;margin:0 0 6px">'
                "Download the molecular structure in a standard chemistry file format.</p>"
                + rdkit_note
            ),
            widgets.HBox(
                [app.export_xyz_btn, app.export_mol_btn, app.export_pdb_btn],
                layout=layout_fn(flex_wrap="wrap", gap="6px"),
            ),
            app.struct_export_status,
            widgets.HTML('<hr style="margin:10px 0 8px">'),
            widgets.HTML(
                '<p style="color:#555;font-size:13px;margin:0 0 6px">'
                "Bundle every file in this result folder into a single zip "
                "for sharing.</p>"
            ),
            widgets.HBox(
                [app._export_bundle_btn, app._export_bundle_status],
                layout=layout_fn(align_items="center", gap="6px"),
            ),
        ]
    )
    app.advanced_accordion = widgets.Accordion(children=[export_content])
    app.advanced_accordion.set_title(0, "Export")
    app.advanced_accordion.selected_index = None

    app._populate_compare_list()


def build_output_tab(app: Any, *, layout_fn: Any) -> None:
    """Build the Output tab panel widgets."""
    app._log_output_html = widgets.HTML(
        '<span style="color:#94a3b8;font-size:13px">'
        "No log yet — run a calculation first, or use "
        "<b>View log</b> in the History tab.</span>"
    )
    app._log_source_lbl = widgets.HTML()
    app._log_clear_btn = widgets.Button(
        description="Clear",
        button_style="",
        icon="times",
        layout=layout_fn(width="80px"),
    )
    app._clear_log_cache_btn = widgets.Button(
        description="Clear Log Cache",
        button_style="",
        icon="trash",
        tooltip=(
            "Delete the session event log (event_log.jsonl). "
            "Calculation performance data is preserved."
        ),
        layout=layout_fn(width="160px"),
    )
    app._clear_log_cache_confirm_btn = widgets.Button(
        description="Confirm clear?",
        button_style="danger",
        layout=layout_fn(width="140px", display="none"),
    )
    # POLISH.8 (M-POLISH, 2026-05-25): the Log tab moved to be an
    # Accordion inside the History tab — rationale in the roadmap. The
    # explanatory text no longer needs to say "Use View log in the
    # History tab" since the user IS in the History tab now.
    app.log_tab_panel = widgets.VBox(
        [
            widgets.HTML(
                '<p style="color:#555;font-size:13px;margin:4px 0 8px">'
                "Raw PySCF output for the most recent calculation or the "
                "currently-selected history result. "
                "Energy-level diagrams, trajectories, and spectra are in the "
                "<b>Analysis</b> tab.</p>"
            ),
            widgets.HBox(
                [app._log_clear_btn],
                layout=layout_fn(margin="0 0 8px"),
            ),
            app._log_source_lbl,
            app._log_output_html,
            app._result_log_accordion,
            widgets.HTML(
                '<hr style="border:none;border-top:1px solid #e2e8f0;margin:16px 0 10px"/>'
                '<p style="color:#94a3b8;font-size:12px;margin:0 0 6px">'
                "Session event log — records molecule loads, calculations, "
                "and issue reports across this session.</p>"
            ),
            widgets.HBox(
                [app._clear_log_cache_btn, app._clear_log_cache_confirm_btn],
                layout=layout_fn(align_items="center", gap="8px"),
            ),
        ],
        layout=layout_fn(padding="8px 0"),
    )

    # POLISH.8: wrap the log panel in an Accordion + append to the
    # History tab. ``history_panel`` was built in
    # ``build_history_section`` earlier in the app-init sequence
    # (see app.py: _build_history_section runs BEFORE _build_output_tab).
    app._history_log_accordion = widgets.Accordion(
        children=[app.log_tab_panel],
        selected_index=None,
    )
    app._history_log_accordion.set_title(0, "PySCF output log")
    app.history_panel.children = (
        *app.history_panel.children,
        app._history_log_accordion,
    )


def build_files_tab(app: Any, *, layout_fn: Any) -> None:
    """Build the read-only Files tab widgets."""
    app._files_root_dd = widgets.Dropdown(
        options=[("(loading)", "")],
        value="",
        description="Root:",
        style={"description_width": "40px"},
        layout=layout_fn(width="520px"),
    )
    app._files_path_html = widgets.HTML(
        value=(
            '<span style="font-size:12px;color:#64748b">'
            "Current folder: (not set)</span>"
        )
    )
    app._files_entries = widgets.Select(
        options=[("(no files)", "")],
        rows=12,
        description="",
        layout=layout_fn(width="100%"),
    )
    app._files_up_btn = widgets.Button(
        description="Up",
        icon="arrow-up",
        layout=layout_fn(width="80px"),
        tooltip="Go to parent folder",
    )
    app._files_open_btn = widgets.Button(
        description="Open",
        button_style="primary",
        icon="folder-open",
        layout=layout_fn(width="100px"),
        tooltip="Open selected folder or preview selected file",
    )
    app._files_refresh_btn = widgets.Button(
        description="Refresh",
        icon="refresh",
        layout=layout_fn(width="100px"),
        tooltip="Refresh roots, folder contents, and preview",
    )
    app._files_status_html = widgets.HTML(
        value=(
            '<span style="font-size:12px;color:#94a3b8">'
            "Select a file to preview it; use Open to enter a folder.</span>"
        )
    )
    app._files_preview_output = widgets.Output(
        layout=layout_fn(
            border="1px solid #cbd5e1",
            min_height="220px",
            max_height="420px",
            overflow="auto",
            padding="6px",
        )
    )

    app.files_tab_panel = widgets.VBox(
        [
            widgets.HTML(
                '<p style="color:#555;font-size:13px;margin:4px 0 8px">'
                "Read-only file browser for result artifacts and logs. "
                "Browsing is limited to approved roots.</p>"
            ),
            app._files_root_dd,
            app._files_path_html,
            widgets.HBox(
                [app._files_up_btn, app._files_open_btn, app._files_refresh_btn],
                layout=layout_fn(gap="8px", margin="6px 0"),
            ),
            app._files_entries,
            app._files_status_html,
            app._files_preview_output,
        ],
        layout=layout_fn(padding="8px 0"),
    )


def build_help_section(app: Any, *, layout_fn: Any) -> None:
    """Build the floating help panel and top-bar help/exit buttons."""
    help_keys = list(HELP_TOPICS.keys())
    help_labels = [HELP_TOPICS[k]["title"] for k in help_keys]
    app.help_topic_dd = widgets.Dropdown(
        options=list(zip(help_labels, help_keys)),
        description="Topic:",
        style={"description_width": "60px"},
        layout=layout_fn(width="460px"),
    )
    app.help_content_html = widgets.HTML()
    app._render_help_topic()

    # POLISH.2 (M-POLISH, 2026-05-25): the single-character "?" was
    # visually noisy and hard to recognise as the global help toggle.
    # Field-level "?" buttons (method_help_btn / basis_help_btn earlier
    # in this file) keep the symbol — for inline-with-input help it's
    # universally understood.
    app._help_btn = widgets.Button(
        description="Help",
        button_style="",
        icon="question-circle",
        tooltip="Help topics",
        layout=layout_fn(width="80px", margin="0 0 0 8px"),
    )

    app._exit_btn = widgets.Button(
        description="Exit",
        button_style="danger",
        tooltip="Shut down the QuantUI server and close this session",
        layout=layout_fn(width="64px", margin="0 0 0 8px"),
    )
    app._exit_output = widgets.Output(layout=layout_fn(height="0px", overflow="hidden"))

    app.help_tab_panel = widgets.VBox(
        [
            widgets.HTML(
                '<p style="color:#555;font-size:13px;margin:4px 0 12px">'
                "Browse help topics below. Click <b>?</b> next to the Method or Basis Set "
                "dropdown in the Calculate tab to jump directly to a relevant topic.</p>"
            ),
            app.help_topic_dd,
            app.help_content_html,
        ],
        layout=layout_fn(
            display="none",
            padding="8px 0",
            border="1px solid #e2e8f0",
            border_radius="6px",
            padding_left="12px",
            margin="0 0 8px",
        ),
    )


def build_issue_widgets(app: Any, *, layout_fn: Any) -> None:
    """Build issue-report widgets shown from the top toolbar."""
    app._issue_btn = widgets.Button(
        description="Report Issue",
        button_style="warning",
        icon="flag",
        tooltip="Report a bug or unexpected behaviour observed in this session",
        layout=layout_fn(width="140px", margin="0 0 0 8px"),
    )
    app._issue_textarea = widgets.Textarea(
        placeholder=(
            "Describe what you observed — what you did, what you expected, "
            "and what actually happened."
        ),
        layout=layout_fn(width="100%", height="90px"),
    )
    app._issue_submit_btn = widgets.Button(
        description="Submit",
        button_style="success",
        layout=layout_fn(width="90px"),
    )
    app._issue_cancel_btn = widgets.Button(
        description="Cancel",
        button_style="",
        layout=layout_fn(width="80px"),
    )
    app._issue_status_html = widgets.HTML()
    app._issue_overlay = widgets.VBox(
        [
            widgets.HTML(
                '<p style="font-size:13px;font-weight:600;margin:0 0 6px;color:#92400e">'
                "&#9872; Report Issue</p>"
                '<p style="font-size:12px;color:#78350f;margin:0 0 8px">'
                "Your report (and a snapshot of the current session state) will be "
                "saved to <code>issues.db</code> and the session event log.</p>"
            ),
            app._issue_textarea,
            widgets.HBox(
                [app._issue_submit_btn, app._issue_cancel_btn],
                layout=layout_fn(margin="6px 0 0", gap="8px"),
            ),
            app._issue_status_html,
        ],
        layout=layout_fn(
            display="none",
            border="1px solid #f59e0b",
            border_radius="6px",
            padding="12px 14px",
            margin="0 0 6px",
            background_color="#fffbeb",
        ),
    )
