# QuantUI — AI Assistant Context

> Stable project context for GitHub Copilot, Claude, and other AI coding assistants.
> Describes what the project IS and how it is built — not where development currently
> stands (see your project STATUS.md for current session state). Update this file when
> architecture or conventions change, not every session.

---

## Overview

QuantUI is an interactive Jupyter/Voilà platform for running PySCF quantum
chemistry workflows end-to-end inside one app: setup, execution, analysis,
visualization, and comparison. It is local-first: calculations run in the
user's own kernel by default. An optional SLURM batch backend (Cluster Jobs
tab, `quantui submit`, the `quantui-batch` login-node launcher;
`quantui/backends/`) sends calculations to a cluster when an operator enables
it (`QUANTUI_ENABLE_SLURM`).

**Primary users:** Undergraduate chemistry students and researchers at North Carolina
Central University and collaborators. The UI runs as a Voilà app so users can run
serious quantum chemistry workflows without needing to work directly in code.

**Strategic direction:** Build an open, powerful, researcher-grade alternative to
closed-source, high-cost GUI quantum chemistry workflows (GaussView-style usability,
with transparent and extensible open tooling).

---

## Repository Structure

```
QuantUI/
├── quantui/                  ← Main Python package (imports as `quantui`)
│   ├── app.py                ← QuantUIApp — orchestration, _do_run, panel registry (_PANEL_META/_PANEL_REGISTRY)
│   ├── app_analysis.py       ← Analysis-tab _pop_* methods, apply_analysis_context, panel switcher
│   ├── app_builders.py       ← _build_* widget construction
│   ├── app_runflow.py        ← run / calc-type / seed / calibration UI handlers
│   ├── app_launcher.py       ← `quantui run app` / `quantui view` Voilà launch
│   ├── app_viewer.py         ← read-only viewer mode (History + Analysis only)
│   ├── app_slurm.py          ← Cluster Jobs tab + SLURM result ingestion
│   ├── app_structure_edit.py ← Edit Structure tools (set bond/angle/dihedral, delete atoms…)
│   ├── app_measurement.py    ← click-to-measure distances/angles in the viewer
│   ├── app_pes_pick.py       ← pick PES-scan atoms in the viewer
│   ├── app_xyz_input.py      ← XYZ Input tab handlers
│   ├── app_visualization.py  ← Trajectory / vib / IR / orbital / PES rendering
│   ├── app_history.py        ← History tab loaders, replay context
│   ├── app_formatters.py     ← Result-card and text formatters
│   ├── app_exports.py        ← Export handlers (XYZ/MOL/PDB/script)
│   ├── molecule.py           ← Molecule dataclass + XYZ/SMILES parsing
│   ├── session_calc.py       ← In-session PySCF runner (run_in_session)
│   ├── optimizer.py          ← QM geometry optimization (ASE-BFGS + PySCF)
│   ├── freq_calc.py          ← Frequency / vibrational analysis
│   ├── tddft_calc.py         ← TD-DFT UV-Vis excited states + ECD rotatory strengths
│   ├── nmr_calc.py           ← NMR shielding (GIAO), returns NMRResult
│   ├── pes_scan.py           ← Potential energy surface (bond/angle/dihedral)
│   ├── ts_search.py          ← Transition-state search (Sella) + frequency check
│   ├── irc.py                ← IRC reaction path (Sella)
│   ├── reorganization_energy.py ← Marcus four-point reorganization energy
│   ├── raman_calc.py         ← Static Raman activities
│   ├── checkpoint.py         ← Checkpoint/resume + SCF warm start
│   ├── cli.py                ← `quantui` command-line toolkit
│   ├── engines/              ← PySCF / PyFock engine dispatch (run_calc(EngineRequest))
│   ├── backends/             ← SLURM batch backend: worker, registry, batch input, cluster config
│   ├── calculator.py         ← PySCFCalculation abstraction
│   ├── comparison.py         ← Side-by-side result comparison table
│   ├── results_storage.py    ← Persist/reload calculation results (JSON + log)
│   ├── calc_log.py           ← Performance + event logging (JSONL)
│   ├── pubchem.py            ← PubChem molecule search
│   ├── visualization_py3dmol.py ← 3D molecular viewer (py3Dmol / plotlyMol)
│   ├── viz_backend_router.py ← Capability-aware backend router (pure function)
│   ├── user_settings.py      ← Persistent user preferences (~/.quantui/settings.json)
│   ├── vib_cache.py          ← On-disk cache of rendered vib-mode HTML
│   ├── ase_bridge.py         ← ASE structure I/O + molecule library
│   ├── preopt.py             ← RDKit MMFF94→UFF (+ optional xtb GFN-FF) pre-optimisation (fast, no PySCF)
│   ├── progress.py           ← StepProgress widget
│   ├── help_content.py       ← HELP_TOPICS dict — in-app educational text
│   ├── orbital_visualization.py ← Orbital energy diagrams, cube file viewer
│   ├── ir_plot.py            ← IR spectrum Plotly figure builder
│   ├── config.py             ← All constants/defaults (methods, basis sets, etc.)
│   ├── utils.py              ← Session resource checks, sanitize_filename, etc.
│   ├── issue_tracker.py      ← In-app issue/bug logging to issues.db
│   ├── log_utils.py          ← Shared logging helpers
│   ├── benchmarks.py         ← Performance benchmarking utilities
│   └── security.py           ← SecurityError exception class
├── notebooks/
│   ├── molecule_computations.ipynb  ← Student-facing Voilà app (thin launcher)
│   └── tutorials/                   ← 01–05 step-by-step tutorial notebooks
├── tests/                    ← pytest suite (~4,100 tests; PySCF-gated tests skip on native Windows)
├── .github/
│   └── copilot-instructions.md  ← This file
├── apptainer/
│   ├── quantui.def / quantui-gpu.def ← CPU and GPU container definitions
│   ├── build.sh / build-gpu.sh       ← Build scripts
│   └── slurm/                        ← SLURM batch guide + templates
├── local-setup/              ← Conda environment YAMLs
├── launchers/                ← Double-click launchers (Windows .bat, macOS .command, Linux/WSL .sh)
├── CHANGELOG.md              ← Release history (Keep a Changelog format)
└── pyproject.toml            ← Package config (name: quantui) + [tool.pytest.ini_options]
```

**Modular UI note.** `QuantUIApp` lives in `app.py`, but most of its logic is
implemented as module-level functions in the 14 `app_*.py` companion modules
listed above. `app.py` keeps thin instance-method wrappers that pass
`self` through (e.g. `def _pop_energies(self, ctx): return _ana_pop_energies(self, ctx)`).
When working on UI changes, prefer editing the relevant companion module
rather than growing `app.py`. The conservative refactor close (DEC-014)
intentionally leaves `_do_run` and a small molecule/theme/pubchem callback
cluster in `app.py`.

---

## Architecture

```
notebooks/molecule_computations.ipynb
    Cell 0: Conda env check (code; tags skip-execution, remove-input)
    Cell 1: from quantui.app import QuantUIApp; QuantUIApp().display()
    Cell 2: (empty)
                │
                ▼
        quantui/app.py — QuantUIApp
      ┌─────────────────────────────────────────────────────────────────────┐
      │  __init__()                                                        │
      │    _build_widgets()                                                │
      │      _build_theme_selector()                                       │
      │      _build_status_panel()                                         │
      │      _build_welcome_header()                                       │
      │      _build_shared_widgets()                                       │
      │      _build_molecule_section()                                     │
      │      _build_calc_setup()                                           │
      │      _build_run_section()                                          │
      │      _build_results_section()   → Analysis accordions + _build_ana_switcher() │
      │      _build_history_section()                                      │
      │      _build_compare_section()                                      │
      │      _build_output_tab()                                           │
      │      _build_slurm_jobs_tab()                                       │
      │      _build_files_tab()                                            │
      │      _build_help_section()                                         │
      │      _build_issue_widgets()                                        │
      │      (viewer mode: folder bar)                                     │
      │    _wire_callbacks()                                               │
      │    _assemble_tabs()            → root_tab (7, or 8 with Cluster Jobs) │
      └─────────────────────────────────────────────────────────────────────┘
                │  _do_run() dispatches by calc_type_dd.value:
                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Single Point  → engines.run_calc(EngineRequest)            │
    │                  (PySCF: session_calc.run_in_session)        │
    │  Geometry Opt  → engines.run_calc(EngineRequest)            │
    │                  (PySCF: optimizer.optimize_geometry)        │
    │  Transition State → ts_search.run_ts_search()               │
    │  IRC           → irc.run_irc()                              │
    │  Frequency     → freq_calc.run_freq_calc()                  │
    │  UV-Vis        → tddft_calc.run_tddft_calc()                │
    │  NMR Shielding → nmr_calc.run_nmr_calc()                    │
    │  PES Scan      → pes_scan.run_pes_scan()                    │
    │  Reorganization Energy → reorganization_energy.run_reorganization_energy() │
    └──────────────────────────────────────────────────────────────┘
                │
                ▼  _apply_analysis_context(_AnalysisContext)
    ┌──────────────────────────────────────────────────────────────┐
    │  _PANEL_REGISTRY[calc_type] → [(_panel_name, _pop_fn, auto)]│
    │  Each _pop_xxx() returns bool → _activate_ana_panel()       │
    └──────────────────────────────────────────────────────────────┘
                │
                ▼
   results_storage.save_result()   calc_log.log_calculation()/log_event()
```

**Root-tab order in the app** (`_root_tab_order`): Calculate → [Cluster Jobs, only
when the SLURM backend is selected] → Results → Analysis → History → Compare →
Files → System Settings. The PySCF output log is an accordion inside History.
Viewer mode (`quantui view`) shows only History and Analysis.

**Help UI:** Help is a floating overlay panel toggled by `[?]`; it is not a root tab.

---

## The Panel Registry Pattern — CRITICAL

This is the core architecture for the Analysis tab. **Any work touching analysis panels
must understand this pattern.**

### How it works

After a calculation completes (live or history), `_apply_analysis_context(ctx)` is the
single entry point for all panel population. It:

1. Calls `_deactivate_all_ana_panels()` to reset all 12 panels — collapses accordions
   and restores "Not available" placeholders (panels remain in the DOM — never hidden)
2. Looks up `_PANEL_REGISTRY[ctx.calc_type]` for an ordered list of
   `(panel_name, populate_method_name, auto_select)` tuples
3. Calls each populate method (e.g. `_pop_energies(ctx)`)
4. If it returns `True`, calls `_activate_ana_panel(panel_name)` to make the panel
   available and optionally auto-select it
5. Updates the Analysis tab context label and navigation button visibility

### `_AnalysisContext` dataclass

```python
@dataclass
class _AnalysisContext:
    calc_type: str           # "single_point" | "geometry_opt" | "frequency" | etc.
    formula: str
    method: str
    basis: str
    live_result: Any = None  # result object from _do_run; None for history
    result_dir: Optional[Any] = None  # Path to saved dir; None before save_result
    molecule: Optional[Any] = None
    spectra_data: dict = field(default_factory=dict)
    preopt_result: Optional[Any] = None  # OptimizationResult from pre-opt step
    timestamp: str = ""      # result timestamp shown in history dropdown labels
    source: str = "live"     # "live" | "history"
```

### `_PANEL_REGISTRY`

A `ClassVar` on `QuantUIApp` in `app.py` (abridged — read the source; the
inline comments explain why order matters):

```python
_PANEL_REGISTRY = {
    "single_point":   [Energies, Isosurface, Populations],
    "geometry_opt":   [Energies(False), Isosurface(True), Trajectory(True), Populations],
    "frequency":      [Vibrational, IR Spectrum, Raman Spectrum, Trajectory (pre-opt), Energies, Isosurface, Populations],
    "transition_state": [Vibrational, IR Spectrum, Trajectory, Energies, Isosurface, Populations],
    "irc":            [PES Scan (energy profile via _pop_irc_profile), Trajectory],
    "tddft":          [UV-Vis, ECD, Energies, Isosurface, Populations],
    "nmr":            [NMR],
    "reorganization_energy": [Energies(False), Geometries(True), Isosurface],
    "pes_scan":       [PES Scan, Trajectory (_pop_pes_trajectory)],
}
```

Two ordering rules: the **first** entry with `auto_select=True` whose populator
returns `True` becomes the default panel, and `_pop_energies` must run before
`_pop_isosurface` (it loads the orbital state Isosurface checks).

### Rules for all panel populate methods (`_pop_xxx`)

- **Must return `bool`** — `True` if data was populated, `False` if data is missing/unavailable
- **Must NOT call `_activate_ana_panel()`** — the registry loop handles activation
- Support both live (use `ctx.live_result`) and history (use `ctx.result_dir` / `ctx.spectra_data`)
- Must be side-effect safe: if data is missing, clear/no-op the widget and return `False`

### Adding a new panel — exactly these steps

1. Create the accordion widget in `_build_results_section()` (follow existing pattern)
2. Add it to the `analysis_tab_panel` VBox in `_build_results_section()`
3. Add `(panel_name, accordion_attr_name, "available after X")` to `_PANEL_META`
   (class-level `ClassVar` — this is the single source of truth for placeholder text
   and accordion attribute lookup; `_build_ana_switcher()` reads it at init time)
4. Write `_pop_xxx(self, ctx: _AnalysisContext) -> bool`
5. Add the entry to `_PANEL_REGISTRY`
6. If history replay needs data: ensure `save_spectra` in `_do_run` saves it

### Live run vs history replay — same code path

Both `_do_run()` and the history loaders (`_on_view_log`, `_history_load_analysis`)
build an `_AnalysisContext` and call `_apply_analysis_context()`. They are guaranteed
to show identical panels for the same result data. Use `_build_history_context(result_dir)`
to build the context from disk.

---

## Visualization Backend Router

3D rendering is dispatched through a pure-function router in
`quantui/viz_backend_router.py`. **Do not call py3Dmol or plotlymol3d
directly from `app_visualization.py` — always go through the router.**

### `VizTask` keys

`MOLECULE_PREVIEW`, `STRUCTURE_VIEW_RESULTS`, `ANALYSIS_STRUCTURE_VIEW`,
`HISTORY_STRUCTURE_REPLAY`, `TRAJECTORY_FRAME`, `TRAJECTORY_EXPORT`,
`VIB_INTERACTIVE`, `VIB_EXPORT`, `ORBITAL_ISOSURFACE`.

### Policy

- Each task has a `(primary, fallback)` tuple.
- User preference (`VizPreference.AUTO | PY3DMOL | PLOTLYMOL`) is respected
  for tasks that support both backends.
- Single-backend tasks **ignore preference** (e.g. `TRAJECTORY_FRAME` is
  py3Dmol-only — Plotly causes flicker; `ORBITAL_ISOSURFACE` is py3Dmol-only).
- `BackendAvailability.from_environment()` probes imports at startup; the
  router falls back gracefully when one backend is missing.

### Usage

```python
from quantui.viz_backend_router import (
    BackendAvailability, VizPreference, VizTask, select_backend,
)
decision = select_backend(
    VizTask.STRUCTURE_VIEW_RESULTS,
    VizPreference("auto"),
    BackendAvailability.from_environment(),
)
# decision.chosen is "py3dmol", "plotlymol", or None (no renderer)
# decision.reason carries a human-readable explanation

# Inside the app, the wrapper logs the decision and returns only the backend:
backend = app._resolve_backend(VizTask.STRUCTURE_VIEW_RESULTS)  # VizBackend | None
```

`Decision` is a frozen dataclass — never mutate it.

### Calculate ↔ Analysis toggle sync

The Calculate and Analysis tabs each have a "Default 3D backend" ToggleButton,
and System Settings has a `viz_default_backend_dd` dropdown. All three are kept
in sync via `_set_viz_preference(...)`, gated by
`_viz_sync_in_progress` to prevent observer echo loops. The preference is
persisted to `~/.quantui/settings.json` (`viz.default_backend`).

---

## Lifecycle Telemetry

Every render-dispatch site in `app_visualization.py` is wrapped in the
`_viz_render_event(app, task, backend, **extras)` context manager. It emits
three event types to `event_log.jsonl`:

| Event | When |
| --- | --- |
| `viz_render_start` | Entry into the context |
| `viz_render_done` | Successful exit, with `elapsed_ms` |
| `viz_render_error` | Exception in the body, with `elapsed_ms` + `error` |

**Every render dispatch site is wrapped** (nine `with _viz_render_event(` sites
today: structure views, trajectory frames, vib modes, the vib cache-hit fast
path, isosurfaces, …). Use this
helper for any new render call so telemetry stays uniform.

---

## Persistent User Settings

`quantui/user_settings.py` provides `UserSettings.load()` /
`UserSettings.save()`, persisted to `~/.quantui/settings.json` (override with
`QUANTUI_SETTINGS_PATH` for testing).

Current schema (`_schema_version = 1`):

| Section | Field | Default | Purpose |
| --- | --- | --- | --- |
| `viz` | `default_backend` | `"auto"` | One of `auto / py3dmol / plotlymol` |
| `viz` | `vib_framerate_fps` | `10` | Clamped `[1, 120]`; included in vib cache key |
| `viz` | `iso_resolution` | `"medium"` | Isosurface grid resolution |
| `viz` | `ir_axis` | `"descending"` | IR wavenumber axis direction |
| `viz` | `dipole_arrow` | `"physics"` | Dipole arrow convention |
| `theme` | `palette` | `"Dark"` | One of the six palettes in `theme.py` |
| `compute` | `gpu_enabled` | `True` | GPU offload when a device is present |
| `compute` | `density_fit` | `False` | Density fitting (RI) |
| `compute` | `execution_backend` | `"local"` | `local` or `slurm` |
| `compute` | `slurm_job_root` | `""` | SLURM job-folder root (empty = default) |
| `compute` | `quantum_engine` | `"auto"` | `auto / pyscf / pyfock` |
| `compute` | `freq_parallel` | `False` | Parallel IR/Raman displacement workers |

**Robustness rules:** atomic writes (`.tmp` + rename); missing file, malformed
JSON, unknown schema version, missing sections, or invalid values all fall
back to defaults with a single warning log — startup never crashes on bad
settings. Schema growth is additive; bump `_schema_version` only for
breaking changes.

---

## Vib-Animation Disk Cache

`quantui/vib_cache.py` stores rendered py3Dmol HTML for each vibrational mode
under `<result_dir>/vib_frames/`:

```
<result_dir>/
└── vib_frames/
    ├── index.json   ← manifest (schema v1)
    ├── mode_001.html
    └── ...
```

**Cache key:** `(result_dir, mode_number, n_frames, amplitude, renderer, fps)`.
Any parameter change → stale → cache miss → re-render.

**Sync cache-hit fast path:** `_swap_vib_output()` atomically swaps in cached
HTML without a transient "Rendering…" placeholder, eliminating flash on
mode switches.

**Staleness guard:** `_vib_render_token` is incremented on every render
request; background renders that finish after a newer token has been issued
bail out instead of overwriting newer output.

---

## Analysis Tab — 12 Panels

All 12 panels are **always in the DOM** (`layout.display=""`, `selected_index=None`).
Unavailable panels show a "Not available — run a X calculation first" placeholder.
`_activate_ana_panel()` swaps the placeholder for real content and expands the accordion.
`_deactivate_all_ana_panels()` restores placeholders and collapses — never hides.

| Panel | Accordion attr | Activated by calc types |
|---|---|---|
Source of truth: `_PANEL_META` / `_PANEL_REGISTRY` in `app.py`.

| Panel | Accordion attr | Activated by calc types |
|---|---|---|
| Energies | `_orb_accordion` | Single Point, Geometry Opt, Frequency, Transition State, UV-Vis, Reorganization Energy |
| Populations | `_mulliken_accordion` | Single Point, Geometry Opt, Frequency, Transition State, UV-Vis |
| Trajectory | `traj_accordion` | Geometry Opt, Transition State, IRC, PES Scan, Frequency (pre-opt) |
| Vibrational | `vib_accordion` | Frequency, Transition State |
| IR Spectrum | `_ir_accordion` | Frequency, Transition State |
| Raman Spectrum | `_raman_accordion` | Frequency |
| PES Scan | `_pes_scan_accordion` | PES Scan, IRC (energy profile) |
| Isosurface | `_iso_accordion` | Single Point, Geometry Opt, Frequency, Transition State, UV-Vis, Reorganization Energy (Linux/WSL) |
| Geometries | `_reorg_geom_accordion` | Reorganization Energy |
| UV-Vis | `_tddft_accordion` | UV-Vis (TD-DFT) |
| ECD | `_ecd_accordion` | UV-Vis (TD-DFT) — rotatory strengths from the same states |
| NMR | `_nmr_accordion` | NMR Shielding |

---

## Key Files

| File | Purpose |
| ---- | ------- |
| `quantui/app.py` | `QuantUIApp`: orchestration, `_do_run`, panel registry. Widget construction and most handlers live in the `app_*.py` companion modules. |
| `quantui/session_calc.py` | `run_in_session()` — PySCF SCF runner; returns `SessionResult` |
| `quantui/molecule.py` | `Molecule` dataclass, `parse_xyz_input()` |
| `quantui/config.py` | `SUPPORTED_METHODS`, `SUPPORTED_BASIS_SETS`, `MOLECULE_LIBRARY`, widget layout constants |
| `quantui/results_storage.py` | `save_result()`, `load_result()`, `list_results()` — result.json schema v2 |
| `quantui/calc_log.py` | `perf_log.jsonl` + `event_log.jsonl` under `~/.quantui/logs/` |
| `quantui/optimizer.py` | `optimize_geometry()` — ASE-BFGS + custom `_QuantUIPySCFCalc` |
| `quantui/freq_calc.py` | `run_freq_calc()` — vibrational analysis via `pyscf.hessian` |
| `quantui/tddft_calc.py` | `run_tddft_calc()` — excited states via `pyscf.tddft` |
| `quantui/nmr_calc.py` | `run_nmr_calc()` — GIAO shielding; returns `NMRResult` |
| `quantui/pes_scan.py` | `run_pes_scan()` — bond/angle/dihedral PES; returns `PESScanResult` |
| `quantui/ir_plot.py` | `plot_ir_spectrum()` — Plotly stick/broadened IR chart |
| `quantui/ecd_plot.py` | `ecd_delta_epsilon()` / `plot_ecd_spectrum()` — Gaussian Δε broadening (Stephens–Harada) and ECD chart |
| `notebooks/molecule_computations.ipynb` | Thin launcher — 3 cells only (do not add logic here) |
| your project `STATUS.md` | **Read this first every session** — current state, git log, open tasks |
| your project `feature-requests.md` | FR backlog |

---

## Critical Constraints

> These are hard rules that must be respected in all implementation work.

1. **PySCF is Linux/macOS/WSL only.** Never assume PySCF is available. All
   PySCF-dependent code must be guarded with `try/except ImportError`. The
   availability flags in `app.py` (`_PYSCF_AVAILABLE`, `_PREOPT_AVAILABLE`,
   `ASE_AVAILABLE`, `VISUALIZATION_AVAILABLE`) are computed once at module
   import — check these instead of importing inline.

2. **No notebook cell logic.** `notebooks/molecule_computations.ipynb` is a
   three-cell thin launcher. Never add widget creation, callbacks, or business
   logic to notebook cells. All logic belongs in `quantui/app.py`.

3. **Thread-safe widget updates only.** `_do_run()` runs in a background thread.
   Widget updates from threads must use `.value =`, `.append_stdout()`, or
   `.append_display_data()`. For Plotly, prefer `plotly.io.to_html(...)` + HTML append
   rather than direct `display(fig)` in a thread.

4. **No new top-level dependencies** without updating both `pyproject.toml`
   and the Apptainer container `apptainer/quantui.def`.

5. **All constants in `config.py`.** Method names, basis sets, layout widths,
   and other shared literals must be defined in `config.py` and imported from
   there — never hard-coded in `app.py` or other modules.

6. **Result schema versioning.** `results_storage.py` uses `_SCHEMA_VERSION = 2`.
   Any new fields must be additive (never remove or rename existing keys). Bump
   `_SCHEMA_VERSION` only when a breaking change is unavoidable.

7. **Panel populate methods must not call `_activate_ana_panel()`.** Only the
   registry loop in `_apply_analysis_context()` activates panels. Helper methods
   (`_show_orbital_diagram`, `_show_vib_animation`, `_show_ir_spectrum`,
   `_show_pes_scan_result`) return `bool` and must not call `_activate_ana_panel()`.

8. **Render in-app Plotly figures with `quantui.plotly_embed.figure_html`.**
   It loads the installed package's plotly.js once per page and embeds each
   figure with `include_plotlyjs=False`. Never use `"cdn"` (fails silently
   offline) or `"require"` (plotly ≥ 6 has no such mode: it inlines the full
   ~4.8 MB bundle per figure). Standalone exported HTML files keep
   `include_plotlyjs=True` so they work on their own. Enforced by
   `tests/test_code_quality.py`.

9. **All `.observe()` callbacks must be wrapped with `_safe_cb`.** Exceptions in
   raw `.observe()` handlers disappear into the kernel console — invisible in Voilà.
   Use `widget.observe(self._safe_cb(self._on_x), names="value")` so exceptions are
   logged to `event_log.jsonl` (via `calc_log.log_event`) instead. See `_safe_cb()`
   in `app.py`.

10. **After significant code changes, run the pre-commit sequence before handoff.**
   For meaningful edits in `quantui/`, run:
   - `pre-commit run --all-files`
   - `python -m pytest tests/ -q --no-cov`

11. **Proactively recommend PR checkpoints for QuantUI codebase work.**
   This applies to QuantUI code work (feature, bugfix, and refactor), especially
   when moving into a different roadmap theme/milestone family. Also recommend a
   PR after completing each major extraction/refactor phase that passes
   validation. Do not proactively recommend PRs for planning-doc-only updates in
   the planning repo (`NCCU-Schultz-Lab/QuantUI-development-tracking`) unless the
   user explicitly asks.

---

## Supported Calculations

| Calc type | Module | Key function | Returns | save_type |
| --- | --- | --- | --- | --- |
| Single Point | `session_calc` (via `engines.run_calc`) | `run_in_session()` | `SessionResult` | `"single_point"` |
| Geometry Opt | `optimizer` (via `engines.run_calc`) | `optimize_geometry()` | `OptimizationResult` | `"geometry_opt"` |
| Transition State | `ts_search` | `run_ts_search()` | `TSResult` | `"transition_state"` |
| IRC | `irc` | `run_irc()` | `IRCResult` | `"irc"` |
| Frequency | `freq_calc` | `run_freq_calc()` | `FreqResult` | `"frequency"` |
| UV-Vis TD-DFT | `tddft_calc` | `run_tddft_calc()` | `TDDFTResult` | `"tddft"` |
| NMR Shielding | `nmr_calc` | `run_nmr_calc()` | `NMRResult` | `"nmr"` |
| PES Scan | `pes_scan` | `run_pes_scan()` | `PESScanResult` | `"pes_scan"` |
| Reorganization Energy | `reorganization_energy` | `run_reorganization_energy()` | `ReorganizationEnergyResult` | `"reorganization_energy"` |

**Supported methods:** RHF, UHF, B3LYP, PBE, PBE0, M06-2X, wB97X-D, CAM-B3LYP,
M06-L, HSE06, PBE-D3, MP2, CCSD, CCSD(T) (defined in `config.SUPPORTED_METHODS`;
post-HF subset in `config.POST_HF_METHODS`, single point only).

**Supported basis sets:** STO-3G, 3-21G, 6-31G, 6-31G\*, 6-31+G\*, 6-31G\*\*,
6-31++G\*\*, cc-pVDZ, aug-cc-pVDZ, cc-pVTZ, def2-SVP, def2-SVPD, def2-TZVP, LANL2DZ
(defined in `config.SUPPORTED_BASIS_SETS`; diffuse subset in `config.DIFFUSE_BASIS_SETS`).

---

## `QuantUIApp` Class (`quantui/app.py`)

### Construction order

```
__init__()
   _build_widgets()
      _build_theme_selector()
      _build_status_panel()
      _build_welcome_header()
      _build_shared_widgets()
      _build_molecule_section()
      _build_calc_setup()
      _build_run_section()
      _build_results_section()    # also calls _build_ana_switcher()
      _build_history_section()
      _build_compare_section()
      _build_output_tab()
      _build_slurm_jobs_tab()
      _build_files_tab()
      _build_help_section()
      _build_issue_widgets()
      (viewer mode: folder bar)
   _wire_callbacks()             # all .observe() and .on_click() wiring
   _assemble_tabs()              # builds self.root_tab (7, or 8 with Cluster Jobs)
```

### Key instance state — Analysis tab

| Attribute | Type | Purpose |
| --- | --- | --- |
| `self._ana_available` | `set[str]` | Names of panels with data; populated by `_activate_ana_panel()` |
| `self._ana_active` | `str` | Currently expanded panel name |
| `self._ana_panel_names` | `list[str]` | Ordered panel names derived from `_PANEL_META` |
| `self._ana_accordions` | `list[Accordion]` | Ordered accordions derived from `_PANEL_META` |
| `self._ana_unavail_html` | `widgets.HTML` | Shared unavailable-state message area (usually hidden) |
| `self._ana_unavail_msgs` | `dict[str, HTML]` | Panel name → "Not available" placeholder widget |
| `self._ana_content_boxes` | `dict[str, VBox]` | Panel name → real content VBox (hidden until activated) |
| `self._pending_traj_result` | `Any` | Trajectory stub; rendered lazily on accordion expand |
| `self._last_orb_info` | `Optional[...]` | Orbital info from last successful `_show_orbital_diagram` |
| `self._last_ir_freqs` | `list[float]` | IR frequencies stored by `_show_ir_spectrum` |
| `self._last_ir_ints` | `list[float]` | IR intensities stored by `_show_ir_spectrum` |
| `self._last_vib_data` | `Optional[VibrationalData]` | plotlyMol vib data for mode selector |
| `self._last_pes_result` | `Optional[...]` | PES result stored by `_show_pes_scan_result` |

### Key instance state — general

| Attribute | Type | Purpose |
| --- | --- | --- |
| `self._molecule` | `Optional[Molecule]` | Currently loaded molecule |
| `self._last_result` | `Optional[...]` | Most recent calculation result |
| `self._results` | `list` | All results from this session |
| `self._pyscf_available` | `bool` | Mirrors module-level `_PYSCF_AVAILABLE` |
| `self.root_tab` | `widgets.Tab` | Top-level displayed widget |
| `self.method_dd` | `widgets.Dropdown` | Selected QC method |
| `self.basis_dd` | `widgets.Dropdown` | Selected basis set |
| `self.calc_type_dd` | `widgets.Dropdown` | Calc type selector |
| `self.result_viz_output` | `widgets.Output` | 3D molecule rendered after every calc |
| `self._last_result_dir` | `Optional[Path]` | Saved result directory from most recent run |

---

## Result Storage (`quantui/results_storage.py`)

Results are saved to timestamped subdirectories:
```
<QUANTUI_RESULTS_DIR>/<timestamp>_<formula>_<method>_<basis>/
    result.json    ← schema v2 (see below)
    pyscf.log      ← raw PySCF stdout (may be absent)
    trajectory.json ← Geo Opt / PES Scan / TS / IRC trajectories
    trajectory.xyz / trajectory.traj ← the same frames for external tools
    result.molden  ← MOs (+ vibrations for Frequency)
    orbitals.npz   ← MO arrays for Single Point / Geo Opt
    thumbnail.png  ← auto-generated molecule thumbnail
```

Default results dir: `Path("results")` relative to cwd, or `$QUANTUI_RESULTS_DIR`.

### result.json schema (version 2)

```json
{
  "_schema_version": 2,
  "timestamp": "YYYY-MM-DD_HH-MM-SS-ffffff",
  "calc_type": "single_point | geometry_opt | transition_state | irc | frequency | tddft | nmr | pes_scan | reorganization_energy",
  "formula": "H2O",
  "method": "RHF",
  "basis": "STO-3G",
  "energy_hartree": -75.0,
  "energy_ev": -2040.8,
  "homo_lumo_gap_ev": 12.3,
  "converged": true,
  "n_iterations": 12,
  "spectra": {
    "ir":     {"frequencies_cm1": [...], "ir_intensities": [...], "zpve_hartree": 0.021, "displacements": [...]},
    "uv_vis": {"excitation_energies_ev": [...], "oscillator_strengths": [...], "wavelengths_nm": [...],
               "rotatory_strengths_velocity": [...], "rotatory_strengths_length": [...]},
    "nmr":    {"atom_symbols": [...], "shielding_iso_ppm": [...], "chemical_shifts_ppm": {"0": 1.2, ...}, "reference_compound": "TMS"},
    "molecule": {"atoms": [...], "coords": [...], "charge": 0, "multiplicity": 1}
  }
}
```

`uv_vis.rotatory_strengths_*` (ECD, SPC2.3) are in 10⁻⁴⁰ esu²·cm²; absent on older results.

Schema rules: new fields must be additive only. Never remove or rename existing keys.

---

## Performance Logging (`quantui/calc_log.py`)

Three JSONL files under `~/.quantui/logs/` (override with `$QUANTUI_LOG_DIR`;
the in-app issue tracker's `issues.db` lives alongside them):

| File | Contents |
| --- | --- |
| `perf_log.jsonl` | One record per converged run: formula, n_atoms, n_electrons, method, basis, elapsed_s, n_basis |
| `event_log.jsonl` | Startup / calc_start / calc_done / calc_error / viz events (entries older than 7 days pruned every 20 writes) |
| `prediction_log.jsonl` | Time-estimate predictions vs actual run times (feeds the analytics "Prediction accuracy" section) |

Key API: `log_calculation()`, `get_perf_history()`, `estimate_time()`, `reset_perf_log()`.
Performance estimation uses a 4-strategy priority chain: N_basis-normalised efficiency →
electron-count scaling → cross-method N_basis → cross-method electron-count.

---

## Naming Conventions

- **Functions:** `verb_noun()` — e.g., `parse_xyz_input()`, `run_in_session()`, `save_result()`
- **Classes:** `PascalCase` — e.g., `QuantUIApp`, `SessionResult`, `FreqResult`
- **Private methods/helpers:** leading underscore — e.g., `_do_run()`, `_set_molecule()`
- **Builder methods in `QuantUIApp`:** `_build_<section>()` — e.g., `_build_calc_setup()`
- **Callback methods:** `_on_<widget>_<event>()` — e.g., `_on_run_clicked()`, `_on_theme_changed()`
- **Panel populate methods:** `_pop_<panel_name>()` — e.g., `_pop_energies()`, `_pop_ir_spectrum()`
- **Module-level availability flags:** `_PYSCF_AVAILABLE`, `_PREOPT_AVAILABLE`, `ASE_AVAILABLE`
- **Config constants:** `ALL_CAPS_SNAKE_CASE` in `config.py`
- **Section banners in `app.py`:** `# ══ SECTION NAME ══` delimiters for VS Code outline navigation

---

## How to Run

```powershell
# Activate environment (Windows PowerShell)
& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate quantui

# Voilà app mode (student-facing — no code visible)
voila notebooks/molecule_computations.ipynb

# JupyterLab (development)
jupyter lab notebooks/molecule_computations.ipynb

# Run tests
python -m pytest tests/ -q --no-cov

# Install/update package
pip install -e ".[dev]"

# Verify app.py import
python -c "from quantui.app import QuantUIApp; print('OK')"
```

**WSL note:** Always `cd` to the repo root before running Python in WSL. The `''`
entry in `sys.path` resolves to the current directory; running from the wrong directory
can shadow the editable install with a different repo's `quantui/` package.

**Python executable:** `C:\Users\<user>\miniconda3\envs\quantui\python.exe`

Note: PySCF calculations will show "unavailable" on Windows — this is expected.
All UI, molecule, visualization, and PubChem features work natively on Windows.

---

## Testing

Test files in `tests/`:

| File | What it covers |
| --- | --- |
| `test_app.py` | QuantUIApp — widgets, panel registry, callbacks, always-visible panels |
| `test_issue_tracker.py` | Issue logging DB integration and context capture |
| `test_molecule.py` | Molecule parsing, validation, formula |
| `test_session_calc.py` | `run_in_session()` — PySCF-gated |
| `test_notebook_interactions.py` | Notebook widget interaction flows |
| `test_notebook_workflows.py` | End-to-end HF/DFT/preopt/thread-safety — PySCF-gated |
| `test_optimizer.py` | `optimize_geometry()` — PySCF + ASE required |
| `test_freq_calc.py` | `run_freq_calc()` — PySCF-gated |
| `test_nmr_calc.py` | `run_nmr_calc()` — PySCF + pyscf.prop required |
| `test_pes_scan.py` | `run_pes_scan()` + PES analysis panel |
| `test_ir_plot.py` | `plot_ir_spectrum()` — stick and broadened modes |
| `test_visualization.py` / `test_visualization_integration.py` | 3D and animation rendering paths |
| `test_templates.py` | Export template/script integrity |
| `test_comparison.py` | Result comparison tables |
| `test_results_storage.py` | Save/load/list round-trip |
| `test_security.py` | `SecurityError`, `sanitize_filename()` |
| `test_code_quality.py` | Static analysis — bans CDN plotlyjs and bare `except/pass` |
| `test_sp_analysis_history.py` | Single-point analysis history replay (end-to-end) |
| `test_geo_opt_analysis_history.py` | Geometry-opt analysis history replay |
| `test_tddft_analysis_history.py` | TD-DFT / UV-Vis analysis history replay |
| `test_nmr_analysis_history.py` | NMR analysis history replay |
| `test_freq_analysis_history.py` | Frequency analysis history replay (Vibrational + IR) |
| `test_pes_scan_analysis_history.py` | PES scan analysis history replay |

**PySCF-gated tests** use `@pytest.mark.skipif(not _PYSCF_AVAILABLE, ...)` or
`pytest.importorskip("pyscf")`. On Windows, these become skips — not failures.

The table above lists the core files only; `tests/` has many more (one per
feature area, e.g. `test_backends_worker.py`, `test_checkpoint_*.py`,
`test_ts_search.py`, `test_irc.py`, `test_cli.py`).

**Baseline (Linux, Python 3.11, 2026-11; `pytest -m "not network"`):** ~4120
passed, 24 skipped.

---

## Dependency Groups

| Group | Packages | Notes |
| --- | --- | --- |
Source of truth: `pyproject.toml`.

| Group | Packages | Notes |
| --- | --- | --- |
| Core (always installed) | `jupyter`, `ipywidgets`, `notebook`, `numpy`, `requests`, `matplotlib`, `plotly`, `plotlymol`, `kaleido`, `py3Dmol` | Base UI + visualization stack |
| `pyscf` extra | `pyscf>=2.13.0`, `pyscf-properties`, `pyscf-dispersion` | SCF/frequency/TDDFT/NMR/PES backends; NMR + Raman; PBE-D3 |
| `pyfock` extra | `pyfock>=0.1.7,<0.3` (+ ase, scipy, numba) | Native-Windows PBE subset |
| `ase` extra | `ase>=3.22.0` | Geometry optimization and structure I/O |
| `xtb` extra | `xtb` | GFN-FF pre-optimization for metals (Linux wheels) |
| `ts` extra | `sella>=2.6`, `ase>=3.26` | Transition State and IRC |
| `app` extra | `voila>=0.5.0,<0.6`, `ipykernel>=6.0.0` | Student-facing Voilà deployment |
| `gpu-cuda12x` / `gpu-cuda13x` | gpu4pyscf + cupy + cutensor | GPU offload (match the driver's CUDA version) |
| `pyfock-gpu-cuda12x` / `-cuda13x` | cupy | PyFock GPU path |
| `notebook` extra | `nbmake>=1.4.0`, `ipykernel>=6.0.0` | Notebook smoke-testing support |
| `dev` extra | `pytest`, `pytest-cov`, `pytest-mock`, `pytest-xdist`, `build`, `mypy`, `types-requests`, `ruff`, `black`, `pre-commit` | Development tooling |

Install all runtime + dev extras: `pip install -e ".[pyscf,ase,app,notebook,dev]"`

---

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `QUANTUI_RESULTS_DIR` | `./results` | Where calculation results are saved |
| `QUANTUI_LOG_DIR` | `~/.quantui/logs` | Where perf_log and event_log live |
| `QUANTUI_FREQ_PARALLEL` | (unset / off) | When truthy (`1`, `true`, `yes`, `on`), fan out IR-intensity finite-difference SCFs across CPU worker processes. Same as the System Settings checkbox; **overrides** the saved setting when set. Reference SCF + Hessian still use gpu4pyscf when available. Requires ≥4 cores and ≥2 atoms. |
| `QUANTUI_SETTINGS_PATH` | `~/.quantui/settings.json` | User-preferences file (`user_settings.py`); override for tests |
| `QUANTUI_DISABLE_GPU` | (unset) | Force CPU even when gpu4pyscf and a device are present |
| `QUANTUI_CHECKPOINT_DIR` | `~/.quantui/checkpoints` | Interactive checkpoint/resume store |
| `QUANTUI_ENABLE_SLURM` | (unset / off) | Offer the SLURM batch backend (needs `sbatch`) |
| `QUANTUI_STAGING_DIR` | System Settings value, else `~/.quantui/staging` | SLURM job-folder root |
| `QUANTUI_SLURM_*` | see `apptainer/slurm/README.md` | Partition, GRES, limits, cooldowns for batch jobs |

---

## Apptainer Container

The container at `apptainer/quantui.def` bundles Python + PySCF + ASE +
py3Dmol + Voilà into a single portable `.sif` file. This is the supported path
for Windows users.

- Build: `bash apptainer/build.sh --clean` (requires Linux/WSL with Apptainer ≥ 1.0)
- Run (app mode): `apptainer run quantui.sif app`
- Run (JupyterLab): `apptainer run quantui.sif`
- Verify: `apptainer test quantui.sif`

The container sets `QUANTUI_RESULTS_DIR=$HOME/.quantui/results` so results survive
across kernel restarts and are accessible from the host (home dir is bind-mounted).

---

## Scope Notes — Intentionally Out of Repo

Cluster execution now exists, in `quantui/backends/` (worker, job registry,
`slurm.py`, `slurm_errors.py`, `cluster_security.py`, batch input) plus
`app_slurm.py` and the `quantui-batch` launcher. The following names from the
pre-backend design are still deliberately absent and should not be
reintroduced:

| File / module | Why it's not here |
| --- | --- |
| `job_manager.py` | Superseded by `quantui/backends/` (registry + dispatch) |
| `storage.py` | Job metadata lives in the backend's job registry and job folders |
| `visualization.py` (the PlotlyMol-fallback module) | Superseded by `viz_backend_router.py` + `visualization_py3dmol.py` |
| SLURM templates in `config.py` | Batch scripts are rendered by `quantui/backends/` |
