# QuantUI — AI Assistant Context

> Stable project context for GitHub Copilot, Claude, and other AI coding assistants.
> Describes what the project IS and how it is built — not where development currently
> stands (see your project STATUS.md for current session state). Update this file when
> architecture or conventions change, not every session.

---

## Overview

QuantUI is an interactive Jupyter/Voilà platform for running PySCF quantum
chemistry workflows end-to-end inside one app: setup, execution, analysis,
visualization, and comparison. It is local-first today (no cluster account, no
SLURM required), and a future roadmap item is to add optional cluster-backed
execution through interactive Jupyter/HPC environments.

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
│   ├── app.py                ← QuantUIApp — orchestration + run dispatch
│   ├── app_analysis.py       ← Analysis-tab panel registry + _pop_* methods
│   ├── app_builders.py       ← _build_* widget construction
│   ├── app_runflow.py        ← _do_run + run/orchestration UI handlers
│   ├── app_visualization.py  ← Trajectory / vib / IR / orbital / PES rendering
│   ├── app_history.py        ← History tab loaders, replay context
│   ├── app_formatters.py     ← Result-card and text formatters
│   ├── app_exports.py        ← Export handlers (XYZ/MOL/PDB/script)
│   ├── molecule.py           ← Molecule dataclass + XYZ/SMILES parsing
│   ├── session_calc.py       ← In-session PySCF runner (run_in_session)
│   ├── optimizer.py          ← QM geometry optimization (ASE-BFGS + PySCF)
│   ├── freq_calc.py          ← Frequency / vibrational analysis
│   ├── tddft_calc.py         ← TD-DFT UV-Vis excited states
│   ├── nmr_calc.py           ← NMR shielding (GIAO), returns NMRResult
│   ├── pes_scan.py           ← Potential energy surface (bond/angle/dihedral)
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
│   ├── preopt.py             ← ASE force-field pre-optimisation (fast, no PySCF)
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
├── tests/                    ← pytest suite (~1000 tests; 97 PySCF-gated skip on Windows)
├── .github/
│   └── copilot-instructions.md  ← This file
├── apptainer/
│   ├── quantui.def     ← Apptainer container definition
│   └── build.sh              ← Build script
├── local-setup/              ← Conda environment YAMLs
├── launchers/                ← Double-click launchers (Windows .bat, macOS .command, Linux/WSL .sh)
├── CHANGELOG.md              ← Release history (Keep a Changelog format)
├── pyproject.toml            ← Package config (name: quantui, imports as quantui)
└── pytest.ini                ← pytest configuration
```

**Modular UI note.** `QuantUIApp` lives in `app.py`, but most of its logic is
implemented as module-level functions in `app_builders.py`, `app_analysis.py`,
`app_runflow.py`, `app_visualization.py`, `app_history.py`, `app_formatters.py`,
and `app_exports.py`. `app.py` keeps thin instance-method wrappers that pass
`self` through (e.g. `def _pop_energies(self, ctx): return _ana_pop_energies(self, ctx)`).
When working on UI changes, prefer editing the relevant companion module
rather than growing `app.py`. The conservative refactor close (DEC-014)
intentionally leaves `_do_run` and a small molecule/theme/pubchem callback
cluster in `app.py`.

---

## Architecture

```
notebooks/molecule_computations.ipynb
    Cell 0: Markdown title
    Cell 1: Conda env check (skip-execution, remove-input)
    Cell 2: from quantui.app import QuantUIApp; QuantUIApp().display()
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
      │      _build_help_section()                                         │
      │      _build_issue_widgets()                                        │
      │    _wire_callbacks()                                               │
      │    _assemble_tabs()            → root_tab (7 tabs)                 │
      └─────────────────────────────────────────────────────────────────────┘
                │  _do_run() dispatches by calc_type_dd.value:
                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Single Point  → session_calc.run_in_session()              │
    │  Geometry Opt  → optimizer.optimize_geometry()              │
    │  Frequency     → freq_calc.run_freq_calc()                  │
    │  UV-Vis        → tddft_calc.run_tddft_calc()                │
    │  NMR Shielding → nmr_calc.run_nmr_calc()                    │
    │  PES Scan      → pes_scan.run_pes_scan()                    │
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

**Root-tab order in the app:** Calculate (0) → Results (1) → Analysis (2) → History (3) → Compare (4) → Log (5) → Status (6)

**Help UI:** Help is a floating overlay panel toggled by `[?]`; it is not a root tab.

---

## The Panel Registry Pattern — CRITICAL

This is the core architecture for the Analysis tab. **Any work touching analysis panels
must understand this pattern.**

### How it works

After a calculation completes (live or history), `_apply_analysis_context(ctx)` is the
single entry point for all panel population. It:

1. Calls `_deactivate_all_ana_panels()` to reset all 8 panels — collapses accordions
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
    result_dir: Optional[Path] = None  # saved result dir
    molecule: Optional[Any] = None
    spectra_data: dict = field(default_factory=dict)
    preopt_result: Optional[Any] = None  # OptimizationResult from Frequency pre-opt
    source: str = "live"     # "live" | "history"
```

### `_PANEL_REGISTRY`

```python
_PANEL_REGISTRY = {
    "single_point": [("Energies", "_pop_energies", True), ("Isosurface", "_pop_isosurface", False)],
    "geometry_opt": [("Trajectory", "_pop_geo_trajectory", True), ("Energies", "_pop_energies", False), ...],
   "frequency":    [("Vibrational", "_pop_vibrational", True), ("IR Spectrum", "_pop_ir_spectrum", True), ("Trajectory", "_pop_preopt_trajectory", False), ("Energies", "_pop_energies", True)],
    "tddft":        [("UV-Vis", "_pop_uv_vis", True)],
    "nmr":          [("NMR", "_pop_nmr_shielding", True)],
    "pes_scan":     [("PES Scan", "_pop_pes_plot", True), ("Trajectory", "_pop_pes_trajectory", False)],
}
```

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
from quantui.viz_backend_router import VizTask, select_backend
decision = app._resolve_backend(VizTask.STRUCTURE_VIEW_RESULTS)
# decision.chosen is "py3dmol", "plotlymol", or None (no renderer)
# decision.reason carries a human-readable explanation
```

`Decision` is a frozen dataclass — never mutate it.

### Calculate ↔ Analysis toggle sync

The Calculate and Analysis tabs each have a "Default 3D backend" ToggleButton.
They are kept in sync via `_set_viz_preference(...)`, gated by
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

**All four render dispatch sites are wrapped:** static structure view,
trajectory frame, vib mode, and the vib sync cache-hit fast path. Use this
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

## Analysis Tab — 8 Panels

All 8 panels are **always in the DOM** (`layout.display=""`, `selected_index=None`).
Unavailable panels show a "Not available — run a X calculation first" placeholder.
`_activate_ana_panel()` swaps the placeholder for real content and expands the accordion.
`_deactivate_all_ana_panels()` restores placeholders and collapses — never hides.

| Panel | Accordion attr | Activated by calc types |
|---|---|---|
| Energies | `_orb_accordion` | Single Point, Geometry Opt |
| Trajectory | `traj_accordion` | Geometry Opt, PES Scan, Frequency (pre-opt only) |
| Vibrational | `vib_accordion` | Frequency |
| IR Spectrum | `_ir_accordion` | Frequency |
| PES Scan | `_pes_scan_accordion` | PES Scan |
| Isosurface | `_iso_accordion` | Single Point (Linux/WSL) |
| UV-Vis | `_tddft_accordion` | UV-Vis (TD-DFT) |
| NMR | `_nmr_accordion` | NMR Shielding |

---

## Key Files

| File | Purpose |
| ---- | ------- |
| `quantui/app.py` | **Primary development target.** All widget logic lives here. |
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

8. **Never use `include_plotlyjs="cdn"` in widget HTML.** CDN requests fail
   silently in offline classroom deployments — the figure div stays blank with no
   error anywhere. Use `include_plotlyjs="require"` (Voilà/RequireJS path) or
   `include_plotlyjs=True` (inline bundle, works everywhere). This rule is enforced
   by `tests/test_code_quality.py::test_no_cdn_plotlyjs`.

9. **All `.observe()` callbacks must be wrapped with `_safe_cb`.** Exceptions in
   raw `.observe()` handlers disappear into the kernel console — invisible in Voilà.
   Use `widget.observe(self._safe_cb(self._on_x), names="value")` so exceptions are
   routed to the Log tab instead. See `_safe_cb()` in `app.py`.

10. **After significant code changes, run the pre-commit sequence before handoff.**
   For meaningful edits in `quantui/`, run:
   - `pre-commit run --all-files`
   - `python -m pytest tests/ -q --no-cov`

11. **Proactively recommend PR checkpoints for QuantUI codebase work.**
   This applies to `repos-PUBLIC/QuantUI` development (feature, bugfix, and refactor
   work), especially when moving into a different roadmap theme/milestone family.
   Also recommend a PR after completing each major extraction/refactor phase that
   passes validation.
   Do not proactively recommend PRs for planning-doc-only updates in
   `repos-WRITING/Research-Project-Admin` unless the user explicitly asks.

---

## Supported Calculations

| Calc type | Module | Key function | Returns | save_type |
| --- | --- | --- | --- | --- |
| Single Point | `session_calc` | `run_in_session()` | `SessionResult` | `"single_point"` |
| Geometry Opt | `optimizer` | `optimize_geometry()` | `OptResult` | `"geometry_opt"` |
| Frequency | `freq_calc` | `run_freq_calc()` | `FreqResult` | `"frequency"` |
| UV-Vis TD-DFT | `tddft_calc` | `run_tddft_calc()` | `TDDFTResult` | `"tddft"` |
| NMR Shielding | `nmr_calc` | `run_nmr_calc()` | `NMRResult` | `"nmr"` |
| PES Scan | `pes_scan` | `run_pes_scan()` | `PESScanResult` | `"pes_scan"` |

**Supported methods:** RHF, UHF, B3LYP, PBE, PBE0, M06-2X, wB97X-D, CAM-B3LYP,
M06-L, HSE06, PBE-D3, MP2 (defined in `config.SUPPORTED_METHODS`).

**Supported basis sets:** STO-3G, 3-21G, 6-31G, 6-31G\*, 6-31G\*\*, cc-pVDZ,
cc-pVTZ, def2-SVP, def2-TZVP (defined in `config.SUPPORTED_BASIS_SETS`).

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
      _build_help_section()
      _build_issue_widgets()
   _wire_callbacks()             # all .observe() and .on_click() wiring
   _assemble_tabs()              # builds self.root_tab (7 tabs)
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
    trajectory.json ← Geo Opt / PES Scan trajectories
    orbitals.npz   ← MO arrays for Single Point / Geo Opt
    thumbnail.png  ← auto-generated molecule thumbnail
```

Default results dir: `Path("results")` relative to cwd, or `$QUANTUI_RESULTS_DIR`.

### result.json schema (version 2)

```json
{
  "_schema_version": 2,
  "timestamp": "YYYY-MM-DD_HH-MM-SS-ffffff",
  "calc_type": "single_point | geometry_opt | frequency | tddft | nmr | pes_scan",
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
    "uv_vis": {"excitation_energies_ev": [...], "oscillator_strengths": [...], "wavelengths_nm": [...]},
    "nmr":    {"atom_symbols": [...], "shielding_iso_ppm": [...], "chemical_shifts_ppm": {"0": 1.2, ...}, "reference_compound": "TMS"},
    "molecule": {"atoms": [...], "coords": [...], "charge": 0, "multiplicity": 1}
  }
}
```

Schema rules: new fields must be additive only. Never remove or rename existing keys.

---

## Performance Logging (`quantui/calc_log.py`)

Two JSONL files under `~/.quantui/logs/` (override with `$QUANTUI_LOG_DIR`):

| File | Contents |
| --- | --- |
| `perf_log.jsonl` | One record per converged run: formula, n_atoms, n_electrons, method, basis, elapsed_s, n_basis |
| `event_log.jsonl` | Startup / calc_start / calc_done / calc_error events (7-day auto-prune) |

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

**PySCF-gated tests** use `@pytest.mark.skipif(not _PYSCF_AVAILABLE, ...)`.
On Windows, these become skips — not failures.

**Baseline (Windows `quantui-win`, 2026-05-22; `python -m pytest tests/ -q --no-cov`):**
1004 passed, 97 skipped (the 97 skips are PySCF-gated Linux-only tests).

---

## Dependency Groups

| Group | Packages | Notes |
| --- | --- | --- |
| Core (always installed) | `jupyter`, `ipywidgets`, `notebook`, `numpy`, `matplotlib`, `plotly`, `plotlymol`, `py3Dmol` | Base UI + visualization stack |
| `pyscf` extra | `pyscf>=2.13.0`, `pyscf-properties` | Enables SCF/frequency/TDDFT/NMR/PES backends |
| `ase` extra | `ase>=3.22.0` | Geometry optimization and structure I/O |
| `app` extra | `voila>=0.5.0`, `ipykernel>=6.0.0` | Student-facing Voilà deployment |
| `notebook` extra | `nbmake>=1.4.0`, `ipykernel>=6.0.0` | Notebook smoke-testing support |
| `dev` extra | `pytest`, `pytest-cov`, `pytest-mock`, `mypy`, `ruff`, `black`, `pre-commit` | Development tooling |

Install all runtime + dev extras: `pip install -e ".[pyscf,ase,app,notebook,dev]"`

---

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `QUANTUI_RESULTS_DIR` | `./results` | Where calculation results are saved |
| `QUANTUI_LOG_DIR` | `~/.quantui/logs` | Where perf_log and event_log live |
| `QUANTUI_FREQ_PARALLEL` | (unset / off) | When truthy (`1`, `true`, `yes`, `on`), fan out IR-intensity finite-difference SCFs across CPU worker processes in `freq_calc.py` (via `freq_ir_workers.py`). Reference SCF + Hessian still use gpu4pyscf when available; workers are CPU-only. Requires ≥4 cores and ≥2 atoms. Useful on NCShare-style nodes (one GPU, many cores). |
| `QUANTUI_SETTINGS_PATH` | `~/.quantui/settings.json` | User-preferences file (`user_settings.py`); override for tests |

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

The following module/file names are deliberately absent from `quantui/` and
should not be reintroduced without an explicit roadmap milestone. They would
only make sense once cluster-backed execution is added (a future roadmap
item, not currently scoped).

| File / module | Why it's not here |
| --- | --- |
| `job_manager.py` | SLURM batch submission belongs to the future cluster-execution path |
| `storage.py` | SLURM job-metadata persistence — same future scope |
| `slurm_errors.py` | SLURM error translation — same future scope |
| `visualization.py` (the PlotlyMol-fallback module) | Superseded by `viz_backend_router.py` + `visualization_py3dmol.py` |
| SLURM-related templates in `config.py` | No cluster orchestration today |
