# Changelog

All notable changes to QuantUI are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-08-06

### Fixed

- **Running the test suite no longer corrupts your time estimates.** QuantUI's
  own tests were writing their runs into `~/.quantui/logs/perf_log.jsonl` — the
  file the runtime estimator learns from. Because those tests use a *simulated*
  calculation, each one recorded a fabricated timing: 2 773 records were in the
  log and roughly four fifths of them were test artifacts, including "water
  frequency" runs whose recorded times ranged from 0.34 s to 143 s for identical
  chemistry. This is the reason the pre-run estimate had been unreliable. Only
  developers running the test suite were affected; the fabricated records are
  now superseded automatically as real runs accumulate, so no manual cleanup is
  needed.
- **Calibration now times the calculation, not the process.** Each calibration
  step runs in a fresh subprocess and its stopwatch started before PySCF was
  even imported, so every calibration record was inflated by startup cost that
  a real run in an open session never pays. Import time is still recorded, just
  separately.

### Added

- **Checkpoints — an interrupted calculation can pick up where it stopped.**
  Geometry Optimization saves its trajectory and the optimizer's accumulated
  curvature after every step; PES Scan banks each point as it finishes. If a
  run is cancelled, crashes, or the machine goes to sleep, the Calculate tab
  offers to resume it — and says how much is already done ("8 of 20 scan points
  already computed") rather than just asking. The offer appears only when the
  calculation you have configured is *exactly* the interrupted one, geometry
  included, so a resume can never splice two different runs together.
- **An "Unfinished calculations" list on the History tab.** After restarting
  QuantUI you no longer have to remember what you were running: every
  interrupted calculation is listed with its molecule, type, level of theory,
  how much finished and how long ago. **Load these settings** puts the molecule
  and all its settings back on the Calculate tab, ready to resume; **Discard**
  removes one you are done with. Hidden entirely when nothing is unfinished.
- **Help topic: "Resuming an interrupted calculation."** Covers how to resume,
  why the offer disappears if you change a setting, which calculation types can
  be resumed, and where checkpoints are stored.
- **Warm-started SCF.** A converged density from an earlier run of the same
  molecule, charge, method and basis is reused as the starting guess, which
  usually cuts several SCF cycles. The geometry does not have to match — a
  density from a nearby geometry is a good guess, which is what a geometry
  optimization relies on internally.
- **Calculation records say how they were measured.** Runs launched from the app
  and runs measured by the calibration tool are now labelled as such, and the
  estimator keeps them apart instead of averaging two populations that measure
  different things. Records also carry a per-stage time breakdown (SCF, Hessian,
  excited-state solve, …), which is the groundwork for stage-aware estimates.
- **`python -m quantui.estimator_eval`** — replays your recorded history through
  the estimator and reports how accurate it would have been, split by
  calculation type. Reports coverage alongside accuracy, so a model that stays
  silent can't look good by refusing to answer.

## [0.6.1] - 2026-08-05

### Fixed

- **Reorganization-energy results now survive History.** Reloading one from
  History showed the calculation's headline numbers as missing — λ, the
  four-point energies and the per-channel breakdown were never written to disk,
  so there was nothing to redisplay. Both the live and History result cards now
  render from the same saved data, so anything visible during a run is visible
  after reloading it.
  **Results saved before this cannot be recovered** — λ requires two geometry
  optimizations and four SCF energies. Those results now say so on the card and
  tell you to re-run, rather than appearing silently incomplete.
- **3-D backgrounds follow the theme** in the new geometry viewers, matching the
  fix applied to the other viewers in 0.6.0.

### Added

- **A Geometries panel** for reorganization energy, on the Analysis tab — which
  previously showed nothing at all for this calculation type. Step through the
  distinct geometries behind the four energies, or overlay two of them with
  arrows marking how each atom moved. The arrows can be scaled ×1–×10 for small
  relaxations; the structures always show true computed positions.
- **Geometry relaxation on the result card** — RMSD and the largest single-atom
  shift between the neutral and ion geometries, which is what λ physically
  measures.
- **Compare λ across saved results.** Selecting several reorganization-energy
  runs in the Compare tab now shows a λ table with a column per channel, for
  screening candidates by how much they reorganize.

### Changed

- The NCShare GPU diagnostic notebook now reports the **CPU affinity mask**
  alongside the core count, and states plainly when CPU timings cannot be
  trusted. A job can be granted 6 cores while still *seeing* 192; whether that
  matters depends on how the scheduler constrains it, and the core count alone
  cannot distinguish the two cases.
- `apptainer/README.md` records the first verified H200 run — driver, compute
  capability, partition, and the measured CPU/GPU crossover.

## [0.6.0] - 2026-08-04

### Added

- **Save orbital isosurfaces as images.** A **Save PNG** button sits under the
  viewer and captures exactly what you are looking at, camera and all. Options
  for the filename, the DPI written into the file (72–600; this sets the
  *print* size, not the pixel count), and a transparent background for dropping
  a figure onto a slide — the transparency applies only to the saved file, so
  the viewer on screen is unchanged.
- **Tunable isosurface resolution** — Coarse/Medium/Fine/Very fine grids,
  remembered between launches.
- **Live isosurface controls**: isovalue, opacity and a colour-scheme dropdown
  (GaussView orange/blue, Avogadro/Jmol red/blue, journal yellow/blue, and
  more). All three update the existing viewer without recomputing anything, so
  the orientation you rotated into is preserved.
- **The isovalue is explained in chemical terms.** Beside the slider, QuantUI
  reports what fraction of the orbital's density the surface actually encloses
  — "encloses 98.4% of the density" — because an amplitude threshold is rarely
  the question anyone has.
- **A Cancel button** for isosurface generation. The underlying PySCF call
  cannot be interrupted, so the computation finishes in the background, but its
  result is discarded and the controls come back immediately.
- **The molecule is shown before you generate anything**, so the panel is never
  empty — and you can orient the structure first, since that view carries into
  the isosurface.

### Changed

- **Orbital isosurfaces are py3Dmol only.** The Plotly renderer is no longer
  selectable for them: it downsamples the volume by construction, and it cannot
  carry the new image export. Plotly remains available for the molecule viewers.
- **Exported animations now match what you saw.** Vibrational modes and
  optimization trajectories both export from py3Dmol — the renderer that drew
  them on screen — instead of being rebuilt in Plotly. Exported files are also
  now complete HTML documents declaring UTF-8, so labels no longer risk
  mojibake when opened from disk.
- **For geometry optimizations, the Orbital Isosurface panel opens by default**
  on the Analysis tab.
- **Isosurface viewers are framed** like the other 3-D viewers, and generating a
  new one dims the current one in place rather than collapsing the panel.

### Fixed

- **3-D backgrounds follow the theme.** The isosurface, vibrational and
  trajectory viewers, and the Analysis-tab molecule viewer, kept their old
  background when switching Light/Dark until something happened to redraw them.
- **The page no longer jumps** when generating an isosurface.
- **Isosurface changes now apply correctly.** Adjusting the isovalue, opacity or
  colours previously layered a new surface on top of the old one, so raising the
  isovalue appeared to do nothing, lowering opacity did nothing, and switching
  palettes made the surface progressively brighter.
- Very large orbital grids are dramatically lighter in the browser — the cube
  data was being embedded three times per render.

## [0.5.2] - 2026-08-03

### Fixed

- **Dark mode now has visible panel edges.** Borders, cards, and dividers were
  effectively invisible in Dark mode — measured at 1.14:1 against the panel
  background, well under the 3:1 that WCAG asks of a UI component. Dark mode is
  a whole-page colour inversion rather than a separate palette, so a light grey
  border inverts to a near-black one on a near-black page and disappears. The
  borders are now a mid-tone that stays visible in *both* modes (3.20:1 light,
  4.30:1 dark). Text contrast was measured too and was never the problem — it
  is 7.24:1 in light mode and slightly *better* inverted — so text colours are
  deliberately unchanged.

### Added

- **Every 3-D viewer now has a border around it**, so it is clear where the
  view ends and where the page can be scrolled without dragging the structure.
  This covers the molecule preview, the results and analysis structure views,
  the geometry-optimization trajectory, the vibrational-mode animation, and the
  classical pre-optimization preview. The frame hugs the plot itself rather
  than spanning the window width, and looks the same whichever rendering
  backend (py3Dmol or plotlymol3d) is active.

## [0.5.1] - 2026-07-31

### Fixed

- **The "Geometry optimization before calculation" checkbox no longer gets
  stuck disabled.** Selecting a seed geometry on Frequency or UV-Vis disables
  that checkbox (a seed is already an optimised geometry, so re-optimising
  first would be redundant) — but switching to a different calculation type
  afterwards used to leave it disabled with no way to re-enable it short of
  going back and clearing the seed selection.
- **The Geometry Opt seed dropdown now updates when you change molecules
  while already on that panel.** It previously only refreshed when you
  switched *into* the Geometry Opt calculation type, so loading a new
  molecule without leaving the panel could leave it listing matches for the
  wrong molecule.

### Changed

- **Internal cleanup:** the Geometry Opt / Frequency / UV-Vis seed-geometry
  dropdowns, refresh buttons, and status notes are now one shared widget group
  instead of three near-identical copies. No user-visible change — the same
  dropdown, filtering, and messages appear on each calculation type as before.

## [0.5.0] - 2026-07-31

First release published to **PyPI** — `pip install quantui`.

### Added

- **Seed geometry for Geometry Optimization runs** — a run can now start from
  the final geometry of a previous optimization instead of the current molecule,
  which makes the standard "optimize at a cheap level of theory, then refine at
  a higher one" workflow a single dropdown choice. Frequency and UV-Vis already
  supported this; Geometry Opt now matches them.
- **A "still working" heartbeat in the live log.** Long silent phases — the
  TD-DFT excited-state solve most of all — could leave the output log unchanged
  for minutes while the calculation was running normally, which reads as a hang.
  The log now reports `… still working — <stage> · <elapsed>` whenever it has
  been quiet for 25 seconds. The saved `pyscf.log` is unaffected; it stays a
  faithful record of the calculation's own output.
- **Basis-set names now show their alternate notation.** `6-31G*` and `6-31G(d)`
  are the same basis set written two ways, and nothing in the UI said so. The
  basis card now notes the equivalent spelling, and the basis-set help topic
  explains the star/parenthesis convention, the `+`/`++` diffuse markers and
  when anions need them, and why Dunning sets (`cc-pVDZ`) have no star at all.
- **The launcher terminal now prints the QuantUI wordmark** instead of two bare
  lines of text.
- **Reorganization energy (Marcus 4-point)** — a new "Reorganization Energy"
  calculation type that computes the internal reorganization energy λ for hole
  (cation) and/or electron (anion) charge transfer. It optimizes the neutral and
  ion geometries and evaluates the four single-point energies of the 4-point
  scheme (λ = [E_ion(R_neutral) − E_ion(R_ion)] + [E_neutral(R_ion) −
  E_neutral(R_neutral)]), reporting λ, its λ₁ (ion) and λ₂ (neutral) relaxation
  components, in eV and kcal/mol. `mode="both"` shares the neutral optimization
  across both channels. Open-shell HF ions are automatically promoted to UHF.
- **One-click "Calc. Reorganization Energy" button** — sets the calculation type
  to Reorganization Energy, defaults the channel to both hole + electron, and
  launches the run in a single click.
- **GPU offload can now be switched off from the UI** — Status tab → Settings →
  "Use GPU when available". The preference persists across launches. This exists
  because GPU offload is not always faster: quantum-chemistry SCF is
  double-precision throughout, and consumer/workstation GPUs gate FP64 to roughly
  1/32–1/64 of their FP32 rate, so offload on such a card can be *slower* than a
  many-core CPU. `QUANTUI_DISABLE_GPU=1` still overrides the setting for scripted
  runs.
- **A warning when a detected GPU is unlikely to help.** `quantui gpu check` and
  the Status tab now flag consumer-class devices with a note that double
  precision is weak on them and offload may be slower than CPU, instead of
  presenting any detected CUDA device as free speed.

### Changed

- **The live calculation log no longer snaps back to the bottom while a run is
  in progress**, so you can scroll up and read earlier output mid-calculation.
  It still follows new output automatically when you are at the bottom. The log
  is now a container QuantUI owns and appends to, rather than a widget that
  rebuilt itself on every line.
- **No pre-optimization preview when the geometry barely moves.** Previously a
  relaxation that changed essentially nothing still opened an animation pane and
  a Keep/Revert choice between two effectively identical structures. Below
  0.05 Å RMS displacement QuantUI now just reports the number and proceeds with
  your geometry.
- **All launcher scripts moved into a `launchers/` folder** to declutter the repo
  root. Behaviour is unchanged — each launcher now resolves the repo root as its
  parent directory. `launchers/launch-app.bat` still finds `quantui.sif` next to
  itself (student download) or in the repo root (dev build). Docs and shortcuts
  that referenced the old top-level paths were updated.
- **Apptainer container now builds on `condaforge/miniforge3` with the `mamba`
  (libmamba) solver.** The previous `continuumio/miniconda3` base used conda's
  classic solver, which could hang for hours resolving the PySCF + RDKit +
  JupyterLab dependency set. The build now installs from conda-forge only
  (`--override-channels`, strict channel priority), which resolves in minutes and
  avoids the base image's implicit `defaults` channel. No change to the shipped
  package set or runtime behavior.

### Fixed

- **The live log and run header render in a fixed-width font again.** A
  system-font rule was applying to the log, which garbled the ASCII wordmark and
  pushed the run header's `Label : value` columns out of alignment.
- **`quantui gpu check` no longer reports a broken CUDA install as "gpu4pyscf not
  installed".** `ModuleNotFoundError` is a subclass of `ImportError`, so catching
  the latter conflated "the package is absent" with "the package is present but
  its CUDA libraries are missing" — the second case was reported as the first,
  sending users back to an install step they had already completed. The two are
  now distinguished, the underlying exception is included in the message (e.g.
  the missing `libnvJitLink.so`), and both cases are logged. The reason string
  now comes from the detection probe itself rather than being re-derived by the
  CLI, so the message can no longer contradict what the run dispatcher decided.
- **Exit is now a two-stage confirmation, so one click can no longer tear down an
  HPC allocation.** Exit shuts the server down by sending `SIGTERM` to the parent
  process. On a laptop that parent is just Voilà, but on a cluster interactive
  session (e.g. NCShare) that parent process *is* the job — so a single stray
  click destroyed the entire allocation. The first click now only arms the
  shutdown: Exit re-labels to "Confirm shutdown" and an inline warning plus a
  Cancel button appear. Only a second click runs the shutdown, which is otherwise
  unchanged. An armed Exit auto-reverts after 15 seconds so a stray click doesn't
  leave the button primed.

## [0.4.1] - 2026-07-16

Bug-fix and hardening release from a full repository audit. No new features and
no breaking changes — it fixes correctness bugs across orbital isosurfaces,
Molden export, the XYZ parser, NMR references, PES scans, and IR intensities,
plus a range of edge-case, packaging, and hygiene issues found by reading the
codebase end to end.

### Added

- Single-atom molecules (e.g. a lone He or Ne atom) are now accepted as XYZ input
  and run end to end — atomic calculations are legitimate targets.

### Changed

- **Python 3.9 is now covered by CI.** The minimum supported version was
  previously claimed but untested; adding it surfaced and fixed real
  geometry-optimization and frequency failures on the older ASE that resolves for
  Python 3.9.
- **Faster `quantui` CLI startup** — the command-line tool no longer imports the
  full notebook/GUI stack (ipywidgets, IPython, the app module) just to tail a
  log.
- **No more logging hijack** — importing `quantui` no longer reconfigures the
  root logger, so it won't override or duplicate logging in a host application or
  notebook.
- **Faster time estimates over long sessions** — the performance log is cached and
  re-read only when it changes, instead of being fully parsed on every UI update.

### Fixed

- **Orbital isosurfaces for charged and open-shell molecules** — the isosurface
  viewer previously errored for every odd-electron system (radicals, and ions
  such as H₃O⁺, NH₄⁺, OH⁻); charge and spin are now carried through to the cube
  generator.
- **Molden export from frequency calculations** — the exported geometry was
  inflated (Bohr coordinates read as Ångström) and internally inconsistent, so it
  rendered wrong in Avogadro/IQmol; frequency geometries are now stored and
  written in the correct units. This also fixes history replay of frequency
  orbital isosurfaces.
- **Export Script and method notes for `wB97X-D`** — "Export Script" failed with
  a "method not supported" error and the educational notes silently disappeared
  for `wB97X-D`; mixed-case method names are now matched correctly.
- **2D structure images from XYZ input** — this path never worked (it always
  returned nothing) and now renders.
- **XYZ files with a blank or comment (`#`) title line** no longer silently drop
  the first atom — a very common file layout was losing an atom with only a
  warning.
- **NMR reference provenance** — chemical shifts no longer silently fall back to
  the B3LYP/6-31G\* TMS constants for other method/basis combinations; the
  reference that was actually applied is now recorded, and the lookup is
  case-insensitive.
- **IR intensities for open-shell / UHF-singlet frequency runs** no longer abort
  the whole IR-intensity step.
- **PES scans** — a failed scan point no longer records a bogus geometry frame or
  poisons the energy/barrier statistics with NaN, and angle/dihedral scans work
  with current ASE releases.
- **UHF calculations** now report a dipole moment and Mulliken charges (these were
  silently skipped).
- **Clearer errors for unsupported input** — elements outside the supported H–Kr
  range, and post-Hartree–Fock methods (MP2 / CCSD / CCSD(T)) requested for
  calculation types that don't support them, now fail with an explicit message
  instead of a misleading one or an uninformative crash.
- **Windows exports** no longer fail when the basis-set name contains `*` (e.g.
  `6-31G*`) — export filenames are sanitized.
- **Method-notes panel** renders bold text correctly instead of leaking literal
  `**` markers.
- **Robustness** — the event log is no longer subject to lost entries under
  concurrent writes; a thumbnail-save failure no longer aborts saving a result;
  PubChem availability checks respect the configured throttle and timeout; and
  results carrying NumPy scalar values now save correctly.

## [0.4.0] - 2026-06-18

Interactive-visualization and offline-readiness release. Adds molecular-orbital
isosurfaces and an interactive pre-optimization preview, reworks the 3D viewers
to preserve camera orientation across frames, makes all 3D rendering work
offline, and substantially speeds up startup.

### Added

- **Molecular-orbital isosurfaces (py3Dmol)** — interactive HOMO / LUMO / MO
  isosurface viewer rendered with py3Dmol; the downsampled Plotly path remains a
  fallback.
- **Interactive classical pre-optimization** — a **Preview** button relaxes the
  geometry with a fast bonded force field (RDKit MMFF94 → UFF) and animates the
  relaxation in place; **Keep this geometry** adopts it as the active structure
  or **Revert** discards it. Stepper controls (play/pause, prev/next, scrub
  slider, and an input ⇄ relaxed flip) let you compare geometries, and the
  captured trajectory is sampled at even RMSD spacing for smooth playback.
- **Cancel button** — stop a running calculation cooperatively at the next SCF
  cycle / optimization step.
- **Live vibrational-animation framerate** — the Vib fps setting updates the
  running animation immediately.

### Changed

- **Single persistent 3D viewers** — the trajectory and vibrational-mode viewers
  now load all frames into one py3Dmol viewer and switch frames/modes
  client-side, so the camera (rotation/zoom) is preserved across steps and modes
  and there is no per-frame rebuild or flicker.
- **Pre-optimization is Preview-only** — the silent "classical pre-optimize"
  checkbox is gone; pre-optimization happens only through the transparent
  Preview → Keep/Revert flow, so nothing relaxes the geometry invisibly. (The
  separate QM "geometry optimization before calculation" option is unchanged.)
- **Offline-first 3D rendering** — 3Dmol.js is vendored and loaded per-view from
  a local `data:` URI instead of a CDN, so every 3D view works with no network
  (the build fails if the vendored asset is missing). Native launchers tolerate
  offline `pip install`.
- **Faster startup** — GPU detection and History/Compare population are deferred
  off the synchronous construction path, so the UI paints in ~1 s instead of
  ~15 s; the GPU status badge and dropdowns fill in shortly after.
- **Clear** of the live calculation log is disabled while a calculation runs.

### Fixed

- **GPU-offloaded result extraction** — HOMO–LUMO gap, dipole moment, and
  Mulliken charges are now reported for GPU runs. CuPy arrays are copied to host
  before extraction, and Mulliken falls back to the CPU object (gpu4pyscf does
  not implement population analysis on the GPU).
- **Vibrational animation glitchiness** — stacked animation loops (a new loop
  started on every mode switch) made playback jittery and too fast and ignored
  the framerate setting; exactly one loop now runs.
- **Cancel status** no longer sticks on "Cancelling…" after a calculation is
  cancelled.
- **Stale run status** — "Pre-optimized geometry accepted." is cleared when a new
  molecule is loaded or a preview is reverted.
- Structure provenance is reported and the input viewer is persisted across
  reloads.

## [0.3.0] - 2026-06-11

Structure-sourcing release. Repairs the external-database structure search and
replaces the 20-entry inline molecule list with an indexed, three-tier bundled
library, alongside visualization and result-card fixes.

### Added

- **External structure search** — resolve molecules by name, PubChem CID, InChI,
  InChIKey, SMILES, or CAS number. Input is routed by type: SMILES/InChI resolve
  locally via RDKit with no network, while names and identifiers query PubChem
  through a hardened client (URL-encoding, request throttling, bounded retry on
  server throttling). NCI CACTUS acts as a fallback resolver, and an offline
  bundled-library fallback keeps the search usable without a network connection.
- **Disambiguation pick-list** — an ambiguous query (e.g. a name with several
  PubChem matches) presents a selectable list instead of silently choosing the
  first hit.
- **Three-tier bundled molecule library** — 20 presets, 156 curated named
  molecules, and ~1,900 bulk QM9 structures (CC0), held in an indexed,
  lazily-loaded package-data store. Bulk entries are reachable via search.
- **Library browse/search tab** — category filter plus name/formula search,
  replacing the flat preset dropdown.

### Changed

- The molecule preset list moved from an inline `config.py` dictionary into the
  indexed library store; `config.MOLECULE_LIBRARY` is preserved as a
  backward-compatible accessor.
- Frequency / UV-Vis seed-geometry dropdown entries are now labeled as optimized
  geometries so their source is explicit.

### Fixed

- **Orbital isosurface could exhaust browser memory** — the full volumetric grid
  was serialized into the Plotly figure. The rendered surface is now downsampled
  to a bounded point count (the saved cube file remains full-resolution).
- **CCSD / MP2 result cards** now show the HF reference and correlation-energy
  breakdown (plus the (T) triples correction for CCSD(T)); these fields are also
  persisted with saved results.
- The live calculation log no longer jumps to the top while streaming output.
- Structures fetched as 2D records are re-embedded in 3D, and salt/counterion
  fragments are separated so bond perception does not misread them.

## [0.2.0] - 2026-05-22

First substantial release after `v0.1.0`. The codebase moved from a single
monolithic `app.py` to a modular package, added six PySCF-backed calculation
types end-to-end, introduced a results-persistence layer with history replay,
and shipped a complete visualization stack (3D viewer with selectable backend,
trajectory animation, IR/UV-Vis/PES plots, orbital isosurfaces, vibrational
mode animation with caching). UI runs as a Voilà app suitable for classroom
deployment.

### Added

#### Calculations

- **Geometry optimization** (`optimizer.py`) — ASE-BFGS driver around a custom
  PySCF calculator; per-step trajectory persisted.
- **Vibrational frequency analysis** (`freq_calc.py`) — Hessian via
  `pyscf.hessian`, ZPVE, thermochemistry (H/S/G at 298 K), IR intensities via
  `pyscf.prop.infrared` or a numerical-derivative fallback for compatibility
  across PySCF versions.
- **TD-DFT UV-Vis** (`tddft_calc.py`) — excitation energies, oscillator
  strengths, wavelengths; full spectrum plot in the Analysis tab.
- **NMR shielding** (`nmr_calc.py`) — GIAO shielding via `pyscf.nmr` (core
  preferred over `pyscf-properties` to dodge a known upstream bug); ¹H/¹³C
  chemical shifts relative to TMS.
- **1D PES scan** (`pes_scan.py`) — bond / angle / dihedral; energy profile +
  per-step geometry animation.
- **PCM implicit solvent** — Water, Ethanol, THF, DMSO, Acetonitrile via a
  single checkbox in the Calculate tab.
- **MP2** post-HF method support.

#### Analysis & visualization

- **Analysis tab with 8 always-in-DOM panels** (Energies, Trajectory,
  Vibrational, IR Spectrum, PES Scan, Isosurface, UV-Vis, NMR) wired through
  a `_PANEL_REGISTRY` so live runs and history replay share one code path.
- **IR spectrum chart** (`ir_plot.py`) — stick plot + Lorentzian-broadened
  curve; broadening toggle and FWHM slider.
- **UV-Vis spectrum plot** — Plotly chart with wavelength/energy axes.
- **Orbital visualization** (`orbital_visualization.py`) — energy-level
  diagram (matplotlib → Plotly HTML) and cube-file isosurface viewer with
  HOMO-1/HOMO/LUMO/LUMO+1 toggle.
- **Trajectory animation** — atomic Output-children swap to avoid
  Voilà-deferred-display blank frames; py3Dmol-only render path with
  prev/next arrow navigation.
- **Vibrational mode animation** — py3Dmol multi-frame XYZ renderer with
  amplitude scaling, prev/next mode nav, and dropdown skipping near-zero
  modes.
- **3D visualization backend router** (`viz_backend_router.py`) — pure
  function that picks py3Dmol or plotlymol3d per `VizTask` based on user
  preference and runtime availability; immutable `Decision` carries chosen
  backend, fallback, and reason.
- **Lifecycle telemetry** — `_viz_render_event` context manager emits
  `viz_render_start` / `viz_render_done` / `viz_render_error` JSONL events
  with backend, task, `elapsed_ms`, and extras at every render dispatch.
- **Side-by-side Compare tab** — pick any two saved calculations and view a
  diff table.

#### Persistence & logging

- **Results storage** (`results_storage.py`) — every run is saved to a
  timestamped directory containing `result.json` (schema v2, additive-only),
  `pyscf.log`, optional `trajectory.json` / `orbitals.npz` / `thumbnail.png`.
- **History tab** — browse and replay saved calculations after a kernel
  restart; replay path is identical to live-run analysis activation.
- **Performance log** (`calc_log.py`) — `perf_log.jsonl` per converged run +
  `event_log.jsonl` for startup/calc/error events; 7-day auto-prune.
- **Time estimator** — 4-strategy priority chain (N_basis-normalised → cross-method
  electron-count) populates "Estimated time" before each run.
- **Benchmark suite** (`benchmarks.py`) — one-click calibration suite to
  populate the time-estimator history with real machine data.
- **Issue tracker** (`issue_tracker.py`) — in-app bug-report UI writing to a
  local `issues.db`.
- **Persistent user settings** (`user_settings.py`) — stored at
  `~/.quantui/settings.json` (override via `QUANTUI_SETTINGS_PATH`). Schema
  is section-based for additive growth, with atomic writes and graceful
  fallback to defaults on corruption.
- **Vibrational-animation disk cache** (`vib_cache.py`) — per-result-dir
  `vib_frames/` of pre-rendered py3Dmol HTML keyed by
  `(mode, n_frames, amplitude, renderer, fps)`. Mode switches on repeat
  visits and history replay are instant.
- **Vib FPS user preference** — `viz.vib_framerate_fps` exposed as an
  IntSlider in the Status tab (clamped 1–120, default 10); included in the
  vib cache key so changing FPS invalidates cleanly.

#### UI

- **Modular UI package** — `app.py` (orchestration) plus `app_analysis.py`,
  `app_builders.py`, `app_exports.py`, `app_formatters.py`, `app_history.py`,
  `app_runflow.py`, `app_visualization.py`.
- **Seven-tab layout** — Calculate, Results, Analysis, History, Compare, Log,
  Status — with a floating Help overlay (not a tab).
- **Light / Dark theme selector** — dark by default on startup.
- **Status tab** — environment info, performance-history accordion (two-step
  reset), default-3D-backend toggle, vib-FPS slider.
- **Files tab + activity indicator** for browsing saved results.
- **Plot export UI** — save IR, UV-Vis, PES, orbital diagram plots as HTML.
- **Scroll guard** for the run output area to keep long PySCF logs from
  jumping the page.
- **Welcome header**, completion banner, structured log header/footer.
- **Compare-tab Copy-path button** (replaced a broken Open-folder action).
- **Result directory label + log accordion** showing inline `pyscf.log`.
- **Structure exports** — XYZ, MOL/SDF, PDB, plus a standalone runnable `.py`
  script export.

#### Tooling & dev

- **Test suite grew from a handful to 1004 passed / 97 skipped** (Windows
  `quantui-win` env baseline; the 97 skips are PySCF-gated Linux-only tests).
- New analysis-history end-to-end tests for every calc type
  (`test_sp_analysis_history.py`, `test_geo_opt_analysis_history.py`,
  `test_freq_analysis_history.py`, `test_tddft_analysis_history.py`,
  `test_nmr_analysis_history.py`, `test_pes_scan_analysis_history.py`).
- `test_code_quality.py` enforces:
  - No `include_plotlyjs="cdn"` anywhere (fails silently in offline Voilà).
  - No bare `except: pass` blocks.
- `test_viz_backend_router.py` + `test_viz_backend_sync.py` — full
  task × preference × availability matrix and Calculate/Analysis toggle sync.
- `test_vib_cache.py`, `test_vib_py3dmol_render.py`,
  `test_viz_render_telemetry.py` — vib animation + telemetry coverage.
- `_layout(...)` helper sanitises `widgets.Layout` kwargs to eliminate a
  4808 → 13 traitlets warning regression.
- `_safe_cb` wrapper around every `.observe()` callback so exceptions surface
  in the Log tab instead of disappearing into the Voilà kernel console.
- Kernel `io_loop` is cached at startup; thread-spawned callbacks are queued
  onto the main thread to avoid `RuntimeError: no current event loop`.
- Native launchers: `launch-native.bat` (Windows / WSL) and
  `launch-native.command` (macOS / Linux) — double-clickable, port `8867`,
  stamp-based editable-install skip, browser auto-open. README documents
  pinning each to the Start menu / Dock as a real app.
- Native JupyterLab launcher (`launch-native-jupyter.bat`) and Apptainer
  launcher improvements.

#### Docs

- `.github/copilot-instructions.md` — canonical AI-assistant context (now
  the single source of truth for any AI assistant working on this repo).
- `CLAUDE.md` — Claude-specific session/workflow context (git-ignored).
- Site favicons (ICO + SVG) for the GitHub Pages docs site.

### Changed

- **Visualization is py3Dmol-first.** `plotlymol3d` remains an optional
  fallback for non-trajectory tasks; trajectory rendering is hard-wired to
  py3Dmol to avoid Plotly/RequireJS flicker.
- **Plotly figures are rendered via `plotly.io.to_html(..., include_plotlyjs="require")`**
  inside `widgets.HTML`, not `display(fig)`, so threaded renders work and
  offline Voilà loads correctly.
- **`pyscf` is now an optional extra** (`pip install quantui[pyscf]`); the
  package imports cleanly on Windows with PySCF unavailable.
- Repo renamed from `QuantUI-local` to `QuantUI`.

### Fixed

- **Trajectory accordion blank on first expand** — switched `traj_output`
  from `Output` to `VBox` and use atomic children-swap so deferred
  widget-display is no longer a blank-frame risk.
- **Vib mode races on rapid switching** — render-token guard (`_vib_render_token`)
  causes stale background renders to bail rather than overwriting newer
  output.
- **Camera state lost on mode switch** — JS hook caches the active
  `$3Dmol.GLViewer` state across atomic HTML swaps; reset only on a
  genuinely new frequency result.
- **PySCF API drift** — robust handling for v2 NMR / thermo API and the
  `pyscf.prop.infrared` rename; both `Infrared.kernel()` and the older
  IR API are supported.
- **Result-dir name collisions** — timestamps now include microseconds; same
  formula + method + basis no longer overwrite each other.
- **IR x-axis** — corrected wavenumber axis on the IR Plotly figure.
- **Plotly figures invisible after accordion show** — figures are re-rendered
  on accordion expand to handle RequireJS / display-deferral edge cases.

### Removed

- `visualization.py` (PlotlyMol fallback) — replaced by the router-backed
  `visualization_py3dmol.py` path.
- All SLURM-era infrastructure already removed during the downstream port:
  `job_manager.py`, `storage.py`, `slurm_errors.py`, SLURM config templates.

## [0.1.0] - 2026

Initial public scaffolding of the QuantUI package: `quantui` package with
`molecule.py`, `pubchem.py`, `config.py`, `visualization_py3dmol.py`,
`calculator.py`, basic notebook launcher, Apptainer container definition,
MIT license, and project metadata.

[Unreleased]: https://github.com/The-Schultz-Lab/QuantUI/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/The-Schultz-Lab/QuantUI/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/The-Schultz-Lab/QuantUI/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/The-Schultz-Lab/QuantUI/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/The-Schultz-Lab/QuantUI/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/The-Schultz-Lab/QuantUI/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/The-Schultz-Lab/QuantUI/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/The-Schultz-Lab/QuantUI/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/The-Schultz-Lab/QuantUI/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/The-Schultz-Lab/QuantUI/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/The-Schultz-Lab/QuantUI/releases/tag/v0.1.0
