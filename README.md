# QuantUI

[![CI](https://github.com/The-Schultz-Lab/QuantUI/actions/workflows/ci.yml/badge.svg)](https://github.com/The-Schultz-Lab/QuantUI/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://the-schultz-lab.github.io/QuantUI/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org)

A powerful open-source frontend for DFT and post-HF quantum chemistry.
QuantUI puts [PySCF](https://pyscf.org) behind an interactive Jupyter/Voilà
UI so you can build molecules, run calculations locally, and visualize the
results — no cluster account, no SLURM, no queueing.

Developed by the
[Schultz Lab, North Carolina Central University](https://github.com/The-Schultz-Lab)
as an open alternative to closed-source GUI workflows. Equally suitable for
research and classroom use.

---

## What it does

- **Molecule input** — paste XYZ coordinates, browse an indexed three-tier
  bundled library (20 presets + 156 curated molecules + ~1,900 QM9 structures,
  searchable by name/formula), or run a structure search by name, SMILES,
  InChI, PubChem CID, InChIKey, or CAS number (PubChem → NCI CACTUS → offline
  bundled-library fallback; SMILES/InChI resolve locally with no network)
- **Offline-first** — runs with no internet: the bundled molecule library and
  the 3D viewer's JavaScript (3Dmol.js) are vendored, so structure lookup and
  every 3D view work in an air-gapped classroom. (Network is used only for the
  optional live PubChem/CACTUS search.)
- **3D visualization** — interactive py3Dmol viewer (py3Dmol-first; optional
  plotlymol3d fallback for non-trajectory tasks). A capability-aware backend
  router picks the right renderer per task, and a Status-tab toggle persists
  your default-backend preference between sessions
- **In-session calculations** — RHF, UHF, 9 DFT functionals, MP2, CCSD,
  CCSD(T), NMR shielding, TD-DFT UV-Vis, and 1D PES scans via PySCF, running
  in your Python kernel (no batch submission)
- **Implicit solvent** — PCM solvation (Water, Ethanol, THF, DMSO,
  Acetonitrile) via a single checkbox
- **Rich results** — total energy, HOMO-LUMO gap, Mulliken charges, dipole
  moment, thermochemistry (H, S, G at 298 K), IR spectrum chart (stick and
  Lorentzian-broadened), ¹H/¹³C NMR chemical shifts, orbital energy-level
  diagram, HOMO/LUMO isosurface (cube-file rendering with toggle for HOMO-1,
  HOMO, LUMO, LUMO+1), and a side-by-side comparison table for multiple
  calculations
- **Geometry optimization** — BFGS optimizer with step-by-step trajectory
  animation; vibrational frequency analysis with animated normal modes,
  user-tunable playback FPS, and a per-result-directory disk cache so mode
  switches on repeat visits and history replay are instant
- **Results persistence** — every calculation is saved automatically to a
  timestamped directory; a built-in browser lets you reload past results
  after a kernel restart; the full `pyscf.log` is shown inline
- **Structure exports** — download XYZ, MOL/SDF, or PDB files alongside the
  saved results; script export for a standalone `.py` file
- **Plot export** — save IR, UV-Vis, PES, and orbital diagrams as standalone
  HTML
- **Optional GPU acceleration** — when [gpu4pyscf](https://github.com/pyscf/gpu4pyscf)
  and a CUDA-capable NVIDIA GPU are present, SCF calculations auto-offload
  via `mf.to_gpu()` (RHF / UHF / RKS / UKS supported; CCSD(T) stays on CPU).
  The Status tab + every result card show which compute device was used.
  Set `QUANTUI_DISABLE_GPU=1` to force CPU even when the GPU is available.
- **Timing calibration** — one-click benchmark suite populates the time
  estimator with real machine data so predictions are accurate from the first run
- **Voilà app mode** — serve the notebook as a polished widget-only UI (no
  code visible), with Light/Dark themes, a dedicated output log, and an
  in-app bug-report form

---

## Platform requirements

| Platform | Works? | Notes |
| --- | --- | --- |
| Linux / macOS | Full | PySCF installs natively |
| WSL (Windows) | Full | Use an Ubuntu WSL environment |
| Windows (native) | Partial | All UI and visualization features work; PySCF calculations require the Apptainer container |

### Windows users: Apptainer container

PySCF does not install on Windows natively. The
[`apptainer/quantui.def`](apptainer/quantui.def) container bundles
the complete environment and runs anywhere Apptainer/Singularity is available.
See [`apptainer/README.md`](apptainer/README.md) for build and run instructions.

---

## Installation

### Option A — conda (recommended for Linux/macOS/WSL)

```bash
# Create a dedicated environment
conda create -n quantui python=3.11
conda activate quantui

# Install with PySCF and ASE
pip install -e ".[pyscf,ase,app]"
```

### Option B — pip only

```bash
python -m pip install quantui[pyscf,ase,app]
```

### Option C — Apptainer container (Windows / reproducible deployment)

See [apptainer/README.md](apptainer/README.md).

### Optional: GPU acceleration (NVIDIA + Linux / WSL)

If you have an NVIDIA GPU, QuantUI can offload SCF calculations to it
through [gpu4pyscf](https://github.com/pyscf/gpu4pyscf). This is **fully
optional** — without these packages QuantUI runs on CPU exactly as
before, and you can re-disable GPU at any time with
`export QUANTUI_DISABLE_GPU=1`.

**Step 1 — check your CUDA driver version:**

```bash
nvidia-smi   # "CUDA Version: 13.x" or "CUDA Version: 12.x" in the top-right
```

> The `CUDA Version` field reports your **driver's** maximum supported
> runtime. You do **not** need to install the CUDA Toolkit — the wheels
> below bundle their own runtime libraries.

**Step 2 — install the CUDA-suffixed wheels matching your driver:**

```bash
# CUDA 13.x driver
pip install gpu4pyscf-cuda13x cupy-cuda13x cutensor-cu13

# CUDA 12.x driver
pip install gpu4pyscf-cuda12x cupy-cuda12x cutensor-cu12
```

> ⚠ **Do not** `pip install gpu4pyscf` or `pip install cupy` (without a
> CUDA suffix). Those are source distributions that try to compile
> against your local CUDA toolkit and will fail with
> `FileNotFoundError: 'nvcc'` on any machine without the full toolkit
> installed. The CUDA-suffixed wheels (`-cuda12x`, `-cuda13x`) are
> prebuilt binaries — no `nvcc`, no compilation, no toolkit required.

**Step 3 — verify the install:**

```bash
python -c "import gpu4pyscf, cupy; print('GPUs:', cupy.cuda.runtime.getDeviceCount())"
```

Should print `GPUs: 1` (or more). Once verified, launch QuantUI as usual
— the Status tab will show "GPU offload: active (NVIDIA {device-name})"
and result cards will display the compute device.

**Method coverage** (per the gpu4pyscf docs):

| Method | GPU offload |
| --- | --- |
| RHF, UHF, RKS, UKS (any DFT functional), TD-DFT | Yes |
| MP2, CCSD | Experimental on GPU (auto-offload) |
| CCSD(T) | CPU only (gpu4pyscf doesn't support GPU triples; QuantUI's dispatcher detects this and skips) |

Whenever gpu4pyscf can't offload a particular call, QuantUI falls back
to CPU automatically and the result card reflects which device ran.

---

## Quick start

```bash
# Activate your environment
conda activate quantui

# JupyterLab (full IDE — shows code)
jupyter lab notebooks/molecule_computations.ipynb

# Voilà app mode (widget-only UI — code hidden)
voila notebooks/molecule_computations.ipynb
```

Open the notebook, pick a molecule, choose a method and basis set, and click
**Run Calculation**. Results appear directly in the notebook.

---

## Launching QuantUI as an app

For the smoothest day-to-day experience, QuantUI ships two double-clickable
launchers that activate the right conda environment, install the editable
package on first run (and only re-install when `pyproject.toml` actually
changes), clear any stale bytecode, start Voilà on port `8867`, and open the
app in your default browser. Edits to `quantui/*.py` are picked up live with
no rebuild.

| Platform | File | Action |
| --- | --- | --- |
| Windows | [`launch-native.bat`](launch-native.bat) | Activates the `quantui` conda env inside WSL Ubuntu, runs Voilà, and opens `http://localhost:8867` |
| macOS / Linux | [`launch-native.command`](launch-native.command) | Activates the local `quantui` conda env directly (no WSL needed) and does the same |

Both launchers reuse port `8867`, so you can keep the same browser tab pinned
across platforms.

### Windows — pin to the Start menu

1. Right-click [`launch-native.bat`](launch-native.bat) in File Explorer
   → **Send to** → **Desktop (create shortcut)**.
2. Rename the shortcut to something friendly like `QuantUI`.
3. *(Optional)* Right-click the shortcut → **Properties** → **Change Icon...**
   and point at `docs\logo.ico` for a proper app icon.
4. Move the shortcut into your Start-menu folder so it appears with normal
   apps. Either:
   - press `Win+R`, paste
     `%APPDATA%\Microsoft\Windows\Start Menu\Programs`, and drop the
     shortcut there *(per-user — recommended)*; or
   - paste `%ProgramData%\Microsoft\Windows\Start Menu\Programs` for an
     all-users install.
5. Open the Start menu, find **QuantUI**, right-click it, and choose
   **Pin to Start** (or **Pin to taskbar**).

You now launch QuantUI like any other Windows app — one click and Voilà opens
in your browser.

### macOS — pin to the Dock / Launchpad

**Quickest:** double-click [`launch-native.command`](launch-native.command)
from Finder. macOS will open Terminal, run the script, and pop the app open
in your browser. The first launch is gated by Gatekeeper: right-click the
file → **Open** → **Open** to clear it (one time only).

**App-like experience (recommended):** wrap the launcher in a tiny Automator
application so it lives in Launchpad and pins to the Dock.

1. Open **Automator** (Spotlight → "Automator") → **New Document** →
   **Application**.
2. In the actions library on the left, find **Run Shell Script** and drag it
   into the workflow pane on the right.
3. Set **Shell** to `/bin/bash` and **Pass input** to **as arguments**, then
   replace the script body with the single line below (adjust the path if
   your clone lives elsewhere):

   ```bash
   "$HOME/path/to/QuantUI/launch-native.command"
   ```

4. **File → Save** → name it `QuantUI` → save into `/Applications`.
5. *(Optional)* Set a custom icon: in Finder, open `docs/logo.svg` in
   **Preview**, **Edit → Select All → Copy**, then in Finder select the
   new `QuantUI.app`, **File → Get Info**, click the small icon in the
   top-left of the Info window, and **Edit → Paste**.
6. Open Launchpad, find **QuantUI**, drag it into the Dock to pin it.

You now have a real `.app` you can launch from Spotlight, Launchpad, or the
Dock — it just runs the `.command` script under the hood, so any
`quantui/*.py` edits take effect immediately on the next launch.

> **Linux users:** the same `launch-native.command` script works from a
> terminal — `./launch-native.command`. To wire it into your desktop
> environment as a pinned app, create a `.desktop` entry pointing at the
> script.

---

## Command-line toolkit

QuantUI ships a small CLI for inspecting state and generating reports
from outside the notebook — useful for verifying GPU offload before a
long run, tailing the event log, and building a usage / speedup
dashboard. After installation:

```bash
quantui log tail -n 50        # last 50 events from event_log.jsonl
quantui gpu check             # is GPU offload available right now?
quantui analytics build --open  # build dashboard.html + open in browser
```

Full reference with all flags and examples: [docs/CLI.md](docs/CLI.md).

---

## Using QuantUI results in other tools

QuantUI's M-EXPORT milestone writes portable companion files alongside
every result so you can hand-off to Avogadro, IQmol, Jmol, VMD, ASE-GUI,
or any spreadsheet without screen-scraping. The quick reference:

| Goal | QuantUI file | Tool |
| --- | --- | --- |
| MOs in 3D, vibrations | `result.molden` | Avogadro 2, IQmol, Jmol |
| Geometry-opt / PES replay | `trajectory.xyz` or `.traj` | VMD, Avogadro, ASE-GUI |
| Orbital isosurface | `isosurfaces/<orb>.cube` | Avogadro, VMD, ChimeraX |
| Spectrum data in Excel | `*_data_*.csv` | Excel, LibreOffice, pandas |
| Share whole result | `<result>.zip` (Export bundle) | Any unzip tool |

Full per-tool walkthrough with troubleshooting: [docs/IMPORTING-INTO-AVOGADRO.md](docs/IMPORTING-INTO-AVOGADRO.md).

---

## Tutorials

Five step-by-step notebooks in [`notebooks/tutorials/`](notebooks/tutorials/):

| Notebook | Topic |
| --- | --- |
| [01_first_calculation.ipynb](notebooks/tutorials/01_first_calculation.ipynb) | Your first RHF calculation |
| [02_basis_set_study.ipynb](notebooks/tutorials/02_basis_set_study.ipynb) | Comparing STO-3G, 6-31G, cc-pVDZ |
| [03_multiplicity_radicals.ipynb](notebooks/tutorials/03_multiplicity_radicals.ipynb) | Open-shell molecules and UHF |
| [04_charged_species.ipynb](notebooks/tutorials/04_charged_species.ipynb) | Ions and charged systems |
| [05_comparing_results.ipynb](notebooks/tutorials/05_comparing_results.ipynb) | Side-by-side result analysis |

---

## Supported calculations

### Methods

| Method | Type | Best for |
| --- | --- | --- |
| RHF | Hartree-Fock | Closed-shell molecules; baseline reference |
| UHF | Hartree-Fock | Radicals and open-shell systems |
| B3LYP | DFT hybrid | General organic chemistry (default DFT choice) |
| PBE | DFT GGA | Large molecules; metals; when speed matters |
| PBE0 | DFT hybrid | Charge-transfer, band gaps |
| M06-2X | DFT meta-hybrid | Thermochemistry, barrier heights |
| wB97X-D | DFT range-sep. + D3 | Non-covalent interactions, excited states |
| CAM-B3LYP | DFT range-sep. | Charge-transfer UV-Vis, Rydberg states |
| M06-L | DFT local meta-GGA | Large molecules; transition metals |
| HSE06 | DFT screened hybrid | Band gaps, large molecules |
| PBE-D3 | DFT GGA + dispersion | Van der Waals complexes, stacking |
| MP2 | Post-HF | Accurate energetics for small molecules (O(N⁵)) |
| CCSD | Post-HF coupled cluster | High-accuracy small-molecule energies (O(N⁶)) |
| CCSD(T) | Post-HF coupled cluster | Benchmark "gold standard" energies (O(N⁷); CPU only) |

### Calculation types

| Type | Output |
| --- | --- |
| Single Point | Energy, HOMO-LUMO gap, Mulliken charges, dipole moment |
| Geometry Opt | Optimised structure, trajectory animation |
| Frequency | Vibrational frequencies, ZPVE, IR intensities, thermochemistry (H/S/G at 298 K), animated normal modes, IR spectrum chart (stick / Lorentzian broadened) |
| UV-Vis (TD-DFT) | Excitation energies, oscillator strengths, UV-Vis spectrum plot |
| NMR Shielding | ¹H and ¹³C chemical shifts relative to TMS via GIAO; tabulated by element |
| PES Scan | 1D potential energy surface along a bond, angle, or dihedral; energy profile chart; geometry animation at each scan point |

### Basis sets

STO-3G (fast, good for learning) → 3-21G → 6-31G / 6-31G\* / 6-31G\*\* →
cc-pVDZ / cc-pVTZ → def2-SVP / def2-TZVP

---

## Running tests

```bash
pip install -e ".[dev]"

# All tests (Linux/macOS — PySCF available)
pytest -m "not network"

# Skip PySCF-dependent tests (Windows without container)
pytest -m "not network" \
  --ignore=tests/test_session_calc.py \
  --ignore=tests/test_optimizer.py \
  --ignore=tests/test_preopt.py
```

---

## Project structure

```text
quantui/                  Main package
  app.py                  QuantUIApp — widget orchestration, run dispatch
  app_analysis.py         Analysis-tab panel registry + _pop_* methods
  app_builders.py         _build_* widget construction
  app_runflow.py          _do_run + run/orchestration UI handlers
  app_visualization.py    Trajectory / vib / IR / orbital / PES rendering
  app_history.py          History tab loaders, replay context
  app_formatters.py       Result-card and text formatters
  app_exports.py          Export handlers (XYZ/MOL/PDB/script)
  molecule.py             Molecule input and validation
  session_calc.py         In-session PySCF runner (RHF/UHF/DFT/MP2/PCM)
  freq_calc.py            Vibrational frequency + thermochemistry
  ir_plot.py              IR spectrum chart (stick / Lorentzian broadened)
  tddft_calc.py           TD-DFT UV-Vis excited-state calculations
  nmr_calc.py             NMR shielding + ¹H/¹³C chemical shifts
  pes_scan.py             1D potential energy surface scan
  optimizer.py            QM geometry optimization with trajectory
  visualization_py3dmol.py  3D viewer (py3Dmol-first; plotlymol fallback)
  viz_backend_router.py   Capability-aware backend router (pure function)
  viz_assets.py           Offline-safe 3Dmol.js loading (vendored, no CDN)
  user_settings.py        Persistent user preferences (~/.quantui/settings.json)
  vib_cache.py            On-disk cache of rendered vib-mode HTML
  orbital_visualization.py  Orbital energy diagrams + cube-file viewer
  pubchem.py              Structure search client (PubChem + RDKit)
  cactus.py               NCI CACTUS resolver (fallback structure source)
  structure_providers.py  Unified resolver chain with offline fallback
  molecule_library.py     Indexed 3-tier bundled molecule library
  comparison.py           Side-by-side result tables
  results_storage.py      Timestamped result persistence (schema v2)
  calc_log.py             Performance + event logging, time estimation
  issue_tracker.py        In-app bug-report DB
  benchmarks.py           Timing calibration benchmark suite
  config.py               Methods, basis sets, solvent/NMR options, presets
  ase_bridge.py           ASE structure I/O
  preopt.py               RDKit MMFF94/UFF force-field pre-optimization
  data/                   Bundled library (SQLite + manifests) + vendored 3Dmol.js
notebooks/
  molecule_computations.ipynb   Main user-facing interface (3-cell launcher)
  tutorials/                    Step-by-step guided notebooks (01–05)
tests/                    pytest test suite (~1500 tests; run in parallel via pytest-xdist)
apptainer/                Container definition for reproducible deployment
local-setup/              Conda environment definition
pyproject.toml            Package metadata and tool config
CHANGELOG.md              Release history (Keep a Changelog format)
```

---

## License

[MIT](LICENSE) — Copyright 2026 The Schultz Lab, North Carolina Central University
