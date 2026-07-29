# QuantUI — Apptainer Container

The Apptainer container packages Python, PySCF, RDKit, ASE, py3Dmol, and Voilà
into a single portable `.sif` file. It is the **recommended path for Windows
users** (via WSL) and for anyone who wants a zero-installation experience —
students copy one file and run it.

**Runs fully offline.** The container bundles everything it needs — the
3-tier molecule library and the 3D viewer's JavaScript (3Dmol.js) are vendored,
so structure lookup and every 3D view (molecules, trajectories, vibrations,
orbital isosurfaces) work with **no internet connection**. Network is only ever
used for the optional live PubChem/CACTUS structure search. Ideal for an
air-gapped or restricted-network classroom.

---

## Contents

| File | Purpose |
| --- | --- |
| `quantui.def` | Container definition file (the "recipe") |
| `build.sh` | Build script with clean/test/fakeroot options |
| `README.md` (this file) | Build, run, and distribution guide |

The compiled `.sif` image is **not** committed to git — it is too large (~4–5 GB).
Build it locally (see below) or download the latest release asset from the
[GitHub Releases page](https://github.com/The-Schultz-Lab/QuantUI/releases).

---

## Getting the container

### Option A — Download a pre-built release (easiest)

1. Go to [Releases](https://github.com/The-Schultz-Lab/QuantUI/releases)
2. Download `quantui.sif` from the latest release
3. Run it directly — no build step needed

### Option B — Build from source

Use the provided build script — see the [Build section](#building-from-source) below.

---

## Prerequisites (for building locally)

- **Apptainer ≥ 1.0** on Linux, macOS, or WSL
- ~6 GB free disk space (build scratch + final image)
- Internet access during build (packages are downloaded from conda-forge and PyPI)

### Install Apptainer on Ubuntu / WSL

```bash
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt-get update
sudo apt-get install -y apptainer
```

If `add-apt-repository` is not available:

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt-get update
sudo apt-get install -y apptainer
```

For other platforms see the
[official Apptainer install docs](https://apptainer.org/docs/admin/main/installation.html).

---

## Building from source

Use the provided `build.sh` script. It handles the correct working directory,
optional clean builds, and post-build testing.

```bash
# From the repo root:
cd /path/to/QuantUI

# Standard build
bash apptainer/build.sh

# Remove old .sif first, then rebuild (recommended after code changes)
bash apptainer/build.sh --clean

# Build + run container tests immediately after
bash apptainer/build.sh --clean --test

# On HPC systems without root (uses --fakeroot)
bash apptainer/build.sh --fakeroot
```

Build time: **~20–40 minutes** (dominated by the conda solve + PySCF download)
on a modern laptop with a good internet connection. Final image size: **~4–5 GB**.

The script must be run from the **repo root** (not from `apptainer/`) because
the `.def` file copies the entire repo root into the container with
`%files . /opt/quantui`.

---

## Running the container

### Windows — double-click launchers (easiest)

Two `.bat` files in the repo's `launchers/` folder handle WSL and browser launch
automatically:

| File | Purpose |
| --- | --- |
| `launchers/launch-app.bat` | Student-facing app — uses the baked-in notebook |
| `launchers/launch-dev.bat` | Development mode — uses local `notebooks/` (no rebuild needed for notebook edits) |

Double-click either file. The browser opens automatically at
[http://localhost:8866](http://localhost:8866).

### Linux / WSL — command line

#### Voilà app mode — recommended for students

Launches the notebook as a clean widget-only interface. Students see no code.

```bash
apptainer run quantui.sif app
```

Then open a browser at [http://localhost:8866](http://localhost:8866).

#### JupyterLab mode — for exploration or development

```bash
apptainer run quantui.sif
```

Then open the URL printed in the terminal (contains a login token).

#### Development mode — hot-reload notebook edits

Uses the local `notebooks/` folder instead of the baked-in copy. Notebook
cell changes take effect on browser refresh — no container rebuild needed.
Note: changes to `quantui/` Python files still require a rebuild.

```bash
cd /path/to/QuantUI
apptainer exec quantui.sif voila notebooks/molecule_computations.ipynb \
    --no-browser --port=8866 \
    --ServerApp.disable_check_xsrf=True
```

Then open [http://localhost:8866](http://localhost:8866).

#### Results directory

Calculation results are saved automatically to `~/.quantui/results/` inside the
container (i.e. your home directory on the host, which Apptainer bind-mounts by
default). Results persist across kernel restarts and container runs. The
in-app **Past Results** browser and **Open folder** button point to this location.

#### Bind a local directory

By default Apptainer binds your current working directory so you can
access local files (e.g. your own XYZ files or saved results) inside the
container:

```bash
# Work from a specific project folder
cd ~/my-calculations
apptainer run /path/to/quantui.sif app
```

#### Custom port

```bash
# Voilà on port 9000
apptainer run quantui.sif app --port=9000
```

---

## Verifying the container

After building (or downloading), run the built-in test to confirm all
packages loaded correctly:

```bash
# Built-in %test section
apptainer test quantui.sif

# Manual import check
apptainer exec quantui.sif python -c "
import quantui, pyscf, ase, py3Dmol
from quantui import Molecule, parse_xyz_input
atoms, coords = parse_xyz_input('O 0 0 0\nH 0.757 0.587 0\nH -0.757 0.587 0')
mol = Molecule(atoms, coords)
print('OK:', mol.get_formula())
"
```

Expected output: `OK: H2O` and package import messages.

### Run the full test suite

```bash
apptainer exec --cleanenv --writable-tmpfs quantui.sif bash -c '
    pip install pytest -q 2>/dev/null
    python -m pytest tests/test_notebook_workflows.py -v --tb=short --override-ini="addopts="
'
```

This installs pytest into a temporary overlay (nothing is written to the `.sif`)
and runs the full notebook workflow tests including HF, DFT, pre-optimization,
and thread-safety checks. Expected: **20 passed** in ~25 seconds.

### Quick calculation check

```bash
apptainer exec --cleanenv quantui.sif python -c "
from quantui.molecule import Molecule
from quantui import run_in_session

atoms = ['O', 'H', 'H']
coords = [[0,0,0],[0.757,0.587,0],[-0.757,0.587,0]]
mol = Molecule(atoms, coords)

# Test RHF
result = run_in_session(mol, method='RHF', basis='STO-3G', verbose=1)
print(f'RHF/STO-3G:   {result.energy_hartree:.6f} Ha  converged: {result.converged}')

# Test DFT
result = run_in_session(mol, method='B3LYP', basis='STO-3G', verbose=1)
print(f'B3LYP/STO-3G: {result.energy_hartree:.6f} Ha  converged: {result.converged}')
"
```

Expected output:

```text
RHF/STO-3G:   -74.963063 Ha  converged: True
B3LYP/STO-3G: -75.312587 Ha  converged: True
```

---

## Supported methods and basis sets

### Methods

| Method | Type | Best for |
| --- | --- | --- |
| `RHF` | Hartree-Fock | Closed-shell molecules (default) |
| `UHF` | Hartree-Fock | Radicals, triplets, open-shell |
| `B3LYP` | DFT hybrid | General organic chemistry |
| `PBE` | DFT GGA | Large molecules, speed-critical |
| `PBE0` | DFT hybrid | Charge transfer, band gaps |
| `M06-2X` | DFT meta-hybrid | Reaction barriers, thermochemistry |
| `wB97X-D`, `CAM-B3LYP` | DFT range-separated | Non-covalent / charge-transfer / UV-Vis |
| `M06-L`, `HSE06`, `PBE-D3` | DFT (meta-GGA / screened / dispersion) | Large systems, band gaps, vdW complexes |
| `MP2` | Post-HF | Accurate small-molecule energetics (O(N⁵)) |
| `CCSD`, `CCSD(T)` | Post-HF coupled cluster | Benchmark-quality energies (O(N⁶)/O(N⁷); (T) is CPU-only) |

Six calculation types run over these: Single Point, Geometry Opt, Frequency
(+ thermochemistry / IR), UV-Vis (TD-DFT), NMR shielding, and 1D PES scan; PCM
implicit solvent (Water, Ethanol, THF, DMSO, Acetonitrile) is a single checkbox.

### Basis sets

| Basis set | Quality | Notes |
| --- | --- | --- |
| `STO-3G` | Minimal | Fast; for learning only |
| `3-21G` | Small | Slightly better than STO-3G |
| `6-31G` | Medium | Good default for HF |
| `6-31G*` | Medium+ | Adds polarization functions |
| `6-31G**` | Medium+ | Polarization on H too |
| `cc-pVDZ` | High | Correlation-consistent double-zeta |
| `cc-pVTZ` | High | Triple-zeta; near-CBS for small molecules |
| `def2-SVP` | Medium | Good default for DFT |
| `def2-TZVP` | High | Near-complete-basis for DFT |

---

## Distributing to students

The `.sif` is a single self-contained file — share it however is convenient:

```bash
# Network drive / shared folder
cp quantui.sif /shared/drive/

# SCP to a department server students can pull from
scp quantui.sif user@server.dept.edu:/shared/tools/

# USB drive
cp quantui.sif /media/usb/
```

Students on Windows download the `.sif` and the `launch-app.bat` file. Then:

1. Place both files in the same folder
2. Double-click `launch-app.bat`
3. Browser opens automatically

Students on Linux/Mac run:

```bash
apptainer run quantui.sif app
```

No Python, no conda, no pip — everything is bundled.

---

## Rebuilding after code changes

```bash
# Pull latest code
git pull origin main

# Rebuild (overwrites existing .sif) and verify
bash apptainer/build.sh --clean --test
```

---

## Troubleshooting

### "No space left on device" during build

Apptainer uses `/tmp` as scratch space. Redirect it to somewhere with more room:

```bash
export APPTAINER_TMPDIR=~/apptainer-tmp
mkdir -p ~/apptainer-tmp
apptainer build quantui.sif apptainer/quantui.def
```

### "Permission denied" or "root required"

Use `--fakeroot` if your HPC or server supports it:

```bash
apptainer build --fakeroot quantui.sif apptainer/quantui.def
```

On a personal machine or in WSL you typically have root access and don't
need this flag.

### PySCF or conda download times out

PySCF is the largest package (~500 MB). If the download keeps timing out:

1. Try building during off-peak hours

2. Pre-download the conda packages:

   ```bash
   conda create -n build-cache pyscf -c conda-forge --download-only
   ```

3. Point Apptainer at a local conda mirror by editing `%post` in the `.def` file

### Container starts but PySCF crashes

PySCF requires OpenMP. If running in a restricted environment:

```bash
export OMP_NUM_THREADS=1
apptainer run quantui.sif app
```

### XSRF 403 warning on shutdown

Fixed in the current container build — `c.ServerApp.disable_check_xsrf = True`
is written to `/etc/jupyter/jupyter_server_config.py` during `%post`. If you
are running an older `.sif`, rebuild with `bash apptainer/build.sh --clean`.

### "Unable to locate package apptainer"

The `apptainer` package is not in the default Ubuntu apt repositories. Add the
PPA first:

```bash
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt-get update
sudo apt-get install -y apptainer
```

---

## What's inside the container

| Layer | Contents |
| --- | --- |
| Base | `continuumio/miniconda3:latest` (Debian + conda) |
| conda-forge | jupyter, jupyterlab, ipywidgets, notebook, pyscf, numpy, scipy, matplotlib, plotly, h5py, rdkit |
| pip | voila, ase, py3dmol, requests |
| QuantUI | installed from `/opt/quantui` (the repo root, copied at build time) — bundles the molecule library + vendored 3Dmol.js for offline use |

The `.git` directory, `__pycache__` folders, and internal dev files are removed
during build to keep the image lean. A build-time check asserts the vendored
3Dmol.js is present and the viewer is CDN-free, so a broken offline build fails
fast instead of shipping blank 3D views.

---

## Updating the container version

Edit `%labels` in `quantui.def` to bump the version string, then rebuild:

```singularity
%labels
  Version "0.3.0"
```

Tag the git commit and push so the version is traceable:

```bash
git tag v0.3.0
git push origin v0.3.0
```
