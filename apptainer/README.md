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
| `quantui.def` | CPU image definition — the local teaching interface |
| `build.sh` | Build script with clean/test/fakeroot options |
| `quantui-gpu.def` | **GPU image** definition (CUDA 12.x + gpu4pyscf) |
| `build-gpu.sh` | GPU build script (`--version`, `--clean`, `--test`, `--fakeroot`) |
| `verify-gpu.sh` | Six-step check that a GPU image really reaches the GPU |
| `slurm/quantui-gpu-test.sbatch` | Batch template for verifying on a cluster |
| `ncshare-gpu-diagnostic.ipynb` | Interactive diagnostic to run **inside** the image on a GPU node |
| `make_ncshare_diagnostic.py` | Generates that notebook (edit here, not the `.ipynb`) |
| `README.md` (this file) | Build, run, and distribution guide |

**Two images, on purpose.** `quantui.def` is CPU-only and conda-based — it is
the "run QuantUI without a cluster" image students copy to a laptop.
`quantui-gpu.def` targets NVIDIA datacenter GPUs under Slurm and differs at
every layer (CUDA base image, all-pip install, a pinned PyPI release rather than
the working tree). Merging them would ship a multi-GB CUDA stack to students who
will never have a GPU. See [GPU image](#gpu-image) below.

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

---

## GPU image

`quantui-gpu.def` builds an image for running PySCF on NVIDIA datacenter GPUs
under Slurm. It was written against **NCShare** (32× H200 SXM 141 GB, `sm_90`,
Ubuntu + Slurm, CUDA 12.8), but nothing in it is site-specific except the two
flagged values in the sbatch template.

### How it differs from the CPU image

| | `quantui.def` | `quantui-gpu.def` |
| --- | --- | --- |
| Base | `condaforge/miniforge3` | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Installer | mamba + pip | **pip only** |
| QuantUI source | working tree (`%files`) | **pinned release from PyPI** |
| GPU | none | `gpu4pyscf` / `cupy` / `cutensor` via the `gpu-cuda12x` extra |
| Needs `--nv` | no | **yes, on every invocation** |
| Default action | launches Voilà or JupyterLab | none — `exec` what you want |

Three choices worth the words:

**One installer, not two.** The CPU image uses conda for the scientific stack
because conda-forge's `pyscf` is prebuilt against conda's BLAS. Mixing conda and
pip for the same package lets pip overwrite a conda-managed install in place,
leaving conda's metadata claiming files it no longer owns. The GPU image avoids
the question entirely: it carries no MPI or HYPRE, so there is no BLAS/OpenMP
linkage to protect and no reason to involve a second installer.

**A pinned PyPI release, not a git commit.** `pip install quantui==0.5.2` names
an immutable artifact anyone can download and diff. Cloning and checking out a
SHA leaves a detached HEAD — reproducible in principle, murkier in practice, and
it ties the image to one person's working tree.

**`cuda12x`, not `cuda13x`.** CUDA's driver API is backward compatible, so the
`cuda12x` wheels run on NCShare's 570-series driver *and* on any 580+ update.
`cuda13x` would hard-fail on 570. H200 is `sm_90` and has prebuilt wheels on
both lines, so no `nvcc` build is needed either way — the `devel` base image is
there for CuPy's runtime JIT (NVRTC), not for compiling this stack.

### Build

```bash
# From the repo root. Installs the version pinned in the def's %arguments.
bash apptainer/build-gpu.sh --test
```

```bash
# Build a different release (must already be published to PyPI)
bash apptainer/build-gpu.sh --version 0.5.3
```

The script checks PyPI for the requested version **before** pulling a multi-GB
base image — a real trap right after cutting a tag, when the GitHub release
exists but the PyPI publish job is still waiting on its environment approval.

`--fakeroot` builds without root, for HPC login nodes.

### Verify

`%test` runs on the *build host*, which usually has no GPU, so it can only prove
the stack imports and the versions are right. Proving the GPU works is a
separate step that has to run where a device exists:

```bash
bash apptainer/verify-gpu.sh quantui-gpu.sif
```

It climbs a six-rung ladder, each rung isolating one layer:

| Step | Proves |
| --- | --- |
| 0 | No environment trap is silently forcing CPU |
| 1 | The host driver sees a GPU at all |
| 2 | `--nv` exposes the device *inside* the container |
| 3 | CuPy can allocate and run FP64 — the wheel/driver ABI matches |
| 4 | `quantui gpu check` — this environment *can* offload |
| 5 | Negative control: `QUANTUI_DISABLE_GPU=1` flips the answer |
| 6 | A real calculation reports `gpu_used: true` |

Fix the lowest-numbered failure first; later rungs assume the earlier ones.

Step 6 is the only one that matters, and steps 1–5 can all pass while
calculations still run on the CPU. Note that a fallback run still **converges to
the correct energy** — a right answer is not evidence the GPU was used. Only
`gpu_used` is. These are three different claims that are routinely conflated:

- `nvidia-smi` — a device is *visible*
- `quantui gpu check` — this environment *can* offload
- `gpu_used: true` — **this calculation actually did**

### Interactive diagnostic notebook

`verify-gpu.sh` runs *outside* the container and answers pass/fail.
`ncshare-gpu-diagnostic.ipynb` runs *inside* it and answers "what is this
node, and where is the crossover" — the questions you want on a first
allocation.

```bash
apptainer exec --nv --env QUANTUI_SETTINGS_PATH=$TMPDIR/q.json \
  quantui-gpu.sif jupyter lab --ip=0.0.0.0 --no-browser
```

Beyond the ladder's checks it records the **driver version and compute
capability**, the Slurm partition and gres strings, verifies the device's
FP64 classification is right for its class, and measures a CPU-vs-GPU
crossover across four systems — each leg in a subprocess so the CPU leg
genuinely starts with the GPU disabled before any import. It writes a JSON
record and a paste-able summary.

Edit `make_ncshare_diagnostic.py` and regenerate rather than editing the
notebook JSON:

```bash
python apptainer/make_ncshare_diagnostic.py
```

### ✅ Verified on NCShare H200, 2026-08-05

The image and this diagnostic have been run end to end on real hardware — all
seven checks green, `gpu_used: true`, negative control flipping correctly.
Recorded so the next build has a known-good pairing to compare against:

| | |
| --- | --- |
| GPU | NVIDIA H200, 143771 MiB, compute capability 9.0 |
| Driver | **580.126.20** (CuPy reports driver API 13000 against runtime 12090) |
| Partition / node | `gpu` / `compute-gpu-02` |
| QuantUI | 0.6.0, `cuda12x` wheels |

The driver/runtime pairing is the point: a 580-series driver running `cuda12x`
wheels is exactly the backward-compatible combination that made `cuda12x` the
right choice over `cuda13x`, which would have hard-failed on the 570-series
driver the hardware notes originally listed.

**Measured crossover** (1 GPU vs 6 CPU cores, affinity-confirmed):

| System | GPU | CPU | Speedup |
| --- | --- | --- | --- |
| H₂O / STO-3G | 1.80 s | 0.35 s | 0.20× |
| H₂O / cc-pVDZ | 2.72 s | 0.48 s | 0.18× |
| C₆H₆ / 6-31G | 2.69 s | 0.77 s | 0.29× |
| C₆H₆ / cc-pVDZ | 2.86 s | 2.74 s | **0.96× — crossover** |
| C₆H₆ / cc-pVTZ | 7.07 s | 42.41 s | **6.00×** |

GPU wall time barely moves across the first four — that is fixed launch and
transfer overhead dominating. Only at cc-pVTZ does the arithmetic grow enough
for the device to matter.

⚠️ **Quote the CPU allocation with any speedup.** These are 1 GPU vs 6 cores;
the node has ~12 physical cores per GPU, so a proportional-share comparison
uses ~12 and would show a smaller factor. Run the diagnostic at both — the
crossover *shape* is identical, but "why only 6 cores?" is the first question
an audience asks, and a number without its denominator invites it.

### On a cluster

```bash
salloc --partition=<PARTITION> --gres=gpu:h200:1 --cpus-per-task=4 \
       --mem=16G --time=00:30:00
bash apptainer/verify-gpu.sh quantui-gpu.sif
```

Or submit the batch template, which sets thread limits and a job-scoped results
directory:

```bash
sbatch --export=ALL,QUANTUI_IMAGE=$HOME/quantui-gpu.sif \
       apptainer/slurm/quantui-gpu-test.sbatch
```

The template's `--partition` and `--gres` are deliberately left as `CHANGEME`
and a `TODO`, because a plausible-looking guess fails later and less clearly
than an obvious placeholder.

### Two ways to end up on the CPU without noticing

Both let every other check pass while calculations quietly run on the CPU, and
neither reports an error — the job converges and the energy is correct.

**1. A persisted preference follows you in.** Apptainer bind-mounts `$HOME`, and
QuantUI stores `compute.gpu_enabled` in `~/.quantui/settings.json`. Switch GPU
offload off on a laptop with a consumer card, and the *cluster* container
inherits that setting. The image does not override it — the same thing happens
with any QuantUI container, and hiding it would hide a real property. To pin a
run to the image's own default:

```bash
apptainer exec --nv --env QUANTUI_SETTINGS_PATH=/tmp/q.json quantui-gpu.sif quantui gpu check
```

**2. `QUANTUI_DISABLE_GPU=1`.** Nothing in the image sets it, and `%test`
asserts that. It is genuinely useful as a negative control — the same command
with and without it is the cleanest demonstration that a GPU run differed.

`quantui gpu check` names the exact reason in both cases, which is why step 0 of
the verification script looks for them before anything else.

### WSL note — local verification cannot get past step 2

**You cannot verify a GPU image on WSL2.** Measured 2026-08-04 (RTX 5060 Ti,
driver 581.95, Apptainer 1.5.3): the container loads the correct `libcuda` —
either `/usr/lib/wsl/lib/libcuda.so.1` or the one `--nv` injects — and
`cuDriverGetVersion` still returns **0**, so every CUDA call fails with
`cudaErrorInsufficientDriver`. WSL reaches the GPU through `/dev/dxg` and a
Windows-side shim that does not function inside an Apptainer container, and
binding the whole of `/usr/lib/wsl` does not change it.

`verify-gpu.sh` detects this, reports steps 3-6 as warnings rather than
failures, and exits **3 (INCONCLUSIVE)** — reporting them as failures would
train you to ignore exactly the failures that matter on a real node.

This is a platform limit, not a defect in the image. Worth knowing is what the
same diagnosis *did* confirm: the container resolves `libcuda` from
`/.singularity.d/libs` — the host driver `--nv` injects — and **not** the
`compat/libcuda.so.570` that the CUDA base image ships alongside it. That
ordering is the thing that would genuinely break on a cluster, and it is
correct.

**Native WSL is unaffected and worth using.** Outside a container the same GPU
works normally — verified on this machine: `driverGetVersion` 13000, a real RHF
run reporting `gpu_used: true`. The distinction is containerisation, not WSL. So
QuantUI's own GPU logic (offload dispatch, `gpu_used` reporting, method
coverage, the FP64 advisory) *can* be exercised locally in a native env:

```bash
pip install "quantui[gpu-cuda13x]"   # match YOUR driver: 13x needs 580+
quantui gpu check
```

Note the wheel line differs from the container's on purpose. A local 580+ driver
takes `cuda13x`; the image targets NCShare's 570-series driver, where `cuda13x`
would hard-fail and `cuda12x` works on both.

So: build locally if you like, exercise QuantUI's GPU path natively, but verify
the **image** on a GPU node.
