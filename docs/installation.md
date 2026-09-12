# Installation

Recommended: **conda** on Linux, macOS, or WSL.

```bash
# Create a dedicated conda environment
conda create -n quantui python=3.11
conda activate quantui

# Install with PySCF, ASE, and Voilà app server
pip install -e ".[pyscf,ase,app]"

# Launch in JupyterLab (full IDE — shows code)
jupyter lab notebooks/molecule_computations.ipynb

# Or in Voilà app mode (widget-only UI — code hidden)
voila notebooks/molecule_computations.ipynb
```

## PyPI install

If you are not developing from source:

```bash
pip install "quantui[pyscf,ase,app]"
```

## Windows

For native Windows and the guarded PyFock single-point subset, use Python 3.11:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install "quantui[pyfock,ase,app]"
```

Automatic engine selection uses PyFock when PySCF is absent. Phase 1 supports
neutral, closed-shell PBE/def2-SVP or PBE/def2-TZVP single points. For all
other methods and workflows, use WSL or the Apptainer container described in
[Platform Support](platforms.md).

## Optional extras

| Extra | What it adds |
| --- | --- |
| `pyscf` | Canonical PySCF backend (full calculation and analysis feature set) |
| `pyfock` | PyFock Phase-1 backend (native-Windows PBE single points) |
| `ase` | ASE bridge for trajectory export and structure I/O |
| `app` | Voilà, JupyterLab, and notebook launcher dependencies |
| `xtb` | GFN-FF metal pre-optimization via xtb |

## Next steps

- Check [Platform Support](platforms.md) for OS-specific notes
- Work through the [Tutorials](tutorials.md) notebooks
- Browse [Supported Methods](methods.md) for calculation types and basis sets
