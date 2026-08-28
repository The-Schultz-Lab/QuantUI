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

PySCF does not install natively on Windows. See [Platform Support](platforms.md) for
WSL and Apptainer container paths.

## Optional extras

| Extra | What it adds |
| --- | --- |
| `pyscf` | PySCF quantum-chemistry backend (required for calculations) |
| `ase` | ASE bridge for trajectory export and structure I/O |
| `app` | Voilà, JupyterLab, and notebook launcher dependencies |
| `xtb` | GFN-FF metal pre-optimization via xtb |

## Next steps

- Check [Platform Support](platforms.md) for OS-specific notes
- Work through the [Tutorials](tutorials.md) notebooks
- Browse [Supported Methods](methods.md) for calculation types and basis sets
