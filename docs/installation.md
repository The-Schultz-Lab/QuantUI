# Installation

Linux, macOS or WSL, Python 3.9–3.11:

```bash
conda create -n quantui python=3.11
conda activate quantui
pip install "quantui[pyscf,ase,app]"
```

Launch:

```bash
quantui run app --open   # starts the Voilà app and opens your browser
```

The PyPI package does not include the notebooks. From a source checkout
(`git clone https://github.com/The-Schultz-Lab/QuantUI.git`), install with
`pip install -e ".[pyscf,ase,app]"` and you can also run:

```bash
voila notebooks/molecule_computations.ipynb       # app mode (code hidden)
jupyter lab notebooks/molecule_computations.ipynb # full JupyterLab
```

## Windows

Native Windows runs a PyFock subset (neutral, closed-shell PBE/def2-SVP or def2-TZVP single points and optimizations). Use Python 3.11:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install "quantui[pyfock,ase,app]"
```

For everything else, use WSL or Apptainer ([Platform Support](platforms.md)).

## Optional extras

| Extra | Adds |
| --- | --- |
| `pyscf` | PySCF backend (full feature set) |
| `pyfock` | PyFock backend (native Windows) |
| `ase` | Geometry optimization (ASE BFGS), trajectory export and structure I/O |
| `app` | Voilà app server (JupyterLab is installed by default) |
| `xtb` | GFN-FF pre-optimization for metals (Linux wheels; use conda-forge `xtb-python` elsewhere) |
| `ts` | Transition states and IRC via [Sella](https://github.com/zadorlab/sella) (about 380 MB) |
| `gpu-cuda12x` / `gpu-cuda13x` | gpu4pyscf GPU offload — pick the one matching your NVIDIA driver's CUDA version |
| `pyfock-gpu-cuda12x` / `pyfock-gpu-cuda13x` | CuPy for PyFock's GPU path |
| `dev` | Tests and linters (pytest, ruff, black, mypy, pre-commit) |

## Next steps

- [Tutorials](tutorials.md)
- [Supported Methods](methods.md)
