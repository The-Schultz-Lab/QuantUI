# Installation

Linux, macOS or WSL, Python 3.9–3.11:

```bash
conda create -n quantui python=3.11
conda activate quantui
pip install "quantui[pyscf,ase,app]"
```

Launch:

```bash
voila notebooks/molecule_computations.ipynb       # app mode (code hidden)
jupyter lab notebooks/molecule_computations.ipynb # full JupyterLab
```

From a source checkout, use `pip install -e ".[pyscf,ase,app]"` instead.

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
| `ase` | Trajectory export and structure I/O |
| `app` | Voilà and JupyterLab |
| `xtb` | GFN-FF pre-optimization for metals |
| `ts` | Transition states and IRC via [Sella](https://github.com/zadorlab/sella) (about 380 MB) |

## Next steps

- [Tutorials](tutorials.md)
- [Supported Methods](methods.md)
