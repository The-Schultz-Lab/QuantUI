# Tutorials

Five guided notebooks in [`notebooks/tutorials/`](https://github.com/The-Schultz-Lab/QuantUI/tree/main/notebooks/tutorials)
— no prior PySCF experience needed. Each runs to completion in under two
minutes on a laptop.

| # | Notebook | Topic |
| --- | --- | --- |
| 01 | [`01_first_calculation.ipynb`](https://github.com/The-Schultz-Lab/QuantUI/blob/main/notebooks/tutorials/01_first_calculation.ipynb) | Your first RHF calculation on water |
| 02 | [`02_basis_set_study.ipynb`](https://github.com/The-Schultz-Lab/QuantUI/blob/main/notebooks/tutorials/02_basis_set_study.ipynb) | Comparing STO-3G, 6-31G, and cc-pVDZ basis sets |
| 03 | [`03_multiplicity_radicals.ipynb`](https://github.com/The-Schultz-Lab/QuantUI/blob/main/notebooks/tutorials/03_multiplicity_radicals.ipynb) | Open-shell molecules and UHF for radicals |
| 04 | [`04_charged_species.ipynb`](https://github.com/The-Schultz-Lab/QuantUI/blob/main/notebooks/tutorials/04_charged_species.ipynb) | Ions and charged species |
| 05 | [`05_comparing_results.ipynb`](https://github.com/The-Schultz-Lab/QuantUI/blob/main/notebooks/tutorials/05_comparing_results.ipynb) | Side-by-side result analysis and comparison |

## How to run

After [installing](installation.md), launch JupyterLab and open any notebook
from the `notebooks/tutorials/` folder:

```bash
jupyter lab notebooks/tutorials/01_first_calculation.ipynb
```

Or start from the main app notebook and follow cross-links from the in-app Help tab.

## What you will learn

1. **First calculation** — load a structure, pick a method/basis, run a single-point energy
2. **Basis sets** — compare convergence across STO-3G, 6-31G, and cc-pVDZ
3. **Radicals** — set multiplicity correctly and run UHF
4. **Charged species** — handle ions with the correct charge and spin
5. **Comparisons** — replay multiple saved results side by side
