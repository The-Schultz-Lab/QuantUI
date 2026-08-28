# Supported Methods

Six calculation types over fourteen methods and nine basis sets, all dispatched
through a single Calculate tab.

## Calculation types

| Calculation type | Output |
| --- | --- |
| **Single Point** | Energy, HOMO–LUMO gap, Mulliken charges, dipole moment |
| **Geometry Opt** | Optimized structure with step-by-step trajectory animation |
| **Frequency** | Vibrational frequencies, ZPVE, IR intensities, thermochemistry (H/S/G at 298 K), animated normal modes |
| **UV-Vis (TD-DFT)** | Excitation energies, oscillator strengths, UV-Vis spectrum plot |
| **NMR Shielding** | ^1^H and ^13^C chemical shifts vs TMS via GIAO |
| **PES Scan** | 1D bond/angle/dihedral scan; energy profile + per-step geometries |

## Methods by family

| Family | Methods |
| --- | --- |
| **Hartree–Fock** | RHF (closed-shell), UHF (open-shell radicals) — baseline reference; fastest path to convergence |
| **DFT** | B3LYP, PBE, PBE0, M06-2X, ωB97X-D, CAM-B3LYP, M06-L, HSE06, PBE-D3 — nine functionals spanning hybrid, GGA, meta-hybrid, range-separated, and dispersion-corrected families |
| **Post-HF** | MP2, CCSD, CCSD(T) — Møller–Plesset (O(N^5^)) for fast post-HF; coupled cluster (O(N^6^) singles+doubles, O(N^7^) with perturbative triples) for benchmark-quality small-molecule energies |
| **Implicit solvent** | PCM — Water, Ethanol, THF, DMSO, Acetonitrile — single checkbox; compatible with any method above |

## Basis sets

From fast iteration to higher accuracy:

`STO-3G` → `3-21G` → `6-31G` → `6-31G*` → `6-31G**` → `cc-pVDZ` → `cc-pVTZ` → `def2-SVP` → `def2-TZVP`

!!! tip "Choosing a basis"
    - **STO-3G** — fast iteration and classroom demos
    - **cc-pVDZ** — common research default
    - **def2-TZVP** — higher accuracy when cost allows

For transition-metal complexes, the in-app guard nudges toward **def2-SVP** or **LANL2DZ**
when a metal is paired with an incompatible all-electron basis.
