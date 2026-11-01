# Supported Methods

Nine calculation types, fourteen methods, fourteen basis sets.

## Calculation types

| Calculation type | Output |
| --- | --- |
| **Single Point** | Energy, HOMO–LUMO gap, charges, dipole |
| **Geometry Opt** | Optimized structure and trajectory; optional frozen atoms |
| **Transition State** | Saddle point (Sella, `quantui[ts]`) plus a frequency check |
| **IRC** | Path from a transition state to the two minima it connects; barriers and energy profile |
| **Frequency** | Frequencies, IR intensities, ZPVE, thermochemistry, animated modes |
| **UV-Vis (TD-DFT)** | Excitation energies, oscillator strengths, spectrum |
| **NMR Shielding** | ^1^H and ^13^C shifts vs TMS (GIAO) |
| **PES Scan** | 1D bond, angle or dihedral scan |
| **Reorganization Energy** | Marcus four-point λ for hole and/or electron transfer |

## Methods

| Family | Methods |
| --- | --- |
| **Hartree–Fock** | RHF, UHF |
| **DFT** | B3LYP, PBE, PBE0, M06-2X, ωB97X-D, CAM-B3LYP, M06-L, HSE06, PBE-D3 |
| **Post-HF** | MP2, CCSD, CCSD(T) |
| **Implicit solvent** | PCM: water, ethanol, THF, DMSO, acetonitrile. All calculation types except NMR and PES scans (no Raman in solvent) |

## Basis sets

Available sets:

`STO-3G` → `3-21G` → `6-31G` → `6-31G*` → `6-31+G*` → `6-31G**` → `6-31++G**` → `cc-pVDZ` → `aug-cc-pVDZ` → `cc-pVTZ` → `def2-SVP` → `def2-SVPD` → `def2-TZVP` → `LANL2DZ`

!!! tip "Choosing a basis"
    - **STO-3G** — quick tests and demos
    - **cc-pVDZ** — common research default
    - **def2-TZVP** — higher accuracy
    - **Anions** — a diffuse set: `6-31+G*`, `6-31++G**`, `aug-cc-pVDZ` or `def2-SVPD`

!!! note "Diffuse basis sets"
    `+`/`++`, `aug-` and trailing `D` sets describe anions and weak interactions. Slower, and SCF may need more cycles. `6-31+G*` stops at Ca; `6-31++G**` and `aug-cc-pVDZ` lack iodine; `def2-SVPD` covers heavy elements. The pre-run check flags missing elements.

For metals, the app suggests **def2-SVP** or **LANL2DZ**.
