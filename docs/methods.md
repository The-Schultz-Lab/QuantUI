# Supported Methods

Nine calculation types, fourteen methods, fourteen basis sets.

## Calculation types

| Calculation type | Output |
| --- | --- |
| **Single Point** | Energy, HOMO–LUMO gap, Mulliken and NPA charges, bond orders, dipole |
| **Geometry Opt** | Optimized structure and trajectory; optional frozen atoms |
| **Transition State** | Saddle point (Sella, `quantui[ts]`) plus a frequency check; optional guess from a reactant + product |
| **IRC** | Path from a transition state to the two minima it connects; barriers and energy profile |
| **Frequency** | Frequencies, IR intensities, Raman activities (gas phase), ZPVE, thermochemistry, animated modes |
| **UV-Vis (TD-DFT)** | Excitation energies, oscillator strengths, UV-Vis and ECD spectra |
| **NMR Shielding** | ¹H and ¹³C shifts vs TMS (GIAO); closed-shell HF/DFT, not M06-2X/M06-L. TMS reference computed by QuantUI at common levels, flagged otherwise |
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

For metals on a basis without parameters, the app suggests **def2-SVP** or **def2-TZVP** (one click switches to def2-SVP); **LANL2DZ** is also available.
