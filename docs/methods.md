# Supported Methods

Eight calculation types over fourteen methods and ten basis sets, all
dispatched through a single Calculate tab.

## Calculation types

| Calculation type | Output |
| --- | --- |
| **Single Point** | Energy, HOMO–LUMO gap, Mulliken charges, dipole moment |
| **Geometry Opt** | Optimized structure with step-by-step trajectory animation; optional frozen atoms |
| **Transition State** | Saddle point near the input geometry (Sella; `pip install "quantui[ts]"`), then a frequency check that says whether it is a transition state (one imaginary frequency), a minimum, or a higher-order saddle point |
| **Frequency** | Vibrational frequencies, ZPVE, IR intensities, thermochemistry at any temperature and pressure, animated normal modes |
| **UV-Vis (TD-DFT)** | Excitation energies, oscillator strengths, UV-Vis spectrum plot |
| **NMR Shielding** | ^1^H and ^13^C chemical shifts vs TMS via GIAO |
| **PES Scan** | 1D bond/angle/dihedral scan; energy profile + per-step geometries |
| **Reorganization Energy** | Marcus four-point internal reorganization energy λ = λ₁ + λ₂ for hole (cation) and/or electron (anion) transfer |

## Methods by family

| Family | Methods |
| --- | --- |
| **Hartree–Fock** | RHF (closed-shell), UHF (open-shell radicals) — baseline reference; fastest path to convergence |
| **DFT** | B3LYP, PBE, PBE0, M06-2X, ωB97X-D, CAM-B3LYP, M06-L, HSE06, PBE-D3 — nine functionals spanning hybrid, GGA, meta-hybrid, range-separated, and dispersion-corrected families |
| **Post-HF** | MP2, CCSD, CCSD(T) — Møller–Plesset (O(N^5^)) for fast post-HF; coupled cluster (O(N^6^) singles+doubles, O(N^7^) with perturbative triples) for benchmark-quality small-molecule energies |
| **Implicit solvent** | PCM (C-PCM) — Water, Ethanol, THF, DMSO, Acetonitrile — single checkbox. Single point, geometry optimization (solvated at every step), frequency (PCM Hessian and IR intensities; Raman is not computed in solvent) and UV-Vis TD-DFT (solvated ground state, non-equilibrium excitations using the solvent's optical dielectric n²). Reorganization energy uses gas-phase optimizations with solvated single points. NMR and PES scans are gas-phase only |

## Basis sets

From fast iteration to higher accuracy:

`STO-3G` → `3-21G` → `6-31G` → `6-31G*` → `6-31+G*` → `6-31G**` → `6-31++G**` → `cc-pVDZ` → `aug-cc-pVDZ` → `cc-pVTZ` → `def2-SVP` → `def2-SVPD` → `def2-TZVP` → `LANL2DZ`

!!! tip "Choosing a basis"
    - **STO-3G** — fast iteration and classroom demos
    - **cc-pVDZ** — common research default
    - **def2-TZVP** — higher accuracy when cost allows
    - **Anions** (including radical anions) — use a diffuse set: `6-31+G*`, `6-31++G**`, `aug-cc-pVDZ` or `def2-SVPD`

!!! note "Diffuse basis sets"
    The `+` / `++` (Pople), `aug-` (Dunning) and trailing `D` (Karlsruhe) markers add
    diffuse functions — widely spread functions that describe an anion's loosely held
    extra electron, lone pairs and weak interactions. They work with every calculation
    type, open-shell (UHF/UKS) included, and with PCM. They are slower than the parent
    set and SCF can need more cycles. `6-31+G*` and `6-31++G**` can also be typed as
    `6-31+G(d)` and `6-31++G(d,p)`. Coverage: `6-31+G*` stops at Ca (no transition
    metals); `6-31++G**` and `aug-cc-pVDZ` have no iodine; `def2-SVPD` carries ECPs and
    covers heavy elements too (lanthanides aside). The pre-run guard flags an element the
    chosen basis lacks.

For transition-metal complexes, the in-app guard nudges toward **def2-SVP** or **LANL2DZ**
when a metal is paired with an incompatible all-electron basis.
