"""
QuantUI Configuration Module

Configuration constants and defaults for the local teaching interface.
SLURM resource limits, job history paths, and cluster settings have been
removed — this version runs calculations in the current Jupyter session.
"""

from pathlib import Path
from typing import Any, Dict

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Supported quantum chemistry methods
SUPPORTED_METHODS = [
    "RHF",
    "UHF",
    "B3LYP",
    "PBE",
    "PBE0",
    "M06-2X",
    "wB97X-D",
    "CAM-B3LYP",
    "M06-L",
    "HSE06",
    "PBE-D3",
    "MP2",
    "CCSD",
    "CCSD(T)",
]

# Educational metadata for each method — shown to students in the UI
METHOD_INFO = {
    "RHF": {
        "type": "hf",
        "label": "RHF — Restricted Hartree-Fock",
        "description": (
            "Classical wavefunction method for closed-shell molecules. "
            "All electrons are paired; no electron correlation beyond mean-field. "
            "Fast and exact within HF theory."
        ),
        "use_for": "Closed-shell molecules with all electrons paired (singlet state).",
    },
    "UHF": {
        "type": "hf",
        "label": "UHF — Unrestricted Hartree-Fock",
        "description": (
            "HF for open-shell systems. Alpha and beta electrons occupy "
            "different spatial orbitals. Can suffer from spin contamination."
        ),
        "use_for": "Radicals, triplet states, any molecule with unpaired electrons.",
    },
    "B3LYP": {
        "type": "dft",
        "label": "B3LYP — DFT Hybrid Functional",
        "description": (
            "The most widely used DFT functional. Mixes HF exchange (20%) with "
            "Becke exchange and Lee-Yang-Parr correlation. Good all-around accuracy "
            "at moderate cost."
        ),
        "use_for": "General organic chemistry, ground state geometries and energetics.",
    },
    "PBE": {
        "type": "dft",
        "label": "PBE — DFT GGA Functional",
        "description": (
            "Generalized gradient approximation functional. No HF exchange, so "
            "faster than hybrids. Works well for larger systems and metals."
        ),
        "use_for": "Large molecules, solid-state systems, when speed matters most.",
    },
    "PBE0": {
        "type": "dft",
        "label": "PBE0 — DFT Hybrid Functional",
        "description": (
            "Hybrid version of PBE with 25% HF exchange. More accurate than "
            "pure PBE for most molecular properties; similar cost to B3LYP."
        ),
        "use_for": "General purpose; better charge-transfer and band gaps than B3LYP.",
    },
    "M06-2X": {
        "type": "dft",
        "label": "M06-2X — Meta-Hybrid DFT",
        "description": (
            "Minnesota meta-hybrid functional with 54% HF exchange. Excellent "
            "for thermochemistry and reaction barrier heights in organic systems. "
            "Not recommended for transition metals."
        ),
        "use_for": "Organic reaction energies, conformational analysis, barrier heights.",
    },
    "wB97X-D": {
        "type": "dft",
        "label": "wB97X-D — Range-Separated Hybrid + D3 Dispersion",
        "description": (
            "Range-separated hybrid functional with empirical D3 dispersion correction. "
            "Excellent for non-covalent interactions, charge-transfer excitations, "
            "and systems where long-range exchange matters."
        ),
        "use_for": "Non-covalent interactions, excited states, large organic molecules.",
    },
    "CAM-B3LYP": {
        "type": "dft",
        "label": "CAM-B3LYP — Coulomb-Attenuating B3LYP",
        "description": (
            "Range-separated version of B3LYP. More reliable than B3LYP for "
            "charge-transfer excited states and Rydberg transitions. "
            "Good general-purpose alternative to B3LYP."
        ),
        "use_for": "Charge-transfer states, UV-Vis spectra, long-range interactions.",
    },
    "M06-L": {
        "type": "dft",
        "label": "M06-L — Local Meta-GGA DFT",
        "description": (
            "Local (no HF exchange) Minnesota meta-GGA. Faster than hybrid "
            "functionals for the same system size. Good for transition metals "
            "and main-group thermochemistry."
        ),
        "use_for": "Larger molecules where hybrid cost is prohibitive; transition metals.",
    },
    "HSE06": {
        "type": "dft",
        "label": "HSE06 — Screened Hybrid DFT",
        "description": (
            "Heyd-Scuseria-Ernzerhof screened hybrid. Uses short-range HF exchange "
            "only, making it efficient for large systems. Often used for solids; "
            "also accurate for molecular band gaps."
        ),
        "use_for": "Band gaps, large molecules, when PBE0 is too expensive.",
    },
    "PBE-D3": {
        "type": "dft",
        "label": "PBE-D3 — PBE + D3 Dispersion Correction",
        "description": (
            "PBE GGA functional with Grimme's D3BJ empirical dispersion correction. "
            "Dramatically improves non-covalent interaction energies over plain PBE "
            "at negligible extra cost."
        ),
        "use_for": "Van der Waals complexes, stacking interactions, large organic molecules.",
    },
    "MP2": {
        "type": "wavefunction",
        "label": "MP2 — 2nd-Order Møller-Plesset",
        "description": (
            "Post-HF wavefunction method that adds electron correlation via 2nd-order "
            "perturbation theory. More accurate than HF for energetics and geometries, "
            "but scales as O(N⁵). Avoid for molecules with > ~20 heavy atoms."
        ),
        "use_for": "Accurate energetics for small closed-shell molecules; bond dissociation.",
    },
    "CCSD": {
        "type": "wavefunction",
        "label": "CCSD — Coupled Cluster with Singles and Doubles",
        "description": (
            "Post-HF coupled-cluster method that includes all single and double "
            "excitations from the HF reference. Often called the gold standard for "
            "single-reference systems — significantly more accurate than MP2 — but "
            "scales as O(N⁶). Memory and runtime both grow steeply with basis size; "
            "expect very small molecules (~10 heavy atoms or fewer) only."
        ),
        "use_for": "High-accuracy benchmarks for small closed-shell molecules.",
    },
    "CCSD(T)": {
        "type": "wavefunction",
        "label": "CCSD(T) — CCSD with Perturbative Triples",
        "description": (
            "Adds a perturbative correction for connected triple excitations on top of "
            "CCSD. Routinely called the 'gold standard' of single-reference electronic "
            "structure when paired with a large basis set. Scales as O(N⁷); the (T) "
            "correction alone is typically the cost bottleneck. Reserve for the "
            "smallest molecules where benchmark-quality energies are required."
        ),
        "use_for": "Reference-quality energies and barrier heights for tiny molecules.",
    },
}

# Supported basis sets
SUPPORTED_BASIS_SETS = [
    "STO-3G",
    "3-21G",
    "6-31G",
    "6-31G*",
    "6-31G**",
    "cc-pVDZ",
    "cc-pVTZ",
    "def2-SVP",
    "def2-TZVP",
]

# Implicit solvent options — name → dielectric constant (ε)
SOLVENT_OPTIONS: Dict[str, float] = {
    "Water": 78.39,
    "Ethanol": 24.55,
    "THF": 7.58,
    "DMSO": 46.70,
    "Acetonitrile": 35.69,
}

# TMS isotropic shielding reference constants for NMR chemical shift computation.
# Key: "method/basis" → {element: σ_TMS (ppm)}.  δ = σ_TMS − σ_molecule.
# Source: Cheeseman et al., J. Chem. Phys. 104 (1996) 5497; CCCBDB.
NMR_REFERENCE_SHIELDINGS: Dict[str, Dict[str, float]] = {
    "B3LYP/6-31G*": {"H": 31.72, "C": 183.71},
    "B3LYP/6-311G**": {"H": 31.60, "C": 188.94},
    "B3LYP/cc-pVDZ": {"H": 31.54, "C": 186.12},
    "B3LYP/def2-SVP": {"H": 31.65, "C": 184.20},
    "RHF/6-31G*": {"H": 32.00, "C": 196.00},
    "RHF/STO-3G": {"H": 30.50, "C": 195.00},
    "PBE0/6-31G*": {"H": 31.60, "C": 184.50},
    "PBE/6-31G*": {"H": 31.50, "C": 185.00},
}
NMR_DEFAULT_REFERENCE: Dict[str, float] = {"H": 31.72, "C": 183.71}  # B3LYP/6-31G*

# Default calculation settings
DEFAULT_METHOD = "RHF"
DEFAULT_BASIS = "6-31G"
DEFAULT_CHARGE = 0
DEFAULT_MULTIPLICITY = 1

# Geometry optimization defaults
DEFAULT_FMAX: float = 0.05  # eV/Å force convergence threshold
DEFAULT_OPT_STEPS: int = 200  # maximum BFGS optimizer steps

# Widget styling
WIDGET_LAYOUT = {
    "width": "400px",
}

DESCRIPTION_WIDTH = "150px"

# ── External structure resolvers (M-STRUCT) ──────────────────────────────────
# Network + throttle settings shared by the PubChem client (and, later, the
# NCI CACTUS resolver). All timeouts/limits live here per constraint #5.
PUBCHEM_TIMEOUT_S: float = 15.0  # per-request HTTP timeout
PUBCHEM_AVAILABILITY_TIMEOUT_S: float = 5.0  # quick reachability probe
PUBCHEM_MAX_RETRIES: int = 3  # bounded retries on 503 / throttling
PUBCHEM_BACKOFF_BASE_S: float = 0.5  # exponential back-off base (×2**attempt)
PUBCHEM_BACKOFF_MAX_S: float = 8.0  # cap on a single back-off sleep
# Proactive client-side rate limit. PUG-REST allows ~5 req/s; stay conservative
# so a classroom of simultaneous users doesn't trip the server-side throttle.
PUBCHEM_MIN_REQUEST_INTERVAL_S: float = 0.25

# NCI CACTUS Chemical Identifier Resolver — chained fallback after PubChem
# (resolves name / CAS / InChI / SMILES → 3D SDF; no API key).
CACTUS_TIMEOUT_S: float = 15.0

# Molecule presets — bundled library (STRUCT.6).
# The former inline literal now lives in the indexed package-data store
# (quantui/data/library/library.sqlite, seeded from
# quantui/data/manifests/presets.json). ``config.MOLECULE_LIBRARY`` is a lazy
# back-compat shim resolved via module ``__getattr__`` (PEP 562): on first
# access it loads the curated/preset entries from the store in the original
# {formula: {atoms, coordinates, charge, multiplicity, description}} shape, so
# every existing consumer keeps working unchanged.
MOLECULE_LIBRARY: Dict[str, Dict[str, Any]]  # populated lazily by __getattr__


def __getattr__(name: str) -> Any:
    if name == "MOLECULE_LIBRARY":
        from quantui.molecule_library import get_preset_dict

        return get_preset_dict()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Valid atomic symbols (periodic table subset commonly used)
VALID_ATOMS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
]

# Quick-start templates for the notebook UI
# The `notes` and `learning_goals` fields are shown to students.
# The `calc_settings` fields pre-fill the method/basis dropdowns.
QUICK_START_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "beginner_water": {
        "name": "Beginner: Water Molecule",
        "description": "Perfect first calculation — fast and reliable",
        "molecule": {
            "xyz": "O  0.0  0.0  0.0\nH  0.757  0.587  0.0\nH  -0.757  0.587  0.0",
            "charge": 0,
            "multiplicity": 1,
        },
        "calc_settings": {"method": "RHF", "basis": "6-31G"},
        "notes": "Water (H2O) with 6-31G basis. Expected energy: ~-76.03 Ha.",
        "learning_goals": [
            "Run your first local calculation",
            "Understand basic molecule structure",
            "See SCF convergence in action",
        ],
    },
    "basis_comparison": {
        "name": "Learn: Basis Set Effect (CO2)",
        "description": "Compare different basis sets on carbon dioxide",
        "molecule": {
            "xyz": "C  0.0  0.0  0.0\nO  1.16  0.0  0.0\nO  -1.16  0.0  0.0",
            "charge": 0,
            "multiplicity": 1,
        },
        "calc_settings": {"method": "RHF", "basis": "STO-3G"},
        "notes": "Try with STO-3G first, then re-run with 6-31G to see the difference!",
        "learning_goals": [
            "Understand basis set accuracy trade-offs",
            "Compare computational costs",
            "See energy convergence with basis size",
        ],
    },
    "radical_oxygen": {
        "name": "Advanced: Oxygen Radical (O2)",
        "description": "Learn about multiplicity with triplet oxygen",
        "molecule": {
            "xyz": "O  0.0  0.0  0.0\nO  1.21  0.0  0.0",
            "charge": 0,
            "multiplicity": 3,
        },
        "calc_settings": {"method": "UHF", "basis": "6-31G"},
        "notes": "O2 has two unpaired electrons (triplet state). Try mult=1 first to see it fail!",
        "learning_goals": [
            "Understand multiplicity for radicals",
            "Learn UHF vs RHF",
            "See why O2 is paramagnetic",
        ],
    },
    "benzene": {
        "name": "Intermediate: Benzene Ring",
        "description": "Medium-sized aromatic molecule",
        "molecule": {
            "xyz": (
                "C  0.000  1.396  0.000\nC  1.209  0.698  0.000\nC  1.209 -0.698  0.000\n"
                "C  0.000 -1.396  0.000\nC -1.209 -0.698  0.000\nC -1.209  0.698  0.000\n"
                "H  0.000  2.479  0.000\nH  2.147  1.240  0.000\nH  2.147 -1.240  0.000\n"
                "H  0.000 -2.479  0.000\nH -2.147 -1.240  0.000\nH -2.147  1.240  0.000"
            ),
            "charge": 0,
            "multiplicity": 1,
        },
        "calc_settings": {"method": "RHF", "basis": "6-31G"},
        "notes": "Benzene (C6H6) — classic aromatic molecule. Larger system, takes longer locally.",
        "learning_goals": [
            "Work with larger molecules",
            "Understand aromatic systems",
            "See how calculation time scales with molecule size",
        ],
    },
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# PySCF calculation script template
# Used by calculator.py to generate a standalone .py file students can
# download and run independently ("Export Script" feature).
PYSCF_SCRIPT_TEMPLATE = """#!/usr/bin/env python3
\"\"\"
PySCF Calculation Script
Generated by QuantUI

Calculation: {job_name}
Method: {method}
Basis: {basis}
\"\"\"

import sys
from pathlib import Path
from pyscf import gto, scf, dft
import numpy as np

def main():
    # Define molecule
    mol = gto.Mole()
    mol.atom = '''
{geometry}
    '''
    mol.basis = '{basis}'
    mol.charge = {charge}
    mol.spin = {spin}
    mol.verbose = 4  # Detailed output
    mol.build()

    print("=" * 60)
    print("Molecule Information")
    print("=" * 60)
    print(f"Number of electrons: {{mol.nelectron}}")
    print(f"Nuclear repulsion energy: {{mol.energy_nuc():.8f}} Ha")
    print()

    # Run calculation
    print("=" * 60)
    print("Starting {method}/{basis} Calculation")
    print("=" * 60)

    try:
        method = '{method}'
        # Display name → PySCF xc string + external D3 dispersion. Matches
        # quantui/session_calc.py resolve_xc + maybe_apply_d3. Important
        # for methods that PySCF doesn't accept directly (notably
        # wB97X-D — on dftd3's black-list; PBE-D3 — D3 must be applied
        # externally via pyscf.dftd3).
        _XC_ALIAS = {{
            'M06-L': 'm06l',
            'wB97X-D': 'wb97x',
            'CAM-B3LYP': 'camb3lyp',
            'PBE-D3': 'pbe',
        }}
        _NEEDS_D3 = {{'PBE-D3', 'wB97X-D'}}

        if method == 'RHF':
            mf = scf.RHF(mol)
        elif method == 'UHF':
            mf = scf.UHF(mol)
        else:
            # DFT: auto-select RKS/UKS based on spin
            mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            mf.xc = _XC_ALIAS.get(method, method)
            if method in _NEEDS_D3:
                try:
                    from pyscf import dftd3 as _dftd3
                    mf = _dftd3.dftd3(mf)
                except ImportError:
                    print(
                        "WARNING: pyscf.dftd3 not available; "
                        "running {{method}} without D3 dispersion."
                    )

        energy = mf.kernel()

        if mf.converged:
            print()
            print("=" * 60)
            print("Calculation Results")
            print("=" * 60)
            print(f"SCF converged: Yes")
            print(f"Total energy: {{energy:.8f}} Ha")
            mo_e = mf.mo_energy if not isinstance(mf.mo_energy, list) else mf.mo_energy[0]
            mo_o = mf.mo_occ   if not isinstance(mf.mo_occ,    list) else mf.mo_occ[0]
            n_occ = int((mo_o > 0).sum())
            if 0 < n_occ < len(mo_e):
                gap = (mo_e[n_occ] - mo_e[n_occ - 1]) * 27.211386
                print(f"HOMO-LUMO gap: {{gap:.4f}} eV")
            print()

            # Save results next to the script so the path is predictable
            results_path = str(Path(__file__).parent / 'results.npz')
            np.savez(results_path,
                     energy=energy,
                     mo_energy=np.array(mf.mo_energy),
                     mo_coeff=np.array(mf.mo_coeff),
                     converged=mf.converged)
            print(f"Results saved to {{results_path}}")
            print("=" * 60)

            sys.exit(0)
        else:
            print("ERROR: SCF did not converge!")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: Calculation failed with exception: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""

# ============================================================================
# Local Configuration Override (optional)
# ============================================================================
# Create config.local.py in the project root to override settings without
# modifying this tracked file. config.local.py is git-ignored.
#
# Supported overrides: PUBCHEM_API_KEY, DEBUG
#
# Example config.local.py:
#   PUBCHEM_API_KEY = "your-key-here"
#   DEBUG = True

import importlib.util  # noqa: E402

_local_config_path = PROJECT_ROOT / "config.local.py"

if _local_config_path.exists():
    spec = importlib.util.spec_from_file_location("config_local", _local_config_path)
    if spec and spec.loader:
        config_local = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_local)

        for attr in ("PUBCHEM_API_KEY", "DEBUG"):
            if hasattr(config_local, attr):
                value = getattr(config_local, attr)
                if value is not None:
                    globals()[attr] = value
