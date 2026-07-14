"""
QuantUI Package

An open-source frontend for DFT and post-HF quantum chemistry with PySCF.
Calculations run locally in the Jupyter session — no cluster or SLURM required.

PySCF requires Linux/macOS/WSL. Windows users should use the Apptainer container.
"""

__version__ = "0.4.0"

import logging
from typing import Any

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .calculator import PySCFCalculation, create_calculation

# Calculation comparison
from .comparison import (
    CalcSummary,
    comparison_table_html,
    plot_comparison,
    summary_from_saved_result,
    summary_from_session_result,
)
from .config import (
    DEFAULT_BASIS,
    DEFAULT_CHARGE,
    DEFAULT_FMAX,
    DEFAULT_METHOD,
    DEFAULT_MULTIPLICITY,
    DEFAULT_OPT_STEPS,
    DESCRIPTION_WIDTH,
    METHOD_INFO,
    MOLECULE_LIBRARY,
    PYSCF_SCRIPT_TEMPLATE,
    QUICK_START_TEMPLATES,
    SUPPORTED_BASIS_SETS,
    SUPPORTED_METHODS,
    VALID_ATOMS,
    WIDGET_LAYOUT,
)
from .molecule import Molecule, parse_xyz_input

# Orbital visualization (matplotlib energy diagrams, cube-file viewer)
from .orbital_visualization import (
    OrbitalInfo,
    load_orbital_info,
    orbital_info_from_arrays,
    orbital_summary_html,
    parse_cube_file,
    plot_orbital_diagram,
)

# Security — catchable exception for constraint violations
from .security import SecurityError
from .utils import (
    get_session_resources,
    get_username,
    sanitize_filename,
    session_can_handle,
)

# ASE bridge (optional — requires ase>=3.22.0)
try:
    from .ase_bridge import (
        ASE_AVAILABLE,
        ASE_MOLECULE_PRESETS,
        ase_molecule_library,
        atoms_to_molecule,
        is_ase_available,
        molecule_to_atoms,
        read_structure_file,
    )
except ImportError:
    ASE_AVAILABLE = False
    ASE_MOLECULE_PRESETS: dict = {}  # type: ignore[misc,no-redef]

# ASE pre-optimization (optional — requires ase_bridge)
try:
    from .preopt import preoptimize
except ImportError:
    pass

# ASE-PySCF in-session calculator (optional — requires ase>=3.22 + pyscf, Linux/WSL)
try:
    from .session_calc import SessionResult, run_in_session
except ImportError:
    pass

# Frequency analysis (optional — requires pyscf, Linux/WSL)
try:
    from .freq_calc import FreqResult, run_freq_calc
except ImportError:
    pass

# TD-DFT excited states (optional — requires pyscf, Linux/WSL)
try:
    from .tddft_calc import TDDFTResult, run_tddft_calc
except ImportError:
    pass

# Results persistence — pure Python, always available
from .results_storage import list_results, load_result, save_result

# QM geometry optimizer (optional — requires ase>=3.22 + pyscf, Linux/WSL)
try:
    from .optimizer import OptimizationResult, optimize_geometry
except ImportError:
    pass

# 1D PES scan (optional — requires ase>=3.22 + pyscf, Linux/WSL)
try:
    from .pes_scan import PESScanResult, run_pes_scan  # noqa: F401
except ImportError:
    pass

# PubChem integration (optional — requires internet)
try:
    from .cactus import fetch_from_cactus
    from .pubchem import (
        MoleculeNotFoundError,
        PubChemError,
        check_pubchem_availability,
        classify_query,
        display_2d_structure,
        fetch_molecule,
        fetch_structure,
        generate_2d_structure_svg,
        get_common_molecules,
        get_smiles_examples,
        inchi_to_xyz,
        search_cid_by_inchikey,
        search_cids_by_name,
        search_pubchem_candidates,
        smiles_to_xyz,
        student_friendly_fetch,
        student_friendly_resolve,
        student_friendly_smiles_to_xyz,
        validate_smiles,
    )
    from .structure_providers import (
        ResolvedStructure,
        resolve_structure,
        search_candidates,
    )

    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False

# Visualization — py3Dmol only (no PlotlyMol fallback)
try:
    from .visualization_py3dmol import (
        display_molecule,
        is_visualization_available,
        visualize_molecule,
    )

    VISUALIZATION_AVAILABLE = True
    PY3DMOL_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    PY3DMOL_AVAILABLE = False

# App class, StepProgress, and help_content are resolved lazily via
# module __getattr__ (PEP 562) below. All three unconditionally pull in
# ipywidgets (app.py additionally pulls in the rest of the GUI stack), and
# eagerly importing them here defeats lightweight consumers like
# ``quantui.cli`` that only need pure-Python submodules (calc_log,
# analytics, gpu_offload) — see cli.py's module docstring.
_LAZY_ATTRS = {
    "QuantUIApp": (".app", "QuantUIApp"),
    "StepProgress": (".progress", "StepProgress"),
    "HELP_TOPICS": (".help_content", "HELP_TOPICS"),
    "VALID_TOPICS": (".help_content", "VALID_TOPICS"),
    "help_panel": (".help_content", "help_panel"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    import importlib

    module = importlib.import_module(module_name, __name__)
    return getattr(module, attr_name)


__all__ = [
    # Config constants
    "MOLECULE_LIBRARY",
    "SUPPORTED_METHODS",
    "METHOD_INFO",
    "SUPPORTED_BASIS_SETS",
    "DEFAULT_METHOD",
    "DEFAULT_BASIS",
    "DEFAULT_CHARGE",
    "DEFAULT_MULTIPLICITY",
    "DEFAULT_FMAX",
    "DEFAULT_OPT_STEPS",
    "VALID_ATOMS",
    "QUICK_START_TEMPLATES",
    "WIDGET_LAYOUT",
    "DESCRIPTION_WIDTH",
    "PYSCF_SCRIPT_TEMPLATE",
    # Utils
    "get_username",
    "sanitize_filename",
    "get_session_resources",
    "session_can_handle",
    # Core
    "Molecule",
    "parse_xyz_input",
    "PySCFCalculation",
    "create_calculation",
    # Security
    "SecurityError",
    # UI components
    "help_panel",
    "HELP_TOPICS",
    "VALID_TOPICS",
    "StepProgress",
    # Orbital visualization
    "OrbitalInfo",
    "load_orbital_info",
    "orbital_info_from_arrays",
    "plot_orbital_diagram",
    "orbital_summary_html",
    "parse_cube_file",
    # App class
    "QuantUIApp",
    # Comparison
    "CalcSummary",
    "summary_from_session_result",
    "summary_from_saved_result",
    "comparison_table_html",
    "plot_comparison",
    # ASE bridge (optional)
    "is_ase_available",
    "molecule_to_atoms",
    "atoms_to_molecule",
    "read_structure_file",
    "ase_molecule_library",
    "ASE_AVAILABLE",
    "ASE_MOLECULE_PRESETS",
    # ASE pre-optimization (optional)
    "preoptimize",
    # In-session calculator (optional — Linux/WSL)
    "SessionResult",
    "run_in_session",
    # Frequency analysis (optional — Linux/WSL)
    "FreqResult",
    "run_freq_calc",
    # TD-DFT excited states (optional — Linux/WSL)
    "TDDFTResult",
    "run_tddft_calc",
    # Results persistence
    "save_result",
    "list_results",
    "load_result",
    # QM geometry optimizer (optional — Linux/WSL)
    "OptimizationResult",
    "optimize_geometry",
    # PubChem (optional)
    "fetch_molecule",
    "fetch_structure",
    "classify_query",
    "student_friendly_fetch",
    "student_friendly_resolve",
    "resolve_structure",
    "ResolvedStructure",
    "search_candidates",
    "fetch_from_cactus",
    "inchi_to_xyz",
    "search_cid_by_inchikey",
    "search_cids_by_name",
    "search_pubchem_candidates",
    "get_common_molecules",
    "check_pubchem_availability",
    "PubChemError",
    "MoleculeNotFoundError",
    "PUBCHEM_AVAILABLE",
    "smiles_to_xyz",
    "student_friendly_smiles_to_xyz",
    "generate_2d_structure_svg",
    "display_2d_structure",
    "get_smiles_examples",
    "validate_smiles",
    # Visualization (optional)
    "is_visualization_available",
    "visualize_molecule",
    "display_molecule",
    "VISUALIZATION_AVAILABLE",
    "PY3DMOL_AVAILABLE",
]
