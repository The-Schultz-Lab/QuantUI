"""
QuantUI Package

An open-source frontend for DFT and post-HF quantum chemistry with PySCF.
Calculations run locally in the Jupyter session — no cluster or SLURM required.

PySCF requires Linux/macOS/WSL. Windows users should use the Apptainer container.
"""

__version__ = "0.10.0"

import logging
from typing import Any

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Every public name is resolved lazily (PEP 562 module ``__getattr__``), so
# ``import quantui`` and lightweight entry points (``quantui.cli``,
# ``quantui.backends.*``, the batch worker's argument parsing) load no
# numpy / rdkit / ase / plotly / IPython until a name that needs them is
# used (M-BATCH2 B2.6). On the NCShare login node the eager imports took
# ~10 s and numpy could fail outright against the per-user thread limit.
# ``from quantui import X`` and ``quantui.X`` work as before. Names are not
# cached here, so a test that patches the defining submodule is seen.
_LAZY_ATTRS = {
    # Config constants
    **{
        name: (".config", name)
        for name in (
            "DEFAULT_BASIS",
            "DEFAULT_CHARGE",
            "DEFAULT_FMAX",
            "DEFAULT_METHOD",
            "DEFAULT_MULTIPLICITY",
            "DEFAULT_OPT_STEPS",
            "DESCRIPTION_WIDTH",
            "METHOD_INFO",
            "MOLECULE_LIBRARY",
            "PYSCF_SCRIPT_TEMPLATE",
            "QUICK_START_TEMPLATES",
            "SUPPORTED_BASIS_SETS",
            "SUPPORTED_METHODS",
            "VALID_ATOMS",
            "WIDGET_LAYOUT",
        )
    },
    "PySCFCalculation": (".calculator", "PySCFCalculation"),
    "create_calculation": (".calculator", "create_calculation"),
    **{
        name: (".comparison", name)
        for name in (
            "CalcSummary",
            "comparison_table_html",
            "plot_comparison",
            "summary_from_saved_result",
            "summary_from_session_result",
        )
    },
    "Molecule": (".molecule", "Molecule"),
    "parse_xyz_input": (".molecule", "parse_xyz_input"),
    **{
        name: (".orbital_visualization", name)
        for name in (
            "OrbitalInfo",
            "load_orbital_info",
            "orbital_info_from_arrays",
            "orbital_summary_html",
            "parse_cube_file",
            "plot_orbital_diagram",
        )
    },
    "SecurityError": (".security", "SecurityError"),
    **{
        name: (".utils", name)
        for name in (
            "get_session_resources",
            "get_username",
            "sanitize_filename",
            "session_can_handle",
        )
    },
    **{
        name: (".ase_bridge", name)
        for name in (
            "ase_molecule_library",
            "atoms_to_molecule",
            "is_ase_available",
            "molecule_to_atoms",
            "read_structure_file",
        )
    },
    "preoptimize": (".preopt", "preoptimize"),
    "SessionResult": (".session_calc", "SessionResult"),
    "run_in_session": (".session_calc", "run_in_session"),
    "FreqResult": (".freq_calc", "FreqResult"),
    "run_freq_calc": (".freq_calc", "run_freq_calc"),
    "TDDFTResult": (".tddft_calc", "TDDFTResult"),
    "run_tddft_calc": (".tddft_calc", "run_tddft_calc"),
    "list_results": (".results_storage", "list_results"),
    "load_result": (".results_storage", "load_result"),
    "save_result": (".results_storage", "save_result"),
    "OptimizationResult": (".optimizer", "OptimizationResult"),
    "optimize_geometry": (".optimizer", "optimize_geometry"),
    "PESScanResult": (".pes_scan", "PESScanResult"),
    "run_pes_scan": (".pes_scan", "run_pes_scan"),
    "ReorganizationEnergyResult": (
        ".reorganization_energy",
        "ReorganizationEnergyResult",
    ),
    "run_reorganization_energy": (
        ".reorganization_energy",
        "run_reorganization_energy",
    ),
    "fetch_from_cactus": (".cactus", "fetch_from_cactus"),
    **{
        name: (".pubchem", name)
        for name in (
            "MoleculeNotFoundError",
            "PubChemError",
            "check_pubchem_availability",
            "classify_query",
            "display_2d_structure",
            "fetch_molecule",
            "fetch_structure",
            "generate_2d_structure_svg",
            "get_common_molecules",
            "get_smiles_examples",
            "inchi_to_xyz",
            "search_cid_by_inchikey",
            "search_cids_by_name",
            "search_pubchem_candidates",
            "smiles_to_xyz",
            "student_friendly_fetch",
            "student_friendly_resolve",
            "student_friendly_smiles_to_xyz",
            "validate_smiles",
        )
    },
    **{
        name: (".structure_providers", name)
        for name in ("ResolvedStructure", "resolve_structure", "search_candidates")
    },
    **{
        name: (".visualization_py3dmol", name)
        for name in (
            "display_molecule",
            "is_visualization_available",
            "visualize_molecule",
        )
    },
    # The GUI (ipywidgets and the rest of the stack).
    "QuantUIApp": (".app", "QuantUIApp"),
    "StepProgress": (".progress", "StepProgress"),
    "HELP_TOPICS": (".help_content", "HELP_TOPICS"),
    "VALID_TOPICS": (".help_content", "VALID_TOPICS"),
    "help_panel": (".help_content", "help_panel"),
}

# Availability flags: True when the optional modules import.
_LAZY_FLAGS = {
    "PUBCHEM_AVAILABLE": (".cactus", ".pubchem", ".structure_providers"),
    "VISUALIZATION_AVAILABLE": (".visualization_py3dmol",),
    "PY3DMOL_AVAILABLE": (".visualization_py3dmol",),
}


def __getattr__(name: str) -> Any:
    import importlib

    if name in _LAZY_FLAGS:
        try:
            for module_name in _LAZY_FLAGS[name]:
                importlib.import_module(module_name, __name__)
        except ImportError:
            return False
        return True
    if name in ("ASE_AVAILABLE", "ASE_MOLECULE_PRESETS"):
        try:
            return getattr(importlib.import_module(".ase_bridge", __name__), name)
        except ImportError:
            return False if name == "ASE_AVAILABLE" else {}
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    try:
        module = importlib.import_module(module_name, __name__)
    except ImportError as exc:
        # Optional backends (e.g. PySCF on native Windows): the name is
        # simply not available, as when these were try/except imports.
        raise AttributeError(
            f"quantui.{name} is unavailable: {module_name[1:]} could not be "
            f"imported ({exc})"
        ) from exc
    return getattr(module, attr_name)


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY_ATTRS) | set(_LAZY_FLAGS))


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
    # Reorganization energy — Marcus 4-point (optional — Linux/WSL)
    "ReorganizationEnergyResult",
    "run_reorganization_energy",
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
