"""
Helpers for building backend requests from QuantUI app state.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from .base import CalculationRequest
from .slurm import SlurmBackend

# Calc types that can consume a History seed geometry (shared ``_seed_dd`` widget).
_SEED_CALC_TYPES = frozenset({"geometry_opt", "frequency", "tddft", "nmr"})
# Calc types that may run a DFT geometry optimization before the main step.
_PREOPT_CALC_TYPES = frozenset({"frequency", "tddft"})


def is_slurm_available() -> bool:
    """Return True when ``sbatch`` is on PATH."""
    return shutil.which("sbatch") is not None


def calc_type_key_from_app(app: Any) -> str:
    from quantui.app_runflow import calc_type_key

    return calc_type_key(app)


def seed_path_from_app(app: Any) -> str | None:
    """Return the selected History seed result dir, if any."""
    seed_dd = getattr(app, "_seed_dd", None)
    if seed_dd is None:
        return None
    path = getattr(seed_dd, "value", "") or ""
    return path or None


def load_seed_molecule(seed_path: str | Path):
    """Load the final frame from a saved geometry-opt ``trajectory.json``."""
    from quantui.results_storage import load_trajectory

    trajectory, _energies = load_trajectory(Path(seed_path))
    if not trajectory:
        raise ValueError(f"Seed trajectory is empty: {seed_path}")
    return trajectory[-1]


def molecule_dict_from_molecule(mol, *, multiplicity: int) -> dict[str, Any]:
    return {
        "atoms": list(mol.atoms),
        "coordinates": [list(c) for c in mol.coordinates],
        "label": mol.get_formula(),
        "charge": int(mol.charge),
        "multiplicity": int(multiplicity),
    }


def build_calculation_request(
    app: Any, *, request_id: str | None = None
) -> CalculationRequest:
    """Serialize the current Calculate-tab configuration into a backend request."""
    mol = app._molecule
    if mol is None:
        raise ValueError("No molecule loaded")

    calc_type = calc_type_key_from_app(app)
    multiplicity = int(app.mult_si.value)
    solvent = app.solvent_dd.value if app.solvent_cb.value else None

    options: dict[str, object] = {}
    if calc_type in ("geometry_opt", "reorganization_energy"):
        options["fmax"] = float(app.fmax_fi.value)
        options["max_steps"] = int(app.max_steps_si.value)
    if calc_type == "tddft":
        options["nstates"] = int(app.nstates_si.value)
    if calc_type == "reorganization_energy":
        options["mode"] = str(app._reorg_mode_dd.value)
    if calc_type == "pes_scan":
        scan_type = str(app._scan_type_dd.value).lower()
        atom_indices = [
            int(app._scan_atom1.value) - 1,
            int(app._scan_atom2.value) - 1,
        ]
        if scan_type in ("angle", "dihedral"):
            atom_indices.append(int(app._scan_atom3.value) - 1)
        if scan_type == "dihedral":
            atom_indices.append(int(app._scan_atom4.value) - 1)
        options.update(
            {
                "scan_type": scan_type,
                "atom_indices": atom_indices,
                "start": float(app._scan_start.value),
                "stop": float(app._scan_stop.value),
                "steps": int(app._scan_steps.value),
            }
        )

    run_context: dict[str, object] = {}
    seed_path = seed_path_from_app(app)
    calc_mol = mol
    if seed_path and calc_type in _SEED_CALC_TYPES:
        calc_mol = load_seed_molecule(seed_path)
        run_context["seed_result_dir"] = str(seed_path)
        run_context["seed_label"] = Path(seed_path).name

    if calc_type in _PREOPT_CALC_TYPES and not seed_path:
        preopt_cb = getattr(app, "_freq_preopt_cb", None)
        if preopt_cb is not None and bool(preopt_cb.value):
            options["preopt_before_run"] = True

    return CalculationRequest(
        request_id=request_id or uuid4().hex[:12],
        calc_type=calc_type,
        method=app.method_dd.value,
        basis=app.basis_dd.value,
        charge=int(calc_mol.charge),
        multiplicity=multiplicity,
        molecule=molecule_dict_from_molecule(calc_mol, multiplicity=multiplicity),
        options=options,
        solvent=solvent,
        run_context=run_context,
    )


def slurm_backend_for_app(app: Any) -> SlurmBackend:
    """Construct a :class:`SlurmBackend` sharing the app's job registry."""
    registry = getattr(app, "_job_registry", None)
    return SlurmBackend(registry=registry)
