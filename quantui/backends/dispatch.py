"""
Helpers for building backend requests from QuantUI app state.
"""

from __future__ import annotations

import shutil
from typing import Any
from uuid import uuid4

from .base import CalculationRequest
from .slurm import SlurmBackend


def is_slurm_available() -> bool:
    """Return True when ``sbatch`` is on PATH."""
    return shutil.which("sbatch") is not None


def calc_type_key_from_app(app: Any) -> str:
    from quantui.app_runflow import calc_type_key

    return calc_type_key(app)


def build_calculation_request(
    app: Any, *, request_id: str | None = None
) -> CalculationRequest:
    """Serialize the current Calculate-tab configuration into a backend request."""
    mol = app._molecule
    if mol is None:
        raise ValueError("No molecule loaded")

    calc_type = calc_type_key_from_app(app)
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

    return CalculationRequest(
        request_id=request_id or uuid4().hex[:12],
        calc_type=calc_type,
        method=app.method_dd.value,
        basis=app.basis_dd.value,
        charge=int(mol.charge),
        multiplicity=int(app.mult_si.value),
        molecule={
            "atoms": list(mol.atoms),
            "coordinates": [list(c) for c in mol.coordinates],
            "label": mol.get_formula(),
            "charge": int(mol.charge),
            "multiplicity": int(app.mult_si.value),
        },
        options=options,
        solvent=solvent,
    )


def slurm_backend_for_app(app: Any) -> SlurmBackend:
    """Construct a :class:`SlurmBackend` sharing the app's job registry."""
    registry = getattr(app, "_job_registry", None)
    return SlurmBackend(registry=registry)
