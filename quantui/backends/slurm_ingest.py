"""
Promote finished SLURM staging artifacts into History-compatible result dirs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .registry import JobRecord


def ingest_staging_success(record: JobRecord, log_text: str = "") -> Path:
    """Read ``result.json`` from staging and save under ``results/``."""
    staging = record.staging_path
    result_path = staging / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing staging result: {result_path}")

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    calc_type = payload.get("calc_type") or record.calc_type

    from quantui import save_result

    if calc_type == "single_point":
        result = SimpleNamespace(
            energy_hartree=float(payload["energy_hartree"]),
            homo_lumo_gap_ev=payload.get("homo_lumo_gap_ev"),
            converged=bool(payload.get("converged", False)),
            n_iterations=int(payload.get("n_iterations", -1)),
            method=str(payload.get("method", record.request_obj.method)),
            basis=str(payload.get("basis", record.request_obj.basis)),
            formula=str(payload.get("formula", "?")),
        )
        saved_dir = save_result(
            result,
            pyscf_log=log_text,
            calc_type="single_point",
        )
        return saved_dir

    if calc_type == "geometry_opt":
        result = SimpleNamespace(
            energy_hartree=float(payload["energy_hartree"]),
            homo_lumo_gap_ev=payload.get("homo_lumo_gap_ev"),
            converged=bool(payload.get("converged", False)),
            n_iterations=int(payload.get("n_iterations", payload.get("n_steps", -1))),
            method=str(payload.get("method", record.request_obj.method)),
            basis=str(payload.get("basis", record.request_obj.basis)),
            formula=str(payload.get("formula", "?")),
        )
        saved_dir = save_result(
            result,
            pyscf_log=log_text,
            calc_type="geometry_opt",
        )
        traj_name = payload.get("trajectory_file", "trajectory.json")
        traj_src = staging / traj_name
        if traj_src.exists():
            shutil.copy2(traj_src, saved_dir / "trajectory.json")
        return saved_dir

    if calc_type == "frequency":
        result = SimpleNamespace(
            energy_hartree=float(payload["energy_hartree"]),
            homo_lumo_gap_ev=payload.get("homo_lumo_gap_ev"),
            converged=bool(payload.get("converged", False)),
            n_iterations=int(payload.get("n_iterations", -1)),
            method=str(payload.get("method", record.request_obj.method)),
            basis=str(payload.get("basis", record.request_obj.basis)),
            formula=str(payload.get("formula", "?")),
        )
        spectra = payload.get("spectra") or {}
        saved_dir = save_result(
            result,
            pyscf_log=log_text,
            calc_type="frequency",
            spectra=spectra,
        )
        ir = spectra.get("ir") or {}
        freqs = ir.get("frequencies_cm1")
        displacements = ir.get("displacements")
        if freqs and displacements:
            from quantui.results_storage import save_molden

            mol_block = spectra.get("molecule") or {}
            atoms = mol_block.get("atoms") or []
            coords = mol_block.get("coords") or []
            pyscf_mol_atom = [[sym, coord] for sym, coord in zip(atoms, coords)]
            save_molden(
                saved_dir,
                pyscf_mol_atom=pyscf_mol_atom,
                pyscf_mol_basis=str(payload.get("basis", record.request_obj.basis)),
                charge=int(mol_block.get("charge", record.request_obj.charge)),
                multiplicity=int(
                    mol_block.get("multiplicity", record.request_obj.multiplicity)
                ),
                frequencies_cm1=freqs,
                normal_modes=displacements,
            )
        return saved_dir

    raise ValueError(f"Unsupported SLURM ingest calc_type={calc_type!r}")


def completion_summary_html(saved_dir: Path, payload: dict[str, Any]) -> str:
    calc_type = payload.get("calc_type", "single_point")
    energy = float(payload.get("energy_hartree", float("nan")))
    lines = [
        f"<b>SLURM calculation complete</b> ({calc_type.replace('_', ' ')})<br>",
        f"Energy: {energy:.6f} Ha<br>",
        f"Saved: <code>{saved_dir}</code>",
    ]
    if calc_type == "geometry_opt":
        lines.insert(
            2,
            f"Steps: {payload.get('n_steps', '?')} — "
            f"converged: {'yes' if payload.get('converged') else 'no'}<br>",
        )
    if calc_type == "frequency":
        freqs = (payload.get("spectra") or {}).get("ir", {}).get(
            "frequencies_cm1"
        ) or []
        lines.insert(2, f"Modes: {len(freqs)}<br>")
    return (
        '<div style="padding:12px;background:#ecfdf5;border-radius:8px;">'
        + "".join(lines)
        + "</div>"
    )
