"""
Promote finished SLURM staging artifacts into History-compatible result dirs.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .registry import JobRecord
from .worker_payload import molecule_from_dict


def _basic_result(payload: dict[str, Any], record: JobRecord) -> SimpleNamespace:
    return SimpleNamespace(
        energy_hartree=float(payload.get("energy_hartree", float("nan"))),
        homo_lumo_gap_ev=payload.get("homo_lumo_gap_ev"),
        converged=bool(payload.get("converged", False)),
        n_iterations=int(payload.get("n_iterations", -1)),
        method=str(payload.get("method", record.request_obj.method)),
        basis=str(payload.get("basis", record.request_obj.basis)),
        formula=str(payload.get("formula", "?")),
    )


def _copy_trajectory(staging: Path, saved_dir: Path, payload: dict[str, Any]) -> None:
    traj_name = payload.get("trajectory_file", "trajectory.json")
    traj_src = staging / traj_name
    if traj_src.exists():
        shutil.copy2(traj_src, saved_dir / "trajectory.json")


def _ingest_frequency(
    staging: Path, payload: dict[str, Any], record: JobRecord, log_text: str
) -> Path:
    from quantui import save_result
    from quantui.results_storage import save_molden

    result = _basic_result(payload, record)
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


def _ingest_reorganization_energy(
    payload: dict[str, Any], record: JobRecord, log_text: str
) -> Path:
    from quantui import save_result

    neutral_geom = payload.get("neutral_geometry") or {}
    neutral_mol = (
        molecule_from_dict(neutral_geom) if neutral_geom.get("atoms") else None
    )
    channels = []
    for ch_data in payload.get("channels") or []:
        ion_mol = None
        ion_geom = ch_data.get("ion_geometry")
        if ion_geom:
            ion_mol = molecule_from_dict(ion_geom)
        channels.append(
            SimpleNamespace(
                kind=ch_data["kind"],
                ion_charge=ch_data["ion_charge"],
                ion_multiplicity=ch_data["ion_multiplicity"],
                e_neutral_at_neutral=ch_data["e_neutral_at_neutral"],
                e_ion_at_ion=ch_data["e_ion_at_ion"],
                e_ion_at_neutral=ch_data["e_ion_at_neutral"],
                e_neutral_at_ion=ch_data["e_neutral_at_ion"],
                lambda1_hartree=ch_data["lambda1_hartree"],
                lambda2_hartree=ch_data["lambda2_hartree"],
                lambda_hartree=ch_data["lambda_hartree"],
                converged=ch_data["converged"],
                ion_molecule=ion_mol,
            )
        )
    result = SimpleNamespace(
        formula=str(payload.get("formula", "?")),
        method=str(payload.get("method", record.request_obj.method)),
        basis=str(payload.get("basis", record.request_obj.basis)),
        energy_hartree=float(payload["energy_hartree"]),
        converged=bool(payload.get("converged", False)),
        n_total_opt_steps=int(payload.get("n_iterations", 0)),
        molecule=neutral_mol,
        channels=channels,
    )
    return save_result(
        result,
        pyscf_log=log_text,
        calc_type="reorganization_energy",
        spectra=payload.get("spectra") or {},
    )


def ingest_staging_success(record: JobRecord, log_text: str = "") -> Path:
    """Read ``result.json`` from staging and save under ``results/``."""
    staging = record.staging_path
    result_path = staging / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing staging result: {result_path}")

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    calc_type = payload.get("calc_type") or record.calc_type

    from quantui import save_result

    if calc_type == "frequency":
        return _ingest_frequency(staging, payload, record, log_text)

    if calc_type == "reorganization_energy":
        return _ingest_reorganization_energy(payload, record, log_text)

    result = _basic_result(payload, record)
    spectra = payload.get("spectra")
    saved_dir = save_result(
        result,
        pyscf_log=log_text,
        calc_type=calc_type,
        spectra=spectra if spectra is not None else {},
    )

    if calc_type in ("geometry_opt", "pes_scan"):
        _copy_trajectory(staging, saved_dir, payload)

    return saved_dir


def completion_summary_html(saved_dir: Path, payload: dict[str, Any]) -> str:
    calc_type = payload.get("calc_type", "single_point")
    energy = float(payload.get("energy_hartree", float("nan")))
    label = calc_type.replace("_", " ")
    lines = [
        f"<b>SLURM calculation complete</b> ({label})<br>",
    ]
    if calc_type != "nmr" and math.isfinite(energy):
        lines.append(f"Energy: {energy:.6f} Ha<br>")
    if calc_type == "geometry_opt":
        lines.append(
            f"Steps: {payload.get('n_steps', '?')} — "
            f"converged: {'yes' if payload.get('converged') else 'no'}<br>"
        )
    if calc_type == "frequency":
        freqs = (payload.get("spectra") or {}).get("ir", {}).get(
            "frequencies_cm1"
        ) or []
        lines.append(f"Modes: {len(freqs)}<br>")
    if calc_type == "tddft":
        excitations = (payload.get("spectra") or {}).get("uv_vis", {}).get(
            "excitation_energies_ev"
        ) or []
        lines.append(f"Excited states: {len(excitations)}<br>")
    if calc_type == "nmr":
        nmr = (payload.get("spectra") or {}).get("nmr", {})
        shifts = nmr.get("chemical_shifts_ppm") or {}
        lines.append(f"Shift entries: {len(shifts)}<br>")
    if calc_type == "pes_scan":
        pes = (payload.get("spectra") or {}).get("pes_scan", {})
        points = pes.get("scan_parameter_values") or []
        lines.append(f"Scan points: {len(points)}<br>")
    if calc_type == "reorganization_energy":
        channels = (
            (payload.get("spectra") or {})
            .get("reorganization_energy", {})
            .get("channels")
            or payload.get("channels")
            or []
        )
        lines.append(f"Channels: {len(channels)}<br>")
    lines.append(f"Saved: <code>{saved_dir}</code>")
    return (
        '<div style="padding:12px;background:#ecfdf5;border-radius:8px;">'
        + "".join(lines)
        + "</div>"
    )
