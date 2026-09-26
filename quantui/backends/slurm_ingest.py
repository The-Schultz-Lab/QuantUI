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

from .registry import JobRecord, JobRegistry, parse_attempt_dir_name
from .worker_payload import molecule_from_dict

# Sidecar files the batch worker may write for Analysis replay parity.
_STAGING_SIDECAR_FILES = (
    "orbitals.npz",
    "orbitals_meta.json",
    "result.molden",
    "trajectory.xyz",
    "trajectory.traj",
    "preopt_trajectory.json",
)


def slurm_provenance(record: JobRecord, staging: Path) -> dict[str, Any]:
    """``result.json`` extras that mark a History entry as a SLURM run.

    M-JOBDIRS JD.5: stored under ``execution_backend`` / ``slurm`` so the
    History list and card can tell SLURM results from local ones. Additive —
    local results and results ingested before this simply lack the keys.
    For a job-dir record, *staging* is the attempt dir and its name gives the
    attempt number and the SLURM job id that produced it (which, for a
    hand-run ``sbatch``, differs from the record's tracked job).
    """
    parsed = parse_attempt_dir_name(staging.name) if record.job_dir else None
    return {
        "execution_backend": "slurm",
        "slurm": {
            "job_id": parsed[1] if parsed else record.slurm_job_id,
            "attempt": parsed[0] if parsed else None,
            "job_dir": record.job_dir or record.staging_dir,
            "attempt_dir": str(staging) if parsed else None,
            "request_id": record.request_id,
        },
    }


def _copy_staging_sidecars(
    staging: Path, saved_dir: Path, payload: dict[str, Any] | None = None
) -> None:
    # Attempt dirs may hold shareable files under descriptive names
    # (result.json["artifact_names"], M-JOBDIRS JD.7); History keeps the
    # fixed names its loaders expect.
    renamed = (payload or {}).get("artifact_names") or {}
    for name in _STAGING_SIDECAR_FILES:
        src = staging / renamed.get(name, name)
        if src.exists():
            shutil.copy2(src, saved_dir / name)


def _finalize_history_entry(saved_dir: Path) -> None:
    """Generate History dropdown thumbnail (matches local ``_do_run``)."""
    from quantui.results_storage import load_result, save_thumbnail

    try:
        save_thumbnail(saved_dir, load_result(saved_dir))
    except Exception:
        pass


def _basic_result(payload: dict[str, Any], record: JobRecord) -> SimpleNamespace:
    """Reconstruct a result-alike object from a worker's staging JSON.

    AUDIT F12 — this used to reconstruct only 7 generic fields even though
    ``session_result_payload()`` (and the other ``*_result_payload``
    builders) already serialize Mulliken charges, dipole, atom symbols,
    SCF variant/rescue provenance, post-HF correlation breakdown, and
    solvent/GPU/density-fit metadata. ``save_result()`` reads every one of
    these via ``getattr(result, ..., default)``, so silently omitting them
    here made ``save_result`` write them as null regardless of whether the
    JSON actually had real values — a real water round trip lost the
    dipole, charges, atom symbols, and RHF provenance.

    Every field below is read defensively (``.get`` with no required key)
    because not every calc type's payload builder sets every field —
    absent ones round-trip as the same ``None``/default ``save_result``
    already treats as "not applicable for this calc type", matching its
    own documented contract.
    """
    return SimpleNamespace(
        energy_hartree=float(payload.get("energy_hartree", float("nan"))),
        homo_lumo_gap_ev=payload.get("homo_lumo_gap_ev"),
        converged=bool(payload.get("converged", False)),
        n_iterations=int(payload.get("n_iterations", -1)),
        method=str(payload.get("method", record.request_obj.method)),
        basis=str(payload.get("basis", record.request_obj.basis)),
        formula=str(payload.get("formula", "?")),
        mulliken_charges=payload.get("mulliken_charges"),
        dipole_moment_debye=payload.get("dipole_moment_debye"),
        dipole_vector_debye=payload.get("dipole_vector_debye"),
        atom_symbols=payload.get("atom_symbols"),
        scf_rescue_stage=payload.get("scf_rescue_stage", "none"),
        scf_variant=payload.get("scf_variant") or None,
        mp2_correlation_hartree=payload.get("mp2_correlation_hartree"),
        ccsd_correlation_hartree=payload.get("ccsd_correlation_hartree"),
        ccsd_t_correction_hartree=payload.get("ccsd_t_correction_hartree"),
        cc_converged=payload.get("cc_converged"),
        td_converged=payload.get("td_converged"),
        n_converged_states=payload.get("n_converged_states"),
        dispersion_applied=payload.get("dispersion_applied"),
        solvent=payload.get("solvent"),
        gpu_used=bool(payload.get("gpu_used", False)),
        gpu_name=payload.get("gpu_name"),
        density_fit=bool(payload.get("density_fit", False)),
    )


def _copy_trajectory(staging: Path, saved_dir: Path, payload: dict[str, Any]) -> None:
    traj_name = payload.get("trajectory_file", "trajectory.json")
    traj_src = staging / traj_name
    if traj_src.exists():
        shutil.copy2(traj_src, saved_dir / "trajectory.json")


def _ingest_frequency(
    staging: Path,
    payload: dict[str, Any],
    record: JobRecord,
    log_text: str,
    extras: dict[str, Any],
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
        extras=extras,
    )
    ir = spectra.get("ir") or {}
    freqs = ir.get("frequencies_cm1")
    displacements = ir.get("displacements")
    _copy_staging_sidecars(staging, saved_dir, payload)
    if not (saved_dir / "result.molden").exists() and freqs and displacements:
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
    _finalize_history_entry(saved_dir)
    return saved_dir


def _ingest_reorganization_energy(
    payload: dict[str, Any],
    record: JobRecord,
    log_text: str,
    extras: dict[str, Any],
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
    saved_dir = save_result(
        result,
        pyscf_log=log_text,
        calc_type="reorganization_energy",
        spectra=payload.get("spectra") or {},
        extras=extras,
    )
    _finalize_history_entry(saved_dir)
    return saved_dir


def _ingest_with_sidecars(
    staging: Path,
    saved_dir: Path,
    payload: dict[str, Any],
) -> Path:
    if calc_type := payload.get("calc_type"):
        if calc_type in ("geometry_opt", "pes_scan"):
            _copy_trajectory(staging, saved_dir, payload)
    _copy_staging_sidecars(staging, saved_dir, payload)
    _finalize_history_entry(saved_dir)
    return saved_dir


def ingest_staging_success(
    record: JobRecord, log_text: str = "", *, attempt_dir: Path | None = None
) -> Path:
    """Read ``result.json`` from staging and save under ``results/``.

    *attempt_dir* selects a specific attempt (used for hand-run attempts the
    record does not track); by default the record's tracked staging path.
    """
    staging = attempt_dir if attempt_dir is not None else record.staging_path
    extras = slurm_provenance(record, staging)
    result_path = staging / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing staging result: {result_path}")

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    calc_type = payload.get("calc_type") or record.calc_type

    from quantui import save_result

    if calc_type == "frequency":
        return _ingest_frequency(staging, payload, record, log_text, extras)

    if calc_type == "reorganization_energy":
        return _ingest_reorganization_energy(payload, record, log_text, extras)

    result = _basic_result(payload, record)
    spectra = payload.get("spectra")
    saved_dir = save_result(
        result,
        pyscf_log=log_text,
        calc_type=calc_type,
        spectra=spectra if spectra is not None else {},
        extras=extras,
    )

    return _ingest_with_sidecars(staging, saved_dir, payload)


def already_ingested(record: JobRecord, staging: Path) -> bool:
    """True when the run in *staging* has already been saved to History.

    Job-dir records list ingested attempt dirs by name. Legacy records
    predate that list; for them, a ``result_dir`` that points at an existing
    History entry (the worker sets it to the staging dir; the app replaces
    it with the saved dir after ingest) means the tracked run was ingested.
    """
    if staging.name in record.ingested_attempts:
        return True
    if record.job_dir is None and record.result_dir:
        saved = Path(record.result_dir)
        return saved != staging and (saved / "result.json").exists()
    return False


def uningested_attempts(record: JobRecord) -> list[Path]:
    """Finished attempt dirs of a job-dir record not yet saved to History.

    ``result.json`` is written only when a run succeeds, so its presence
    marks a finished, successful attempt (M-JOBDIRS: successes only).
    Includes hand-run ``sbatch submit.slurm`` attempts the record never
    tracked (JD.11). Legacy records return ``[]``: their single run is
    ingested by the monitor / reconnect path, as before.
    """
    if record.job_dir is None:
        return []
    return [
        d
        for d in record.attempt_dirs()
        if (d / "result.json").exists() and not already_ingested(record, d)
    ]


def ingest_attempt(
    registry: JobRegistry, record: JobRecord, attempt_dir: Path | None = None
) -> Path:
    """Save one attempt to History and record it as ingested.

    *attempt_dir* defaults to the record's tracked staging path. The
    record's ``result_dir`` is updated only for the tracked attempt, so it
    keeps pointing at the History entry of the run the record reports on.
    """
    staging = attempt_dir if attempt_dir is not None else record.staging_path
    log_path = staging / "live.log"
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )
    saved_dir = ingest_staging_success(record, log_text, attempt_dir=staging)
    tracked = staging == record.staging_path
    registry.mark_ingested(
        record.request_id,
        staging.name,
        result_dir=str(saved_dir) if tracked else None,
    )
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
