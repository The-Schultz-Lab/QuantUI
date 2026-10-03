"""
Build ``CalculationRequest`` objects for ``quantui submit`` from files.

``quantui submit`` started out (CL2.7) taking only hand-written
CalculationRequest JSON. Writing atoms and coordinates as nested JSON arrays
is the hardest part of running QuantUI from a terminal, so an ``.xyz`` file
plus a few flags is accepted too: the same request the Calculate tab would
build, without the app.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import CALC_TYPES, CalculationRequest

# Calc types whose worker runner honours options["preopt_before_run"]
# (single point and NMR since ISSUE.19 #3).
PREOPT_CALC_TYPES = frozenset({"single_point", "nmr", "frequency", "tddft", "pes_scan"})
# Calc types the batch worker runs with a PCM solvent; it fails any other
# calc type that asks for one (worker._SOLVENT_SUPPORTED_CALC_TYPES), so a
# request is refused here, before it waits in the queue. Frequency and
# TD-DFT joined with B2.2 (DEC-023); NMR and PES scans stay gas-phase.
SOLVENT_CALC_TYPES = frozenset(
    {"single_point", "geometry_opt", "frequency", "tddft", "reorganization_energy"}
)


class BatchInputError(ValueError):
    """An input file or flag combination that cannot become a request."""


def parse_option_pairs(pairs: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Turn ``KEY=VALUE`` strings into an options dict.

    Values are read as JSON when they parse (``nstates=10`` -> 10,
    ``atom_indices=[0,1]`` -> list, ``x=true`` -> True) and kept as strings
    otherwise (``scan_type=bond``).
    """
    options: Dict[str, Any] = {}
    for pair in pairs or ():
        key, sep, raw = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise BatchInputError(f"--option expects KEY=VALUE, got {pair!r}")
        try:
            options[key] = json.loads(raw)
        except json.JSONDecodeError:
            options[key] = raw
    return options


def _file_label(path: Path) -> str:
    """The file's stem, made safe for ids and folder names."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_.-") or "molecule"


def _request_id_for(path: Path) -> str:
    return f"{_file_label(path)[:40]}-{uuid.uuid4().hex[:8]}"


def request_from_xyz(
    path: Path,
    *,
    calc_type: Optional[str],
    method: str,
    basis: str,
    charge: int = 0,
    multiplicity: int = 1,
    solvent: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    preopt: bool = False,
) -> Tuple[CalculationRequest, List[str]]:
    """Build a request from an XYZ file; return it with any warnings.

    Raises :class:`BatchInputError` for anything the worker would reject:
    a missing or unknown calc type, an unreadable XYZ, or a charge and
    multiplicity that cannot both be true for this electron count.
    """
    from quantui.xyz_input import load_molecule_from_xyz_text

    if not calc_type:
        raise BatchInputError(
            f"{path.name}: an .xyz input needs --calc " f"({', '.join(CALC_TYPES)})."
        )
    if calc_type not in CALC_TYPES:
        raise BatchInputError(
            f"unknown --calc {calc_type!r}; choose one of {', '.join(CALC_TYPES)}."
        )

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BatchInputError(f"{path}: could not read file — {exc}") from exc
    try:
        mol, spin_note = load_molecule_from_xyz_text(
            text, charge=charge, multiplicity=multiplicity
        )
    except Exception as exc:  # noqa: BLE001 — parser errors are user input errors
        raise BatchInputError(f"{path}: could not read XYZ — {exc}") from exc
    if spin_note:
        raise BatchInputError(
            f"{path.name}: charge {charge} and multiplicity {multiplicity} "
            f"do not fit this molecule. {spin_note}"
        )

    warnings: List[str] = []
    from quantui import config

    known_methods = {m.upper() for m in config.SUPPORTED_METHODS}
    if method.upper() not in known_methods:
        warnings.append(
            f"method {method!r} is not in QuantUI's method list; PySCF may still "
            "accept it, but check the spelling."
        )

    opts = dict(options or {})
    if preopt:
        if calc_type in PREOPT_CALC_TYPES:
            opts["preopt_before_run"] = True
        else:
            warnings.append(
                f"--preopt has no effect for {calc_type} "
                f"(only {', '.join(sorted(PREOPT_CALC_TYPES))})."
            )

    request = CalculationRequest(
        request_id=_request_id_for(path),
        calc_type=calc_type,
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        molecule={
            "atoms": list(mol.atoms),
            "coords": [[float(c) for c in row] for row in mol.coordinates],
            # Names the job folder (default_job_name), e.g. water_opt_B3LYP_def2-SVP.
            "label": _file_label(path),
            "charge": charge,
            "multiplicity": multiplicity,
        },
        options=opts,
        solvent=solvent,
        run_context={"source_file": path.name},
    )
    return request, warnings


def check_solvent(request: CalculationRequest) -> None:
    """Normalise ``request.solvent`` to QuantUI's spelling, or refuse it."""
    if not request.solvent:
        request.solvent = None
        return
    from quantui import config

    names = {name.lower(): name for name in config.SOLVENT_OPTIONS}
    canonical = names.get(str(request.solvent).lower())
    if canonical is None:
        raise BatchInputError(
            f"unknown solvent {request.solvent!r}; choose one of "
            f"{', '.join(sorted(config.SOLVENT_OPTIONS))}."
        )
    if request.calc_type not in SOLVENT_CALC_TYPES:
        raise BatchInputError(
            f"a solvent works only with {', '.join(sorted(SOLVENT_CALC_TYPES))} "
            f"in batch jobs, not {request.calc_type}; drop --solvent."
        )
    request.solvent = canonical


def load_request(
    path: Path,
    *,
    calc_type: Optional[str] = None,
    method: Optional[str] = None,
    basis: Optional[str] = None,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    solvent: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    preopt: bool = False,
) -> Tuple[CalculationRequest, List[str]]:
    """Read *path* (``.xyz`` or request JSON) into a request.

    For JSON, flags left as ``None`` keep the file's values; flags that are
    given override them. For XYZ, unset method/basis fall back to the app's
    defaults and charge/multiplicity to a neutral singlet.
    """
    if path.suffix.lower() == ".xyz":
        from quantui import config

        request, warnings = request_from_xyz(
            path,
            calc_type=calc_type,
            method=method or config.DEFAULT_METHOD,
            basis=basis or config.DEFAULT_BASIS,
            charge=0 if charge is None else charge,
            multiplicity=1 if multiplicity is None else multiplicity,
            solvent=solvent,
            options=options,
            preopt=preopt,
        )
        check_solvent(request)
        return request, warnings

    data = json.loads(path.read_text())
    request = CalculationRequest.from_dict(data)
    warnings = []  # same List[str] as the .xyz branch above
    if calc_type is not None:
        if calc_type not in CALC_TYPES:
            raise BatchInputError(
                f"unknown --calc {calc_type!r}; choose one of {', '.join(CALC_TYPES)}."
            )
        request.calc_type = calc_type
    if method is not None:
        request.method = method
    if basis is not None:
        request.basis = basis
    if charge is not None:
        request.charge = charge
        request.molecule["charge"] = charge
    if multiplicity is not None:
        request.multiplicity = multiplicity
        request.molecule["multiplicity"] = multiplicity
    if solvent is not None:
        request.solvent = solvent
    if options:
        request.options.update(options)
    if preopt and request.calc_type in PREOPT_CALC_TYPES:
        request.options["preopt_before_run"] = True
    check_solvent(request)
    return request, warnings
