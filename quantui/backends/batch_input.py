"""
Build ``CalculationRequest`` objects for ``quantui submit`` from files.

``quantui submit`` started out taking only hand-written
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

# Calc types whose worker runner honours options["preopt_before_run"].
PREOPT_CALC_TYPES = frozenset({"single_point", "nmr", "frequency", "tddft", "pes_scan"})
# Calc types the batch worker runs with a PCM solvent; it fails any other
# calc type that asks for one (worker._SOLVENT_SUPPORTED_CALC_TYPES), so a
# request is refused here, before it waits in the queue. NMR and PES scans
# stay gas-phase.
SOLVENT_CALC_TYPES = frozenset(
    {
        "single_point",
        "geometry_opt",
        "transition_state",
        "irc",
        "frequency",
        "tddft",
        "reorganization_energy",
    }
)
# Calc types driven by analytic gradients: a post-HF method (no gradients
# here) is refused before the job queues, not after it starts.
GRADIENT_CALC_TYPES = frozenset({"transition_state", "irc"})
# The request options each calc type's worker runner reads
# (backends/worker.py); any other key is stored in request.json and ignored,
# so it gets a warning at submit. ``preopt_before_run`` is added for
# PREOPT_CALC_TYPES below.
CALC_OPTION_KEYS: Dict[str, frozenset] = {
    "single_point": frozenset(),
    "geometry_opt": frozenset({"fmax", "max_steps", "frozen_atoms"}),
    "transition_state": frozenset({"fmax", "max_steps"}),
    "irc": frozenset({"max_steps", "dx", "fmax"}),
    "frequency": frozenset(),
    "tddft": frozenset({"nstates"}),
    "nmr": frozenset(),
    "pes_scan": frozenset(
        {
            "scan_type",
            "atom_indices",
            "start",
            "stop",
            "steps",
            "grid",
            "fmax",
            "max_opt_steps",
        }
    ),
    "reorganization_energy": frozenset({"mode", "fmax", "max_steps"}),
}
# Options the worker reads with bool(): the string "no" would be True.
BOOLEAN_OPTION_KEYS = frozenset({"scf_rescue", "preopt_before_run"})
_TRUE_WORDS = frozenset({"true", "yes", "on", "1"})
_FALSE_WORDS = frozenset({"false", "no", "off", "0"})
# A request id names the job record file, so no path separators or "..".
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")


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
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    # A ".." in the stem would be refused as a path component later.
    return re.sub(r"\.{2,}", ".", label).strip("_.-") or "molecule"


def _request_id_for(path: Path) -> str:
    return f"{_file_label(path)[:40]}-{uuid.uuid4().hex[:8]}"


def xyz_header_atom_count(text: str) -> Optional[int]:
    """The atom count on an XYZ file's first line, if it has one.

    Same rule as ``molecule.parse_xyz_input``: the first line that is not
    blank and not a ``#``/``!`` comment, if it is a bare integer.
    """
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", "!")):
            continue
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def xyz_count_mismatch_message(name: str, declared: int, listed: int) -> str:
    return (
        f"{name}: the first line says {declared} atoms but the file lists "
        f"{listed}. The file may be cut off or have extra lines; fix the count "
        "line, or delete the first two lines to submit the atoms as listed."
    )


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
    declared = xyz_header_atom_count(text)
    if declared is not None and declared != len(mol.atoms):
        # parse_xyz_input already rejects a header/atom-count mismatch;
        # kept as a backstop, since an unattended job would run a
        # truncated or padded file without anyone seeing it.
        # Checked before the spin check: a cut-off file is the real cause
        # of the odd electron count it usually also has.
        raise BatchInputError(
            xyz_count_mismatch_message(path.name, declared, len(mol.atoms))
        )
    if spin_note:
        raise BatchInputError(
            f"{path.name}: charge {charge} and multiplicity {multiplicity} "
            f"do not fit this molecule. {spin_note}"
        )

    warnings: List[str] = []

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


def check_method(request: CalculationRequest) -> None:
    """Normalise the method's spelling, or refuse one the worker cannot run.

    The worker fails any method outside ``config.SUPPORTED_METHODS``
    (``session_calc``, ``calculator``), so an unknown one ("HF", a typo) is
    refused here, before it waits in the queue; so is a post-HF method for a
    gradient-driven calc type (a TS search or IRC).
    """
    from quantui import config

    names = {m.upper(): m for m in config.SUPPORTED_METHODS}
    given = request.method.strip()
    canonical = names.get(given.upper())
    if canonical is None:
        hint = (
            " (Hartree-Fock is RHF for closed shells, UHF for open shells)"
            if given.upper() == "HF"
            else ""
        )
        raise BatchInputError(
            f"unknown method {given!r}{hint}; choose one of "
            f"{', '.join(config.SUPPORTED_METHODS)}."
        )
    request.method = canonical
    if request.calc_type not in GRADIENT_CALC_TYPES:
        return
    if canonical.upper() in config.POST_HF_METHODS:
        raise BatchInputError(
            f"{request.method} has no analytic gradients here, so it cannot drive "
            f"a {request.calc_type} job; use RHF/UHF or a DFT functional."
        )


def option_keys_for(calc_type: str) -> frozenset:
    """The request option keys the worker reads for *calc_type*."""
    keys = {"scf_rescue"} | set(CALC_OPTION_KEYS.get(calc_type, ()))
    if calc_type in PREOPT_CALC_TYPES:
        keys.add("preopt_before_run")
    return frozenset(keys)


def check_options(request: CalculationRequest) -> List[str]:
    """Normalise boolean options; return warnings for keys the job ignores.

    ``scf_rescue=no`` arrives as the string ``"no"``, which the worker's
    ``bool()`` reads as True, so yes/no/on/off/true/false/1/0 are turned into
    booleans and anything else is refused.
    """
    for key in BOOLEAN_OPTION_KEYS:
        if key not in request.options:
            continue
        value = request.options[key]
        word = str(value).strip().lower()
        if isinstance(value, bool):
            continue
        if word in _TRUE_WORDS:
            request.options[key] = True
        elif word in _FALSE_WORDS:
            request.options[key] = False
        else:
            raise BatchInputError(
                f"option {key}={value!r} must be true or false (yes/no also work)."
            )
    allowed = option_keys_for(request.calc_type)
    unknown = sorted(k for k in request.options if k not in allowed)
    if not unknown:
        return []
    return [
        f"option(s) {', '.join(unknown)} are ignored by a {request.calc_type} "
        f"job (it reads: {', '.join(sorted(allowed))})."
    ]


def known_basis_sets() -> List[str]:
    """QuantUI's basis sets, plus the parenthesis spelling of starred Pople sets."""
    from quantui import config

    names = list(config.SUPPORTED_BASIS_SETS)
    names += [config.pople_notation_alias(b) for b in config.SUPPORTED_BASIS_SETS]
    return [n for n in names if n]


def check_basis(request: CalculationRequest) -> List[str]:
    """Normalise the basis spelling; warn about one QuantUI does not list.

    Same policy as ``calculator``: a basis outside the list is allowed (PySCF
    knows many more), but the job fails if PySCF does not know it, so the
    user is told at submit rather than after the queue wait.
    """
    names = {b.lower(): b for b in known_basis_sets()}
    given = str(request.basis or "").strip()
    canonical = names.get(given.lower())
    if canonical is not None:
        request.basis = canonical
        return []
    if not given:
        raise BatchInputError("the request has no basis set; give --basis.")
    return [
        f"basis {given!r} is not one of QuantUI's basis sets; the job fails "
        "if PySCF does not know it (check the spelling)."
    ]


def check_json_request(request: CalculationRequest) -> None:
    """The checks an ``.xyz`` input gets while it is parsed, for request JSON.

    Request id (it names the job record file), calc type, atoms and
    coordinates, and a charge/multiplicity that fit the electron count. The
    worker takes charge and multiplicity from the molecule when it has
    them, so those are the values checked.
    """
    from quantui.utils import validate_atom_symbol, validate_coordinates
    from quantui.xyz_input import spin_compatibility_note

    rid = request.request_id
    if rid and (not isinstance(rid, str) or not REQUEST_ID_RE.match(rid)):
        raise BatchInputError(
            f"request_id {rid!r} may only use letters, digits and _ . - "
            "(at most 64 characters)."
        )
    if rid and ".." in rid:
        raise BatchInputError(f"request_id {rid!r} may not contain '..'.")
    if request.calc_type not in CALC_TYPES:
        raise BatchInputError(
            f"unknown calc_type {request.calc_type!r}; choose one of "
            f"{', '.join(CALC_TYPES)}."
        )
    mol = request.molecule
    atoms = mol.get("atoms")
    coords = mol.get("coords") or mol.get("coordinates")
    if not isinstance(atoms, list) or not atoms:
        raise BatchInputError("molecule.atoms must be a non-empty list.")
    if not isinstance(coords, list) or len(coords) != len(atoms):
        raise BatchInputError(
            "molecule.coords must list one [x, y, z] per atom " f"({len(atoms)} atoms)."
        )
    # The worker's Molecule() rules (utils.validate_atom_symbol /
    # validate_coordinates).
    for i, (atom, row) in enumerate(zip(atoms, coords)):
        if not isinstance(atom, str) or not validate_atom_symbol(atom):
            raise BatchInputError(f"molecule.atoms[{i}]: unknown element {atom!r}.")
        if not validate_coordinates(row):
            raise BatchInputError(
                f"molecule.coords[{i}] must be three finite numbers, got {row!r}."
            )
    charge = int(mol.get("charge", request.charge))
    mult = int(mol.get("multiplicity", request.multiplicity))
    note = spin_compatibility_note(atoms, charge, mult)
    if note:
        raise BatchInputError(
            f"charge {charge} and multiplicity {mult} do not fit this "
            f"molecule. {note}"
        )


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
        check_method(request)
        warnings += check_basis(request)
        warnings += check_options(request)
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
    check_json_request(request)
    check_solvent(request)
    check_method(request)
    warnings += check_basis(request)
    warnings += check_options(request)
    return request, warnings
