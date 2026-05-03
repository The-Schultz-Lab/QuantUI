"""
Performance and event logging for QuantUI.

Two separate log files, both stored in ``~/.quantui/logs/`` by default
(override with the ``QUANTUI_LOG_DIR`` environment variable):

``perf_log.jsonl``
    One record per completed calculation.  Kept indefinitely — the full
    history is needed to build reliable time-prediction models.

``event_log.jsonl``
    General app events (startup, calculation lifecycle, errors).
    Auto-pruned: entries older than 7 days are removed on every write.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()

# Rough relative cost of each method per SCF iteration.
# Used when no exact method+basis match exists in the history.
_METHOD_COST: dict[str, float] = {
    "RHF": 1.0,
    "UHF": 1.0,
    "B3LYP": 2.5,
    "PBE": 2.0,
    "PBE0": 2.5,
    "M06-2X": 3.0,
    "wB97X-D": 3.0,
    "CAM-B3LYP": 2.5,
    "M06-L": 2.0,
    "HSE06": 2.5,
    "PBE-D3": 2.1,
    "MP2": 8.0,
}

# Contracted basis function counts per element per basis set (spherical harmonics,
# PySCF default).  Used by count_basis_functions() and estimate_time().
_BASIS_FUNCTIONS: dict[str, dict[str, int]] = {
    "STO-3G": {
        "H": 1,
        "He": 1,
        "Li": 5,
        "Be": 5,
        "B": 5,
        "C": 5,
        "N": 5,
        "O": 5,
        "F": 5,
        "Ne": 5,
        "Na": 9,
        "Mg": 9,
        "Al": 9,
        "Si": 9,
        "P": 9,
        "S": 9,
        "Cl": 9,
        "Ar": 9,
    },
    "3-21G": {
        "H": 2,
        "He": 2,
        "Li": 9,
        "Be": 9,
        "B": 9,
        "C": 9,
        "N": 9,
        "O": 9,
        "F": 9,
        "Ne": 9,
        "Na": 13,
        "Mg": 13,
        "Al": 15,
        "Si": 15,
        "P": 15,
        "S": 15,
        "Cl": 15,
        "Ar": 15,
    },
    "6-31G": {
        "H": 2,
        "He": 2,
        "Li": 9,
        "Be": 9,
        "B": 9,
        "C": 9,
        "N": 9,
        "O": 9,
        "F": 9,
        "Ne": 9,
        "Na": 13,
        "Mg": 13,
        "Al": 15,
        "Si": 15,
        "P": 15,
        "S": 15,
        "Cl": 15,
        "Ar": 15,
    },
    "6-31G*": {
        "H": 2,
        "He": 2,
        "Li": 9,
        "Be": 9,
        "B": 14,
        "C": 14,
        "N": 14,
        "O": 14,
        "F": 14,
        "Ne": 14,
        "Na": 13,
        "Mg": 13,
        "Al": 20,
        "Si": 20,
        "P": 20,
        "S": 20,
        "Cl": 20,
        "Ar": 20,
    },
    "6-31G**": {
        "H": 5,
        "He": 2,
        "Li": 9,
        "Be": 9,
        "B": 14,
        "C": 14,
        "N": 14,
        "O": 14,
        "F": 14,
        "Ne": 14,
        "Na": 13,
        "Mg": 13,
        "Al": 20,
        "Si": 20,
        "P": 20,
        "S": 20,
        "Cl": 20,
        "Ar": 20,
    },
    "cc-pVDZ": {
        "H": 5,
        "He": 5,
        "Li": 9,
        "Be": 9,
        "B": 14,
        "C": 14,
        "N": 14,
        "O": 14,
        "F": 14,
        "Ne": 14,
        "Na": 18,
        "Mg": 18,
        "Al": 23,
        "Si": 23,
        "P": 23,
        "S": 23,
        "Cl": 23,
        "Ar": 23,
    },
    "cc-pVTZ": {
        "H": 14,
        "He": 14,
        "Li": 20,
        "Be": 20,
        "B": 30,
        "C": 30,
        "N": 30,
        "O": 30,
        "F": 30,
        "Ne": 30,
        "Na": 35,
        "Mg": 35,
        "Al": 43,
        "Si": 43,
        "P": 43,
        "S": 43,
        "Cl": 43,
        "Ar": 43,
    },
    "def2-SVP": {
        "H": 5,
        "He": 5,
        "Li": 9,
        "Be": 9,
        "B": 14,
        "C": 14,
        "N": 14,
        "O": 14,
        "F": 14,
        "Ne": 14,
        "Na": 18,
        "Mg": 18,
        "Al": 23,
        "Si": 23,
        "P": 23,
        "S": 23,
        "Cl": 23,
        "Ar": 23,
    },
    "def2-TZVP": {
        "H": 14,
        "He": 14,
        "Li": 20,
        "Be": 20,
        "B": 30,
        "C": 30,
        "N": 30,
        "O": 30,
        "F": 30,
        "Ne": 30,
        "Na": 35,
        "Mg": 35,
        "Al": 43,
        "Si": 43,
        "P": 43,
        "S": 43,
        "Cl": 43,
        "Ar": 43,
    },
}

# Formal scaling exponents in N_basis.  HF/DFT: formally O(N³–N⁴), empirically
# ~3.5 in the student size range.  Correlated methods scale more steeply.
_METHOD_SCALE_EXP: dict[str, float] = {
    "RHF": 3.5,
    "UHF": 3.5,
    "B3LYP": 3.5,
    "PBE": 3.5,
    "PBE0": 3.5,
    "M06-2X": 3.5,
    "wB97X-D": 3.5,
    "CAM-B3LYP": 3.5,
    "M06-L": 3.5,
    "HSE06": 3.5,
    "PBE-D3": 3.5,
    "MP2": 5.0,
    "CCSD": 6.0,
    "CCSD(T)": 7.0,
}


def _log_dir() -> Path:
    env = os.environ.get("QUANTUI_LOG_DIR")
    return Path(env) if env else Path.home() / ".quantui" / "logs"


def _perf_path() -> Path:
    return _log_dir() / "perf_log.jsonl"


def _event_path() -> Path:
    return _log_dir() / "event_log.jsonl"


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)


def _read_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with _LOCK:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    try:
                        records.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
    return records


# ---------------------------------------------------------------------------
# Basis function utilities
# ---------------------------------------------------------------------------


def count_basis_functions(atoms: list[str], basis: str) -> Optional[int]:
    """
    Return the total number of contracted basis functions for a molecule.

    Args:
        atoms: Element symbols (e.g. ``["O", "H", "H"]``).
        basis: Basis set name (e.g. ``"STO-3G"``).

    Returns:
        Total basis function count, or ``None`` if the basis set or any
        element is not in the lookup table.
    """
    table = _BASIS_FUNCTIONS.get(basis)
    if table is None:
        return None
    total = 0
    for atom in atoms:
        n = table.get(atom)
        if n is None:
            return None
        total += n
    return total


# ---------------------------------------------------------------------------
# Performance log
# ---------------------------------------------------------------------------


def log_calculation(
    formula: str,
    n_atoms: int,
    n_electrons: int,
    method: str,
    basis: str,
    n_iterations: Optional[int],
    elapsed_s: float,
    converged: bool,
    n_basis: Optional[int] = None,
    n_cores: Optional[int] = None,
    calc_type: Optional[str] = None,
) -> None:
    """Append one performance record to ``perf_log.jsonl``."""
    record: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "formula": formula,
        "n_atoms": n_atoms,
        "n_electrons": n_electrons,
        "method": method,
        "basis": basis,
        "n_iterations": n_iterations,
        "elapsed_s": round(elapsed_s, 3),
        "converged": converged,
    }
    if n_basis is not None:
        record["n_basis"] = n_basis
    if n_cores is not None:
        record["n_cores"] = n_cores
    if calc_type is not None:
        record["calc_type"] = calc_type
    _append(_perf_path(), record)


def estimate_time(
    n_atoms: int,
    n_electrons: int,
    method: str,
    basis: str,
    n_basis: Optional[int] = None,
    n_cores: Optional[int] = None,
    calc_type: Optional[str] = None,
) -> Optional[dict]:
    """
    Return a time estimate dict, or ``None`` if there is insufficient data.

    The returned dict has keys:

    * ``seconds``    – estimated wall time as a float
    * ``confidence`` – ``"high"``, ``"medium"``, or ``"low"``
    * ``n_samples``  – number of historical records used

    Prediction strategy (in priority order):

    1. **Exact method + basis, basis-function efficiency** (≥ 2 records with
       ``n_basis``): Computes a normalised efficiency
       ``eff = elapsed_s × n_cores_hist / n_basis_hist^β`` for each record,
       then predicts ``median(eff) × n_basis_new^β / n_cores_current``.
       β is method-specific (RHF/DFT ≈ 3.5, MP2 = 5.0, CCSD = 6.0, …).
       Confidence: high (≥ 5 samples) or medium (2–4 samples).

    2. **Exact method + basis, electron-count fallback** (≥ 2 records):
       Median elapsed time scaled by ``(n_electrons / median_n_e)^2.7``.
       Used when older records lack ``n_basis``.
       Confidence: high / medium.

    3. **Same basis, any method, basis-function efficiency** (≥ 2 records with
       ``n_basis``): Like strategy 1, plus a method-cost correction factor.
       Confidence: low.

    4. **Same basis, any method, electron-count fallback** (≥ 2 records):
       Same as the original strategy 2.  Confidence: low.

    ``calc_type`` narrows the candidate pool so that expensive workflows
    (for example, Frequency) are not predicted from cheap workflows
    (for example, Single Point). Legacy records without ``calc_type`` are
    only included when estimating ``single_point``.

    Returns ``None`` when fewer than 2 converged records are available for
    the scoped candidate pool.
    """
    records = _read_all(_perf_path())
    converged = [r for r in records if r.get("converged")]
    if not converged:
        return None

    if calc_type is None:
        scoped = converged
    elif calc_type == "single_point":
        # Back-compat bridge: older records did not store calc_type.
        scoped = [
            r
            for r in converged
            if r.get("calc_type") == "single_point" or r.get("calc_type") is None
        ]
    else:
        scoped = [r for r in converged if r.get("calc_type") == calc_type]

    if len(scoped) < 2:
        return None

    beta_new = _METHOD_SCALE_EXP.get(method, 3.5)
    n_cores_current = n_cores if n_cores is not None else 1

    def _eff(r: dict) -> Optional[float]:
        """Normalised efficiency: elapsed_s × n_cores / n_basis^β."""
        nb: float = float(r.get("n_basis") or 0)
        if not nb:
            return None
        rc: float = float(r.get("n_cores") or 1)
        r_method: str = str(r.get("method") or method)
        beta: float = _METHOD_SCALE_EXP.get(r_method, 3.5)
        elapsed: float = float(r["elapsed_s"])
        return float(elapsed * rc / (nb**beta))

    # ── Strategy 1: exact method + basis, basis-function efficiency ──────────
    if n_basis is not None:
        exact_nb = [
            r
            for r in scoped
            if r.get("method") == method
            and r.get("basis") == basis
            and r.get("n_basis") is not None
        ]
        effs = [e for r in exact_nb for e in [_eff(r)] if e is not None]
        if len(effs) >= 2:
            predicted = statistics.median(effs) * (n_basis**beta_new) / n_cores_current
            return {
                "seconds": predicted,
                "confidence": "high" if len(effs) >= 5 else "medium",
                "n_samples": len(effs),
            }

    # ── Strategy 2: exact method + basis, electron-count fallback ────────────
    exact = [r for r in scoped if r.get("method") == method and r.get("basis") == basis]
    if len(exact) >= 2:
        median_ne = statistics.median(r["n_electrons"] for r in exact)
        median_t = statistics.median(r["elapsed_s"] for r in exact)
        scale = (n_electrons / median_ne) ** 2.7 if median_ne > 0 else 1.0
        return {
            "seconds": median_t * scale,
            "confidence": "high" if len(exact) >= 5 else "medium",
            "n_samples": len(exact),
        }

    # ── Strategy 3: same basis, any method, basis-function efficiency ─────────
    if n_basis is not None:
        same_basis_nb = [
            r
            for r in scoped
            if r.get("basis") == basis and r.get("n_basis") is not None
        ]
        effs = [e for r in same_basis_nb for e in [_eff(r)] if e is not None]
        if len(effs) >= 2:
            ref_cost = statistics.median(
                _METHOD_COST.get(r.get("method", "RHF"), 1.0) for r in same_basis_nb
            )
            tgt_cost = _METHOD_COST.get(method, 1.0)
            cost_factor = tgt_cost / ref_cost if ref_cost > 0 else 1.0
            predicted = (
                statistics.median(effs)
                * (n_basis**beta_new)
                * cost_factor
                / n_cores_current
            )
            return {
                "seconds": predicted,
                "confidence": "low",
                "n_samples": len(effs),
            }

    # ── Strategy 4: same basis, any method, electron-count fallback ───────────
    same_basis = [r for r in scoped if r.get("basis") == basis]
    if len(same_basis) >= 2:
        median_ne = statistics.median(r["n_electrons"] for r in same_basis)
        median_t = statistics.median(r["elapsed_s"] for r in same_basis)
        ref_cost = statistics.median(
            _METHOD_COST.get(r.get("method", "RHF"), 1.0) for r in same_basis
        )
        tgt_cost = _METHOD_COST.get(method, 1.0)
        ne_scale = (n_electrons / median_ne) ** 2.7 if median_ne > 0 else 1.0
        cost_scale = tgt_cost / ref_cost if ref_cost > 0 else 1.0
        return {
            "seconds": median_t * ne_scale * cost_scale,
            "confidence": "low",
            "n_samples": len(same_basis),
        }

    return None


def format_estimate(est: Optional[dict]) -> str:
    """
    Return an HTML string summarising *est* for display in the notebook.

    Returns an empty string when *est* is ``None``.
    """
    if est is None:
        return ""
    s = est["seconds"]
    conf = est["confidence"]
    n = est["n_samples"]

    if s < 5:
        time_str = "&lt; 5 s"
    elif s < 60:
        time_str = f"~{int(s)} s"
    elif s < 3600:
        time_str = f"~{int(s / 60)} min"
    else:
        time_str = f"~{s / 3600:.1f} hr"

    colour = {"high": "#22c55e", "medium": "#f59e0b", "low": "#94a3b8"}[conf]
    return (
        f'<span style="font-size:12px;color:#64748b">'
        f'Estimated time: <b style="color:{colour}">{time_str}</b>'
        f'&ensp;<span style="color:#94a3b8">({conf} confidence, {n} similar '
        f'run{"s" if n != 1 else ""})</span></span>'
    )


def get_perf_history() -> list[dict]:
    """Return all records from ``perf_log.jsonl`` as a list of dicts."""
    return _read_all(_perf_path())


def reset_perf_log() -> None:
    """Delete all records from ``perf_log.jsonl``.

    Removes the file entirely.  A fresh file is created automatically on the
    next :func:`log_calculation` call.  Time estimates will return ``None``
    until enough new records accumulate.
    """
    path = _perf_path()
    with _LOCK:
        if path.exists():
            path.unlink()


def clear_event_log() -> None:
    """Delete the session event log (``event_log.jsonl``).

    Removes the file entirely.  A fresh file is created automatically on the
    next :func:`log_event` call.  ``perf_log.jsonl`` and ``issues.db`` are
    **not** affected.
    """
    path = _event_path()
    with _LOCK:
        if path.exists():
            path.unlink()


# ---------------------------------------------------------------------------
# Event log (7-day TTL)
# ---------------------------------------------------------------------------


def log_event(event_type: str, message: str, **extra: object) -> None:
    """
    Append one event to ``event_log.jsonl`` and prune entries > 7 days old.

    Args:
        event_type: Short category string, e.g. ``"startup"``, ``"calc_done"``,
                    ``"calc_error"``.
        message:    Human-readable description.
        **extra:    Any additional key-value pairs to include in the record.
    """
    record: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "message": message,
        **extra,
    }
    _append(_event_path(), record)
    prune_events()


def prune_events(days: int = 7) -> None:
    """Remove event-log entries older than *days* days (default: 7)."""
    path = _event_path()
    if not path.exists():
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    records = _read_all(path)
    kept: list[dict] = []
    for r in records:
        try:
            ts = datetime.fromisoformat(r["timestamp"])
            # fromisoformat on Python < 3.11 doesn't handle 'Z' suffix
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                kept.append(r)
        except (KeyError, ValueError):
            kept.append(r)  # keep malformed entries rather than silently drop
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_recent_events(n: int = 100) -> list[dict]:
    """Return the *n* most recent entries from ``event_log.jsonl``."""
    return _read_all(_event_path())[-n:]
