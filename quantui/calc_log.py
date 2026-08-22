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

from . import theme as _theme

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
    # CCSD scales O(N⁶); CCSD(T) adds the perturbative-triples step that
    # scales O(N⁷). Cost ratios here are illustrative — actual runtimes are
    # extracted from the perf log when available.
    "CCSD": 30.0,
    "CCSD(T)": 100.0,
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
        "He": 5,
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


def _prediction_log_path() -> Path:
    """Path to ``prediction_log.jsonl`` — the file capturing one record
    per ``_do_run`` invocation with the estimator's pre-run prediction
    and the actual wall-clock outcome.

    Kept indefinitely (like ``perf_log.jsonl``) so the analytics
    dashboard can plot prediction accuracy over time without manual
    pruning. Lives in the same dir as the other logs; honours
    ``QUANTUI_LOG_DIR`` for tests.
    """
    return _log_dir() / "prediction_log.jsonl"


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)


_READ_ALL_CACHE: dict[str, tuple[float, int, list[dict]]] = {}


def _read_all(path: Path) -> list[dict]:
    """Parse every JSON line in *path*, caching on (mtime, size).

    ``estimate_time()`` calls this on every UI refresh (widget-change
    callback), re-parsing the entire (indefinitely-kept) perf log each
    time even though it usually hasn't changed since the last call.
    Cache the parsed records keyed by the file's mtime + size so repeat
    calls between writes skip the read + per-line ``json.loads`` entirely;
    a write bumps both, so the cache invalidates correctly.
    """
    key = str(path)
    with _LOCK:
        if not path.exists():
            _READ_ALL_CACHE.pop(key, None)
            return []
        stat = path.stat()
        cached = _READ_ALL_CACHE.get(key)
        if (
            cached is not None
            and cached[0] == stat.st_mtime
            and cached[1] == stat.st_size
        ):
            return list(cached[2])
        records: list[dict] = []
        # errors="replace": a concurrent/interrupted append can leave a partial
        # multibyte UTF-8 sequence at EOF; strict decoding would raise
        # UnicodeDecodeError and abort the whole read. Replacing the bad bytes
        # lets that one line fail json.loads and be skipped, matching the
        # deliberate malformed-entry tolerance below (and under pytest -n=auto).
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    try:
                        records.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
        _READ_ALL_CACHE[key] = (stat.st_mtime, stat.st_size, records)
        return list(records)


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
# Statistical helpers (2026-05-25)
# ---------------------------------------------------------------------------


def _iqr_filter(values: list[float]) -> list[float]:
    """Discard outliers outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR].

    The classic Tukey fence catches cold-cache outliers (single slow
    runs that landed before BLAS / DFT grids were resident) and
    thermal-throttled runs (a single overheated run pulled the median
    high) without being overly aggressive on the legitimate spread
    you'd expect across the perf-log timeline.

    Returns the unmodified list when there are fewer than 4 samples —
    IQR isn't meaningful on small N, and the median-based predictors
    upstream already handle small-N gracefully.
    """
    if len(values) < 4:
        return list(values)
    sorted_v = sorted(values)
    # Use the "inclusive" method (matches numpy/pandas default linear
    # interpolation). "exclusive" places quartiles BETWEEN data points
    # via n*p/(n+1) which lets a single small-N outlier pull Q3 high
    # enough that its own value falls inside the fence — defeating the
    # filter. "inclusive" anchors quartiles AT data points so the
    # fence cleanly excludes the outlier.
    q1 = statistics.quantiles(sorted_v, n=4, method="inclusive")[0]
    q3 = statistics.quantiles(sorted_v, n=4, method="inclusive")[2]
    iqr = q3 - q1
    if iqr == 0:
        # All-equal pool — no outliers to reject.
        return list(values)
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return [v for v in values if low <= v <= high]


def _coefficient_of_variation(values: list[float]) -> float:
    """Return σ / |μ|. Returns 0.0 when the mean is zero or N < 2."""
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / abs(mean)


def _confidence_label(values: list[float], n_samples: int) -> str:
    """Variance-aware confidence label.

    Combines coefficient of variation (CV) with sample count:

    - CV < 0.15        → "high"
    - 0.15 ≤ CV < 0.35 → "medium"
    - CV ≥ 0.35        → "low"

    Then capped by sample count: n < 3 always reports "low" (CV is
    noisy on tiny pools); n < 5 caps at "medium" regardless of CV.

    This is what catches the 1-min-predicted / 5-min-actual class —
    even with many samples, a high-variance pool should report "low"
    confidence so the user knows the prediction has wide error bars.
    """
    if n_samples < 3:
        return "low"
    cv = _coefficient_of_variation(values)
    if cv < 0.15:
        base = "high"
    elif cv < 0.35:
        base = "medium"
    else:
        base = "low"
    # Sample-count cap.
    if n_samples < 5 and base == "high":
        return "medium"
    return base


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
    gpu_used: Optional[bool] = None,
    gpu_name: Optional[str] = None,
    n_steps: Optional[int] = None,
    source: Optional[str] = None,
    warm: Optional[bool] = None,
    import_s: Optional[float] = None,
    stages: Optional[dict] = None,
    density_fit: Optional[bool] = None,
) -> None:
    """Append one performance record to ``perf_log.jsonl``.

    ``gpu_used`` / ``gpu_name`` (added 2026-05-25) record whether GPU
    offload was active for the run; reading these back lets
    ``quantui.analytics.build_dashboard`` compute GPU-vs-CPU speedups
    across runs of the same (method, basis, formula) tuple.

    ``source`` / ``warm`` / ``import_s`` / ``stages`` (added 2026-08-05,
    M-PROGRESS Phase C) describe **how the timing was measured**, which
    turned out to matter more than the cost model itself:

    * ``source`` — ``"app"`` for a run the user launched in the UI,
      ``"calibration"`` for one the benchmark harness measured in a fresh
      subprocess. Those two populations have very different wall times for
      the same chemistry (see :func:`estimate_time`), so mixing them was
      the dominant error term in the pre-Phase-C estimator.
    * ``warm`` — whether this process had already completed a calc of the
      same ``calc_type``. The first frequency of a session pays PySCF's
      Hessian-module import; later ones don't.
    * ``import_s`` — measured import cost, when the caller runs in a fresh
      process and can separate it from compute.
    * ``stages`` — ``{stage_label: seconds}`` wall-time breakdown, used by
      the stage-aware frequency model.

    All four are additive and optional: records written before they existed
    simply lack the keys, and every reader treats "absent" as "unknown"
    rather than assuming a default.

    ``density_fit`` (added 2026-08-15, M-DF) records whether density fitting
    (RI) was applied to the SCF. It follows the same additive convention and is
    partitioned in :func:`estimate_time` exactly as ``gpu_used`` is — mixing
    fitted and unfitted runs of the same chemistry under one key would give the
    estimator a bimodal distribution it cannot see.
    """
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
    if gpu_used is not None:
        record["gpu_used"] = bool(gpu_used)
    if gpu_name is not None:
        record["gpu_name"] = gpu_name
    if density_fit is not None:
        record["density_fit"] = bool(density_fit)
    # Outer-loop step count (geom-opt BFGS steps / PES scan points). Enables
    # a history-based "step k / ~N" prior for the live progress fraction.
    if n_steps is not None:
        record["n_steps"] = n_steps
    if source is not None:
        record["source"] = str(source)
    if warm is not None:
        record["warm"] = bool(warm)
    if import_s is not None:
        record["import_s"] = round(float(import_s), 3)
    if stages:
        record["stages"] = {str(k): round(float(v), 3) for k, v in stages.items()}
    _append(_perf_path(), record)


#: Hessian-cost multipliers used by the frequency cost model.
#: PySCF's analytical Hessian for HF/DFT runs in ~2-3× SCF time; for
#: post-HF methods it falls back to numerical Hessian which is much
#: more expensive (effectively 6N SCFs by itself, on top of the IR
#: intensity 6N SCFs). The constants below are empirical defaults that
#: tier-3/4 calibration data can refine — they're load-bearing only
#: when no direct frequency-calc history exists for the (method, basis)
#: tuple. Once the user has run a tier-4 freq, strategies 1-4 use real
#: data and the cost model is skipped entirely.
_HESSIAN_MULTIPLIER_HF_DFT: float = 2.0
_HESSIAN_MULTIPLIER_POST_HF: float = 6.0
_POST_HF_METHODS: frozenset = frozenset({"MP2", "CCSD", "CCSD(T)"})


def _estimate_frequency_cost(
    records: Optional[list[dict]] = None,
    *,
    n_atoms: int,
    n_electrons: int,
    method: str,
    basis: str,
    n_basis: Optional[int] = None,
    n_cores: Optional[int] = None,
    gpu_used: Optional[bool] = None,
    source: Optional[str] = None,
) -> Optional[dict]:
    """Structured frequency-time estimate from an SP anchor.

    Decomposition::

        freq_total ≈ scf_anchor + hessian_term + ir_intensity_term

    where:

    - ``scf_anchor`` — predicted single-point time for the same
      ``(method, basis, n_atoms, gpu_used)`` profile, derived via
      :func:`estimate_time` with ``calc_type="single_point"``.
    - ``hessian_term`` — empirical multiple of ``scf_anchor`` (~2× for
      HF/DFT analytical, ~6× for post-HF numerical).
    - ``ir_intensity_term`` — the 6N inner SCFs that compute ∂μ/∂R for
      IR intensities, divided by ``effective_workers`` when the
      ``QUANTUI_FREQ_PARALLEL`` cross-displacement worker pool is gated
      on (requires no GPU + ≥4 cores + ≥6 displacements). On a GPU host
      the inner SCFs are already accelerated by gpu4pyscf, so parallel
      adds little and stays serial.

    Returns ``None`` when the SP anchor can't be produced (no usable
    history for the SP profile). In that case ``estimate_time``'s
    overall return value stays ``None`` and the UI shows
    "no estimate available — run a calibration".

    The model's confidence is inherited from the SP anchor — we don't
    have direct freq variance data to claim independently, and the
    cost decomposition itself is a fixed structural assumption.
    """
    if n_atoms <= 0:
        return None

    # ``records=None`` means "use the live history". The replay harness and
    # ``estimate_time_from_records`` always pass an explicit slice, so the
    # model itself stays pure; the default exists for direct callers that
    # just want the same answer the app would give.
    if records is None:
        records = _read_all(_perf_path())

    sp_est = estimate_time_from_records(
        records,
        n_atoms=n_atoms,
        n_electrons=n_electrons,
        method=method,
        basis=basis,
        n_basis=n_basis,
        n_cores=n_cores,
        calc_type="single_point",
        gpu_used=gpu_used,
        source=source,
    )
    if sp_est is None:
        return None
    scf_anchor_s = float(sp_est["seconds"])

    # Hessian term.
    method_upper = method.strip().upper()
    hessian_mult = (
        _HESSIAN_MULTIPLIER_POST_HF
        if method_upper in _POST_HF_METHODS
        else _HESSIAN_MULTIPLIER_HF_DFT
    )
    hessian_term_s = hessian_mult * scf_anchor_s

    # IR intensity term — 6N inner SCFs, possibly parallelized.
    displacement_count = 6 * n_atoms
    effective_workers = 1
    try:
        from quantui.freq_ir_workers import (
            parallel_enabled_for_run,
            pick_worker_count,
        )

        cpu_count = n_cores if n_cores is not None else (os.cpu_count() or 1)
        if parallel_enabled_for_run(
            cpu_count=cpu_count,
            displacement_count=displacement_count,
            gpu_available=bool(gpu_used),
        ):
            effective_workers = pick_worker_count(cpu_count, displacement_count)
    except Exception:  # noqa: BLE001 — gating is best-effort
        effective_workers = 1
    ir_term_s = displacement_count * scf_anchor_s / max(1, effective_workers)

    total_s = scf_anchor_s + hessian_term_s + ir_term_s
    return {
        "seconds": total_s,
        # Cost model adds structural assumptions but no new data — don't
        # claim more confidence than the SP anchor it leans on.
        "confidence": sp_est["confidence"],
        "n_samples": sp_est["n_samples"],
    }


def estimate_time(
    n_atoms: int,
    n_electrons: int,
    method: str,
    basis: str,
    n_basis: Optional[int] = None,
    n_cores: Optional[int] = None,
    calc_type: Optional[str] = None,
    gpu_used: Optional[bool] = None,
    source: Optional[str] = None,
    density_fit: Optional[bool] = None,
) -> Optional[dict]:
    """Estimate wall time for an upcoming calculation from ``perf_log.jsonl``.

    Thin wrapper: reads the performance history, then defers every
    modelling decision to :func:`estimate_time_from_records`. The split
    exists so the offline replay harness
    (:mod:`quantui.estimator_eval`) can score the *same* predictor
    against an arbitrary historical slice without touching the real log
    file — which is what makes Phase C's model changes measurable rather
    than a matter of taste.
    """
    return estimate_time_from_records(
        _read_all(_perf_path()),
        n_atoms=n_atoms,
        n_electrons=n_electrons,
        method=method,
        basis=basis,
        n_basis=n_basis,
        n_cores=n_cores,
        calc_type=calc_type,
        gpu_used=gpu_used,
        source=source,
        density_fit=density_fit,
    )


def estimate_time_from_records(
    records: list[dict],
    *,
    n_atoms: int,
    n_electrons: int,
    method: str,
    basis: str,
    n_basis: Optional[int] = None,
    n_cores: Optional[int] = None,
    calc_type: Optional[str] = None,
    gpu_used: Optional[bool] = None,
    source: Optional[str] = None,
    density_fit: Optional[bool] = None,
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

    **GPU-aware filtering** (2026-05-25): when ``gpu_used``
    is passed, the candidate pool is partitioned by device — GPU-history
    predicts GPU runs and CPU-history predicts CPU runs. Older records
    don't have ``gpu_used`` at all; those are treated
    as "device unknown" and admitted only when ``gpu_used=False`` is
    requested (the conservative assumption, since QuantUI was CPU-only
    before GPU offload shipped). When ``gpu_used=None`` (default), the device
    axis is ignored and all records are eligible — back-compat with
    callers that don't know which device the upcoming run will use.

    If GPU partitioning leaves fewer than 2 records in the pool, the
    function falls back to the unpartitioned pool with the confidence
    label downgraded one notch — better an approximate estimate from
    cross-device data than no estimate at all.

    **Provenance filtering** (2026-08-05, M-PROGRESS Phase C): ``source``
    partitions the pool the same way ``gpu_used`` does, and for the same
    reason — the two populations measure different things. A calibration
    record times a fresh subprocess, so it includes PySCF's import cost;
    an app record times a calculation inside an already-warm kernel. On
    small molecules that import dominates, which is why the pre-Phase-C
    estimator was systematically wrong despite an unbiased median: mixing
    the populations inflated the spread rather than the centre. Records
    written before this field existed carry no ``source`` and are treated
    as "provenance unknown" — admitted only on the fallback path, with
    confidence downgraded, so the pool self-heals as tagged records
    accumulate rather than needing the user's history to be discarded.

    Returns ``None`` when fewer than 2 converged records are available for
    the scoped candidate pool.
    """
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
        # Frequency calcs can still produce a prediction via the
        # SP-anchored cost model even when direct freq history is empty.
        # The cost model lives at the end of this function — fall through
        # for freq, bail for everything else.
        if calc_type != "frequency":
            return None
        # Continue with empty/small ``scoped``: the four direct strategies
        # will all no-op (their pool checks require len >= 2), and the
        # freq cost-model fallback at the end will fire.

    # Partition by device when the caller specified one.
    # Older records don't carry ``gpu_used`` — admit them
    # only into the CPU pool, since QuantUI was CPU-only when they were
    # written. Track whether we downgraded for the fall-back path below.
    _gpu_filtered = False
    if gpu_used is True:
        gpu_scoped = [r for r in scoped if r.get("gpu_used") is True]
        if len(gpu_scoped) >= 2:
            scoped = gpu_scoped
            _gpu_filtered = True
        # else: fall through to the unpartitioned pool; caller's
        # confidence will be downgraded below.
    elif gpu_used is False:
        cpu_scoped = [
            r for r in scoped if r.get("gpu_used") is False or "gpu_used" not in r
        ]
        if len(cpu_scoped) >= 2:
            scoped = cpu_scoped
            _gpu_filtered = True

    # Partition by provenance. Same shape as the device partition above:
    # prefer a pool that was measured the same way the upcoming run will
    # be, fall back to everything when that pool is too thin.
    _source_filtered = False
    if source is not None:
        same_source = [r for r in scoped if r.get("source") == source]
        if len(same_source) >= 2:
            scoped = same_source
            _source_filtered = True

    # Partition by density fitting (M-DF). Same shape as the device partition:
    # older records predate DF and were all unfitted, so admit "absent" into
    # the unfitted pool. Mixing fitted and unfitted runs of the same chemistry
    # would give the estimator a bimodal distribution it cannot see.
    _df_filtered = False
    if density_fit is True:
        df_scoped = [r for r in scoped if r.get("density_fit") is True]
        if len(df_scoped) >= 2:
            scoped = df_scoped
            _df_filtered = True
    elif density_fit is False:
        nodf_scoped = [
            r for r in scoped if r.get("density_fit") is False or "density_fit" not in r
        ]
        if len(nodf_scoped) >= 2:
            scoped = nodf_scoped
            _df_filtered = True

    def _maybe_downgrade(conf: str) -> str:
        """Downgrade confidence one notch per partition that fell back.

        A fall-back means the pool contains records measured on a
        different device, or measured a different way, than the run being
        predicted. Either alone is worth a notch; both together should
        not read as merely "medium", so the downgrades compose.
        """
        order = ["high", "medium", "low"]
        idx = order.index(conf)
        if gpu_used is not None and not _gpu_filtered:
            idx += 1
        if source is not None and not _source_filtered:
            idx += 1
        if density_fit is not None and not _df_filtered:
            idx += 1
        return order[min(idx, len(order) - 1)]

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
            # Drop Tukey outliers before computing the predictor.
            # The variance of the *filtered* pool drives confidence.
            filtered_effs = _iqr_filter(effs)
            predicted = (
                statistics.median(filtered_effs) * (n_basis**beta_new) / n_cores_current
            )
            return {
                "seconds": predicted,
                "confidence": _maybe_downgrade(
                    _confidence_label(filtered_effs, len(filtered_effs))
                ),
                "n_samples": len(filtered_effs),
            }

    # ── Strategy 2: exact method + basis, electron-count fallback ────────────
    exact = [r for r in scoped if r.get("method") == method and r.get("basis") == basis]
    if len(exact) >= 2:
        elapsed_values = [float(r["elapsed_s"]) for r in exact]
        filtered_elapsed = _iqr_filter(elapsed_values)
        # Recompute electron-count median against the same filtered pool
        # so the scale factor is consistent with the time median.
        filtered_records = [
            r for r in exact if float(r["elapsed_s"]) in filtered_elapsed
        ]
        median_ne = statistics.median(
            r["n_electrons"] for r in (filtered_records or exact)
        )
        median_t = statistics.median(filtered_elapsed)
        scale = (n_electrons / median_ne) ** 2.7 if median_ne > 0 else 1.0
        return {
            "seconds": median_t * scale,
            "confidence": _maybe_downgrade(
                _confidence_label(filtered_elapsed, len(filtered_elapsed))
            ),
            "n_samples": len(filtered_elapsed),
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

    # ── Frequency cost-model fallback ─────────────────────────────────────────
    # When all four direct-history strategies fail for a freq calc, fall
    # back to the structural decomposition: SP anchor + Hessian + 6N
    # inner SCFs. The SP anchor comes from the much richer single-point
    # history pool, which is usually populated even on a fresh install
    # (tier 1 is SP-only). Confidence is inherited from the SP anchor.
    if calc_type == "frequency":
        cost_est = _estimate_frequency_cost(
            records,
            n_atoms=n_atoms,
            n_electrons=n_electrons,
            method=method,
            basis=basis,
            n_basis=n_basis,
            n_cores=n_cores,
            gpu_used=gpu_used,
            source=source,
        )
        if cost_est is not None:
            return cost_est

    return None


def estimate_opt_steps(
    method: str, basis: str, calc_type: str = "geometry_opt"
) -> Optional[float]:
    """Median historical outer-step count for *calc_type* (progress prior).

    Reads ``perf_log`` for converged records of *calc_type* that recorded
    ``n_steps``. Prefers exact method+basis (>= 2 records), falls back
    to same-basis (>= 2), then to all matching-calc_type records. Returns the
    median or ``None`` when there is no usable history — a rough prior used only
    to seed / floor the live progress fraction, not a hard prediction.
    """
    try:
        records = _read_all(_perf_path())
    except Exception:
        return None

    def _usable(r: dict) -> bool:
        return (
            r.get("calc_type") == calc_type
            and bool(r.get("converged"))
            and isinstance(r.get("n_steps"), (int, float))
            and r["n_steps"] > 0
        )

    pool = [r for r in records if _usable(r)]
    if not pool:
        return None
    exact = [r for r in pool if r.get("method") == method and r.get("basis") == basis]
    same_basis = [r for r in pool if r.get("basis") == basis]
    if len(exact) >= 2:
        chosen = exact
    elif len(same_basis) >= 2:
        chosen = same_basis
    else:
        chosen = pool
    try:
        return float(statistics.median(r["n_steps"] for r in chosen))
    except Exception:
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

    colour = {
        "high": "#22c55e",
        "medium": _theme.ACCENT_WARNING_LIGHT,
        "low": _theme.TEXT_SUBTLE,
    }[conf]
    return (
        f'<span style="font-size:12px;color:{_theme.TEXT_SLATE}">'
        f'Estimated time: <b style="color:{colour}">{time_str}</b>'
        f'&ensp;<span style="color:{_theme.TEXT_SUBTLE}">({conf} confidence, {n} similar '
        f'run{"s" if n != 1 else ""})</span></span>'
    )


def get_perf_history() -> list[dict]:
    """Return all records from ``perf_log.jsonl`` as a list of dicts."""
    return _read_all(_perf_path())


# ---------------------------------------------------------------------------
# Prediction log (2026-05-25)
# ---------------------------------------------------------------------------
#
# Captures one record per ``_do_run`` invocation with the estimator's
# pre-run prediction + the actual wall-clock outcome. Lets the analytics
# dashboard show prediction accuracy over time, broken down by calc-type
# and device, so the user can tell at a glance whether the estimator is
# working or whether it's time to re-calibrate.


def log_prediction(
    predicted_s: Optional[float],
    actual_s: float,
    *,
    method: str,
    basis: str,
    calc_type: str,
    formula: str = "",
    confidence: str = "unknown",
    gpu_used: Optional[bool] = None,
) -> None:
    """Append one prediction record to ``prediction_log.jsonl``.

    ``predicted_s`` is ``None`` when the estimator returned no estimate
    (insufficient history at run-time). Both columns are still logged
    so the dashboard can count "no-estimate" runs separately from
    "estimate-was-way-off" runs — both are meaningful failure modes
    for the predictor.

    ``actual_s`` should match the value passed to ``log_calculation``
    for the same run; the dashboard cross-references them via the
    ``timestamp`` key. The two writes are not transactional — if one
    side fails we'd rather have the perf-log record than no record
    at all, so ``log_prediction`` is best-effort and the caller does
    not depend on its return.
    """
    record: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_s": (
            round(float(predicted_s), 3) if predicted_s is not None else None
        ),
        "actual_s": round(float(actual_s), 3),
        "method": method,
        "basis": basis,
        "calc_type": calc_type,
        "formula": formula,
        "confidence": confidence,
    }
    if gpu_used is not None:
        record["gpu_used"] = bool(gpu_used)
    # Derived: signed error percentage. ``None`` when we had no estimate.
    if predicted_s is not None and predicted_s > 0:
        record["error_pct"] = round(
            100.0 * (float(actual_s) - float(predicted_s)) / float(predicted_s), 1
        )
    else:
        record["error_pct"] = None
    _append(_prediction_log_path(), record)


def get_prediction_history() -> list[dict]:
    """Return all records from ``prediction_log.jsonl`` as a list of dicts."""
    return _read_all(_prediction_log_path())


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

# Audit fix (2026-07-14): log_event() used to call prune_events() after
# every single append, and prune_events() itself read the file (acquiring
# and releasing _LOCK) and only later reacquired _LOCK to rewrite it. Two
# problems:
#
# 1. Race: an append from another thread landing in the gap between the
#    read and the rewrite got silently discarded when the rewrite replaced
#    the whole file with the (now-stale) `kept` list computed before that
#    append happened.
# 2. Cost: reading + rewriting the entire event log on every single write
#    is O(file size) per event, i.e. O(N^2) over a session as the log
#    grows — noticeable once a session has logged more than a few hundred
#    events.
#
# Fixed by (a) making prune_events() read + filter + rewrite as a single
# lock-held critical section, so a concurrent append either completes
# before the prune starts or blocks until it finishes — it can never be
# silently lost — and (b) only running the full prune every
# _PRUNE_EVERY_N_EVENTS appends instead of on every single one.
_PRUNE_EVERY_N_EVENTS = 20
_events_since_prune = 0


def log_event(event_type: str, message: str, **extra: object) -> None:
    """
    Append one event to ``event_log.jsonl``; prune entries > 7 days old
    periodically (every :data:`_PRUNE_EVERY_N_EVENTS` calls, not every one).

    Args:
        event_type: Short category string, e.g. ``"startup"``, ``"calc_done"``,
                    ``"calc_error"``.
        message:    Human-readable description.
        **extra:    Any additional key-value pairs to include in the record.
    """
    global _events_since_prune

    record: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "message": message,
        **extra,
    }
    _append(_event_path(), record)

    with _LOCK:
        _events_since_prune += 1
        due_for_prune = _events_since_prune >= _PRUNE_EVERY_N_EVENTS
        if due_for_prune:
            _events_since_prune = 0
    if due_for_prune:
        prune_events()


def prune_events(days: int = 7) -> None:
    """Remove event-log entries older than *days* days (default: 7).

    Reads, filters, and rewrites the file as a single lock-held critical
    section so a concurrent :func:`log_event` append can never be silently
    lost between the read and the rewrite (see module-level note above).
    """
    path = _event_path()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with _LOCK:
        if not path.exists():
            return
        records: list[dict] = []
        # errors="replace" for the same reason as _read_all: tolerate a partial
        # multibyte sequence from a concurrent/interrupted append rather than
        # abort the prune.
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    try:
                        records.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass

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

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_recent_events(n: int = 100) -> list[dict]:
    """Return the *n* most recent entries from ``event_log.jsonl``."""
    return _read_all(_event_path())[-n:]
