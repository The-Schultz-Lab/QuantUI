"""
Timing calibration benchmark suite for QuantUI.

Runs a fixed set of small calculations that span the student-relevant
method/basis/molecule-size space.  Each completed step is logged to
``perf_log.jsonl`` via :func:`~quantui.calc_log.log_calculation` so that
:func:`~quantui.calc_log.estimate_time` immediately becomes useful on a
fresh install.

Four tiers (M-EST / EST.4, 2026-05-25)
--------------------------------------

The calibration suite is now a **four-tier cascade** rather than the
original short/long pair. Users pick the depth that matches their setup-
time tolerance:

- **Tier 1 — Quick** (~15 s): SP only, smoke-test PySCF + bootstrap
  predictor. Same molecules as the historical "short" suite.
- **Tier 2 — Standard** (~3–5 min): SP only, expanded method × basis
  grid so the predictor has multiple anchors per `(method, basis)` tuple.
- **Tier 3 — Mixed** (~10–15 min): tier 2 + 2–3 small geometry
  optimizations + 1–2 small frequency calcs. First reliable GeoOpt +
  Freq predictions.
- **Tier 4 — Deep** (up to 30 min): tier 3 + medium GeoOpt + medium
  Freq (ethanol, benzene) + MP2 / CCSD anchors. Lets the estimator
  predict every calc-type × device combo within ±25%.

Back-compat: the legacy ``mode="short"`` / ``mode="long"`` strings still
work as aliases for tier 1 / tier 2 respectively. New code should use
``mode="tier1"`` … ``mode="tier4"``.

Entry format
------------

Each tier is a list of 7-tuples (single-point calcs) or 8-tuples (when
the 8th element overrides the calc-type, e.g. ``"geometry_opt"`` /
``"frequency"``). ``_normalize_entry()`` unpacks either shape.

Typical usage (from the UI)::

    import threading
    from quantui.benchmarks import run_calibration

    stop = threading.Event()
    result = run_calibration(
        progress_cb=lambda *a: print(a),
        stop_event=stop,
        timeout_per_step=120,
        mode="tier3",  # or "tier1"/"tier2"/"tier4"
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Benchmark suite definition
# ---------------------------------------------------------------------------

#: Each entry: (label, atoms, coordinates, charge, multiplicity, method, basis)
#: Molecules are kept deliberately small so the full suite finishes quickly on
#: any modern laptop.
BENCHMARK_SUITE: list[tuple] = [
    (
        "H₂  RHF/STO-3G",
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        0,
        1,
        "RHF",
        "STO-3G",
    ),
    (
        "H₂O  RHF/STO-3G",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "RHF",
        "STO-3G",
    ),
    (
        "H₂O  B3LYP/STO-3G",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "B3LYP",
        "STO-3G",
    ),
    (
        "H₂O  RHF/6-31G*",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "RHF",
        "6-31G*",
    ),
    (
        "CH₄  RHF/STO-3G",
        ["C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ],
        0,
        1,
        "RHF",
        "STO-3G",
    ),
    (
        "C₂H₄  RHF/STO-3G",
        ["C", "C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.670],
            [0.0, 0.0, -0.670],
            [0.0, 0.924, 1.241],
            [0.0, -0.924, 1.241],
            [0.0, 0.924, -1.241],
            [0.0, -0.924, -1.241],
        ],
        0,
        1,
        "RHF",
        "STO-3G",
    ),
    (
        "C₂H₆O (ethanol)  RHF/STO-3G",
        ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        [
            [-1.232, 0.026, 0.000],
            [0.281, 0.026, 0.000],
            [0.829, 1.310, 0.000],
            [-1.566, 1.059, 0.000],
            [-1.609, -0.506, 0.880],
            [-1.609, -0.506, -0.880],
            [0.668, -0.497, 0.890],
            [0.668, -0.497, -0.890],
            [1.802, 1.311, 0.000],
        ],
        0,
        1,
        "RHF",
        "STO-3G",
    ),
    (
        "C₂H₆O (ethanol)  B3LYP/6-31G*",
        ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        [
            [-1.232, 0.026, 0.000],
            [0.281, 0.026, 0.000],
            [0.829, 1.310, 0.000],
            [-1.566, 1.059, 0.000],
            [-1.609, -0.506, 0.880],
            [-1.609, -0.506, -0.880],
            [0.668, -0.497, 0.890],
            [0.668, -0.497, -0.890],
            [1.802, 1.311, 0.000],
        ],
        0,
        1,
        "B3LYP",
        "6-31G*",
    ),
]

#: Extended suite for a full calibration run (~3–6 min on a modern laptop).
#: Includes the short suite plus larger molecules and more expensive methods
#: to anchor the efficiency model across the student-relevant size range.
BENCHMARK_SUITE_LONG: list[tuple] = [
    *BENCHMARK_SUITE,
    # ── Additional entries ─────────────────────────────────────────────────
    (
        "H₂O  RHF/cc-pVDZ",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "RHF",
        "cc-pVDZ",
    ),
    (
        "C₂H₆O (ethanol)  RHF/6-31G*",
        ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        [
            [-1.232, 0.026, 0.000],
            [0.281, 0.026, 0.000],
            [0.829, 1.310, 0.000],
            [-1.566, 1.059, 0.000],
            [-1.609, -0.506, 0.880],
            [-1.609, -0.506, -0.880],
            [0.668, -0.497, 0.890],
            [0.668, -0.497, -0.890],
            [1.802, 1.311, 0.000],
        ],
        0,
        1,
        "RHF",
        "6-31G*",
    ),
    (
        "C₆H₆ (benzene)  RHF/STO-3G",
        ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
        [
            [1.395, 0.000, 0.000],
            [0.698, 1.209, 0.000],
            [-0.698, 1.209, 0.000],
            [-1.395, 0.000, 0.000],
            [-0.698, -1.209, 0.000],
            [0.698, -1.209, 0.000],
            [2.479, 0.000, 0.000],
            [1.240, 2.147, 0.000],
            [-1.240, 2.147, 0.000],
            [-2.479, 0.000, 0.000],
            [-1.240, -2.147, 0.000],
            [1.240, -2.147, 0.000],
        ],
        0,
        1,
        "RHF",
        "STO-3G",
    ),
    (
        "C₆H₆ (benzene)  RHF/6-31G*",
        ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
        [
            [1.395, 0.000, 0.000],
            [0.698, 1.209, 0.000],
            [-0.698, 1.209, 0.000],
            [-1.395, 0.000, 0.000],
            [-0.698, -1.209, 0.000],
            [0.698, -1.209, 0.000],
            [2.479, 0.000, 0.000],
            [1.240, 2.147, 0.000],
            [-1.240, 2.147, 0.000],
            [-2.479, 0.000, 0.000],
            [-1.240, -2.147, 0.000],
            [1.240, -2.147, 0.000],
        ],
        0,
        1,
        "RHF",
        "6-31G*",
    ),
    (
        "C₆H₆ (benzene)  B3LYP/6-31G*",
        ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
        [
            [1.395, 0.000, 0.000],
            [0.698, 1.209, 0.000],
            [-0.698, 1.209, 0.000],
            [-1.395, 0.000, 0.000],
            [-0.698, -1.209, 0.000],
            [0.698, -1.209, 0.000],
            [2.479, 0.000, 0.000],
            [1.240, 2.147, 0.000],
            [-1.240, 2.147, 0.000],
            [-2.479, 0.000, 0.000],
            [-1.240, -2.147, 0.000],
            [1.240, -2.147, 0.000],
        ],
        0,
        1,
        "B3LYP",
        "6-31G*",
    ),
    (
        "C₁₀H₈ (naphthalene)  RHF/STO-3G",
        [
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "H",
            "H",
            "H",
            "H",
            "H",
            "H",
            "H",
            "H",
        ],
        [
            [1.243, 1.400, 0.000],
            [2.440, 0.725, 0.000],
            [2.440, -0.725, 0.000],
            [1.243, -1.400, 0.000],
            [0.000, -0.720, 0.000],
            [0.000, 0.720, 0.000],
            [-1.243, 1.400, 0.000],
            [-2.440, 0.725, 0.000],
            [-2.440, -0.725, 0.000],
            [-1.243, -1.400, 0.000],
            [1.237, 2.488, 0.000],
            [3.377, 1.244, 0.000],
            [3.377, -1.244, 0.000],
            [1.237, -2.488, 0.000],
            [-1.237, -2.488, 0.000],
            [-3.377, -1.244, 0.000],
            [-3.377, 1.244, 0.000],
            [-1.237, 2.488, 0.000],
        ],
        0,
        1,
        "RHF",
        "STO-3G",
    ),
    # ── M-EST / EST.4 expansion (2026-05-25) ──────────────────────────────
    # Additional SP entries that broaden the method × basis grid coverage,
    # extending tier 2's expected wall-clock to the 3-5 min target.
    (
        "H₂O  B3LYP/6-31G*",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "B3LYP",
        "6-31G*",
    ),
    (
        "H₂O  wB97X-D/6-31G*",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "wB97X-D",
        "6-31G*",
    ),
    (
        "CH₄  B3LYP/6-31G*",
        ["C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ],
        0,
        1,
        "B3LYP",
        "6-31G*",
    ),
    (
        "NH₃  RHF/cc-pVDZ",
        ["N", "H", "H", "H"],
        [
            [0.000, 0.000, 0.111],
            [0.000, 0.940, -0.260],
            [0.814, -0.470, -0.260],
            [-0.814, -0.470, -0.260],
        ],
        0,
        1,
        "RHF",
        "cc-pVDZ",
    ),
    (
        "NH₃  B3LYP/cc-pVDZ",
        ["N", "H", "H", "H"],
        [
            [0.000, 0.000, 0.111],
            [0.000, 0.940, -0.260],
            [0.814, -0.470, -0.260],
            [-0.814, -0.470, -0.260],
        ],
        0,
        1,
        "B3LYP",
        "cc-pVDZ",
    ),
    (
        "H₂CO (formaldehyde)  B3LYP/6-31G*",
        ["C", "O", "H", "H"],
        [
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 1.207],
            [0.000, 0.943, -0.589],
            [0.000, -0.943, -0.589],
        ],
        0,
        1,
        "B3LYP",
        "6-31G*",
    ),
]


# ---------------------------------------------------------------------------
# Tier 3 — Mixed (~10-15 min): tier 2 + small GeoOpts + small Freqs
# ---------------------------------------------------------------------------
#
# 8-tuple entries override the default ``"single_point"`` calc-type. The 8th
# element is one of ``"geometry_opt"`` / ``"frequency"``.
#
# Small geometry opts (3-5 atoms) and the cheapest realistic frequency calc
# (H₂O / B3LYP / STO-3G) anchor the multi-calc-type predictions without
# blowing the time budget.

BENCHMARK_SUITE_TIER3: list[tuple] = [
    *BENCHMARK_SUITE_LONG,
    # ── Small GeoOpts ─────────────────────────────────────────────────────
    (
        "H₂O  B3LYP/STO-3G  [GeoOpt]",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "B3LYP",
        "STO-3G",
        "geometry_opt",
    ),
    (
        "H₂CO  B3LYP/6-31G*  [GeoOpt]",
        ["C", "O", "H", "H"],
        [
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 1.207],
            [0.000, 0.943, -0.589],
            [0.000, -0.943, -0.589],
        ],
        0,
        1,
        "B3LYP",
        "6-31G*",
        "geometry_opt",
    ),
    (
        "CH₄  B3LYP/6-31G*  [GeoOpt]",
        ["C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ],
        0,
        1,
        "B3LYP",
        "6-31G*",
        "geometry_opt",
    ),
    # ── Small Freqs (cheapest realistic anchors for the 6N inner-SCF model) ──
    (
        "H₂O  B3LYP/STO-3G  [Freq]",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "B3LYP",
        "STO-3G",
        "frequency",
    ),
    (
        "H₂CO  B3LYP/6-31G*  [Freq]",
        ["C", "O", "H", "H"],
        [
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 1.207],
            [0.000, 0.943, -0.589],
            [0.000, -0.943, -0.589],
        ],
        0,
        1,
        "B3LYP",
        "6-31G*",
        "frequency",
    ),
]


# ---------------------------------------------------------------------------
# Tier 4 — Deep (up to 30 min): tier 3 + medium GeoOpt + medium Freq + MP2/CCSD
# ---------------------------------------------------------------------------
#
# Medium-size geometry opt + medium-size frequency anchors the predictor
# across realistic molecule sizes. MP2 + CCSD entries on H₂O / cc-pVDZ
# anchor the β=5.0 (MP2) and β=6.0 (CCSD) scaling exponents in
# ``calc_log._METHOD_SCALE_EXP``. The benzene frequency is the workhorse
# parallel-IR test — 12 atoms × 6 = 72 inner SCFs.

BENCHMARK_SUITE_TIER4: list[tuple] = [
    *BENCHMARK_SUITE_TIER3,
    # ── Medium GeoOpt ─────────────────────────────────────────────────────
    (
        "C₂H₆O (ethanol)  B3LYP/6-31G*  [GeoOpt]",
        ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        [
            [-1.232, 0.026, 0.000],
            [0.281, 0.026, 0.000],
            [0.829, 1.310, 0.000],
            [-1.566, 1.059, 0.000],
            [-1.609, -0.506, 0.880],
            [-1.609, -0.506, -0.880],
            [0.668, -0.497, 0.890],
            [0.668, -0.497, -0.890],
            [1.802, 1.311, 0.000],
        ],
        0,
        1,
        "B3LYP",
        "6-31G*",
        "geometry_opt",
    ),
    # ── Medium Freq ───────────────────────────────────────────────────────
    (
        "C₂H₆O (ethanol)  B3LYP/6-31G*  [Freq]",
        ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        [
            [-1.232, 0.026, 0.000],
            [0.281, 0.026, 0.000],
            [0.829, 1.310, 0.000],
            [-1.566, 1.059, 0.000],
            [-1.609, -0.506, 0.880],
            [-1.609, -0.506, -0.880],
            [0.668, -0.497, 0.890],
            [0.668, -0.497, -0.890],
            [1.802, 1.311, 0.000],
        ],
        0,
        1,
        "B3LYP",
        "6-31G*",
        "frequency",
    ),
    (
        "C₆H₆ (benzene)  B3LYP/6-31G*  [Freq]",
        ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
        [
            [1.395, 0.000, 0.000],
            [0.698, 1.209, 0.000],
            [-0.698, 1.209, 0.000],
            [-1.395, 0.000, 0.000],
            [-0.698, -1.209, 0.000],
            [0.698, -1.209, 0.000],
            [2.479, 0.000, 0.000],
            [1.240, 2.147, 0.000],
            [-1.240, 2.147, 0.000],
            [-2.479, 0.000, 0.000],
            [-1.240, -2.147, 0.000],
            [1.240, -2.147, 0.000],
        ],
        0,
        1,
        "B3LYP",
        "6-31G*",
        "frequency",
    ),
    # ── Post-HF anchors ───────────────────────────────────────────────────
    (
        "H₂O  MP2/cc-pVDZ",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "MP2",
        "cc-pVDZ",
    ),
    (
        "H₂O  CCSD/cc-pVDZ",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        0,
        1,
        "CCSD",
        "cc-pVDZ",
    ),
]


# Aliases — keep BENCHMARK_SUITE / BENCHMARK_SUITE_LONG for back-compat
# (existing tests + app.py imports). New code should reference the
# tier-named constants for clarity.
BENCHMARK_SUITE_TIER1: list[tuple] = BENCHMARK_SUITE
BENCHMARK_SUITE_TIER2: list[tuple] = BENCHMARK_SUITE_LONG


# ---------------------------------------------------------------------------
# Mode-string → suite mapping
# ---------------------------------------------------------------------------
#
# ``run_calibration(mode=)`` accepts any of these strings. The legacy
# ``"short"`` / ``"long"`` aliases are kept so older callers (including
# pinned UI state) keep working.

_MODE_TO_SUITE: dict = {
    "tier1": BENCHMARK_SUITE_TIER1,
    "tier2": BENCHMARK_SUITE_TIER2,
    "tier3": BENCHMARK_SUITE_TIER3,
    "tier4": BENCHMARK_SUITE_TIER4,
    "short": BENCHMARK_SUITE_TIER1,
    "long": BENCHMARK_SUITE_TIER2,
}


def _normalize_entry(entry: tuple) -> dict:
    """Unpack a 7-tuple or 8-tuple benchmark entry into a uniform dict.

    7-tuple: ``(label, atoms, coords, charge, mult, method, basis)`` —
    defaults ``calc_type`` to ``"single_point"``.

    8-tuple: ``(label, atoms, coords, charge, mult, method, basis, calc_type)``
    — used by tier 3 + tier 4 entries that need ``"geometry_opt"`` or
    ``"frequency"`` dispatch.
    """
    if len(entry) == 7:
        label, atoms, coords, charge, mult, method, basis = entry
        calc_type = "single_point"
    elif len(entry) == 8:
        label, atoms, coords, charge, mult, method, basis, calc_type = entry
    else:
        raise ValueError(
            f"Benchmark entry must have 7 or 8 fields, got {len(entry)}: {entry!r}"
        )
    return {
        "label": label,
        "atoms": atoms,
        "coords": coords,
        "charge": charge,
        "multiplicity": mult,
        "method": method,
        "basis": basis,
        "calc_type": calc_type,
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

_STATUS_OK = "ok"
_STATUS_TIMEOUT = "timed_out"
_STATUS_STOPPED = "stopped"
_STATUS_ERROR = "error"


@dataclass
class BenchmarkStep:
    """Result for a single benchmark step."""

    label: str
    method: str
    basis: str
    n_atoms: int
    n_electrons: int
    status: str  # "ok" | "timed_out" | "stopped" | "error"
    elapsed_s: float = 0.0
    error_msg: str = ""
    n_basis: Optional[int] = None
    # M-EST / EST.4: track which calc-type this step ran so tier 3+4
    # entries can be distinguished in summaries.
    calc_type: str = "single_point"


@dataclass
class CalibrationResult:
    """Summary result from :func:`run_calibration`."""

    timestamp: str
    steps: List[BenchmarkStep] = field(default_factory=list)
    stopped_early: bool = False
    mode: str = "tier1"

    @property
    def n_completed(self) -> int:
        return sum(1 for s in self.steps if s.status == _STATUS_OK)

    @property
    def n_total(self) -> int:
        return len(_MODE_TO_SUITE.get(self.mode, BENCHMARK_SUITE_TIER1))


# ---------------------------------------------------------------------------
# Main calibration runner
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int, str, str, float], None]
"""progress_cb(step_n, total, label, status, elapsed_s)"""


def _count_electrons(atoms: list[str], charge: int) -> int:
    """Rough electron count: sum of atomic numbers minus charge."""
    _Z = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
    }
    return sum(_Z.get(a, 6) for a in atoms) - charge


# ---------------------------------------------------------------------------
# Subprocess worker (M-EST follow-up, 2026-05-25)
# ---------------------------------------------------------------------------
#
# Originally calibration ran each step in a ThreadPoolExecutor with a
# ``future.result(timeout=...)`` block. That had three blockers exposed by
# the user's tier-4 attempt (session 55):
#
#   1. The Stop button only checked between steps, so an in-flight 5-minute
#      freq calc could not be killed mid-run.
#   2. There was no per-step progress signal beyond a single "running"
#      label — the user couldn't tell whether a slow step had frozen the
#      kernel.
#   3. ``calibration.json`` was only flushed at the END of the loop, so
#      stopping at step 25/30 lost the partial-state marker.
#
# The fix runs each step in a child process via ``multiprocessing.Process``
# so ``worker.terminate()`` works reliably cross-platform. The worker pipes
# PySCF's progress stream to a calibration log file the main process tails
# every 500 ms for the live status display, and ``calibration.json`` is
# rewritten after each completed step.


def _calibration_worker(
    atoms: list,
    coords: list,
    charge: int,
    mult: int,
    method: str,
    basis: str,
    calc_type: str,
    log_path_str: str,
    result_queue,
) -> None:
    """Run one calibration step in a child process.

    Picklable (top-level function, primitive args + a Queue). Pipes
    PySCF progress to ``log_path_str`` (append mode) so the parent can
    tail it. Puts a dict with status / formula / n_iterations /
    converged / elapsed_s on ``result_queue`` when done.

    On exception, puts ``{"status": "error", "error_msg": ...}``. The
    parent treats absence of a queue entry (after worker exit) as a
    crashed worker — distinct from a step-level error.
    """
    import time as _t
    from datetime import datetime as _dt
    from pathlib import Path as _P

    log_path = _P(log_path_str)
    t0 = _t.perf_counter()
    label = f"{method}/{basis}  ({calc_type})"

    try:
        # Line-buffered append so the parent's tail sees output as it
        # arrives. ``buffering=1`` requires text mode (which we use).
        with open(log_path, "a", encoding="utf-8", buffering=1) as log_fh:
            log_fh.write(
                f"\n========= {_dt.utcnow().isoformat()} :: {label} =========\n"
            )

            from quantui.molecule import Molecule as _Molecule

            mol = _Molecule(atoms, coords, charge=charge, multiplicity=mult)

            if calc_type == "geometry_opt":
                from quantui.optimizer import optimize_geometry as _opt

                res = _opt(
                    molecule=mol,
                    method=method,
                    basis=basis,
                    progress_stream=log_fh,
                )
                formula = res.molecule.get_formula()
                converged = bool(res.converged)
                n_iterations = int(getattr(res, "n_steps", -1))
            elif calc_type == "frequency":
                from quantui.freq_calc import run_freq_calc as _freq

                res = _freq(
                    molecule=mol,
                    method=method,
                    basis=basis,
                    progress_stream=log_fh,
                )
                formula = res.formula
                converged = bool(res.converged)
                n_iterations = int(res.n_iterations)
            else:  # single_point
                from quantui.session_calc import run_in_session as _sp

                # verbose=3 gives per-iteration SCF energies in the log —
                # enough signal to confirm the worker hasn't frozen on a
                # slow tier-4 entry. (Was verbose=0 pre-session-55.)
                res = _sp(
                    mol,
                    method=method,
                    basis=basis,
                    verbose=3,
                    progress_stream=log_fh,
                )
                formula = res.formula
                converged = bool(res.converged)
                n_iterations = int(res.n_iterations)

            elapsed = _t.perf_counter() - t0
            log_fh.write(f"\n[QuantUI_STATUS] COMPLETED in {elapsed:.2f} s\n")

            result_queue.put(
                {
                    "status": "ok",
                    "formula": formula,
                    "converged": converged,
                    "n_iterations": n_iterations,
                    "elapsed_s": elapsed,
                }
            )
    except Exception as exc:
        result_queue.put(
            {
                "status": "error",
                "error_msg": str(exc)[:500],
                "elapsed_s": _t.perf_counter() - t0,
            }
        )


def _tail_last_status_line(log_path) -> str:
    """Return the last meaningful progress line from the calibration log.

    Prefers ``[QuantUI_STATUS] ...`` markers emitted by ``freq_calc``;
    falls back to any non-blank line. Truncated to ~120 chars so the
    UI widget renders cleanly. Returns "" on any IO failure (best-
    effort).
    """
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return ""
    # Walk backwards looking for the best candidate.
    status_line = ""
    fallback_line = ""
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if "[QuantUI_STATUS]" in stripped:
            status_line = stripped
            break
        if not fallback_line:
            fallback_line = stripped
    best = status_line or fallback_line
    if len(best) > 120:
        best = best[-120:]
    return best


def _calibration_log_path(timestamp: str) -> Path:
    """Return the path to the per-run calibration log file.

    Filename includes the run timestamp so multiple runs don't clobber
    each other. Lives under ``~/.quantui/logs/`` (honours
    ``QUANTUI_LOG_DIR``) alongside the event + perf logs.
    """
    import os as _os

    env = _os.environ.get("QUANTUI_LOG_DIR")
    base = Path(env) if env else Path.home() / ".quantui" / "logs"
    # Make a filename-safe timestamp.
    safe_ts = timestamp.replace(":", "-").replace(".", "-")
    return base / f"calibration_{safe_ts}.log"


def _save_calibration_json(result: CalibrationResult, log_path: Path) -> None:
    """Persist the current ``CalibrationResult`` snapshot to disk.

    Called after EVERY completed step (not just at end-of-run) so an
    interrupted tier-4 still records the partial-state marker the user
    can see next session. Includes the log file path so the "last
    calibration" UI can link to the per-run log.
    """
    import json as _json

    cal_path = Path.home() / ".quantui" / "calibration.json"
    try:
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        cal_path.write_text(
            _json.dumps(
                {
                    "timestamp": result.timestamp,
                    "mode": result.mode,
                    "stopped_early": result.stopped_early,
                    "log_path": str(log_path),
                    "n_completed": result.n_completed,
                    "n_total": result.n_total,
                    "steps": [
                        {
                            "label": s.label,
                            "method": s.method,
                            "basis": s.basis,
                            "n_atoms": s.n_atoms,
                            "n_electrons": s.n_electrons,
                            "n_basis": s.n_basis,
                            "status": s.status,
                            "elapsed_s": round(s.elapsed_s, 3),
                            "error_msg": s.error_msg,
                            "calc_type": s.calc_type,
                        }
                        for s in result.steps
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        # Disk full / permission denied — best-effort. The perf log is
        # the canonical record; calibration.json is just a UI summary.
        pass


def run_calibration(
    progress_cb: Optional[ProgressCallback] = None,
    stop_event=None,
    timeout_per_step: float = 120.0,
    mode: str = "tier1",
) -> CalibrationResult:
    """Run the benchmark suite and populate ``perf_log.jsonl``.

    Each step runs in a child process so the Stop button can terminate
    a long-running calc mid-run. Per-step progress is piped to a log
    file under ``~/.quantui/logs/calibration_<timestamp>.log`` and the
    parent tails it every 500 ms to drive the live status display.
    ``~/.quantui/calibration.json`` is rewritten after every completed
    step, so an interrupted run still records partial state.

    Args:
        progress_cb: Called periodically with
            ``(step_n, total, label, status, elapsed_s)`` and optionally
            ``live_message=<latest log line>`` during slow steps. The
            terminal call after each step uses status in
            ``ok / timed_out / stopped / error``; intermediate "running"
            ticks fire while the step is in-flight.
        stop_event: A :class:`threading.Event`; checked every 500 ms.
            When set, the in-flight worker is terminated immediately
            and the current step is marked ``"stopped"``.
        timeout_per_step: Wall-clock seconds allowed per step. Defaults
            to 120 s — fine for tier 1 / tier 2 (SP only). Caller
            should bump for tier 3 (~900 s) and tier 4 (~1800 s).
        mode: One of ``"tier1"`` / ``"tier2"`` / ``"tier3"`` / ``"tier4"``.
            Legacy aliases ``"short"`` / ``"long"`` map to tier1 / tier2.
            Unknown modes fall back to tier1 with a warning.

    Returns:
        :class:`CalibrationResult` with per-step outcomes.
    """
    import multiprocessing as _mp
    import queue as _queue

    from quantui import calc_log as _calc_log

    _pyscf_available = False
    try:
        import pyscf  # noqa: F401

        _pyscf_available = True
    except ImportError:
        pass

    if mode not in _MODE_TO_SUITE:
        import logging as _log

        _log.getLogger(__name__).warning(
            "run_calibration: unknown mode %r, falling back to tier1", mode
        )
        mode = "tier1"
    suite = _MODE_TO_SUITE[mode]
    timestamp = datetime.now(timezone.utc).isoformat()
    result = CalibrationResult(timestamp=timestamp, mode=mode)
    total = len(suite)

    # Per-run calibration log file. The worker appends; the parent tails.
    log_path = _calibration_log_path(timestamp)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"QuantUI calibration log\n"
                f"started   : {timestamp}\n"
                f"mode      : {mode}\n"
                f"suite size: {total} entries\n"
                f"timeout/step: {timeout_per_step:.0f} s\n"
            )
    except OSError:
        # No log file is non-fatal — calibration still runs, just without
        # the per-step progress trail.
        pass

    # Use ``spawn`` everywhere (session 55 follow-up): ``fork`` from a
    # background thread (run_calibration runs inside ``_do_calibration``
    # which is itself a daemon thread) collides hard with CUDA contexts
    # that the parent process may have initialized via the GPU-detection
    # probe — every step would die at ~0.04 s with no useful error.
    # ``spawn`` adds ~1-2 s startup overhead per step but isolates the
    # worker from the parent's interpreter state entirely, so CUDA / MPI /
    # any C-extension global is freshly initialized. Sub-2-second-per-step
    # overhead is a great trade for "the Stop button works AND nothing
    # crashes for opaque reasons".
    _ctx = _mp.get_context("spawn")

    def _emit_progress(*args, live_message=None, step=None) -> None:
        """Wrap progress_cb to tolerate callers that pre-date the
        ``live_message`` / ``step`` kwargs (notably the test-suite
        lambdas that accept ``*args`` only). Falls back through each
        new kwarg in turn on ``TypeError``."""
        if progress_cb is None:
            return
        # Try newest signature first, peel off kwargs the caller can't
        # accept. Modern callers (do_calibration) take both; tests pass
        # ``lambda *a: ...``.
        try:
            progress_cb(*args, live_message=live_message, step=step)
            return
        except TypeError:
            pass
        try:
            progress_cb(*args, live_message=live_message)
            return
        except TypeError:
            pass
        progress_cb(*args)

    stopped_mid_step = False
    for step_n, entry in enumerate(suite, start=1):
        normalized = _normalize_entry(entry)
        label = normalized["label"]
        atoms = normalized["atoms"]
        coords = normalized["coords"]
        charge = normalized["charge"]
        mult = normalized["multiplicity"]
        method = normalized["method"]
        basis = normalized["basis"]
        calc_type = normalized["calc_type"]

        # Honour stop request BEFORE starting a new step.
        if stop_event is not None and stop_event.is_set():
            result.stopped_early = True
            break

        nb = _calc_log.count_basis_functions(atoms, basis)
        step = BenchmarkStep(
            label=label,
            method=method,
            basis=basis,
            n_atoms=len(atoms),
            n_electrons=_count_electrons(atoms, charge),
            status=_STATUS_ERROR,
            n_basis=nb,
            calc_type=calc_type,
        )

        if not _pyscf_available:
            step.error_msg = "PySCF not available"
            result.steps.append(step)
            _save_calibration_json(result, log_path)
            _emit_progress(step_n, total, label, step.status, 0.0, step=step)
            continue

        # Spawn the worker.
        result_queue = _ctx.Queue()
        worker = _ctx.Process(
            target=_calibration_worker,
            args=(
                atoms,
                coords,
                charge,
                mult,
                method,
                basis,
                calc_type,
                str(log_path),
                result_queue,
            ),
            daemon=True,
        )
        t_start = time.perf_counter()
        worker.start()

        # Poll loop — finish naturally OR hit timeout OR receive stop signal.
        poll_interval = 0.5
        worker_done_normally = False
        while True:
            worker.join(timeout=poll_interval)
            elapsed = time.perf_counter() - t_start

            if not worker.is_alive():
                worker_done_normally = True
                break

            if elapsed > timeout_per_step:
                worker.terminate()
                worker.join(timeout=5)
                step.status = _STATUS_TIMEOUT
                step.elapsed_s = elapsed
                step.error_msg = f"exceeded {timeout_per_step:.0f}s timeout"
                break

            if stop_event is not None and stop_event.is_set():
                worker.terminate()
                worker.join(timeout=5)
                step.status = _STATUS_STOPPED
                step.elapsed_s = elapsed
                result.stopped_early = True
                stopped_mid_step = True
                break

            # Live-tick: pull the latest log line for the UI.
            live_msg = _tail_last_status_line(log_path)
            _emit_progress(
                step_n, total, label, "running", elapsed, live_message=live_msg
            )

        if worker_done_normally:
            try:
                msg = result_queue.get(timeout=2.0)
            except _queue.Empty:
                # Worker process exited (either crashed during import,
                # raised before reaching the worker's try/except, or
                # was killed by the OS) without putting anything on
                # the queue. Capture the exit code + the tail of the
                # calibration log so the user can see what actually
                # happened — "worker exited without result" alone is
                # useless for diagnosis (the original session-55
                # symptom of every step failing at 0.04 s).
                _exitcode = getattr(worker, "exitcode", None)
                _tail = _tail_last_status_line(log_path) or "(no log output)"
                _hint = ""
                if _exitcode is not None and _exitcode != 0:
                    # On Unix, negative exit codes encode the signal
                    # that killed the process (-9 = SIGKILL, -11 = SEGV).
                    if _exitcode < 0:
                        import signal as _sig

                        try:
                            _sig_name = _sig.Signals(-_exitcode).name
                            _hint = f" (killed by {_sig_name})"
                        except (ValueError, AttributeError):
                            _hint = f" (signal {-_exitcode})"
                msg = {
                    "status": "error",
                    "error_msg": (
                        f"worker exited (exitcode={_exitcode}){_hint}; "
                        f"last log line: {_tail}"
                    )[:500],
                    "elapsed_s": time.perf_counter() - t_start,
                }
            if msg.get("status") == "ok":
                step.status = _STATUS_OK
                step.elapsed_s = float(msg["elapsed_s"])
                # Log to perf_log.jsonl so estimate_time() picks it up.
                _calc_log.log_calculation(
                    formula=msg["formula"],
                    n_atoms=step.n_atoms,
                    n_electrons=step.n_electrons,
                    method=method,
                    basis=basis,
                    n_iterations=int(msg.get("n_iterations", -1)),
                    elapsed_s=float(msg["elapsed_s"]),
                    converged=bool(msg["converged"]),
                    n_basis=step.n_basis,
                    n_cores=1,
                    calc_type=calc_type,
                )
            else:
                step.status = _STATUS_ERROR
                step.error_msg = msg.get("error_msg", "unknown")
                step.elapsed_s = float(
                    msg.get("elapsed_s", time.perf_counter() - t_start)
                )

        result.steps.append(step)
        # Fix 2: persist after EVERY step so an interrupt at step N
        # still leaves a partial-state record on disk.
        _save_calibration_json(result, log_path)

        # Terminal call for this step — pass the full BenchmarkStep so
        # the UI callback can append it to the incremental results table.
        _emit_progress(step_n, total, label, step.status, step.elapsed_s, step=step)

        if stopped_mid_step:
            break

    # Final write (idempotent — same content as the last per-step write
    # unless the loop broke via the top-of-loop stop check).
    _save_calibration_json(result, log_path)
    return result


def load_last_calibration() -> Optional[dict]:
    """Return the last calibration summary dict, or ``None`` if absent."""
    import json

    path = Path.home() / ".quantui" / "calibration.json"
    if not path.exists():
        return None
    try:
        data: dict = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception:
        return None
