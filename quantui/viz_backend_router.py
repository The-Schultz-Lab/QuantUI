"""
3D visualization backend router for QuantUI.

Resolves which 3D rendering backend (py3Dmol or plotlymol3d) to use for a given
render task, taking into account user preference and installed-package
availability. Pure function — no I/O, no widget state, no app reference.

The routing policy mirrors a capability-based routing policy table.

Typical usage
-------------
>>> from quantui.viz_backend_router import (
...     BackendAvailability, VizPreference, VizTask, select_backend,
... )
>>> avail = BackendAvailability.from_environment()
>>> decision = select_backend(
...     task=VizTask.TRAJECTORY_FRAME,
...     preference=VizPreference.AUTO,
...     availability=avail,
... )
>>> if decision.chosen == "py3dmol":
...     ...  # render via py3Dmol
... elif decision.chosen == "plotlymol":
...     ...  # render via plotlymol3d
... else:
...     ...  # no renderer available
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum

# StrEnum landed in the stdlib in Python 3.11. Provide a behaviour-preserving
# shim on 3.10 (and 3.9, which pyproject still claims to support) so the
# router can use the cleaner inherited class. The shim matches StrEnum's key
# observable behaviours: members compare equal to plain strings, and both
# ``str(member)`` and ``f"{member}"`` return the underlying value rather than
# the dotted enum name. ``__str__ = str.__str__`` works because each member is
# already a real ``str`` instance whose content is the declared value.
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        __str__ = str.__str__


class VizTask(StrEnum):
    """Internal routing key for each visualization context."""

    MOLECULE_PREVIEW = "molecule_preview"
    STRUCTURE_VIEW_RESULTS = "structure_view_results"
    ANALYSIS_STRUCTURE_VIEW = "analysis_structure_view"
    HISTORY_STRUCTURE_REPLAY = "history_structure_replay"
    TRAJECTORY_FRAME = "trajectory_frame"
    TRAJECTORY_EXPORT = "trajectory_export"
    VIB_INTERACTIVE = "vib_interactive"
    VIB_EXPORT = "vib_export"
    ORBITAL_ISOSURFACE = "orbital_isosurface"


class VizPreference(StrEnum):
    """User-selectable backend preference. `AUTO` defers to the task's primary."""

    AUTO = "auto"
    PY3DMOL = "py3dmol"
    PLOTLYMOL = "plotlymol"


class VizBackend(StrEnum):
    """Concrete backend identifier returned by `select_backend`."""

    PY3DMOL = "py3dmol"
    PLOTLYMOL = "plotlymol"


@dataclass(frozen=True)
class BackendAvailability:
    """Which backends are importable in the current Python environment."""

    py3dmol: bool
    plotlymol: bool

    @classmethod
    def from_environment(cls) -> BackendAvailability:
        """Probe imports to detect installed backends. Call once at app startup."""
        try:
            import py3Dmol  # noqa: F401

            py3dmol_ok = True
        except ImportError:
            py3dmol_ok = False
        try:
            import plotlymol3d  # noqa: F401

            plotlymol_ok = True
        except ImportError:
            plotlymol_ok = False
        return cls(py3dmol=py3dmol_ok, plotlymol=plotlymol_ok)

    def supports(self, backend: VizBackend) -> bool:
        if backend == VizBackend.PY3DMOL:
            return self.py3dmol
        if backend == VizBackend.PLOTLYMOL:
            return self.plotlymol
        return False


@dataclass(frozen=True)
class Decision:
    """Result of a routing decision.

    `chosen` is None when no available backend can serve the requested task —
    callers should render a graceful unavailable-state message in that case.
    `fallback` reports the secondary option available for that task at the time
    of the decision (informational; not a promise to retry automatically).
    """

    chosen: VizBackend | None
    fallback: VizBackend | None
    reason: str


# Capability routing policy.
#
# Maps task -> (primary backend, optional fallback backend).
# Tasks whose fallback is None are single-backend — user preference is ignored
# for these. Single-backend rationale by task:
#   - TRAJECTORY_FRAME: py3Dmol-only. Plotlymol's RequireJS-driven re-render
#     pattern causes flicker when frames swap rapidly; py3Dmol's WebGL path
#     is the only viable real-time trajectory backend in this app.
#   - TRAJECTORY_EXPORT / VIB_EXPORT: plotlymol produces self-contained HTML
#     animations with embedded controls, which is the export contract.
# ORBITAL_ISOSURFACE is py3Dmol-ONLY as of 2026-08-04 (user decision). It was
# dual-backend, with the Plotly cube path selectable by preference. Two reasons
# it is not:
#
#   - Quality. Plotly renders isosurfaces by striding the volume down to a point
#     cap before building a mesh, so it is downsampled by construction, while
#     py3Dmol isosurfaces the cube in-browser at full resolution.
#   - Capability. Only py3Dmol can carry the Save-PNG capture (ORBX.1), and only
#     it can be told what the user is actually looking at, camera included.
#
# User: "plotlymol is really only good for the molecule viewing windows."
#
# ⚠️ This is deliberately reversible: restore the fallback by putting
# VizBackend.PLOTLYMOL back as the second element. The Plotly isosurface code in
# orbital_visualization.plot_cube_isosurface is kept and still tested — it is
# unreachable through the router, not deleted — so reverting is a one-line
# change rather than an archaeology exercise.
#
# py3Dmol is a REQUIRED dependency (pyproject: py3Dmol>=2.0.0,<3), so dropping
# the fallback does not create a realistic "no backend" hole; an install without
# it is already broken in ways an isosurface fallback would not rescue.
_TASK_POLICY: dict[VizTask, tuple[VizBackend, VizBackend | None]] = {
    VizTask.MOLECULE_PREVIEW: (VizBackend.PY3DMOL, VizBackend.PLOTLYMOL),
    VizTask.STRUCTURE_VIEW_RESULTS: (VizBackend.PY3DMOL, VizBackend.PLOTLYMOL),
    VizTask.ANALYSIS_STRUCTURE_VIEW: (VizBackend.PY3DMOL, VizBackend.PLOTLYMOL),
    VizTask.HISTORY_STRUCTURE_REPLAY: (VizBackend.PY3DMOL, VizBackend.PLOTLYMOL),
    VizTask.TRAJECTORY_FRAME: (VizBackend.PY3DMOL, None),
    # Same change and same reasoning as VIB_EXPORT (2026-08-04): the exported
    # file should be the animation the user was watching. The trajectory viewer
    # is py3Dmol, so the export reuses build_trajectory_viewer_html rather than
    # rebuilding it in Plotly.
    VizTask.TRAJECTORY_EXPORT: (VizBackend.PY3DMOL, None),
    VizTask.VIB_INTERACTIVE: (VizBackend.PY3DMOL, VizBackend.PLOTLYMOL),
    # VIB_EXPORT was PLOTLYMOL-only, on the reasoning that a Plotly animation
    # with embedded controls is the canonical "export quality" artifact. That
    # traded away the more important property: the exported file should be the
    # animation the user was looking at when they clicked Export. py3Dmol is
    # what renders the live vibrational viewer, so it is what exports now
    # (user decision 2026-08-04).
    VizTask.VIB_EXPORT: (VizBackend.PY3DMOL, None),
    VizTask.ORBITAL_ISOSURFACE: (VizBackend.PY3DMOL, None),
}


def select_backend(
    task: VizTask,
    preference: VizPreference,
    availability: BackendAvailability,
) -> Decision:
    """Resolve the backend to use for a given render task.

    Resolution order:

    1. If the task is single-backend (export and isosurface paths), user
       preference is ignored and the task's required backend is used if
       available; otherwise `Decision.chosen` is None.
    2. Otherwise, `preference` selects between py3Dmol and plotlymol3d.
       `AUTO` resolves to the task's primary backend.
    3. If the preferred backend is unavailable, fall back to the task's
       fallback backend if it is available.
    4. If neither is available, `Decision.chosen` is None.
    """
    primary, fallback_policy = _TASK_POLICY[task]

    # Single-backend tasks: preference is ignored. Used for export-quality
    # renders (trajectory/vib export HTML) and the orbital isosurface path.
    if fallback_policy is None:
        if availability.supports(primary):
            return Decision(
                chosen=primary,
                fallback=None,
                reason=f"task '{task}' requires {primary}",
            )
        return Decision(
            chosen=None,
            fallback=None,
            reason=f"task '{task}' requires {primary} but it is unavailable",
        )

    # Multi-backend tasks: resolve preference -> preferred backend.
    if preference == VizPreference.AUTO:
        preferred = primary
        reason_prefix = f"auto -> task primary ({primary})"
    elif preference == VizPreference.PY3DMOL:
        preferred = VizBackend.PY3DMOL
        reason_prefix = "user preference (py3dmol)"
    elif preference == VizPreference.PLOTLYMOL:
        preferred = VizBackend.PLOTLYMOL
        reason_prefix = "user preference (plotlymol)"
    else:  # defensive — should be unreachable given the StrEnum
        preferred = primary
        reason_prefix = f"unknown preference '{preference}' -> task primary ({primary})"

    if availability.supports(preferred):
        # Report the other backend (if available) as fallback for transparency.
        other = fallback_policy if preferred == primary else primary
        reported_fallback = other if availability.supports(other) else None
        return Decision(
            chosen=preferred,
            fallback=reported_fallback,
            reason=reason_prefix,
        )

    # Preferred is unavailable — try the policy fallback if it differs.
    if fallback_policy != preferred and availability.supports(fallback_policy):
        return Decision(
            chosen=fallback_policy,
            fallback=None,
            reason=(
                f"preferred {preferred} unavailable -> "
                f"fell back to {fallback_policy}"
            ),
        )

    # Some installs may have the policy primary but not the requested
    # preference's fallback. Try the primary as a last resort if it differs.
    if preferred != primary and availability.supports(primary):
        return Decision(
            chosen=primary,
            fallback=None,
            reason=(
                f"preferred {preferred} unavailable -> "
                f"fell back to task primary ({primary})"
            ),
        )

    return Decision(
        chosen=None,
        fallback=None,
        reason=(
            f"no available backend for task '{task}' "
            f"(preferred={preferred}, policy fallback={fallback_policy})"
        ),
    )
