"""Unit tests for `quantui.viz_backend_router.select_backend`.

The router is a pure function — no I/O, no widget state — so these tests are
exhaustive across the availability × preference × task matrix without any
mocking infrastructure.
"""

from __future__ import annotations

import dataclasses

import pytest

from quantui.viz_backend_router import (
    BackendAvailability,
    Decision,
    VizBackend,
    VizPreference,
    VizTask,
    select_backend,
)

BOTH = BackendAvailability(py3dmol=True, plotlymol=True)
ONLY_PY3DMOL = BackendAvailability(py3dmol=True, plotlymol=False)
ONLY_PLOTLYMOL = BackendAvailability(py3dmol=False, plotlymol=True)
NEITHER = BackendAvailability(py3dmol=False, plotlymol=False)


# Tasks whose policy permits either backend (multi-backend tasks).
_DUAL_BACKEND_TASKS = [
    VizTask.MOLECULE_PREVIEW,
    VizTask.STRUCTURE_VIEW_RESULTS,
    VizTask.ANALYSIS_STRUCTURE_VIEW,
    VizTask.HISTORY_STRUCTURE_REPLAY,
    VizTask.VIB_INTERACTIVE,
]

# Tasks that require plotlymol3d regardless of preference.
_PLOTLYMOL_ONLY_TASKS = [
    VizTask.TRAJECTORY_EXPORT,
    VizTask.VIB_EXPORT,
    VizTask.ORBITAL_ISOSURFACE,
]

# Tasks that require py3Dmol regardless of preference.
_PY3DMOL_ONLY_TASKS = [
    VizTask.TRAJECTORY_FRAME,
]


class TestStrEnumBehavior:
    """StrEnum members must act as str — verifies the choice made in VIZBACK.1
    so log events and JSON serialization don't need `.value` unwrapping."""

    def test_viz_task_is_str(self):
        assert VizTask.TRAJECTORY_FRAME == "trajectory_frame"
        assert f"task={VizTask.TRAJECTORY_FRAME}" == "task=trajectory_frame"

    def test_viz_preference_is_str(self):
        assert VizPreference.AUTO == "auto"
        assert VizPreference.PY3DMOL == "py3dmol"
        assert VizPreference.PLOTLYMOL == "plotlymol"

    def test_viz_backend_is_str(self):
        assert VizBackend.PY3DMOL == "py3dmol"
        assert VizBackend.PLOTLYMOL == "plotlymol"


class TestBackendAvailability:
    def test_supports_py3dmol(self):
        assert ONLY_PY3DMOL.supports(VizBackend.PY3DMOL) is True
        assert ONLY_PY3DMOL.supports(VizBackend.PLOTLYMOL) is False

    def test_supports_plotlymol(self):
        assert ONLY_PLOTLYMOL.supports(VizBackend.PY3DMOL) is False
        assert ONLY_PLOTLYMOL.supports(VizBackend.PLOTLYMOL) is True

    def test_supports_neither(self):
        assert NEITHER.supports(VizBackend.PY3DMOL) is False
        assert NEITHER.supports(VizBackend.PLOTLYMOL) is False

    def test_from_environment_returns_bool_fields(self):
        """from_environment should always return an instance (boolean fields
        reflect the runtime environment — we don't assert on their values)."""
        avail = BackendAvailability.from_environment()
        assert isinstance(avail.py3dmol, bool)
        assert isinstance(avail.plotlymol, bool)


class TestDualBackendTasksAuto:
    """Auto preference should pick py3Dmol primary for all dual-backend tasks."""

    @pytest.mark.parametrize("task", _DUAL_BACKEND_TASKS)
    def test_auto_with_both_available_picks_py3dmol(self, task):
        decision = select_backend(task, VizPreference.AUTO, BOTH)
        assert decision.chosen == VizBackend.PY3DMOL
        assert decision.fallback == VizBackend.PLOTLYMOL
        assert "auto" in decision.reason

    @pytest.mark.parametrize("task", _DUAL_BACKEND_TASKS)
    def test_auto_with_only_py3dmol(self, task):
        decision = select_backend(task, VizPreference.AUTO, ONLY_PY3DMOL)
        assert decision.chosen == VizBackend.PY3DMOL
        assert decision.fallback is None

    @pytest.mark.parametrize("task", _DUAL_BACKEND_TASKS)
    def test_auto_with_only_plotlymol_falls_back(self, task):
        decision = select_backend(task, VizPreference.AUTO, ONLY_PLOTLYMOL)
        assert decision.chosen == VizBackend.PLOTLYMOL
        assert decision.fallback is None
        assert "fell back" in decision.reason

    @pytest.mark.parametrize("task", _DUAL_BACKEND_TASKS)
    def test_auto_with_neither_returns_none(self, task):
        decision = select_backend(task, VizPreference.AUTO, NEITHER)
        assert decision.chosen is None
        assert decision.fallback is None
        assert "no available backend" in decision.reason


class TestDualBackendTasksExplicitPreference:
    @pytest.mark.parametrize("task", _DUAL_BACKEND_TASKS)
    def test_explicit_py3dmol_with_both(self, task):
        decision = select_backend(task, VizPreference.PY3DMOL, BOTH)
        assert decision.chosen == VizBackend.PY3DMOL
        assert decision.fallback == VizBackend.PLOTLYMOL
        assert "user preference" in decision.reason

    @pytest.mark.parametrize("task", _DUAL_BACKEND_TASKS)
    def test_explicit_plotlymol_with_both(self, task):
        decision = select_backend(task, VizPreference.PLOTLYMOL, BOTH)
        assert decision.chosen == VizBackend.PLOTLYMOL
        assert decision.fallback == VizBackend.PY3DMOL
        assert "user preference" in decision.reason

    @pytest.mark.parametrize("task", _DUAL_BACKEND_TASKS)
    def test_explicit_py3dmol_when_only_plotlymol_available(self, task):
        decision = select_backend(task, VizPreference.PY3DMOL, ONLY_PLOTLYMOL)
        assert decision.chosen == VizBackend.PLOTLYMOL
        assert decision.fallback is None
        assert "unavailable" in decision.reason

    @pytest.mark.parametrize("task", _DUAL_BACKEND_TASKS)
    def test_explicit_plotlymol_when_only_py3dmol_available(self, task):
        decision = select_backend(task, VizPreference.PLOTLYMOL, ONLY_PY3DMOL)
        assert decision.chosen == VizBackend.PY3DMOL
        assert decision.fallback is None
        assert "unavailable" in decision.reason


class TestSingleBackendTasksIgnorePreference:
    """Export and isosurface tasks require plotlymol3d; preference must not
    change that decision (only availability can)."""

    @pytest.mark.parametrize("task", _PLOTLYMOL_ONLY_TASKS)
    @pytest.mark.parametrize(
        "preference",
        [VizPreference.AUTO, VizPreference.PY3DMOL, VizPreference.PLOTLYMOL],
    )
    def test_picks_plotlymol_when_available(self, task, preference):
        decision = select_backend(task, preference, BOTH)
        assert decision.chosen == VizBackend.PLOTLYMOL
        assert decision.fallback is None
        assert "requires" in decision.reason

    @pytest.mark.parametrize("task", _PLOTLYMOL_ONLY_TASKS)
    @pytest.mark.parametrize(
        "preference",
        [VizPreference.AUTO, VizPreference.PY3DMOL, VizPreference.PLOTLYMOL],
    )
    def test_returns_none_when_plotlymol_unavailable(self, task, preference):
        decision = select_backend(task, preference, ONLY_PY3DMOL)
        assert decision.chosen is None
        assert decision.fallback is None
        assert "unavailable" in decision.reason


class TestPy3DmolOnlyTasksIgnorePreference:
    """Trajectory frame browsing requires py3Dmol regardless of preference —
    plotlymol is blocked from real-time trajectory rendering to avoid the
    RequireJS flicker issue."""

    @pytest.mark.parametrize("task", _PY3DMOL_ONLY_TASKS)
    @pytest.mark.parametrize(
        "preference",
        [VizPreference.AUTO, VizPreference.PY3DMOL, VizPreference.PLOTLYMOL],
    )
    def test_picks_py3dmol_when_available(self, task, preference):
        decision = select_backend(task, preference, BOTH)
        assert decision.chosen == VizBackend.PY3DMOL
        assert decision.fallback is None
        assert "requires" in decision.reason

    @pytest.mark.parametrize("task", _PY3DMOL_ONLY_TASKS)
    @pytest.mark.parametrize(
        "preference",
        [VizPreference.AUTO, VizPreference.PY3DMOL, VizPreference.PLOTLYMOL],
    )
    def test_returns_none_when_py3dmol_unavailable(self, task, preference):
        decision = select_backend(task, preference, ONLY_PLOTLYMOL)
        assert decision.chosen is None
        assert decision.fallback is None
        assert "unavailable" in decision.reason


class TestDecisionShape:
    def test_decision_is_immutable(self):
        decision = select_backend(VizTask.MOLECULE_PREVIEW, VizPreference.AUTO, BOTH)
        with pytest.raises(dataclasses.FrozenInstanceError):
            decision.chosen = VizBackend.PLOTLYMOL  # type: ignore[misc]

    def test_decision_reason_is_nonempty(self):
        """Every decision must explain itself for log telemetry."""
        for task in VizTask:
            for preference in VizPreference:
                for avail in (BOTH, ONLY_PY3DMOL, ONLY_PLOTLYMOL, NEITHER):
                    decision = select_backend(task, preference, avail)
                    assert decision.reason, (
                        f"empty reason for task={task} "
                        f"preference={preference} availability={avail}"
                    )


class TestFullMatrix:
    """Exhaustive matrix: every task × every preference × every availability
    state should return a valid Decision with consistent fields."""

    def test_every_combination_returns_decision(self):
        for task in VizTask:
            for preference in VizPreference:
                for avail in (BOTH, ONLY_PY3DMOL, ONLY_PLOTLYMOL, NEITHER):
                    decision = select_backend(task, preference, avail)
                    assert isinstance(decision, Decision)
                    # chosen is either None or a real VizBackend
                    assert decision.chosen is None or isinstance(
                        decision.chosen, VizBackend
                    )
                    # fallback is None or a real VizBackend that differs from chosen
                    if decision.fallback is not None:
                        assert isinstance(decision.fallback, VizBackend)
                        assert decision.fallback != decision.chosen
                    # If chosen is set, the corresponding availability is True
                    if decision.chosen == VizBackend.PY3DMOL:
                        assert avail.py3dmol
                    if decision.chosen == VizBackend.PLOTLYMOL:
                        assert avail.plotlymol
