"""Tests for VIZBACK.3 — Analysis-tab backend toggle parity and sync.

Verifies that the Calculate-tab and Analysis-tab backend toggles stay in
sync without observer feedback loops, that ``_set_viz_backend`` is a
single source of truth, and that ``_analysis_displayed_molecule`` tracking
allows the Analysis-tab viewer to re-render on toggle.
"""

from __future__ import annotations

import pytest

from quantui.app import QuantUIApp


@pytest.fixture
def app(tmp_path, monkeypatch):
    # Isolate settings (reflections/10 Rule 6): these tests set the persisting
    # Settings-dropdown backend, which would otherwise write the real
    # ~/.quantui/settings.json and cross-contaminate other tests' QuantUIApp()
    # construction under pytest-xdist (nondeterministic worker ordering).
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    return QuantUIApp()


class TestToggleParityWhenBothBackendsAvailable:
    """When both py3Dmol and plotlymol3d are installed, both toggles should
    exist and be wired together."""

    def test_analysis_toggle_exists_when_both_backends_available(self, app):
        # If the Calculate-tab toggle exists, the Analysis-tab toggle must
        # also exist (and vice versa).
        if app.viz_backend_toggle is not None:
            assert app.viz_backend_toggle_ana is not None
        else:
            assert app.viz_backend_toggle_ana is None

    def test_toggles_start_in_sync(self, app):
        if app.viz_backend_toggle is None:
            pytest.skip("Single-backend environment; sync N/A")
        assert app.viz_backend_toggle.value == app.viz_backend_toggle_ana.value
        assert app.viz_backend_toggle.value == app._viz_backend


class TestSyncBehavior:
    """Changing one toggle should update the other without echo loops."""

    def test_calculate_toggle_change_syncs_analysis(self, app):
        if app.viz_backend_toggle is None:
            pytest.skip("Single-backend environment; sync N/A")
        # Pick a value different from the current one.
        current = app.viz_backend_toggle.value
        new_value = "py3dmol" if current != "py3dmol" else "plotlymol"
        app.viz_backend_toggle.value = new_value
        assert app._viz_backend == new_value
        assert app.viz_backend_toggle_ana.value == new_value

    def test_analysis_toggle_change_syncs_calculate(self, app):
        if app.viz_backend_toggle_ana is None:
            pytest.skip("Single-backend environment; sync N/A")
        current = app.viz_backend_toggle_ana.value
        new_value = "py3dmol" if current != "py3dmol" else "plotlymol"
        app.viz_backend_toggle_ana.value = new_value
        assert app._viz_backend == new_value
        assert app.viz_backend_toggle.value == new_value

    def test_no_observer_echo_loop(self, app):
        """Repeated toggles must not cause unbounded recursion or duplicate
        state updates — flag should always end in cleared state."""
        if app.viz_backend_toggle is None:
            pytest.skip("Single-backend environment; sync N/A")
        for _ in range(3):
            app.viz_backend_toggle.value = "py3dmol"
            assert app._viz_sync_in_progress is False
            app.viz_backend_toggle.value = "plotlymol"
            assert app._viz_sync_in_progress is False
            app.viz_backend_toggle_ana.value = "py3dmol"
            assert app._viz_sync_in_progress is False

    def test_setter_idempotent_on_same_value(self, app):
        """Setting the preference to its current value should be a no-op."""
        if app.viz_backend_toggle is None:
            pytest.skip("Single-backend environment; sync N/A")
        before_pref = app._viz_backend_preference
        before_backend = app._viz_backend
        app._set_viz_preference(before_pref, persist=False)
        assert app._viz_backend_preference == before_pref
        assert app._viz_backend == before_backend
        assert app._viz_sync_in_progress is False


class TestAnalysisDisplayedMoleculeTracking:
    def test_initial_state_is_none(self, app):
        assert app._analysis_displayed_molecule is None

    def test_show_result_3d_tracks_analysis_molecule(self, app):
        """When show_result_3d renders into _analysis_mol_output, it should
        cache the molecule so the toggle can re-render later."""
        # Skip if visualization isn't available on this platform.
        try:
            from quantui.molecule import Molecule
        except ImportError:
            pytest.skip("Molecule unavailable")
        mol = Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        )
        # Direct invocation of the wrapper method.
        app._show_result_3d(mol, extra_output=app._analysis_mol_output)
        assert app._analysis_displayed_molecule is mol

    def test_show_result_3d_without_analysis_output_does_not_set_state(self, app):
        try:
            from quantui.molecule import Molecule
        except ImportError:
            pytest.skip("Molecule unavailable")
        mol = Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        )
        app._show_result_3d(mol, extra_output=None)
        assert app._analysis_displayed_molecule is None


class TestPersistedPreferenceAppliedAtStartup:
    """The persisted preference should drive the runtime effective backend
    at startup (bridge for VIZBACK.4). Concrete preferences are honoured
    when the requested backend is available; "auto" leaves the runtime
    default alone for later per-task routing."""

    def test_concrete_py3dmol_preference_applied_at_init(self, tmp_path, monkeypatch):
        # Write a settings file requesting py3dmol, then construct an app.
        import json

        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"_schema_version": 1, "viz": {"default_backend": "py3dmol"}})
        )
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(path))
        from quantui.app import QuantUIApp

        a = QuantUIApp()
        if not a._viz_availability.py3dmol:
            pytest.skip("py3dmol not available in this environment")
        assert a._viz_backend == "py3dmol"
        assert a._viz_backend_preference == "py3dmol"
        if a.viz_backend_toggle is not None:
            assert a.viz_backend_toggle.value == "py3dmol"
        if a.viz_backend_toggle_ana is not None:
            assert a.viz_backend_toggle_ana.value == "py3dmol"

    def test_concrete_plotlymol_preference_applied_at_init(self, tmp_path, monkeypatch):
        import json

        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"_schema_version": 1, "viz": {"default_backend": "plotlymol"}})
        )
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(path))
        from quantui.app import QuantUIApp

        a = QuantUIApp()
        if not a._viz_availability.plotlymol:
            pytest.skip("plotlymol not available in this environment")
        assert a._viz_backend == "plotlymol"
        assert a._viz_backend_preference == "plotlymol"

    def test_auto_preference_resolves_per_task_at_startup(self, tmp_path, monkeypatch):
        """Post-VIZBACK.4: 'auto' preference is resolved through the router
        at startup. _viz_backend reflects the router's MOLECULE_PREVIEW
        resolution (py3Dmol when both backends are available)."""
        import json

        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"_schema_version": 1, "viz": {"default_backend": "auto"}})
        )
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(path))
        from quantui.app import QuantUIApp

        a = QuantUIApp()
        assert a._viz_backend_preference == "auto"
        if a._viz_availability.py3dmol:
            # Auto resolves to py3Dmol primary for MOLECULE_PREVIEW.
            assert a._viz_backend == "py3dmol"
        elif a._viz_availability.plotlymol:
            assert a._viz_backend == "plotlymol"


class TestSettingsWidgetChangeAppliesRuntime:
    """Toggling the Settings widget should immediately update the runtime
    effective backend (when preference is concrete) and sync both toggles."""

    def test_settings_change_to_concrete_updates_runtime(self, app):
        if not (app._viz_availability.py3dmol and app._viz_availability.plotlymol):
            pytest.skip("Both backends needed for this test")
        # Pick a target different from current backend.
        target = "py3dmol" if app._viz_backend != "py3dmol" else "plotlymol"
        app.viz_default_backend_dd.value = target
        assert app._viz_backend == target
        assert app._viz_backend_preference == target
        if app.viz_backend_toggle is not None:
            assert app.viz_backend_toggle.value == target
        if app.viz_backend_toggle_ana is not None:
            assert app.viz_backend_toggle_ana.value == target

    def test_settings_change_to_auto_resolves_via_router(self, app):
        """Post-VIZBACK.4: setting preference to 'auto' resolves _viz_backend
        through the router. For MOLECULE_PREVIEW with both backends
        available, that's py3Dmol."""
        if not (app._viz_availability.py3dmol and app._viz_availability.plotlymol):
            pytest.skip("Both backends needed")
        # Force preference to a concrete plotlymol first.
        app._set_viz_preference("plotlymol", persist=False)
        assert app._viz_backend == "plotlymol"
        # Switch to auto.
        app.viz_default_backend_dd.value = "auto"
        assert app._viz_backend_preference == "auto"
        # Router resolves MOLECULE_PREVIEW (auto, both available) -> py3Dmol.
        assert app._viz_backend == "py3dmol"


class TestSyncLockState:
    """The sync flag must always be cleared after a set, even on the
    no-change short-circuit path."""

    def test_flag_cleared_after_change(self, app):
        if app.viz_backend_toggle is None:
            pytest.skip("Single-backend environment; sync N/A")
        new = "py3dmol" if app._viz_backend != "py3dmol" else "plotlymol"
        app._set_viz_preference(new, persist=False)
        assert app._viz_sync_in_progress is False

    def test_flag_cleared_after_no_op(self, app):
        if app.viz_backend_toggle is None:
            pytest.skip("Single-backend environment; sync N/A")
        app._set_viz_preference(app._viz_backend_preference, persist=False)
        assert app._viz_sync_in_progress is False


class TestRouterDrivenRendering:
    """Render sites should call the router instead of using _viz_backend
    directly. Verifies VIZBACK.4 (static) and VIZBACK.5 (trajectory)."""

    def test_auto_preference_routes_static_to_py3dmol(self, tmp_path, monkeypatch):
        """With preference='auto' and both backends available, the router
        should resolve STRUCTURE_VIEW_RESULTS / ANALYSIS_STRUCTURE_VIEW to
        py3Dmol per the routing policy."""
        import json

        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"_schema_version": 1, "viz": {"default_backend": "auto"}})
        )
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(path))
        from quantui.app import QuantUIApp

        a = QuantUIApp()
        if not (a._viz_availability.py3dmol and a._viz_availability.plotlymol):
            pytest.skip("Both backends needed")
        # _viz_backend should be py3dmol after initialization (auto -> py3dmol
        # primary for MOLECULE_PREVIEW per the policy).
        assert a._viz_backend == "py3dmol"
        # Settings widget still shows "auto" (preference, not effective backend).
        assert a.viz_default_backend_dd.value == "auto"
        # Effective toggles display the resolved value.
        if a.viz_backend_toggle is not None:
            assert a.viz_backend_toggle.value == "py3dmol"
        if a.viz_backend_toggle_ana is not None:
            assert a.viz_backend_toggle_ana.value == "py3dmol"

    def test_toggle_click_commits_concrete_preference(self, app):
        """Clicking a Calculate/Analysis toggle should set preference to the
        chosen concrete value (no longer "auto")."""
        if app.viz_backend_toggle is None:
            pytest.skip("Single-backend environment; sync N/A")
        # Start by forcing preference to auto.
        app._set_viz_preference("auto", persist=False)
        # Click the toggle to plotlymol.
        app.viz_backend_toggle.value = "plotlymol"
        assert app._viz_backend_preference == "plotlymol"
        assert app._viz_backend == "plotlymol"

    def test_analysis_label_widget_exists_when_both_backends_available(self, app):
        if app.viz_backend_toggle is None:
            pytest.skip("Single-backend environment")
        assert getattr(app, "viz_backend_label_ana", None) is not None
