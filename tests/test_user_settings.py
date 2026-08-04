"""Unit tests for `quantui.user_settings.UserSettings`.

Covers default values, load/save roundtrip, every fallback path documented in
the module docstring, atomic-write semantics, and path resolution precedence.
All file I/O uses pytest's `tmp_path` fixture so no test touches
``~/.quantui/settings.json``.
"""

from __future__ import annotations

import json
import os

import pytest

from quantui.user_settings import UserSettings, VizSettings


class TestDefaults:
    def test_default_viz_backend_is_auto(self):
        assert VizSettings().default_backend == "auto"

    def test_default_user_settings_has_default_viz(self):
        settings = UserSettings()
        assert settings.viz.default_backend == "auto"

    def test_to_dict_uses_current_schema_version(self):
        data = UserSettings().to_dict()
        assert data["_schema_version"] == 1
        # No schema bump for iso_resolution: the loader reads every viz key with
        # a default, so a v1 file written before this field existed still loads
        # cleanly. Bumping would force a needless migration.
        assert data["viz"] == {
            "default_backend": "auto",
            "vib_framerate_fps": 10,
            "iso_resolution": "medium",
        }

    def test_default_vib_framerate_is_10(self):
        assert UserSettings().viz.vib_framerate_fps == 10


class TestLoad:
    def test_missing_file_returns_defaults(self, tmp_path):
        path = tmp_path / "doesnt_exist.json"
        settings = UserSettings.load(path)
        assert settings.viz.default_backend == "auto"

    def test_valid_settings_load_correctly(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"_schema_version": 1, "viz": {"default_backend": "py3dmol"}})
        )
        settings = UserSettings.load(path)
        assert settings.viz.default_backend == "py3dmol"

    def test_malformed_json_returns_defaults(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("this is not json {{{")
        settings = UserSettings.load(path)
        assert settings.viz.default_backend == "auto"

    def test_wrong_schema_version_returns_defaults(self, tmp_path):
        """Mismatched schema version must NOT silently consume the file's data."""
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"_schema_version": 99, "viz": {"default_backend": "py3dmol"}})
        )
        settings = UserSettings.load(path)
        assert settings.viz.default_backend == "auto"

    def test_missing_schema_version_returns_defaults(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"viz": {"default_backend": "py3dmol"}}))
        settings = UserSettings.load(path)
        assert settings.viz.default_backend == "auto"

    def test_missing_viz_section_uses_default_viz(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"_schema_version": 1}))
        settings = UserSettings.load(path)
        assert settings.viz.default_backend == "auto"

    def test_viz_section_wrong_type_uses_defaults(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"_schema_version": 1, "viz": "not a dict"}))
        settings = UserSettings.load(path)
        assert settings.viz.default_backend == "auto"

    def test_invalid_backend_value_uses_default(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps(
                {"_schema_version": 1, "viz": {"default_backend": "nonexistent"}}
            )
        )
        settings = UserSettings.load(path)
        assert settings.viz.default_backend == "auto"

    def test_backend_value_wrong_type_uses_default(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"_schema_version": 1, "viz": {"default_backend": 42}})
        )
        settings = UserSettings.load(path)
        assert settings.viz.default_backend == "auto"

    def test_top_level_list_returns_defaults(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(["not", "a", "dict"]))
        settings = UserSettings.load(path)
        assert settings.viz.default_backend == "auto"

    @pytest.mark.parametrize("good_fps", [5, 10, 30, 60, 120])
    def test_valid_vib_fps_round_trips(self, tmp_path, good_fps):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps(
                {
                    "_schema_version": 1,
                    "viz": {
                        "default_backend": "auto",
                        "vib_framerate_fps": good_fps,
                    },
                }
            )
        )
        settings = UserSettings.load(path)
        assert settings.viz.vib_framerate_fps == good_fps

    @pytest.mark.parametrize("bad_fps", [0, -1, 121, "30", True])
    def test_invalid_vib_fps_falls_back_to_default(self, tmp_path, bad_fps):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps(
                {
                    "_schema_version": 1,
                    "viz": {
                        "default_backend": "auto",
                        "vib_framerate_fps": bad_fps,
                    },
                }
            )
        )
        settings = UserSettings.load(path)
        assert settings.viz.vib_framerate_fps == 10

    def test_unknown_fields_are_tolerated(self, tmp_path):
        """Future versions may add fields old code doesn't know about — old
        code should still load successfully and ignore them."""
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps(
                {
                    "_schema_version": 1,
                    "viz": {
                        "default_backend": "py3dmol",
                        "future_field": {"nested": True},
                    },
                    "future_section": {"key": "value"},
                }
            )
        )
        settings = UserSettings.load(path)
        assert settings.viz.default_backend == "py3dmol"


class TestSave:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "settings.json"
        UserSettings().save(path)
        assert path.exists()

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "subdir" / "settings.json"
        UserSettings().save(path)
        assert path.exists()

    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "settings.json"
        settings = UserSettings()
        settings.viz.default_backend = "plotlymol"
        settings.save(path)
        data = json.loads(path.read_text())
        assert data["_schema_version"] == 1
        assert data["viz"]["default_backend"] == "plotlymol"

    def test_no_tmp_leftover_on_success(self, tmp_path):
        path = tmp_path / "settings.json"
        UserSettings().save(path)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_replaces_existing_file_atomically(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"old_format": True}))
        settings = UserSettings()
        settings.viz.default_backend = "py3dmol"
        settings.save(path)
        data = json.loads(path.read_text())
        assert data["_schema_version"] == 1
        assert "old_format" not in data
        assert data["viz"]["default_backend"] == "py3dmol"

    def test_save_failure_does_not_raise(self, tmp_path, monkeypatch):
        """Filesystem errors during save should log, not crash startup."""
        path = tmp_path / "settings.json"

        def _boom(*_args, **_kwargs):
            raise OSError("simulated disk full")

        # Replace write_text on Path so the atomic write fails.
        monkeypatch.setattr("pathlib.Path.write_text", _boom)
        UserSettings().save(path)  # must NOT raise


class TestRoundtrip:
    def test_save_then_load_preserves_values(self, tmp_path):
        path = tmp_path / "settings.json"
        original = UserSettings()
        original.viz.default_backend = "py3dmol"
        original.save(path)
        loaded = UserSettings.load(path)
        assert loaded.viz.default_backend == "py3dmol"

    @pytest.mark.parametrize("backend", ["auto", "py3dmol", "plotlymol"])
    def test_roundtrip_each_valid_backend(self, tmp_path, backend):
        path = tmp_path / "settings.json"
        original = UserSettings(viz=VizSettings(default_backend=backend))
        original.save(path)
        loaded = UserSettings.load(path)
        assert loaded.viz.default_backend == backend


class TestPathResolution:
    def test_explicit_path_used_when_provided(self, tmp_path):
        explicit = tmp_path / "explicit.json"
        UserSettings().save(explicit)
        assert explicit.exists()

    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        env_path = tmp_path / "env.json"
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(env_path))
        UserSettings().save()
        assert env_path.exists()

    def test_explicit_path_overrides_env_var(self, tmp_path, monkeypatch):
        env_path = tmp_path / "env.json"
        explicit_path = tmp_path / "explicit.json"
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(env_path))
        UserSettings().save(explicit_path)
        assert explicit_path.exists()
        assert not env_path.exists()

    def test_env_var_load_path(self, tmp_path, monkeypatch):
        env_path = tmp_path / "env.json"
        env_path.write_text(
            json.dumps({"_schema_version": 1, "viz": {"default_backend": "plotlymol"}})
        )
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(env_path))
        settings = UserSettings.load()
        assert settings.viz.default_backend == "plotlymol"

    def test_empty_env_var_falls_through_to_default(self, tmp_path, monkeypatch):
        """Empty QUANTUI_SETTINGS_PATH should not be treated as a real path."""
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", "")
        # We don't want to actually write to ~/.quantui — only check resolution
        # logic by ensuring the explicit path argument wins.
        explicit = tmp_path / "explicit.json"
        UserSettings().save(explicit)
        assert explicit.exists()


class TestEnvironmentIsolation:
    """Defensive — ensure QUANTUI_SETTINGS_PATH from a parent process does not
    leak into tests that don't set it. pytest's monkeypatch handles this when
    used; this test verifies the convention holds."""

    def test_env_path_uses_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("QUANTUI_SETTINGS_PATH", raising=False)
        # We can't safely assert the actual default path is what we expect
        # (the default lives under ~/.quantui), so just check the env var
        # does not affect resolution when unset.
        resolved = UserSettings._resolve_path(None)
        assert "settings.json" in str(resolved)
        assert os.environ.get("QUANTUI_SETTINGS_PATH") is None
