"""
User preference persistence for QuantUI.

Settings are stored at ``~/.quantui/settings.json`` (override with the
``QUANTUI_SETTINGS_PATH`` environment variable for testing). The schema is
section-based for additive growth — new feature areas (theme, defaults,
etc.) add their own top-level sections without breaking existing readers.

Robustness rules
----------------
- **Atomic writes** — write to ``settings.json.tmp`` then rename, so a
  crash mid-write cannot corrupt the file.
- **Graceful fallback** — missing file, malformed JSON, unknown schema
  version, missing sections, or invalid field values all silently fall
  back to defaults with a single warning log. Startup never crashes on
  bad settings.
- **Additive-friendly** — new fields use defaults if absent in older
  saved files. Unknown fields are tolerated on read (no crash).
- **Schema versioning** — ``_schema_version`` is bumped only for breaking
  changes; additive changes keep the same version. Mismatched versions
  fall back to defaults.

Typical usage
-------------
>>> from quantui.user_settings import UserSettings
>>> settings = UserSettings.load()              # at app startup
>>> settings.viz.default_backend = "py3dmol"
>>> settings.save()                              # on settings change
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

_SCHEMA_VERSION = 1
_LOG = logging.getLogger(__name__)

# Valid values for VizSettings.default_backend. Kept in sync with
# quantui.viz_backend_router.VizPreference values; not imported here to keep
# this module zero-dependency for unit testing.
_VALID_VIZ_BACKENDS = ("auto", "py3dmol", "plotlymol")

# Vibrational animation playback rate. Clamped to a sensible range on load
# so a corrupt value can't produce a 0-ms or absurdly high interval.
_VIB_FPS_MIN = 1
_VIB_FPS_MAX = 120
_VIB_FPS_DEFAULT = 10

# Default settings path. The QUANTUI_SETTINGS_PATH env var overrides for tests.
DEFAULT_SETTINGS_PATH = Path.home() / ".quantui" / "settings.json"


@dataclass
class VizSettings:
    """Visualization-related user preferences."""

    default_backend: str = "auto"  # one of _VALID_VIZ_BACKENDS
    vib_framerate_fps: int = _VIB_FPS_DEFAULT  # py3Dmol vib-animation fps


@dataclass
class ComputeSettings:
    """Compute-related user preferences."""

    # Whether GPU offload may engage when a CUDA device is detected. Default on
    # — the historical behavior. Users on consumer cards (weak FP64, see
    # gpu_offload.is_low_fp64_device) may want this off, since offload can be
    # slower than a many-core CPU there. ``QUANTUI_DISABLE_GPU=1`` still wins
    # over this setting.
    gpu_enabled: bool = True


@dataclass
class UserSettings:
    """Root user settings container — section-based for additive growth."""

    viz: VizSettings = field(default_factory=VizSettings)
    compute: ComputeSettings = field(default_factory=ComputeSettings)

    @classmethod
    def load(cls, path: Path | None = None) -> UserSettings:
        """Load settings from disk, falling back to defaults on any failure.

        Missing file, malformed JSON, unknown schema version, malformed
        sections, and invalid field values all return the default
        ``UserSettings`` with one warning log per failure mode.
        """
        resolved = cls._resolve_path(path)
        if not resolved.exists():
            return cls()
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning(
                "Failed to read settings at %s (%s); using defaults",
                resolved,
                exc,
            )
            return cls()
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: object) -> UserSettings:
        """Parse a deserialized JSON object into a UserSettings instance."""
        if not isinstance(data, dict):
            _LOG.warning(
                "Settings root is not a JSON object (got %s); using defaults",
                type(data).__name__,
            )
            return cls()

        version = data.get("_schema_version")
        if version != _SCHEMA_VERSION:
            _LOG.warning(
                "Settings schema version %r does not match current %r; "
                "using defaults",
                version,
                _SCHEMA_VERSION,
            )
            return cls()

        viz_section = data.get("viz", {})
        if not isinstance(viz_section, dict):
            _LOG.warning(
                "Settings 'viz' section is not an object (got %s); "
                "using viz defaults",
                type(viz_section).__name__,
            )
            viz_section = {}

        viz = VizSettings()
        candidate_backend = viz_section.get("default_backend", viz.default_backend)
        if (
            isinstance(candidate_backend, str)
            and candidate_backend in _VALID_VIZ_BACKENDS
        ):
            viz.default_backend = candidate_backend
        else:
            _LOG.warning(
                "Invalid viz.default_backend %r; using %r",
                candidate_backend,
                viz.default_backend,
            )

        candidate_fps = viz_section.get("vib_framerate_fps", viz.vib_framerate_fps)
        if (
            isinstance(candidate_fps, int)
            and not isinstance(candidate_fps, bool)
            and _VIB_FPS_MIN <= candidate_fps <= _VIB_FPS_MAX
        ):
            viz.vib_framerate_fps = candidate_fps
        else:
            _LOG.warning(
                "Invalid viz.vib_framerate_fps %r; using %r",
                candidate_fps,
                viz.vib_framerate_fps,
            )

        compute_section = data.get("compute", {})
        if not isinstance(compute_section, dict):
            _LOG.warning(
                "Settings 'compute' section is not an object (got %s); "
                "using compute defaults",
                type(compute_section).__name__,
            )
            compute_section = {}

        compute = ComputeSettings()
        if "gpu_enabled" in compute_section:
            candidate_gpu = compute_section["gpu_enabled"]
            if isinstance(candidate_gpu, bool):
                compute.gpu_enabled = candidate_gpu
            else:
                _LOG.warning(
                    "Invalid compute.gpu_enabled %r; using %r",
                    candidate_gpu,
                    compute.gpu_enabled,
                )

        return cls(viz=viz, compute=compute)

    def to_dict(self) -> dict:
        """Serialize to a dict for JSON storage with the current schema version."""
        return {
            "_schema_version": _SCHEMA_VERSION,
            "viz": asdict(self.viz),
            "compute": asdict(self.compute),
        }

    def save(self, path: Path | None = None) -> None:
        """Write settings to disk atomically (write to .tmp then rename).

        Does not raise on filesystem failure — logs a warning and continues.
        Callers should not assume the save succeeded on a hostile filesystem;
        the next load will fall back to defaults if the file is missing.
        """
        resolved = self._resolve_path(path)
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _LOG.warning(
                "Failed to create settings parent dir %s (%s); save aborted",
                resolved.parent,
                exc,
            )
            return

        tmp = resolved.with_suffix(resolved.suffix + ".tmp")
        try:
            tmp.write_text(
                json.dumps(self.to_dict(), indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, resolved)
        except OSError as exc:
            _LOG.warning(
                "Failed to save settings to %s (%s)",
                resolved,
                exc,
            )
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    @classmethod
    def _resolve_path(cls, path: Path | None) -> Path:
        if path is not None:
            return Path(path)
        env_override = os.environ.get("QUANTUI_SETTINGS_PATH")
        if env_override:
            return Path(env_override)
        return DEFAULT_SETTINGS_PATH
