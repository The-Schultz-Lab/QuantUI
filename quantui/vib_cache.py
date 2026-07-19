"""
Vibrational animation disk cache.

Caches rendered vibrational-animation HTML per calculation result directory
so mode switches on repeat visits and history replay are instant — no
re-render cost.

Layout
------
::

    <result_dir>/
    └── vib_frames/
        ├── index.json   ← manifest with schema version, render params, modes
        ├── mode_001.html
        ├── mode_002.html
        └── ...

`index.json` schema (version 1)::

    {
      "_schema_version": 1,
      "n_frames": 24,
      "amplitude": 0.4,
      "renderer": "py3dmol",
      "modes": {
        "1": {"cached": true, "file": "mode_001.html", "freq_cm1": 1623.4},
        "2": {"cached": false}
      }
    }

Cache key
---------
``(result_dir, mode_number, n_frames, amplitude, renderer)``. If any render
parameter changes between save and read (e.g. user customises amplitude
later), the cache is treated as stale and a miss is returned.

Robustness
----------
- Atomic writes (write-to-`.tmp` then `os.replace`) so a crash mid-write
  cannot leave the cache in a corrupted state.
- Graceful fallback: missing/malformed index → empty dict → cache miss.
  All filesystem errors log a warning and return ``None`` / ``False`` —
  never raise. Caller falls back to a fresh render.
- Schema-version mismatch is treated as a full cache invalidation (saved
  index discarded on the next write).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

_SCHEMA_VERSION = 1
_LOG = logging.getLogger(__name__)

# Floating-point tolerance when comparing saved vs requested amplitude. Avoids
# spurious cache misses when 0.4 round-trips through JSON serialization.
_AMPLITUDE_TOL = 1e-6


def cache_dir(result_dir: Path) -> Path:
    """Return the `vib_frames/` directory inside a result directory."""
    return Path(result_dir) / "vib_frames"


def load_index(result_dir: Path) -> dict:
    """Load the cache index manifest. Returns ``{}`` on missing/malformed."""
    path = cache_dir(result_dir) / "index.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning(
            "Failed to read vib cache index at %s (%s); ignoring",
            path,
            exc,
        )
        return {}
    return data if isinstance(data, dict) else {}


def has_cached(
    result_dir: Path,
    mode_number: int,
    *,
    n_frames: int,
    amplitude: float,
    renderer: str,
    fps: int,
) -> bool:
    """Return True if a cached HTML exists matching all render parameters."""
    idx = load_index(result_dir)
    if idx.get("_schema_version") != _SCHEMA_VERSION:
        return False
    if idx.get("n_frames") != n_frames:
        return False
    if not _amplitude_matches(idx.get("amplitude"), amplitude):
        return False
    if idx.get("renderer") != renderer:
        return False
    if idx.get("fps") != fps:
        return False
    modes = idx.get("modes", {})
    entry = modes.get(str(mode_number))
    if not isinstance(entry, dict) or not entry.get("cached"):
        return False
    fname = entry.get("file")
    if not isinstance(fname, str):
        return False
    return (cache_dir(result_dir) / fname).exists()


def get_cached_html(
    result_dir: Path,
    mode_number: int,
    *,
    n_frames: int,
    amplitude: float,
    renderer: str,
    fps: int,
) -> str | None:
    """Return the cached HTML string for a mode, or None on any miss."""
    if not has_cached(
        result_dir,
        mode_number,
        n_frames=n_frames,
        amplitude=amplitude,
        renderer=renderer,
        fps=fps,
    ):
        return None
    idx = load_index(result_dir)
    entry = idx["modes"][str(mode_number)]
    html_path = cache_dir(result_dir) / entry["file"]
    try:
        return html_path.read_text(encoding="utf-8")
    except OSError as exc:
        _LOG.warning(
            "Failed to read cached html at %s (%s)",
            html_path,
            exc,
        )
        return None


def save_cached_html(
    result_dir: Path,
    mode_number: int,
    html: str,
    *,
    freq_cm1: float | None,
    n_frames: int,
    amplitude: float,
    renderer: str,
    fps: int,
) -> None:
    """Save HTML to the cache and update the manifest. Non-fatal on failure.

    If the existing manifest has different render parameters (different
    n_frames, amplitude, or renderer), it is discarded and replaced — the
    previous mode HTML files are left on disk but become unreferenced
    (cleaned up next time the user deletes the result dir).
    """
    cdir = cache_dir(result_dir)
    try:
        cdir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _LOG.warning(
            "Failed to create vib cache dir %s (%s); skipping save",
            cdir,
            exc,
        )
        return

    idx = load_index(result_dir)
    if (
        idx.get("_schema_version") != _SCHEMA_VERSION
        or idx.get("n_frames") != n_frames
        or not _amplitude_matches(idx.get("amplitude"), amplitude)
        or idx.get("renderer") != renderer
        or idx.get("fps") != fps
    ):
        idx = {
            "_schema_version": _SCHEMA_VERSION,
            "n_frames": n_frames,
            "amplitude": amplitude,
            "renderer": renderer,
            "fps": fps,
            "modes": {},
        }

    fname = f"mode_{mode_number:03d}.html"
    html_path = cdir / fname
    tmp_html = html_path.with_suffix(html_path.suffix + ".tmp")
    try:
        tmp_html.write_text(html, encoding="utf-8")
        os.replace(tmp_html, html_path)
    except OSError as exc:
        _LOG.warning(
            "Failed to save cached html to %s (%s)",
            html_path,
            exc,
        )
        try:
            if tmp_html.exists():
                tmp_html.unlink()
        except OSError:
            pass
        return

    modes = idx.setdefault("modes", {})
    modes[str(mode_number)] = {
        "cached": True,
        "file": fname,
        "freq_cm1": float(freq_cm1) if freq_cm1 is not None else None,
    }

    index_path = cdir / "index.json"
    tmp_idx = index_path.with_suffix(index_path.suffix + ".tmp")
    try:
        tmp_idx.write_text(json.dumps(idx, indent=2), encoding="utf-8")
        os.replace(tmp_idx, index_path)
    except OSError as exc:
        _LOG.warning(
            "Failed to save cache index to %s (%s)",
            index_path,
            exc,
        )
        try:
            if tmp_idx.exists():
                tmp_idx.unlink()
        except OSError:
            pass


def _amplitude_matches(saved: object, requested: float) -> bool:
    """Compare amplitudes with tolerance to avoid float-equality issues."""
    if saved is None:
        return False
    try:
        return abs(float(saved) - requested) < _AMPLITUDE_TOL
    except (TypeError, ValueError):
        return False
