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
      "fps": 12,
      "bgcolor": "#ffffff",
      "modes": {
        "1": {"cached": true, "file": "mode_001.html", "mac": "<hex>",
              "freq_cm1": 1623.4},
        "2": {"cached": false}
      }
    }

Cache key
---------
``(result_dir, mode_number, n_frames, amplitude, renderer, fps, bgcolor)``.
If any render parameter changes between save and read (e.g. user customises
amplitude later, or toggles the Light/Dark theme, which changes the scene
background baked into the HTML), the cache is treated as stale and a miss is
returned.

Trust
-----
A result directory can come from someone else (a shared folder, a zip opened
in the Viewer), and a cached file is pasted into the page as HTML. So each
entry carries ``"mac"``: an HMAC-SHA256 of the HTML under a per-user secret
(``vib_cache.key`` next to the settings file, created 0600 on first use).
A cached file is served only when its name is a plain ``mode_NNN.html`` in
``vib_frames/`` and the MAC verifies; anything else, including every cache
written by another user or an older QuantUI, is a miss and is re-rendered.

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

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Optional, cast

_SCHEMA_VERSION = 1
_LOG = logging.getLogger(__name__)

# Floating-point tolerance when comparing saved vs requested amplitude. Avoids
# spurious cache misses when 0.4 round-trips through JSON serialization.
_AMPLITUDE_TOL = 1e-6


_FILE_RE = re.compile(r"mode_\d{3,}\.html")
_KEY_BYTES = 32


def _key_path() -> Path:
    from .user_settings import UserSettings

    return UserSettings._resolve_path(None).parent / "vib_cache.key"


def _cache_key() -> Optional[bytes]:
    """This user's secret for signing cached HTML (created on first use)."""
    path = _key_path()
    try:
        key = path.read_bytes()
        if len(key) == _KEY_BYTES:
            return key
    except OSError:
        pass
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_bytes(_KEY_BYTES)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
        os.replace(tmp, path)
        # Another process may have won the race; use what is on disk.
        key = path.read_bytes()
        return key if len(key) == _KEY_BYTES else None
    except OSError as exc:
        _LOG.warning("No vib cache key at %s (%s); cache disabled", path, exc)
        return None


def _mac(html: str) -> Optional[str]:
    key = _cache_key()
    if key is None:
        return None
    return hmac.new(key, html.encode("utf-8"), hashlib.sha256).hexdigest()


def _safe_file_name(fname: object) -> Optional[str]:
    """*fname* if it is a plain ``mode_NNN.html`` name, else ``None``."""
    if isinstance(fname, str) and _FILE_RE.fullmatch(fname):
        return fname
    return None


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
    bgcolor: str | None = None,
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
    if idx.get("bgcolor") != bgcolor:
        return False
    modes = idx.get("modes", {})
    entry = modes.get(str(mode_number))
    if not isinstance(entry, dict) or not entry.get("cached"):
        return False
    fname = _safe_file_name(entry.get("file"))
    if fname is None or not isinstance(entry.get("mac"), str):
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
    bgcolor: str | None = None,
) -> str | None:
    """Return the cached HTML string for a mode, or None on any miss."""
    if not has_cached(
        result_dir,
        mode_number,
        n_frames=n_frames,
        amplitude=amplitude,
        renderer=renderer,
        fps=fps,
        bgcolor=bgcolor,
    ):
        return None
    idx = load_index(result_dir)
    entry = idx["modes"][str(mode_number)]
    fname = cast(str, _safe_file_name(entry.get("file")))
    html_path = cache_dir(result_dir) / fname
    try:
        html = html_path.read_text(encoding="utf-8")
    except OSError as exc:
        _LOG.warning(
            "Failed to read cached html at %s (%s)",
            html_path,
            exc,
        )
        return None
    expected = _mac(html)
    if expected is None or not hmac.compare_digest(expected, str(entry["mac"])):
        # Not written by this user (or altered since): never paste it into
        # the page. The caller re-renders from the result's data instead.
        _LOG.info("Ignoring unsigned or foreign vib cache file %s", html_path)
        return None
    return html


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
    bgcolor: str | None = None,
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
        or idx.get("bgcolor") != bgcolor
    ):
        idx = {
            "_schema_version": _SCHEMA_VERSION,
            "n_frames": n_frames,
            "amplitude": amplitude,
            "renderer": renderer,
            "fps": fps,
            "bgcolor": bgcolor,
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

    mac = _mac(html)
    if mac is None:
        return
    modes = idx.setdefault("modes", {})
    modes[str(mode_number)] = {
        "cached": True,
        "file": fname,
        "mac": mac,
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
        # saved is deliberately object (untyped JSON) — float() may reject
        # it at runtime for any non-numeric value, which the except below
        # already handles; that's the real type check, not this cast.
        return abs(float(saved) - requested) < _AMPLITUDE_TOL  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
