"""Bundled molecule library — indexed package-data store + lazy loader (STRUCT.6).

The library lives in ``quantui/data/`` as two committed artifacts:

- ``manifests/presets.json`` — the human-readable, reviewable source of record.
- ``library/library.sqlite`` — an indexed, lazily-queried store generated from
  the manifest(s). Coordinates are packed as ``int16`` @ 0.001 Å (7–8 bytes per
  atom) so the store stays compact enough to hold the eventual ~10 MB hybrid
  library (STRUCT.7/.8) without bloating import time.

Design goals:

- **Lazy + indexed** — search/get touch only the rows they need, so the store
  scales to tens of thousands of bulk entries without loading them all.
- **Back-compat** — :func:`get_preset_dict` returns the exact legacy
  ``MOLECULE_LIBRARY`` shape, so ``config.MOLECULE_LIBRARY`` and every existing
  consumer keep working unchanged.
- **Robust** — if the SQLite artifact is missing or unreadable, every entry
  point transparently falls back to the JSON manifest, so ``import quantui``
  never fails on a fresh/odd checkout.

When STRUCT.7/.8 add curated + bulk content, they only add manifests + rebuild
the store; nothing in this module's public API changes.
"""

import json
import logging
import sqlite3
import struct
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Coordinate codec ─────────────────────────────────────────────────────────
# Per-atom record: 2 bytes element symbol (ascii, 2nd byte 0 if 1-char) +
# 3× int16 little-endian coordinate in milli-ångström. 8 bytes/atom.
_COORD_SCALE = 1000  # 0.001 Å resolution; ±32.767 Å range (int16)
_ATOM_RECORD = "<BBhhh"
_ATOM_RECORD_SIZE = struct.calcsize(_ATOM_RECORD)  # 8

# Categories that are NOT part of the browsable preset dict (reached via search
# only). Bulk QM9 entries (STRUCT.8) will carry "bulk-qm9".
_BULK_CATEGORIES = frozenset({"bulk-qm9"})


def encode_coords(atoms: List[str], coords: List[List[float]]) -> bytes:
    """Pack ``(atoms, coords)`` into the compact per-atom binary record."""
    buf = bytearray()
    for sym, (x, y, z) in zip(atoms, coords):
        s = sym.encode("ascii")
        if not 1 <= len(s) <= 2:
            raise ValueError(f"Unsupported element symbol: {sym!r}")
        c0 = s[0]
        c1 = s[1] if len(s) > 1 else 0
        buf += struct.pack(
            _ATOM_RECORD,
            c0,
            c1,
            round(float(x) * _COORD_SCALE),
            round(float(y) * _COORD_SCALE),
            round(float(z) * _COORD_SCALE),
        )
    return bytes(buf)


def decode_coords(blob: bytes) -> Tuple[List[str], List[List[float]]]:
    """Unpack a coordinate blob back into ``(atoms, coords)``."""
    atoms: List[str] = []
    coords: List[List[float]] = []
    for offset in range(0, len(blob), _ATOM_RECORD_SIZE):
        c0, c1, xi, yi, zi = struct.unpack(
            _ATOM_RECORD, blob[offset : offset + _ATOM_RECORD_SIZE]
        )
        sym = chr(c0) + (chr(c1) if c1 else "")
        atoms.append(sym)
        coords.append([xi / _COORD_SCALE, yi / _COORD_SCALE, zi / _COORD_SCALE])
    return atoms, coords


# ── Paths ────────────────────────────────────────────────────────────────────
def _data_dir() -> Path:
    return Path(str(files("quantui"))) / "data"


def db_path() -> Path:
    return _data_dir() / "library" / "library.sqlite"


def manifest_paths() -> List[Path]:
    """All manifest JSON files that seed the store (STRUCT.7/.8 add more).

    ``presets.json`` is ordered first so the original teaching molecules lead
    the (interim) flat dropdown; remaining manifests follow alphabetically.
    """
    manifest_dir = _data_dir() / "manifests"
    if not manifest_dir.is_dir():
        return []
    return sorted(
        manifest_dir.glob("*.json"),
        key=lambda p: (p.name != "presets.json", p.name),
    )


# ── Manifest loading (always-available fallback source) ──────────────────────
@lru_cache(maxsize=1)
def _manifest_entries() -> Tuple[Dict[str, Any], ...]:
    """Load every manifest entry. Cached. The JSON is the source of record."""
    entries: List[Dict[str, Any]] = []
    for path in manifest_paths():
        try:
            entries.extend(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # pragma: no cover - corrupt manifest
            logger.error(f"Failed to read library manifest {path}: {exc}")
    return tuple(entries)


def _normalize_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Fill derived fields (id/formula/n_atoms/...) for a manifest entry."""
    atoms = raw["atoms"]
    formula = raw.get("formula") or raw.get("id") or raw["name"]
    return {
        "id": raw.get("id", formula),
        "name": raw.get("name", formula),
        "formula": formula,
        "category": raw.get("category", "preset"),
        "n_atoms": len(atoms),
        "n_heavy": sum(1 for a in atoms if a != "H"),
        "charge": int(raw.get("charge", 0)),
        "multiplicity": int(raw.get("multiplicity", 1)),
        "source": raw.get("source", "preset"),
        "smiles": raw.get("smiles"),
        "inchikey": raw.get("inchikey"),
        "synonyms": raw.get("synonyms", ""),
        "description": raw.get("description", ""),
        "atoms": atoms,
        "coordinates": raw["coordinates"],
    }


# ── Store build (run by scripts/build_library.py or on-demand) ───────────────
_SCHEMA = """
CREATE TABLE molecule (
    id           TEXT PRIMARY KEY,
    name         TEXT,
    formula      TEXT NOT NULL,
    category     TEXT NOT NULL,
    n_heavy      INTEGER NOT NULL,
    n_atoms      INTEGER NOT NULL,
    charge       INTEGER NOT NULL DEFAULT 0,
    multiplicity INTEGER NOT NULL DEFAULT 1,
    source       TEXT NOT NULL,
    smiles       TEXT,
    inchikey     TEXT,
    synonyms     TEXT,
    description  TEXT,
    coords       BLOB NOT NULL
);
CREATE INDEX idx_name ON molecule(name);
CREATE INDEX idx_formula ON molecule(formula);
CREATE INDEX idx_category ON molecule(category);
CREATE INDEX idx_inchikey ON molecule(inchikey);
"""


def build_store(entries: List[Dict[str, Any]], target: Path) -> Path:
    """(Re)build the SQLite store from a list of (raw) manifest entries.

    Deterministic: entries are inserted in the order given, so a rebuild from
    the same manifest yields a stable file. Overwrites ``target``.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    con = sqlite3.connect(target)
    try:
        con.executescript(_SCHEMA)
        rows = []
        for raw in entries:
            e = _normalize_entry(raw)
            rows.append(
                (
                    e["id"],
                    e["name"],
                    e["formula"],
                    e["category"],
                    e["n_heavy"],
                    e["n_atoms"],
                    e["charge"],
                    e["multiplicity"],
                    e["source"],
                    e["smiles"],
                    e["inchikey"],
                    e["synonyms"],
                    e["description"],
                    encode_coords(e["atoms"], e["coordinates"]),
                )
            )
        con.executemany(
            "INSERT INTO molecule (id, name, formula, category, n_heavy, n_atoms, "
            "charge, multiplicity, source, smiles, inchikey, synonyms, description, "
            "coords) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        con.commit()
    finally:
        con.close()
    logger.info(f"Built molecule library store: {target} ({len(entries)} entries)")
    return target


def build_from_manifests(target: Optional[Path] = None) -> Path:
    """Build the store from all committed manifests. Returns the store path."""
    target = target or db_path()
    return build_store([dict(e) for e in _manifest_entries()], target)


def _ensure_store() -> Optional[Path]:
    """Best-effort: ensure the SQLite store exists. Returns its path or None.

    Never raises — if the package dir is read-only (installed wheel) and the
    store is somehow absent, callers fall back to the JSON manifest.
    """
    path = db_path()
    if path.exists():
        return path
    try:
        if _manifest_entries():
            return build_from_manifests(path)
    except Exception as exc:  # pragma: no cover - read-only install w/o store
        logger.warning(f"Could not build library store on demand: {exc}")
    return None


# ── Query API ────────────────────────────────────────────────────────────────
def _connect_ro() -> Optional[sqlite3.Connection]:
    """Open a fresh read-only connection (thread-safe per-call), or None."""
    path = _ensure_store()
    if path is None or not path.exists():
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _row_to_entry(row: sqlite3.Row) -> Dict[str, Any]:
    atoms, coords = decode_coords(row["coords"])
    return {
        "id": row["id"],
        "name": row["name"],
        "formula": row["formula"],
        "category": row["category"],
        "n_heavy": row["n_heavy"],
        "n_atoms": row["n_atoms"],
        "charge": row["charge"],
        "multiplicity": row["multiplicity"],
        "source": row["source"],
        "smiles": row["smiles"],
        "inchikey": row["inchikey"],
        "synonyms": row["synonyms"],
        "description": row["description"],
        "atoms": atoms,
        "coordinates": coords,
    }


@lru_cache(maxsize=1)
def get_preset_dict() -> Dict[str, Dict[str, Any]]:
    """Return the curated/preset entries in the legacy ``MOLECULE_LIBRARY`` shape.

    Keyed by entry id (the formula key, e.g. ``"H2O"``); each value carries the
    legacy ``atoms`` / ``coordinates`` / ``charge`` / ``multiplicity`` /
    ``description`` fields. Bulk categories are excluded (they are reached via
    :func:`search`, never the browse dropdown). Preserves manifest order.

    Prefers the SQLite store; falls back to the JSON manifest if the store is
    unavailable, so this never hard-fails.
    """
    out: Dict[str, Dict[str, Any]] = {}
    con = _connect_ro()
    if con is not None:
        try:
            placeholders = ",".join("?" * len(_BULK_CATEGORIES))
            rows = con.execute(
                f"SELECT * FROM molecule WHERE category NOT IN ({placeholders}) "
                "ORDER BY rowid",
                tuple(_BULK_CATEGORIES),
            ).fetchall()
            for row in rows:
                e = _row_to_entry(row)
                out[e["id"]] = _legacy_view(e)
            return out
        except Exception as exc:  # pragma: no cover - corrupt store
            logger.warning(f"Library store unreadable, using manifest: {exc}")
        finally:
            con.close()
    # JSON fallback.
    for raw in _manifest_entries():
        e = _normalize_entry(raw)
        if e["category"] in _BULK_CATEGORIES:
            continue
        out[e["id"]] = _legacy_view(e)
    return out


def _legacy_view(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Project a full entry down to the legacy MOLECULE_LIBRARY value shape."""
    return {
        "atoms": entry["atoms"],
        "coordinates": entry["coordinates"],
        "charge": entry["charge"],
        "multiplicity": entry["multiplicity"],
        "description": entry["description"],
    }


def get(entry_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one full entry (incl. decoded coordinates) by id, or None."""
    con = _connect_ro()
    if con is not None:
        try:
            row = con.execute(
                "SELECT * FROM molecule WHERE id = ?", (entry_id,)
            ).fetchone()
            return _row_to_entry(row) if row else None
        finally:
            con.close()
    for raw in _manifest_entries():
        e = _normalize_entry(raw)
        if e["id"] == entry_id:
            return e
    return None


def search(
    query: str = "", *, category: Optional[str] = None, limit: int = 50
) -> List[Dict[str, Any]]:
    """Search the library by name / formula / synonyms (substring, case-insensitive).

    Returns lightweight rows (no coordinates) for listing; call :func:`get` to
    materialize the chosen entry. Empty ``query`` lists entries (optionally
    filtered by ``category``).
    """
    q = f"%{query.strip().lower()}%"
    con = _connect_ro()
    if con is not None:
        try:
            sql = (
                "SELECT id, name, formula, category, n_heavy, n_atoms, charge, "
                "multiplicity, source FROM molecule WHERE "
                "(lower(name) LIKE ? OR lower(formula) LIKE ? OR lower(synonyms) LIKE ?)"
            )
            params: List[Any] = [q, q, q]
            if category:
                sql += " AND category = ?"
                params.append(category)
            sql += " ORDER BY n_atoms, id LIMIT ?"
            params.append(limit)
            return [dict(r) for r in con.execute(sql, params).fetchall()]
        finally:
            con.close()
    # JSON fallback (linear scan).
    needle = query.strip().lower()
    results = []
    for raw in _manifest_entries():
        e = _normalize_entry(raw)
        hay = f"{e['name']} {e['formula']} {e['synonyms']}".lower()
        if needle in hay and (category is None or e["category"] == category):
            results.append(
                {
                    k: e[k]
                    for k in (
                        "id",
                        "name",
                        "formula",
                        "category",
                        "n_heavy",
                        "n_atoms",
                        "charge",
                        "multiplicity",
                        "source",
                    )
                }
            )
    results.sort(key=lambda r: (r["n_atoms"], r["id"]))
    return results[:limit]


def categories() -> List[str]:
    """Distinct category labels present in the library."""
    con = _connect_ro()
    if con is not None:
        try:
            return [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT category FROM molecule ORDER BY category"
                ).fetchall()
            ]
        finally:
            con.close()
    return sorted({_normalize_entry(e)["category"] for e in _manifest_entries()})


def count() -> int:
    """Total number of entries in the library."""
    con = _connect_ro()
    if con is not None:
        try:
            return int(con.execute("SELECT COUNT(*) FROM molecule").fetchone()[0])
        finally:
            con.close()
    return len(_manifest_entries())


def iter_entries():
    """Yield every entry (full, with decoded coordinates) in store order.

    Single connection — efficient for whole-library governance/round-trip
    checks (STRUCT.10). Falls back to the JSON manifest if the store is absent.
    """
    con = _connect_ro()
    if con is None:
        for raw in _manifest_entries():
            yield _normalize_entry(raw)
        return
    try:
        for row in con.execute("SELECT * FROM molecule ORDER BY rowid"):
            yield _row_to_entry(row)
    finally:
        con.close()
