"""
Calculation checkpoints — the ``.chk`` parity feature (M-CHECKPOINT).

A long calculation that dies partway currently throws away everything it
finished. A Geometry Optimization eleven steps into a twelve-step run, a PES
Scan on its last point, a Frequency job interrupted by a laptop lid — all of
them start again from nothing. This module is the storage layer that lets them
pick up instead.

Two different things are worth persisting, and conflating them is the easy
mistake:

**Resume** — continue *this* interrupted run. Only valid if the inputs are
identical, geometry included: resuming a scan of a different molecule into the
same directory would silently mix two calculations. Keyed by
:attr:`CalcIdentity.resume_key`.

**Warm start** — reuse a converged SCF density as the *initial guess* for a
different-but-related run. Here an exact geometry match is the wrong test: a
density from a nearby geometry is an excellent guess, which is precisely what a
geometry optimization exploits internally. Keyed by
:attr:`CalcIdentity.warm_start_key`, which deliberately omits coordinates.

Layout, under ``~/.quantui/checkpoints`` (override with
``QUANTUI_CHECKPOINT_DIR``)::

    <root>/<resume_key>/
        meta.json      identity + status, written atomically
        scf.chk        PySCF chkfile (CHK.1)
        opt.traj       ASE trajectory, appended per step (CHK.2)
        opt.restart    BFGS Hessian state (CHK.2)
        points.jsonl   one line per completed scan point (CHK.3)

Checkpoints live outside the results directory on purpose: a result directory
is created when a calculation *succeeds*, and the runs that most need a
checkpoint are the ones that never get there.

Design constraint that outranks the feature itself: **a checkpoint must never
break a calculation.** Every write is best-effort and every read is defensive.
A corrupt, truncated, or unreadable checkpoint has to behave exactly like no
checkpoint at all — the calculation still runs, it just starts from scratch.
:func:`Checkpoint.load_state` returning ``None`` is the normal failure mode,
not an exception.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

#: Bumped when the on-disk layout changes incompatibly. Checkpoints written by
#: a different version are ignored rather than migrated — they represent at
#: most a few minutes of recomputation, which is not worth a migration path.
CHECKPOINT_SCHEMA_VERSION = 1

#: Status values recorded in ``meta.json``.
STATUS_RUNNING = "running"
STATUS_INTERRUPTED = "interrupted"
STATUS_COMPLETE = "complete"

#: Default retention. A checkpoint's whole value is being recent enough that
#: resuming beats re-running; past this it is just disk.
DEFAULT_MAX_AGE_DAYS = 14
DEFAULT_MAX_CHECKPOINTS = 40


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


def checkpoint_root() -> Path:
    """Return the directory holding all checkpoints.

    Honours ``QUANTUI_CHECKPOINT_DIR`` so tests — and anyone on a machine
    where ``$HOME`` is not the right place for scratch state — can redirect it.
    """
    env = os.environ.get("QUANTUI_CHECKPOINT_DIR")
    return Path(env) if env else Path.home() / ".quantui" / "checkpoints"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _coords_digest(coords: Optional[Sequence[Sequence[float]]]) -> str:
    """Stable digest of a coordinate array.

    Rounded to 6 decimal places (well below any chemically meaningful
    difference, well above float round-trip noise) so a geometry that survives
    a save/load cycle still matches itself.
    """
    if not coords:
        return ""
    rounded = [[round(float(v), 6) for v in row] for row in coords]
    return hashlib.sha256(
        json.dumps(rounded, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


@dataclass(frozen=True)
class CalcIdentity:
    """Everything that decides whether two calculations are the same one."""

    calc_type: str
    method: str
    basis: str
    charge: int = 0
    multiplicity: int = 1
    atom_symbols: tuple = ()
    coords: tuple = ()

    @classmethod
    def from_molecule(
        cls, molecule: Any, *, calc_type: str, method: str, basis: str
    ) -> CalcIdentity:
        """Build an identity from a :class:`~quantui.molecule.Molecule`."""
        coords = getattr(molecule, "coordinates", None)
        if coords is None:
            coords = getattr(molecule, "coords", None)
        try:
            coord_rows = tuple(tuple(float(v) for v in row) for row in (coords or []))
        except (TypeError, ValueError):
            coord_rows = ()
        return cls(
            calc_type=str(calc_type),
            method=str(method),
            basis=str(basis),
            charge=int(getattr(molecule, "charge", 0) or 0),
            multiplicity=int(getattr(molecule, "multiplicity", 1) or 1),
            atom_symbols=tuple(str(a) for a in (getattr(molecule, "atoms", []) or [])),
            coords=coord_rows,
        )

    @property
    def warm_start_key(self) -> str:
        """Key for "same system, same level of theory" — geometry excluded.

        An SCF density from a nearby geometry is a good initial guess, so
        requiring identical coordinates here would throw away the reuse this
        exists to enable. Calc type is excluded too: the converged density of a
        single point is just as good a guess for an optimization.
        """
        parts = [
            str(CHECKPOINT_SCHEMA_VERSION),
            self.method.upper(),
            self.basis.upper(),
            str(self.charge),
            str(self.multiplicity),
            ",".join(self.atom_symbols),
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    @property
    def resume_key(self) -> str:
        """Key for "the identical run" — geometry and calc type included.

        Resuming into the wrong checkpoint would silently splice two different
        calculations together, which is worse than not resuming at all. So this
        key is deliberately strict.
        """
        parts = [
            self.warm_start_key,
            self.calc_type,
            _coords_digest(self.coords),
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        """Short human-readable label, for logs and the resume prompt."""
        formula = "".join(self.atom_symbols[:6]) or "?"
        return f"{formula} {self.method}/{self.basis} ({self.calc_type})"


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write *payload* to *path* so a crash can't leave a half-file.

    A partially-written ``meta.json`` would be read back as corrupt on the next
    run — and the next run is by definition the one recovering from a crash,
    so this is the exact case that has to hold.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class Checkpoint:
    """One calculation's checkpoint directory.

    Constructing this does **not** touch the disk; :meth:`begin` creates the
    directory. That keeps a checkpoint object cheap enough to build
    speculatively when asking "is there anything to resume?".
    """

    def __init__(self, identity: CalcIdentity, root: Optional[Path] = None) -> None:
        self.identity = identity
        self._root = Path(root) if root is not None else checkpoint_root()

    # ── Paths ───────────────────────────────────────────────────────────────

    @property
    def dir(self) -> Path:
        return self._root / self.identity.resume_key

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def scf_chkfile(self) -> Path:
        return self.dir / "scf.chk"

    @property
    def trajectory_path(self) -> Path:
        return self.dir / "opt.traj"

    @property
    def optimizer_restart_path(self) -> Path:
        return self.dir / "opt.restart"

    @property
    def points_path(self) -> Path:
        return self.dir / "points.jsonl"

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def exists(self) -> bool:
        return self.meta_path.is_file()

    def begin(self, **extra: Any) -> bool:
        """Create the directory and mark the run as in progress.

        Returns ``True`` on success, ``False`` if the checkpoint could not be
        created — in which case the caller carries on without one.
        """
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "status": STATUS_RUNNING,
                "started_at": time.time(),
                "updated_at": time.time(),
                "warm_start_key": self.identity.warm_start_key,
                "resume_key": self.identity.resume_key,
                "calc_type": self.identity.calc_type,
                "method": self.identity.method,
                "basis": self.identity.basis,
                "charge": self.identity.charge,
                "multiplicity": self.identity.multiplicity,
                "atom_symbols": list(self.identity.atom_symbols),
                "label": self.identity.describe(),
            }
            payload.update(extra)
            _atomic_write_json(self.meta_path, payload)
            return True
        except Exception as exc:  # noqa: BLE001 — never break the calculation
            logger.debug("checkpoint begin failed for %s: %s", self.dir, exc)
            return False

    def load_state(self) -> Optional[dict]:
        """Return the stored metadata, or ``None`` if unusable.

        Unusable covers missing, unreadable, corrupt, and written by a
        different schema version. All four mean the same thing to a caller —
        there is nothing to resume from — so they collapse to one return value
        rather than four exception types.
        """
        try:
            raw = self.meta_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("discarding corrupt checkpoint metadata at %s", self.meta_path)
            return None
        if not isinstance(state, dict):
            return None
        if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            return None
        return state

    def update(self, **fields: Any) -> None:
        """Merge *fields* into the metadata. Best-effort."""
        state = self.load_state()
        if state is None:
            return
        # Stamp first so an explicitly-passed ``updated_at`` still wins.
        # Stamping afterwards would silently discard the caller's value, which
        # is the kind of quiet override that is very hard to notice.
        state["updated_at"] = time.time()
        state.update(fields)
        try:
            _atomic_write_json(self.meta_path, state)
        except Exception as exc:  # noqa: BLE001 — never break the calculation
            logger.debug("checkpoint update failed for %s: %s", self.dir, exc)

    def mark_interrupted(self) -> None:
        """Record that the run stopped before finishing — the resumable state."""
        self.update(status=STATUS_INTERRUPTED)

    def mark_complete(self) -> None:
        """Record success. The checkpoint stays for warm-starting a later run."""
        self.update(status=STATUS_COMPLETE)

    def discard(self) -> None:
        """Delete the checkpoint directory entirely. Best-effort."""
        try:
            shutil.rmtree(self.dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001 — cleanup is never fatal
            logger.debug("checkpoint discard failed for %s: %s", self.dir, exc)

    # ── Resumability ────────────────────────────────────────────────────────

    def resumable_state(self) -> Optional[dict]:
        """Return the state only if this checkpoint can actually be resumed.

        A checkpoint is resumable when it stopped mid-run **and** left
        something behind worth continuing from. A run that was interrupted
        during its very first SCF has a directory but no progress, and
        offering to "resume" it would promise the user a saving that does not
        exist.
        """
        state = self.load_state()
        if state is None:
            return None
        if state.get("status") == STATUS_COMPLETE:
            return None
        if not self.has_progress():
            return None
        return state

    def has_progress(self) -> bool:
        """True when some finished work is stored, not merely a directory."""
        if self.completed_points():
            return True
        traj = self.trajectory_path
        try:
            return traj.is_file() and traj.stat().st_size > 0
        except OSError:
            return False

    # ── Scan points (CHK.3) ─────────────────────────────────────────────────

    def append_point(self, payload: dict) -> None:
        """Record one completed scan point. Best-effort, append-only.

        Append-only rather than rewrite-the-file: a crash during a rewrite
        could lose every completed point, whereas a crash during an append
        loses at most the partial final line, which :meth:`completed_points`
        discards.
        """
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.points_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 — never break the calculation
            logger.debug("checkpoint point append failed for %s: %s", self.dir, exc)

    def completed_points(self) -> list[dict]:
        """Return every complete point record, skipping any truncated tail."""
        try:
            raw = self.points_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        points: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A half-written final line is exactly what a crash leaves.
                continue
            if isinstance(record, dict):
                points.append(record)
        return points


# ---------------------------------------------------------------------------
# Discovery + retention
# ---------------------------------------------------------------------------


def _iter_checkpoint_dirs(root: Optional[Path] = None) -> Iterable[Path]:
    base = Path(root) if root is not None else checkpoint_root()
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return []
    return [p for p in entries if p.is_dir()]


def load_all(root: Optional[Path] = None) -> list[dict]:
    """Return metadata for every readable checkpoint, newest first.

    Each entry gains a ``"dir"`` key with its path. Unreadable directories are
    skipped silently — this feeds a UI listing, and one bad checkpoint should
    not hide the rest.
    """
    out: list[dict] = []
    for path in _iter_checkpoint_dirs(root):
        try:
            raw = (path / "meta.json").read_text(encoding="utf-8")
            state = json.loads(raw)
        except Exception:  # noqa: BLE001 — skip anything unreadable
            continue
        if not isinstance(state, dict):
            continue
        if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            continue
        state["dir"] = str(path)
        out.append(state)
    out.sort(key=lambda s: float(s.get("updated_at") or 0), reverse=True)
    return out


def find_warm_start_chkfile(
    identity: CalcIdentity, root: Optional[Path] = None
) -> Optional[Path]:
    """Return an SCF chkfile usable as an initial guess for *identity*.

    Matches on :attr:`CalcIdentity.warm_start_key`, so a density converged at a
    different geometry — or by a different calc type — still qualifies. Prefers
    the most recently updated match, which is the one most likely to be near
    the geometry about to be run.
    """
    wanted = identity.warm_start_key
    for state in load_all(root):
        if state.get("warm_start_key") != wanted:
            continue
        candidate = Path(state["dir"]) / "scf.chk"
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    return None


def find_resumable(
    identity: CalcIdentity, root: Optional[Path] = None
) -> Optional[Checkpoint]:
    """Return the checkpoint for *identity* if it has resumable progress."""
    ckpt = Checkpoint(identity, root=root)
    return ckpt if ckpt.resumable_state() is not None else None


def prune(
    root: Optional[Path] = None,
    *,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
) -> int:
    """Delete stale checkpoints; return how many were removed.

    Two limits, because either alone leaves a hole: an age limit lets a busy
    week fill the disk, and a count limit lets a single ancient checkpoint
    linger forever. Age is applied first so the count limit operates on what
    survives.
    """
    removed = 0
    cutoff = time.time() - max_age_days * 86400.0
    states = load_all(root)

    survivors: list[dict] = []
    for state in states:
        if float(state.get("updated_at") or 0) < cutoff:
            shutil.rmtree(state["dir"], ignore_errors=True)
            removed += 1
        else:
            survivors.append(state)

    # load_all() is newest-first, so anything past the cap is the oldest.
    for state in survivors[max_checkpoints:]:
        shutil.rmtree(state["dir"], ignore_errors=True)
        removed += 1
    return removed
