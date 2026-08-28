"""Calculation checkpoints — M-CHECKPOINT.

The feature exists so an interrupted long run doesn't discard the work it
finished. That makes the failure cases the interesting ones: a checkpoint is
read precisely when something has already gone wrong, so every read has to cope
with a directory left behind by a process that died mid-write.

The rule these tests enforce above all others: **a checkpoint must never break
a calculation.** Corrupt metadata, a truncated append, a wrong schema version
and an unwritable directory all have to behave exactly like "no checkpoint" —
the calculation runs, it just starts from scratch.

Platform-independent: no PySCF, no ASE, no widgets front-end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from quantui import checkpoint as C

# ══ Fixtures ═════════════════════════════════════════════════════════════════


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("QUANTUI_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    return tmp_path / "ckpt"


class _FakeMolecule:
    def __init__(self, atoms=("O", "H", "H"), coords=None, charge=0, multiplicity=1):
        self.atoms = list(atoms)
        self.coordinates = coords or [
            [0.0, 0.0, 0.0],
            [0.76, 0.59, 0.0],
            [-0.76, 0.59, 0.0],
        ]
        self.charge = charge
        self.multiplicity = multiplicity


def _identity(**overrides) -> C.CalcIdentity:
    base = {
        "calc_type": "geometry_opt",
        "method": "RHF",
        "basis": "6-31G",
        "charge": 0,
        "multiplicity": 1,
        "atom_symbols": ("O", "H", "H"),
        "coords": ((0.0, 0.0, 0.0), (0.76, 0.59, 0.0), (-0.76, 0.59, 0.0)),
    }
    base.update(overrides)
    return C.CalcIdentity(**base)


# ══ Identity ═════════════════════════════════════════════════════════════════


class TestResumeKey:
    """Resuming into the wrong checkpoint silently splices two calculations."""

    def test_identical_inputs_share_a_key(self):
        assert _identity().resume_key == _identity().resume_key

    def test_geometry_change_changes_the_key(self):
        moved = _identity(
            coords=((0.0, 0.0, 0.0), (0.80, 0.59, 0.0), (-0.76, 0.59, 0.0))
        )
        assert moved.resume_key != _identity().resume_key

    def test_calc_type_change_changes_the_key(self):
        assert _identity(calc_type="pes_scan").resume_key != _identity().resume_key

    def test_basis_change_changes_the_key(self):
        assert _identity(basis="STO-3G").resume_key != _identity().resume_key

    def test_charge_change_changes_the_key(self):
        assert _identity(charge=1).resume_key != _identity().resume_key

    def test_multiplicity_change_changes_the_key(self):
        assert _identity(multiplicity=3).resume_key != _identity().resume_key

    def test_key_survives_float_round_tripping(self):
        """A geometry saved and reloaded must still match itself.

        Coordinates make a round trip through JSON on every save; if the key
        were sensitive to the last bits of a float, resume would fail exactly
        when it was needed.
        """
        original = _identity()
        reloaded_coords = tuple(
            tuple(json.loads(json.dumps(v)) for v in row) for row in original.coords
        )
        assert _identity(coords=reloaded_coords).resume_key == original.resume_key


class TestWarmStartKey:
    """A density from a nearby geometry is a good guess — that's the point."""

    def test_geometry_change_does_not_change_the_key(self):
        moved = _identity(
            coords=((0.0, 0.0, 0.0), (0.90, 0.59, 0.0), (-0.76, 0.59, 0.0))
        )
        assert moved.warm_start_key == _identity().warm_start_key

    def test_calc_type_change_does_not_change_the_key(self):
        """A single point's converged density is a fine guess for an optimization."""
        assert (
            _identity(calc_type="pes_scan").warm_start_key == _identity().warm_start_key
        )

    def test_method_change_changes_the_key(self):
        assert _identity(method="B3LYP").warm_start_key != _identity().warm_start_key

    def test_basis_change_changes_the_key(self):
        """A density in one basis is not a density in another."""
        assert _identity(basis="cc-pVDZ").warm_start_key != _identity().warm_start_key

    def test_different_molecule_changes_the_key(self):
        assert _identity(atom_symbols=("C", "H", "H")).warm_start_key != (
            _identity().warm_start_key
        )


class TestIdentityFromMolecule:
    def test_reads_atoms_charge_and_multiplicity(self):
        ident = C.CalcIdentity.from_molecule(
            _FakeMolecule(charge=-1, multiplicity=2),
            calc_type="single_point",
            method="UHF",
            basis="6-31G",
        )
        assert ident.atom_symbols == ("O", "H", "H")
        assert ident.charge == -1
        assert ident.multiplicity == 2

    def test_matches_a_hand_built_identity(self):
        from_mol = C.CalcIdentity.from_molecule(
            _FakeMolecule(), calc_type="geometry_opt", method="RHF", basis="6-31G"
        )
        assert from_mol.resume_key == _identity().resume_key

    def test_survives_a_molecule_without_coordinates(self):
        """Never raise while merely identifying a calculation."""

        class _Bare:
            atoms = ["H", "H"]

        ident = C.CalcIdentity.from_molecule(
            _Bare(), calc_type="single_point", method="RHF", basis="STO-3G"
        )
        assert ident.resume_key


# ══ Lifecycle ════════════════════════════════════════════════════════════════


class TestLifecycle:
    def test_constructing_touches_no_disk(self, root):
        C.Checkpoint(_identity())
        assert not root.exists()

    def test_begin_creates_the_directory_and_metadata(self, root):
        ckpt = C.Checkpoint(_identity())
        assert ckpt.begin() is True
        assert ckpt.meta_path.is_file()

    def test_begin_records_running_status(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        assert ckpt.load_state()["status"] == C.STATUS_RUNNING

    def test_begin_stores_the_warm_start_key(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        assert ckpt.load_state()["warm_start_key"] == _identity().warm_start_key

    def test_extra_fields_are_stored(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin(total_points=12)
        assert ckpt.load_state()["total_points"] == 12

    def test_update_merges_without_dropping_fields(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.update(steps_done=4)
        state = ckpt.load_state()
        assert state["steps_done"] == 4
        assert state["calc_type"] == "geometry_opt"

    def test_mark_complete_is_recorded(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.write_run_log("partial output")
        ckpt.mark_complete()
        assert ckpt.load_state()["status"] == C.STATUS_COMPLETE
        assert ckpt.read_run_log() == ""

    def test_discard_removes_the_directory(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.discard()
        assert not ckpt.dir.exists()

    def test_discard_is_safe_when_nothing_exists(self, root):
        C.Checkpoint(_identity()).discard()  # must not raise

    def test_update_on_a_missing_checkpoint_is_a_no_op(self, root):
        C.Checkpoint(_identity()).update(steps_done=3)  # must not raise


# ══ Provenance in the run log ════════════════════════════════════════════════


class _Stream:
    """Stand-in for the run's output log."""

    def __init__(self) -> None:
        self.text = ""

    def write(self, chunk: str) -> None:
        self.text += chunk


class TestCheckpointLogging:
    """Checkpoint events belong in the archived ``pyscf.log``.

    Unlike the Phase D liveness heartbeat, which is deliberately kept out of
    the archive, these lines are *provenance*: without them the log of a
    resumed run reads as a calculation that started from the geometry at the
    top of the file, which is untrue.
    """

    def _logged(self, root):
        stream = _Stream()
        ckpt = C.Checkpoint(_identity(), log_stream=stream)
        return ckpt, stream

    def test_opening_is_logged_with_its_location(self, root):
        ckpt, stream = self._logged(root)
        ckpt.begin()
        assert "opened" in stream.text
        assert str(ckpt.dir) in stream.text

    def test_each_save_is_logged(self, root):
        ckpt, stream = self._logged(root)
        ckpt.begin()
        ckpt.update(steps_done=1)
        ckpt.update(steps_done=2)
        assert stream.text.count("saved") == 2

    def test_the_saved_line_names_what_changed(self, root):
        ckpt, stream = self._logged(root)
        ckpt.begin()
        ckpt.update(steps_done=7)
        assert "steps_done=7" in stream.text

    def test_scan_points_are_logged_by_index(self, root):
        ckpt, stream = self._logged(root)
        ckpt.begin()
        ckpt.append_point({"index": 4, "value": 1.2})
        assert "scan point 4" in stream.text

    def test_completion_is_logged(self, root):
        ckpt, stream = self._logged(root)
        ckpt.begin()
        ckpt.mark_complete()
        assert "status=complete" in stream.text

    def test_discarding_is_logged(self, root):
        ckpt, stream = self._logged(root)
        ckpt.begin()
        ckpt.discard()
        assert "discarded" in stream.text

    def test_a_failed_write_is_not_logged_as_a_save(self, root, monkeypatch):
        """A log line claiming work was saved when it wasn't is worse than
        silence — it is exactly the line someone would rely on later."""
        ckpt, stream = self._logged(root)
        ckpt.begin()
        monkeypatch.setattr(
            C, "_atomic_write_json", lambda *a, **k: (_ for _ in ()).throw(OSError())
        )
        ckpt.update(steps_done=3)
        assert "steps_done=3" not in stream.text

    def test_resume_banner_marks_where_continuation_began(self, root):
        """A resumed run must not pretend the log started from scratch."""
        ckpt, stream = self._logged(root)
        ckpt.begin()
        ckpt.log_resumed("continuing from step 12")
        assert "RESUMED" in stream.text
        assert "continuing from step 12" in stream.text
        assert "continuation began" in stream.text

    def test_no_stream_attached_is_silent_and_safe(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.update(steps_done=1)
        ckpt.log_resumed("x")  # must not raise

    def test_a_raising_stream_never_breaks_the_checkpoint(self, root):
        """``_log`` runs from a ``finally`` during teardown and from the
        optimizer's per-step callback, where the stream may already be
        raising cancellation."""

        class _Exploding:
            def write(self, _chunk):
                raise RuntimeError("cancelled")

        ckpt = C.Checkpoint(_identity(), log_stream=_Exploding())
        assert ckpt.begin() is True
        ckpt.update(steps_done=1)
        assert ckpt.load_state()["steps_done"] == 1

    def test_attach_log_routes_later_events(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        stream = _Stream()
        ckpt.attach_log(stream)
        ckpt.update(steps_done=5)
        assert "steps_done=5" in stream.text


# ══ Robustness — the cases that actually happen after a crash ════════════════


class TestCorruptState:
    def test_missing_metadata_reads_as_none(self, root):
        assert C.Checkpoint(_identity()).load_state() is None

    def test_truncated_json_reads_as_none(self, root):
        """A half-written meta.json is exactly what a crash leaves behind."""
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.meta_path.write_text('{"status": "run', encoding="utf-8")
        assert ckpt.load_state() is None

    def test_non_dict_json_reads_as_none(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.meta_path.write_text("[1, 2, 3]", encoding="utf-8")
        assert ckpt.load_state() is None

    def test_foreign_schema_version_is_ignored(self, root):
        """Old checkpoints are discarded, not migrated — they're worth minutes."""
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        state = json.loads(ckpt.meta_path.read_text(encoding="utf-8"))
        state["schema_version"] = C.CHECKPOINT_SCHEMA_VERSION + 99
        ckpt.meta_path.write_text(json.dumps(state), encoding="utf-8")
        assert ckpt.load_state() is None

    def test_corrupt_checkpoint_is_not_resumable(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.trajectory_path.write_bytes(b"frames")
        ckpt.meta_path.write_text("{oh no", encoding="utf-8")
        assert ckpt.resumable_state() is None


class TestAtomicWrites:
    def test_no_temp_file_is_left_behind(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.update(steps_done=1)
        assert not list(ckpt.dir.glob("*.tmp"))

    def test_a_failed_write_leaves_the_previous_state_readable(self, root, monkeypatch):
        """os.replace is what makes this hold — a partial write is never visible."""
        ckpt = C.Checkpoint(_identity())
        ckpt.begin(steps_done=7)

        def _boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(C, "_atomic_write_json", _boom)
        ckpt.update(steps_done=8)  # must not raise
        assert ckpt.load_state()["steps_done"] == 7


# ══ Resumability ═════════════════════════════════════════════════════════════


class TestResumable:
    def test_a_fresh_checkpoint_has_nothing_to_resume(self, root):
        """A directory is not progress. Offering to resume it promises nothing."""
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        assert ckpt.resumable_state() is None

    def test_a_trajectory_counts_as_progress(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.trajectory_path.write_bytes(b"some frames")
        assert ckpt.resumable_state() is not None

    def test_an_empty_trajectory_is_not_progress(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.trajectory_path.write_bytes(b"")
        assert ckpt.resumable_state() is None

    def test_completed_points_count_as_progress(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.append_point({"index": 1, "value": 0.5})
        assert ckpt.resumable_state() is not None

    def test_a_completed_run_is_not_resumable(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.trajectory_path.write_bytes(b"frames")
        ckpt.mark_complete()
        assert ckpt.resumable_state() is None

    def test_find_resumable_returns_the_checkpoint(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.trajectory_path.write_bytes(b"frames")
        assert C.find_resumable(_identity()) is not None

    def test_find_resumable_is_none_for_a_different_geometry(self, root):
        """The strict key is the guard against resuming the wrong run."""
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.trajectory_path.write_bytes(b"frames")
        other = _identity(
            coords=((9.0, 9.0, 9.0), (0.76, 0.59, 0.0), (-0.76, 0.59, 0.0))
        )
        assert C.find_resumable(other) is None


# ══ Scan points ══════════════════════════════════════════════════════════════


class TestPoints:
    def test_points_round_trip(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.append_point({"index": 1, "energy_hartree": -1.1})
        ckpt.append_point({"index": 2, "energy_hartree": -1.2})
        assert [p["index"] for p in ckpt.completed_points()] == [1, 2]

    def test_a_truncated_final_line_is_skipped_not_fatal(self, root):
        """Append-only means a crash costs at most the last partial line."""
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.append_point({"index": 1, "energy_hartree": -1.1})
        with open(ckpt.points_path, "a", encoding="utf-8") as fh:
            fh.write('{"index": 2, "energy_hart')
        points = ckpt.completed_points()
        assert len(points) == 1
        assert points[0]["index"] == 1

    def test_blank_lines_are_ignored(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.append_point({"index": 1})
        with open(ckpt.points_path, "a", encoding="utf-8") as fh:
            fh.write("\n\n")
        assert len(ckpt.completed_points()) == 1

    def test_missing_points_file_reads_as_empty(self, root):
        assert C.Checkpoint(_identity()).completed_points() == []

    def test_append_creates_the_directory_if_needed(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.append_point({"index": 1})
        assert ckpt.completed_points()


# ══ Warm start discovery ═════════════════════════════════════════════════════


class TestWarmStartDiscovery:
    def _seed(self, identity, *, chk_bytes=b"density"):
        ckpt = C.Checkpoint(identity)
        ckpt.begin()
        if chk_bytes is not None:
            ckpt.scf_chkfile.write_bytes(chk_bytes)
        return ckpt

    def test_finds_a_chkfile_from_a_different_geometry(self, root):
        """The whole point: a nearby geometry's density is a good guess."""
        self._seed(_identity())
        moved = _identity(
            coords=((0.0, 0.0, 0.0), (0.85, 0.59, 0.0), (-0.76, 0.59, 0.0))
        )
        assert C.find_warm_start_chkfile(moved) is not None

    def test_finds_a_chkfile_from_a_different_calc_type(self, root):
        self._seed(_identity(calc_type="single_point"))
        assert (
            C.find_warm_start_chkfile(_identity(calc_type="geometry_opt")) is not None
        )

    def test_does_not_cross_basis_sets(self, root):
        self._seed(_identity(basis="STO-3G"))
        assert C.find_warm_start_chkfile(_identity(basis="cc-pVDZ")) is None

    def test_does_not_cross_methods(self, root):
        self._seed(_identity(method="RHF"))
        assert C.find_warm_start_chkfile(_identity(method="B3LYP")) is None

    def test_does_not_cross_molecules(self, root):
        self._seed(_identity())
        assert C.find_warm_start_chkfile(_identity(atom_symbols=("C", "O"))) is None

    def test_an_empty_chkfile_is_not_offered(self, root):
        """A zero-byte file is what an interrupted write leaves."""
        self._seed(_identity(), chk_bytes=b"")
        assert C.find_warm_start_chkfile(_identity()) is None

    def test_a_missing_chkfile_is_not_offered(self, root):
        self._seed(_identity(), chk_bytes=None)
        assert C.find_warm_start_chkfile(_identity()) is None

    def test_prefers_the_most_recent_match(self, root):
        older = self._seed(_identity(calc_type="single_point"))
        newer = self._seed(_identity(calc_type="pes_scan"))
        older.update(updated_at=1.0)
        newer.update(updated_at=time.time() + 1000)
        found = C.find_warm_start_chkfile(_identity())
        assert found == newer.scf_chkfile

    def test_no_history_returns_none(self, root):
        assert C.find_warm_start_chkfile(_identity()) is None


# ══ Listing + retention ══════════════════════════════════════════════════════


class TestLoadAll:
    def test_lists_every_readable_checkpoint(self, root):
        C.Checkpoint(_identity(calc_type="geometry_opt")).begin()
        C.Checkpoint(_identity(calc_type="pes_scan")).begin()
        assert len(C.load_all()) == 2

    def test_one_corrupt_checkpoint_does_not_hide_the_others(self, root):
        good = C.Checkpoint(_identity(calc_type="geometry_opt"))
        good.begin()
        bad = C.Checkpoint(_identity(calc_type="pes_scan"))
        bad.begin()
        bad.meta_path.write_text("{ not json", encoding="utf-8")
        assert len(C.load_all()) == 1

    def test_entries_carry_their_directory(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        assert C.load_all()[0]["dir"] == str(ckpt.dir)

    def test_newest_first(self, root):
        first = C.Checkpoint(_identity(calc_type="geometry_opt"))
        first.begin()
        second = C.Checkpoint(_identity(calc_type="pes_scan"))
        second.begin()
        first.update(updated_at=1.0)
        second.update(updated_at=time.time() + 500)
        assert C.load_all()[0]["calc_type"] == "pes_scan"

    def test_missing_root_reads_as_empty(self, root):
        assert C.load_all() == []


class TestPrune:
    def test_old_checkpoints_are_removed(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.update(updated_at=time.time() - 60 * 86400)
        assert C.prune(max_age_days=14) == 1
        assert C.load_all() == []

    def test_recent_checkpoints_are_kept(self, root):
        C.Checkpoint(_identity()).begin()
        assert C.prune(max_age_days=14) == 0
        assert len(C.load_all()) == 1

    def test_count_limit_removes_the_oldest(self, root):
        for i in range(5):
            ckpt = C.Checkpoint(_identity(calc_type=f"type_{i}"))
            ckpt.begin()
            ckpt.update(updated_at=time.time() - i)
        assert C.prune(max_checkpoints=2) == 3
        kept = {s["calc_type"] for s in C.load_all()}
        assert kept == {"type_0", "type_1"}

    def test_age_limit_applies_before_the_count_limit(self, root):
        """Otherwise a single ancient checkpoint could evict fresh ones."""
        fresh = C.Checkpoint(_identity(calc_type="fresh"))
        fresh.begin()
        stale = C.Checkpoint(_identity(calc_type="stale"))
        stale.begin()
        stale.update(updated_at=time.time() - 90 * 86400)
        C.prune(max_age_days=14, max_checkpoints=1)
        assert [s["calc_type"] for s in C.load_all()] == ["fresh"]

    def test_prune_on_an_empty_root_is_a_no_op(self, root):
        assert C.prune() == 0


# ══ Run-log persistence (CHK.8b / ISSUE.9) ═══════════════════════════════════


class TestRunLogPersistence:
    def test_write_read_round_trip(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.write_run_log("first chunk\nline two\n")
        assert ckpt.read_run_log() == "first chunk\nline two\n"

    def test_clear_run_log(self, root):
        ckpt = C.Checkpoint(_identity())
        ckpt.begin()
        ckpt.write_run_log("chunk")
        ckpt.clear_run_log()
        assert ckpt.read_run_log() == ""

    def test_read_missing_log_is_empty(self, root):
        ckpt = C.Checkpoint(_identity())
        assert ckpt.read_run_log() == ""


# ══ Location ═════════════════════════════════════════════════════════════════


class TestCheckpointRoot:
    def test_env_var_overrides_the_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_CHECKPOINT_DIR", str(tmp_path / "elsewhere"))
        assert C.checkpoint_root() == tmp_path / "elsewhere"

    def test_default_lives_under_the_quantui_home(self, monkeypatch):
        monkeypatch.delenv("QUANTUI_CHECKPOINT_DIR", raising=False)
        assert C.checkpoint_root() == Path.home() / ".quantui" / "checkpoints"

    def test_checkpoints_are_not_inside_the_results_directory(self, monkeypatch):
        """Results dirs are created on success; checkpoints exist for failure."""
        monkeypatch.delenv("QUANTUI_CHECKPOINT_DIR", raising=False)
        assert "results" not in C.checkpoint_root().parts
