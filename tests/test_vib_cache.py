"""Unit tests for `quantui.vib_cache` — vibrational animation disk cache.

Covers cache hit / miss / fallback paths, schema-version invalidation,
parameter-mismatch invalidation, atomic-write semantics, and graceful
failure modes (missing dir, malformed JSON, OS errors). All file I/O uses
pytest's ``tmp_path`` fixture so no test touches a real result directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantui import vib_cache


def _make_result_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "result"
    rd.mkdir()
    return rd


class TestCacheDir:
    def test_cache_dir_is_vib_frames_subfolder(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        assert vib_cache.cache_dir(rd) == rd / "vib_frames"


class TestLoadIndex:
    def test_missing_returns_empty(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        assert vib_cache.load_index(rd) == {}

    def test_valid_index_loads(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        cdir = vib_cache.cache_dir(rd)
        cdir.mkdir()
        (cdir / "index.json").write_text(
            json.dumps(
                {
                    "_schema_version": 1,
                    "n_frames": 24,
                    "amplitude": 0.4,
                    "renderer": "py3dmol",
                    "modes": {"1": {"cached": True, "file": "mode_001.html"}},
                }
            )
        )
        idx = vib_cache.load_index(rd)
        assert idx["_schema_version"] == 1
        assert idx["n_frames"] == 24

    def test_malformed_json_returns_empty(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        cdir = vib_cache.cache_dir(rd)
        cdir.mkdir()
        (cdir / "index.json").write_text("not valid json {")
        assert vib_cache.load_index(rd) == {}

    def test_non_dict_root_returns_empty(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        cdir = vib_cache.cache_dir(rd)
        cdir.mkdir()
        (cdir / "index.json").write_text(json.dumps([1, 2, 3]))
        assert vib_cache.load_index(rd) == {}


class TestSaveAndLoadRoundtrip:
    def test_save_creates_files(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            1,
            "<html>mode 1</html>",
            freq_cm1=1600.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        cdir = vib_cache.cache_dir(rd)
        assert (cdir / "mode_001.html").exists()
        assert (cdir / "index.json").exists()

    def test_save_then_get_returns_html(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            1,
            "<html>mode 1</html>",
            freq_cm1=1600.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        loaded = vib_cache.get_cached_html(
            rd, 1, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=10
        )
        assert loaded == "<html>mode 1</html>"

    def test_index_has_freq_cm1(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            3,
            "<html/>",
            freq_cm1=1623.4,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        idx = vib_cache.load_index(rd)
        assert idx["modes"]["3"]["freq_cm1"] == pytest.approx(1623.4)

    def test_multiple_modes_share_index(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        for mode, freq in [(1, 1600.0), (2, 3800.0), (3, 3850.0)]:
            vib_cache.save_cached_html(
                rd,
                mode,
                f"<html>mode {mode}</html>",
                freq_cm1=freq,
                n_frames=24,
                amplitude=0.4,
                renderer="py3dmol",
                fps=10,
            )
        idx = vib_cache.load_index(rd)
        assert set(idx["modes"].keys()) == {"1", "2", "3"}
        # Each mode's HTML loads correctly
        for mode in (1, 2, 3):
            loaded = vib_cache.get_cached_html(
                rd, mode, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=10
            )
            assert loaded == f"<html>mode {mode}</html>"


class TestHasCachedMatching:
    def setup_method(self):
        # Fixtures populated in each test
        pass

    def test_matching_params_returns_true(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            1,
            "<html/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        assert vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=10
        )

    def test_different_n_frames_invalidates(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            1,
            "<html/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        assert not vib_cache.has_cached(
            rd, 1, n_frames=48, amplitude=0.4, renderer="py3dmol", fps=10
        )

    def test_different_amplitude_invalidates(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            1,
            "<html/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        assert not vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.8, renderer="py3dmol", fps=10
        )

    def test_different_renderer_invalidates(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            1,
            "<html/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        assert not vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.4, renderer="plotlymol", fps=10
        )

    def test_amplitude_tolerance_allows_float_roundtrip(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            1,
            "<html/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        # 0.4 should match itself across JSON round-trip even with tiny FP noise
        assert vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.4 + 1e-9, renderer="py3dmol", fps=10
        )

    def test_mode_not_in_index_returns_false(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            1,
            "<html/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        assert not vib_cache.has_cached(
            rd, 99, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=10
        )

    def test_missing_html_file_returns_false(self, tmp_path):
        """If index claims cached but the HTML file is deleted, treat as miss."""
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            1,
            "<html/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        # Delete the HTML file but leave the index
        (vib_cache.cache_dir(rd) / "mode_001.html").unlink()
        assert not vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=10
        )


class TestSchemaVersionInvalidation:
    def test_old_schema_version_returns_miss(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        cdir = vib_cache.cache_dir(rd)
        cdir.mkdir()
        # Write an index from a hypothetical older schema
        (cdir / "index.json").write_text(
            json.dumps(
                {
                    "_schema_version": 0,
                    "n_frames": 24,
                    "amplitude": 0.4,
                    "renderer": "py3dmol",
                    "modes": {"1": {"cached": True, "file": "mode_001.html"}},
                }
            )
        )
        # Create the html file too
        (cdir / "mode_001.html").write_text("<html/>")
        assert not vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=10
        )

    def test_resaving_with_changed_params_rebuilds_index(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        # Save with amplitude=0.4
        vib_cache.save_cached_html(
            rd,
            1,
            "<html/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        assert vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=10
        )
        # Re-save with amplitude=0.6 — should reset the index
        vib_cache.save_cached_html(
            rd,
            1,
            "<html v2/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.6,
            renderer="py3dmol",
            fps=10,
        )
        # Old amplitude no longer matches
        assert not vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=10
        )
        # New amplitude matches
        assert vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.6, renderer="py3dmol", fps=10
        )


class TestFpsInvalidation:
    """fps is a cache key parameter; mismatch must yield a miss + rebuild."""

    def test_different_fps_invalidates(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            1,
            "<html/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        # Different fps → cache miss
        assert not vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=30
        )
        # Same fps → cache hit
        assert vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=10
        )

    def test_resaving_with_new_fps_rebuilds_index(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            1,
            "<html fps10/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        # User changes fps to 30 — saving the same mode should reset the
        # index's fps field and invalidate the old fps=10 entry.
        vib_cache.save_cached_html(
            rd,
            1,
            "<html fps30/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=30,
        )
        assert not vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=10
        )
        assert vib_cache.has_cached(
            rd, 1, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=30
        )
        # Index records the new fps
        assert vib_cache.load_index(rd)["fps"] == 30


class TestAtomicWrites:
    def test_no_tmp_leftover_on_success(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            1,
            "<html/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        cdir = vib_cache.cache_dir(rd)
        tmp_leftovers = list(cdir.glob("*.tmp"))
        assert tmp_leftovers == []

    def test_save_failure_does_not_raise(self, tmp_path, monkeypatch):
        rd = _make_result_dir(tmp_path)

        def _boom(self, *_a, **_kw):
            raise OSError("simulated disk full")

        monkeypatch.setattr("pathlib.Path.write_text", _boom)
        # Should NOT raise — non-fatal
        vib_cache.save_cached_html(
            rd,
            1,
            "<html/>",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )


class TestGetCachedHtml:
    def test_miss_returns_none(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        assert (
            vib_cache.get_cached_html(
                rd, 1, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=10
            )
            is None
        )

    def test_hit_returns_html_content(self, tmp_path):
        rd = _make_result_dir(tmp_path)
        vib_cache.save_cached_html(
            rd,
            5,
            "PAYLOAD",
            freq_cm1=1.0,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        result = vib_cache.get_cached_html(
            rd, 5, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=10
        )
        assert result == "PAYLOAD"
