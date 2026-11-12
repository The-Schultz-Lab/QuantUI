"""A results folder is data, not page code (audit VIZ-1).

The Viewer exists to open other people's result folders, and the app
replays History from disk. Three paths pasted folder content into the page:
cached vibrational-animation HTML (with an unchecked file name), the saved
HOMO/LUMO cube (unchecked file name, embedded in a ``<script>`` with
``json.dumps``, which leaves ``</script>`` intact), and mode-symmetry labels
(raw into a widgets.HTML table).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantui import vib_cache
from quantui.symmetry import html_gamma, html_group, html_irrep
from quantui.viz_assets import script_json

PAYLOAD = "<img src=x onerror=alert(1)>"
SCRIPT_BREAK = "x</script><script>window.PWNED=1</script>"


def _save(rd: Path, html: str = "<p>mode 1</p>") -> None:
    vib_cache.save_cached_html(
        rd,
        1,
        html,
        freq_cm1=1600.0,
        n_frames=24,
        amplitude=0.4,
        renderer="py3dmol",
        fps=24,
    )


def _get(rd: Path):
    return vib_cache.get_cached_html(
        rd, 1, n_frames=24, amplitude=0.4, renderer="py3dmol", fps=24
    )


class TestVibCache:
    def test_own_cache_round_trips(self, tmp_path):
        _save(tmp_path)
        assert _get(tmp_path) == "<p>mode 1</p>"

    def test_altered_file_is_not_served(self, tmp_path):
        _save(tmp_path)
        (vib_cache.cache_dir(tmp_path) / "mode_001.html").write_text(
            "<script>window.PWNED=2</script>", encoding="utf-8"
        )
        assert _get(tmp_path) is None

    def test_foreign_cache_is_not_served(self, tmp_path):
        # A folder from another user: a well-formed index, but its MAC was
        # made with a key this user does not have.
        _save(tmp_path)
        idx_path = vib_cache.cache_dir(tmp_path) / "index.json"
        idx = json.loads(idx_path.read_text())
        idx["modes"]["1"]["mac"] = "0" * 64
        idx_path.write_text(json.dumps(idx))
        assert _get(tmp_path) is None

    @pytest.mark.parametrize("name", ["../evil.html", "/etc/passwd", "x/mode_001.html"])
    def test_index_file_must_be_a_plain_mode_name(self, tmp_path, name):
        _save(tmp_path)
        (tmp_path / "evil.html").write_text("<script>bad()</script>")
        idx_path = vib_cache.cache_dir(tmp_path) / "index.json"
        idx = json.loads(idx_path.read_text())
        idx["modes"]["1"]["file"] = name
        idx_path.write_text(json.dumps(idx))
        assert _get(tmp_path) is None


class TestFrontierCube:
    def _folder(self, tmp_path, file_name):
        from quantui.orbital_visualization import FRONTIER_CUBES_DIR

        d = tmp_path / "result" / FRONTIER_CUBES_DIR
        d.mkdir(parents=True)
        (tmp_path / "outside.gz").write_bytes(gzip.compress(b"secret"))
        (d / "index.json").write_text(
            json.dumps({"orbitals": [{"index": 4, "spin": "", "file": file_name}]})
        )
        return tmp_path / "result"

    @pytest.mark.parametrize("name", ["../../outside.gz", "/tmp/outside.gz", ".."])
    def test_index_file_outside_the_folder_is_refused(self, tmp_path, name):
        from quantui.orbital_visualization import load_frontier_cube

        rd = self._folder(tmp_path, name)
        dest = tmp_path / "out.cube"
        assert load_frontier_cube(rd, 4, beta=False, dest=dest) is False
        assert not dest.exists()

    def test_script_breaking_cube_text_stays_inside_the_script(self):
        from quantui.orbital_visualization import _build_iso_viewer

        html = _build_iso_viewer(
            SCRIPT_BREAK, data_format="cube", scene_key="k", with_surfaces=False
        )
        assert "</script><script>window.PWNED" not in html


def test_script_json_round_trips_and_cannot_close_a_script():
    value = {"a": SCRIPT_BREAK + "<!--"}
    text = script_json(value)
    assert "</" not in text and "<!--" not in text
    assert json.loads(text) == value


class TestSymmetryLabels:
    def test_helpers_escape(self):
        assert "<img" not in html_irrep(PAYLOAD)
        assert "<img" not in html_gamma(f"2{PAYLOAD} + A1")
        assert "<img" not in html_group(PAYLOAD)
        assert html_irrep("a1g") == "a<sub>1g</sub>"

    def test_vib_mode_table_escapes_saved_labels(self):
        from quantui.app_visualization import vib_mode_table_html

        result = SimpleNamespace(frequencies_cm1=[1600.0], ir_intensities=[1.0])
        irreps = SimpleNamespace(labels=[PAYLOAD], caption_html=lambda: "")
        assert "<img" not in vib_mode_table_html(result, irreps)
