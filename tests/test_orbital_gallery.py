"""Orbital gallery: small linked viewers around the gap (SURF.1, DEC-024)."""

from __future__ import annotations

import io
import re
import time

import numpy as np
import pytest

from quantui.molecule import Molecule
from quantui.orbital_visualization import (
    GalleryTile,
    build_orbital_gallery_html,
    compact_cube_text,
    gallery_caption,
    gallery_orbital_indices,
    pretty_irrep,
)

WATER = Molecule(
    ["O", "H", "H"], [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]]
)


class TestSelection:
    def test_symmetric_window(self):
        assert gallery_orbital_indices(5, 13, 3) == [
            ("HOMO−2", 2),
            ("HOMO−1", 3),
            ("HOMO", 4),
            ("LUMO", 5),
            ("LUMO+1", 6),
            ("LUMO+2", 7),
        ]

    def test_clipped_at_both_ends(self):
        assert gallery_orbital_indices(1, 2, 3) == [("HOMO", 0), ("LUMO", 1)]


class TestCaptions:
    def test_pretty_irrep(self):
        assert pretty_irrep("3a1") == "3a₁"
        assert pretty_irrep("1b2g") == "1b₂g"
        assert pretty_irrep("2πu") == "2πu"

    def test_two_lines(self):
        title, detail = gallery_caption("HOMO−1", 3, -0.5612, "3a1")
        assert title == "HOMO−1 · 3a₁"
        assert detail == "−0.561 Eh (−15.27 eV) · MO 3"
        assert gallery_caption("LUMO", 5, None) == ("LUMO", "MO 5")


def _cube(tmp_path):
    pytest.importorskip("pyscf")
    from quantui.orbital_visualization import generate_cube_from_arrays
    from quantui.session_calc import run_in_session

    r = run_in_session(WATER, "RHF", "STO-3G", progress_stream=io.StringIO())
    path = tmp_path / "homo.cube"
    generate_cube_from_arrays(
        r.pyscf_mol_atom, "STO-3G", r.mo_coeff, 4, path, nx=12, ny=12, nz=12
    )
    return path.read_text()


class TestCompactCube:
    def test_same_grid_fewer_bytes(self, tmp_path):
        text = _cube(tmp_path)
        small = compact_cube_text(text)
        assert len(small) < 0.85 * len(text)
        a, b = text.splitlines(), small.splitlines()
        assert a[:9] == b[:9]  # 2 comments, origin, 3 axes, 3 atoms
        va = np.array(" ".join(a[9:]).split(), dtype=float)
        vb = np.array(" ".join(b[9:]).split(), dtype=float)
        assert va.shape == vb.shape == (12**3,)
        assert np.allclose(va, vb, rtol=1e-3, atol=1e-8)


class TestGalleryHtml:
    def test_one_library_copy_unique_viewers_linked(self, tmp_path):
        text = _cube(tmp_path)
        tiles = [GalleryTile(f"T{i}", f"T{i}", text, "d") for i in range(6)]
        html = build_orbital_gallery_html(tiles)
        # The vendored 3Dmol.js travels once, not once per tile.
        assert html.count("data:text/javascript;base64,") == 1
        uids = re.findall(r'id="3dmolviewer_(\w+)"', html)
        assert len(uids) == 6 and len(set(uids)) == 6
        assert "linkViewer" in html
        assert "repeat(3," in html  # 6 tiles → 3 columns × 2 rows

    def test_column_count_follows_tile_count(self):
        for n, cols in ((2, 1), (4, 2), (8, 4)):
            tiles = [GalleryTile("x", "x", "", "") for _ in range(n)]
            assert f"repeat({cols}," in build_orbital_gallery_html(tiles)


class TestAppGallery:
    @pytest.fixture(autouse=True)
    def _results_in_tmp(self, tmp_path, monkeypatch):
        # Gallery cubes go to the results folder; keep them out of the repo.
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))

    def _app(self, mol, method, basis):
        pytest.importorskip("pyscf")
        from quantui.app import QuantUIApp
        from quantui.app_visualization import show_orbital_diagram
        from quantui.session_calc import run_in_session

        r = run_in_session(mol, method, basis, progress_stream=io.StringIO())
        app = QuantUIApp()
        app._set_molecule(mol, "test")
        show_orbital_diagram(app, r)
        return app

    def test_water_tiles_and_symmetry_labels(self):
        from quantui.app_visualization import build_gallery_for_app

        html = build_gallery_for_app(self._app(WATER, "RHF", "6-31G"), 3)
        titles = re.findall(
            r'font-weight:600;margin:2px 0 0;text-align:center">([^<]+)<', html
        )
        assert titles == [
            "HOMO−2 · 1b₂",
            "HOMO−1 · 3a₁",
            "HOMO · 1b₁",
            "LUMO · 4a₁",
            "LUMO+1 · 2b₂",
            "LUMO+2 · 3b₂",
        ]

    def test_beta_channel(self):
        from quantui.app_visualization import build_gallery_for_app

        oh = Molecule(["O", "H"], [[0, 0, 0], [0, 0, 0.97]], multiplicity=2)
        app = self._app(oh, "UHF", "STO-3G")
        app._orb_spin_toggle.value = "beta"
        html = build_gallery_for_app(app, 2)
        assert "HOMO (β)" in html and "LUMO (β)" in html

    def test_button_builds_in_the_background(self):
        app = self._app(WATER, "RHF", "STO-3G")
        captured = {}
        orig = app._set_html_output

        def _grab(out, html):
            if out is app._orb_gallery_output:
                captured["html"] = html
            return orig(out, html)

        app._set_html_output = _grab
        app._orb_gallery_span_dd.value = 2
        app._orb_gallery_btn.click()
        deadline = time.time() + 120
        while app._orb_gallery_btn.disabled and time.time() < deadline:
            time.sleep(0.1)
        assert not app._orb_gallery_btn.disabled
        assert captured["html"].count("quantui-gallery-tile") == 4

    def test_needs_orbitals(self):
        from quantui.app import QuantUIApp
        from quantui.app_visualization import build_gallery_for_app

        assert "needs orbitals" in build_gallery_for_app(QuantUIApp(), 3)

    def test_gallery_shown_only_for_orbitals(self):
        app = self._app(WATER, "RHF", "STO-3G")
        app._iso_surface_dd.value = "density"
        assert app._orb_gallery_box.layout.display == "none"
        app._iso_surface_dd.value = "orbital"
        assert app._orb_gallery_box.layout.display == ""
