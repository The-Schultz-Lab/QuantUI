"""Regression tests for the Low-priority visualization audit items (VIZ-7..14)."""

from __future__ import annotations

import numpy as np
import pytest

# ── VIZ-7: generate_cube_file must not unpickle results.npz ────────────────


def _h2_npz(path, *, mol_atom):
    np.savez(
        path,
        mo_coeff=np.array([[0.55, 1.2], [0.55, -1.2]]),
        mo_occ=np.array([2.0, 0.0]),
        mol_atom=mol_atom,
        mol_basis="STO-3G",
    )


class TestGenerateCubeFileNoPickle:
    def test_object_array_in_npz_is_refused_not_unpickled(self, tmp_path):
        pytest.importorskip("pyscf")
        from quantui.orbital_visualization import generate_cube_file

        npz = tmp_path / "results.npz"
        # A list of (symbol, coords) tuples is saved as a pickled object array.
        _h2_npz(
            npz,
            mol_atom=np.array([("H", (0, 0, 0)), ("H", (0, 0, 0.74))], dtype=object),
        )
        with pytest.raises(ValueError, match="allow_pickle"):
            generate_cube_file(npz, 0, tmp_path / "mo.cube", nx=4, ny=4, nz=4)

    def test_plain_string_npz_still_works(self, tmp_path):
        pytest.importorskip("pyscf")
        from quantui.orbital_visualization import generate_cube_file

        npz = tmp_path / "results.npz"
        _h2_npz(npz, mol_atom="H 0 0 0; H 0 0 0.74")
        out = generate_cube_file(npz, 0, tmp_path / "mo.cube", nx=4, ny=4, nz=4)
        assert out.exists()


# ── VIZ-8: unscoped JS globals ──────────────────────────────────────────────

_H2_XYZ = "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74"
_GOTO_GLOBAL = 'window.__quantuiTrajGoTo=window["__quantuiTrajGoTo_"+UID];'


def _h2_molecule():
    from quantui.molecule import Molecule

    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


class TestPesClickTargetsOnlyTheTrajectoryStepper:
    """A PES-plot click calls ``window.__quantuiTrajGoTo``. Every stepper used
    to claim that name, so the click drove whichever was built last."""

    def test_trajectory_viewer_registers_the_click_target(self):
        pytest.importorskip("py3Dmol")
        from quantui.app_visualization import build_trajectory_viewer_html

        html = build_trajectory_viewer_html([_H2_XYZ, _H2_XYZ])
        assert _GOTO_GLOBAL in html

    def test_reorg_and_preopt_steppers_do_not(self):
        pytest.importorskip("py3Dmol")
        from quantui.app_visualization import (
            build_preopt_preview_html,
            build_reorg_geometry_viewer_html,
        )

        geoms = [
            {"label": lbl, "atoms": ["H", "H"], "coordinates": c}
            for lbl, c in (
                ("A", [[0, 0, 0], [0, 0, 0.74]]),
                ("B", [[0, 0, 0], [0, 0, 0.80]]),
            )
        ]
        reorg = build_reorg_geometry_viewer_html(geoms)
        preopt = build_preopt_preview_html(
            ["H", "H"], [[[0, 0, 0], [0, 0, 0.74]], [[0, 0, 0], [0, 0, 0.80]]]
        )
        for html in (reorg, preopt):
            assert "__quantuiTrajGoTo_" in html  # the uid-scoped one stays
            assert "window.__quantuiTrajGoTo=" not in html


def _iso_script(html: str) -> str:
    import re

    for body in re.findall(r"<script>(.*?)</script>", html, flags=re.S):
        if "__quantuiIsoUpdate_" in body:
            return body
    raise AssertionError("iso viewer script not found")


_FAKE_VIEWER_JS = """
var window = globalThis; var volCalls = 0;
globalThis.document = {getElementById: function(){ return null; },
                       querySelector: function(){ return null; }};
globalThis.$3Dmol = {VolumeData: function(){}};
var noop = function(){ return {}; };
window.viewer___UID__ = new Proxy({}, {get: function(_t, name){
  if(name === "addVolumetricData"){ return function(){ volCalls++; return {}; }; }
  if(name === "getView"){ return function(){ return [0,0,0,0,0,0,0,1]; }; }
  return noop;
}});
"""


class TestPlaceholderIgnoresSurfaceUpdates:
    """Before the first Generate the iso panel shows the molecule (XYZ). A
    slider move used to rebuild 'surfaces' from that XYZ text as if it were a
    cube."""

    def _run(self, html: str, tmp_path) -> int:
        import json
        import re
        import shutil
        import subprocess

        node = shutil.which("node")
        if node is None:
            pytest.skip("node not available")
        uid = re.search(r"3dmolviewer_(\w+)", html).group(1)
        script = (
            _FAKE_VIEWER_JS.replace("__UID__", uid)
            + _iso_script(html)
            + "\nvar before = volCalls;"
            + "\nwindow.__quantuiIsoUpdate({iso: 0.05, op: 0.5});"
            + "\nconsole.log(JSON.stringify(volCalls - before));"
        )
        path = tmp_path / "iso.js"
        path.write_text(script, encoding="utf-8")
        out = subprocess.run([node, str(path)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        return int(json.loads(out.stdout.strip().splitlines()[-1]))

    def test_update_on_placeholder_adds_no_volumetric_data(self, tmp_path):
        pytest.importorskip("py3Dmol")
        from quantui.orbital_visualization import render_molecule_placeholder_py3dmol

        html = render_molecule_placeholder_py3dmol(_h2_molecule())
        assert self._run(html, tmp_path) == 0

    def test_update_on_a_real_cube_still_rebuilds(self, tmp_path):
        # Guard against the harness passing for the wrong reason.
        pytest.importorskip("py3Dmol")
        from quantui.orbital_visualization import _build_iso_viewer

        cube = (
            "c\nc\n1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n2 0.0 1.0 0.0\n2 0.0 0.0 1.0\n"
            "1 0.0 0.0 0.0 0.0\n0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8\n"
        )
        html = _build_iso_viewer(
            cube, data_format="cube", scene_key="k", with_surfaces=True
        )
        assert self._run(html, tmp_path) == 2  # + and - lobes
