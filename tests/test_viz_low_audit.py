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


# ── VIZ-9: Cancel must not double-decrement the activity counter ──────────


class _CountingApp:
    """Just enough of QuantUIApp for on_iso_generate / on_iso_cancel."""

    def __init__(self):
        from unittest.mock import Mock

        self.count = 0
        self._iso_render_token = 0
        self._orb_toggle = Mock(value="HOMO")
        self._iso_js_bridge = None
        self._render_orbital_isosurface = Mock()

    def _activity_begin(self, message="", kind="ui"):
        self.count += 1

    def _activity_end(self, kind="ui"):
        self.count = max(0, self.count - 1)

    def _queue_main_thread_callback(self, fn, *args):
        fn(*args)


class TestIsoCancelBalancesActivity:
    def test_cancel_then_late_completion_ends_activity_once(self, monkeypatch):
        from unittest.mock import Mock

        from quantui import app_visualization as av

        started = []
        monkeypatch.setattr(av, "_start_daemon", lambda fn: started.append(fn) or True)
        app = _CountingApp()
        app._activity_begin("a calculation is running", kind="compute")

        av.on_iso_generate(app, Mock())
        assert app.count == 2
        av.on_iso_cancel(app, None)
        assert app.count == 1
        started[0]()  # the abandoned cubegen worker finishes later
        assert app.count == 1, "toolbar would show Idle while the calc runs"

    def test_normal_completion_still_ends_activity(self, monkeypatch):
        from unittest.mock import Mock

        from quantui import app_visualization as av

        started = []
        monkeypatch.setattr(av, "_start_daemon", lambda fn: started.append(fn) or True)
        app = _CountingApp()
        av.on_iso_generate(app, Mock())
        started[0]()
        assert app.count == 0


# ── VIZ-10: placeholder camera carries into the first isosurface ──────────


def _water_molecule():
    from quantui.molecule import Molecule

    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


def _cube_text_for(molecule, tmp_path) -> str:
    gto = pytest.importorskip("pyscf.gto")
    from pyscf import scf
    from pyscf.tools import cubegen

    mol = gto.M(
        atom=[(s, c) for s, c in zip(molecule.atoms, molecule.coordinates)],
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    path = tmp_path / "mo.cube"
    cubegen.orbital(mol, str(path), mf.mo_coeff[:, 0], nx=4, ny=4, nz=4)
    return path.read_text()


class TestSceneKeysMatchAcrossPlaceholderAndCube:
    def test_placeholder_and_cube_of_the_same_molecule_share_a_key(self, tmp_path):
        from quantui.orbital_visualization import _scene_key, _scene_key_from_xyz

        water = _water_molecule()
        xyz = f"{len(water.atoms)}\n{water.get_formula()}\n{water.to_xyz_string()}"
        assert _scene_key(_cube_text_for(water, tmp_path)) == _scene_key_from_xyz(xyz)

    def test_a_different_molecule_does_not_match(self, tmp_path):
        from quantui.orbital_visualization import _scene_key, _scene_key_from_xyz

        assert _scene_key(_cube_text_for(_water_molecule(), tmp_path)) != (
            _scene_key_from_xyz(_H2_XYZ)
        )

    def test_rendered_viewers_carry_the_same_scene(self, tmp_path):
        pytest.importorskip("py3Dmol")
        import re

        from quantui.orbital_visualization import (
            render_molecule_placeholder_py3dmol,
            render_orbital_isosurface_py3dmol,
        )

        water = _water_molecule()
        cube = tmp_path / "w.cube"
        cube.write_text(_cube_text_for(water, tmp_path))
        scene = re.compile(r"SCENE=(\"[0-9a-f]+\")")
        a = scene.search(render_molecule_placeholder_py3dmol(water)).group(1)
        b = scene.search(render_orbital_isosurface_py3dmol(cube)).group(1)
        assert a == b


# ── VIZ-11: a failed plot must not leave the previous figure on screen ────


def _html_sink_app(**attrs):
    from unittest.mock import Mock

    app = Mock()
    shown = {}
    app._set_html_output.side_effect = lambda out, html: shown.__setitem__(
        id(out), html
    )
    for k, v in attrs.items():
        setattr(app, k, v)
    return app, shown


class TestFailedPlotsReplaceTheStaleFigure:
    def test_ir_failure_shows_an_escaped_error(self, monkeypatch):
        from quantui import app_visualization as av
        from quantui import ir_plot

        def boom(*a, **k):
            raise ValueError("bad <data>")

        monkeypatch.setattr(ir_plot, "plot_ir_spectrum", boom)
        app, shown = _html_sink_app(_last_ir_freqs=[1.0], _last_ir_ints=[1.0])
        av.update_ir_figure(app, "stick", 10.0)
        html = shown[id(app._ir_fig)]
        assert "IR spectrum could not be plotted" in html
        assert "bad &lt;data&gt;" in html
        assert app._last_ir_fig is None

    def test_nmr_failure_shows_an_error(self, monkeypatch):
        from quantui import app_visualization as av
        from quantui import nmr_plot

        def boom(*a, **k):
            raise RuntimeError("nope")

        monkeypatch.setattr(nmr_plot, "plot_nmr_spectrum", boom)
        app, shown = _html_sink_app(
            _last_nmr_atom_symbols=["O", "H", "H"],
            _last_nmr_h_shifts=[(1, 1.0)],
        )
        av.update_nmr_figure(app, "¹H")
        assert "NMR spectrum could not be plotted" in shown[id(app._nmr_fig)]

    def test_pes_failure_shows_an_error(self):
        from types import SimpleNamespace

        from quantui import app_visualization as av

        app, shown = _html_sink_app()
        av.show_pes_scan_result(app, SimpleNamespace())  # no energies at all
        assert (
            "PES energy profile could not be plotted" in shown[id(app._pes_plot_html)]
        )


# ── VIZ-12: exception/path text is escaped before it reaches HTML ─────────

_EVIL = "<img src=x onerror=alert(1)>"


class TestErrorTextIsEscaped:
    def test_vib_error_message(self, monkeypatch):
        from unittest.mock import Mock

        from quantui import app_visualization as av

        seen = []
        monkeypatch.setattr(av, "_swap_vib_output", lambda app, h: seen.append(h))
        av._vib_err(Mock(), f"render failed: {_EVIL}")
        assert _EVIL not in seen[0]
        assert "&lt;img" in seen[0]

    def test_standalone_export_title(self):
        from quantui.viz_assets import standalone_html

        doc = standalone_html("<div></div>", title=f"Geo Opt: {_EVIL}")
        assert _EVIL not in doc
        assert "<div></div>" in doc  # the body itself is not escaped

    def test_orbital_summary_formula(self):
        from quantui.orbital_visualization import OrbitalInfo, orbital_summary_html

        info = OrbitalInfo(
            mo_energies_ev=np.array([-10.0, 2.0]),
            n_occupied=1,
            homo_energy_ev=-10.0,
            lumo_energy_ev=2.0,
            homo_lumo_gap_ev=12.0,
            formula=_EVIL,
        )
        assert _EVIL not in orbital_summary_html(info)

    def test_molecule_viewer_failure(self, monkeypatch):
        from quantui import visualization_py3dmol as vp

        def boom(*a, **k):
            raise RuntimeError(_EVIL)

        monkeypatch.setattr(vp, "is_visualization_available", lambda: True)
        monkeypatch.setattr(vp, "visualize_molecule", boom)
        html = vp.render_molecule_html(_h2_molecule(), show_info=False)
        assert "Visualization failed" in html
        assert _EVIL not in html

    def test_trajectory_viewer_failure(self, monkeypatch):
        # Inside show_opt_trajectory a local ``html`` string shadows the
        # module, so the message must not be built with html.escape there.
        from types import SimpleNamespace
        from unittest.mock import Mock

        import ipywidgets as widgets

        from quantui import app_visualization as av

        def boom(*a, **k):
            raise RuntimeError(_EVIL)

        monkeypatch.setattr(av, "build_trajectory_viewer_html", boom)
        app = Mock()
        app._traj_render_token = 0
        app._plotly_theme_colors.return_value = {"scene_bgcolor": "white"}
        h2 = _h2_molecule()
        result = SimpleNamespace(
            trajectory=[h2, h2], energies_hartree=[-1.0, -1.1], formula="H2"
        )
        av.show_opt_trajectory(app, result, layout_fn=widgets.Layout)
        outs = [
            c
            for c in app.traj_output.children
            if isinstance(c, widgets.Output) and c.outputs
        ]
        page = " ".join(o.outputs[0]["data"]["text/html"] for o in outs)
        assert "Trajectory viewer failed" in page
        assert _EVIL not in page
