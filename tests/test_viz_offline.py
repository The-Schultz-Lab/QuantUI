"""Offline-safety tests for py3Dmol rendering.

py3Dmol fetches 3Dmol.js from the jsDelivr CDN by default, which blanks every
3D view with no network (offline classroom) or under a CSP. ``viz_assets``
vendors the bundle and loads it from a ``data:`` URI instead. These tests are
platform-independent (no PySCF) but DO require py3Dmol, which is a hard runtime
dependency. The contract they lock in:

- the vendored bundle ships and is non-trivial;
- the one-time page bootstrap embeds the bundle (no CDN URL);
- every viewer built via ``make_view`` is CDN-free and reuses the bootstrap;
- the orbital isosurface (M-ORBVIZ) renderer is CDN-free;
- exported animation HTML is self-contained (inline bootstrap).

See reflections/01-voila-rendering-and-display.md Rule 1 (CDN ban) and
reflections/06-visualization-backend-policy.md.
"""

import textwrap

import pytest

from quantui import viz_assets

# py3Dmol is a hard dependency; skip cleanly if a stripped env lacks it.
py3Dmol = pytest.importorskip("py3Dmol")

_CDN = "cdn.jsdelivr.net"


def test_bundle_present_and_nontrivial():
    raw = viz_assets._JS_PATH.read_bytes()
    # The real 3Dmol-min.js 2.5.4 build is ~0.5 MB; guard against an empty or
    # placeholder file slipping into the wheel.
    assert len(raw) > 100_000, f"vendored 3Dmol.js looks too small ({len(raw)} B)"


def test_data_uri_is_base64_javascript():
    uri = viz_assets._js_data_uri()
    assert uri.startswith("data:text/javascript;base64,")
    assert _CDN not in uri


def test_make_view_loads_vendored_js_not_cdn():
    """A make_view viewer loads 3Dmol from the local data: URI, never the CDN —
    and crucially carries NO startup-time page bootstrap (the loader runs
    per-view, after page load, exactly like py3Dmol's native path)."""
    view = viz_assets.make_view(width=200, height=200)
    view.addModel("3\nH2O\nO 0 0 0\nH 0 0 1\nH 0 1 0", "xyz")
    view.setStyle({"stick": {}})
    view.zoomTo()
    html = view._make_html()
    assert _CDN not in html, "make_view leaked the 3Dmol CDN URL into a view"
    assert "data:text/javascript;base64," in html
    # py3Dmol's own per-view loader (the proven mechanism), local source.
    assert "loadScriptAsync" in html


def test_bare_py3dmol_view_still_uses_cdn_regression():
    """Documents WHY make_view exists: a raw py3Dmol.view DOES embed the CDN.

    If a future py3Dmol release stops defaulting to a CDN this test will fail
    loudly, prompting us to revisit whether the local-js override is still
    load-bearing."""
    view = py3Dmol.view(width=100, height=100)
    view.addModel("1\nH\nH 0 0 0", "xyz")
    assert _CDN in view._make_html()


def test_standalone_html_self_contained_via_view():
    """Views are self-contained (carry the vendored loader), so standalone_html
    is a pass-through and the exported HTML plays offline."""
    view = viz_assets.make_view(width=200, height=200)
    view.addModel("1\nH\nH 0 0 0", "xyz")
    bundled = viz_assets.standalone_html(view._make_html())
    assert "data:text/javascript;base64," in bundled
    assert _CDN not in bundled


def test_orbital_isosurface_renderer_is_cdn_free(tmp_path):
    from quantui import orbital_visualization as ov

    cube = tmp_path / "homo.cube"
    cube.write_text(
        textwrap.dedent(
            """\
            Comment line 1
            Comment line 2
             1  0.000000  0.000000  0.000000
             2  0.500000  0.000000  0.000000
             2  0.000000  0.500000  0.000000
             2  0.000000  0.000000  0.500000
             1  0.000000  0.000000  0.000000  0.000000
             0.1  0.2  0.3  0.4
             0.5  0.6  0.7  0.8
            """
        )
    )
    html = ov.render_orbital_isosurface_py3dmol(cube, isovalue=0.02)
    assert _CDN not in html
    # Two lobes (M-ORBVIZ contract) + loads vendored 3Dmol offline.
    assert html.count("addVolumetricData") == 2
    assert "data:text/javascript;base64," in html


def test_trajectory_viewer_is_single_viewer_stepper():
    # The trajectory viewer loads ALL steps into ONE viewer (addModelsAsFrames)
    # and navigates client-side via setFrame, so the camera persists across
    # steps (vs the old per-frame rebuild). Offline-safe + energy-annotated.
    from quantui.app_visualization import build_trajectory_viewer_html

    xyzblocks = [
        "2\nH2\nH 0 0 0\nH 0 0 0.74",
        "2\nH2\nH 0 0 0\nH 0 0 0.77",
        "2\nH2\nH 0 0 0\nH 0 0 0.80",
    ]
    html = build_trajectory_viewer_html(
        xyzblocks,
        formula="H2",
        energies=[-1.10, -1.13, -1.12],
        rel_e=[0.0, -18.8, -12.5],
        bgcolor="white",
    )
    assert _CDN not in html  # offline-safe
    assert "addModelsAsFrames" in html  # one viewer, all frames preloaded
    assert "setFrame" in html  # client-side navigation
    assert 'type="range"' in html and 'max="2"' in html  # scrub slider spans frames
    assert "Final geometry" in html  # start <-> final A/B flip
    assert "EABS" in html and "EREL" in html  # per-step energy label data


def test_trajectory_viewer_single_frame_is_static():
    from quantui.app_visualization import build_trajectory_viewer_html

    html = build_trajectory_viewer_html(["1\nH\nH 0 0 0"])
    assert _CDN not in html
    assert "setFrame" not in html  # nothing to step through


def test_vib_viewer_is_single_viewer_all_modes():
    # All vibrational modes share ONE viewer; modes switch client-side via
    # window.__quantuiVibSetMode on the same instance, so the camera persists
    # across modes (vs the old per-mode rebuild). Offline-safe.
    from types import SimpleNamespace

    from quantui.app_visualization import build_vib_viewer_html
    from quantui.molecule import Molecule

    mol = Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    displ = [
        [[0, 0, 0.1], [0, 0, -0.4], [0, 0, -0.4]],
        [[0.1, 0, 0], [-0.4, 0.2, 0], [-0.4, -0.2, 0]],
        [[0, 0.1, 0], [0, -0.4, 0], [0, -0.4, 0]],
    ]
    freq = SimpleNamespace(
        displacements=displ, frequencies_cm1=[1600.0, 3700.0, 3800.0]
    )
    html = build_vib_viewer_html(mol, freq, [1, 2, 3], 1, fps=10)

    assert _CDN not in html  # offline-safe
    assert "window.__quantuiVibSetMode" in html  # client-side mode switch fn
    assert "removeAllModels" in html  # swaps frames on the SAME viewer instance
    assert "addModelsAsFrames" in html
    # stopAnimate before each animate() prevents stacked animation loops — the
    # glitchy / too-fast playback after repeated setMode calls.
    assert "stopAnimate" in html
    # Live framerate change without a rebuild (camera preserved).
    assert "window.__quantuiVibSetFps" in html
    # All three modes' displacement vectors are embedded for client-side frames.
    assert '"1":' in html and '"2":' in html and '"3":' in html


def test_vib_viewer_requires_displacements():
    from types import SimpleNamespace

    import pytest

    from quantui.app_visualization import build_vib_viewer_html
    from quantui.molecule import Molecule

    mol = Molecule(atoms=["H", "H"], coordinates=[[0, 0, 0], [0, 0, 0.74]])
    freq = SimpleNamespace(displacements=None, frequencies_cm1=[4400.0])
    with pytest.raises(ValueError):
        build_vib_viewer_html(mol, freq, [1], 1)
