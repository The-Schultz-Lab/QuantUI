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


def test_bootstrap_embeds_bundle_not_cdn():
    html = viz_assets.offline_bootstrap_html()
    assert "$3Dmolpromise" in html
    assert "data:text/javascript;base64," in html
    # The whole point: the bootstrap must NOT reach the network.
    assert _CDN not in html
    assert "loadScriptAsync" in html  # mirrors py3Dmol's own loader


def test_make_view_is_cdn_free_and_uses_sentinel():
    view = viz_assets.make_view(width=200, height=200)
    view.addModel("3\nH2O\nO 0 0 0\nH 0 0 1\nH 0 1 0", "xyz")
    view.setStyle({"stick": {}})
    view.zoomTo()
    html = view._make_html()
    assert _CDN not in html, "make_view leaked the 3Dmol CDN URL into a view"
    assert viz_assets._INAPP_SENTINEL in html
    # The view guards on $3Dmolpromise, so when the bootstrap has run it never
    # loads the sentinel — it just reuses the page-global promise.
    assert "if(typeof $3Dmolpromise === 'undefined')" in html


def test_bare_py3dmol_view_still_uses_cdn_regression():
    """Documents WHY make_view exists: a raw py3Dmol.view DOES embed the CDN.

    If a future py3Dmol release stops defaulting to a CDN this test will fail
    loudly, prompting us to revisit whether the offline bootstrap is still
    needed (it would then be belt-and-suspenders, not load-bearing)."""
    view = py3Dmol.view(width=100, height=100)
    view.addModel("1\nH\nH 0 0 0", "xyz")
    assert _CDN in view._make_html()


def test_standalone_html_is_self_contained():
    view = viz_assets.make_view(width=200, height=200)
    view.addModel("1\nH\nH 0 0 0", "xyz")
    bundled = viz_assets.standalone_html(view._make_html())
    # Inline bootstrap travels with the file so it plays offline standalone.
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
    # Two lobes (M-ORBVIZ contract) + the offline sentinel.
    assert html.count("addVolumetricData") == 2
    assert viz_assets._INAPP_SENTINEL in html
