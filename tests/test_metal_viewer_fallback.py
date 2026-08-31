"""Viewer never hard-errors on a metal complex (M-METAL MET.3).

PlotlyMol runs RDKit valence perception, which raises "Atom N has no valences
defined" on a transition metal. The backend router must fall back to py3Dmol
(which renders straight from coordinates) instead of crashing, and the HTML
renderer must show the structure rather than a red failure box. Organic
molecules must still render through PlotlyMol unchanged.
"""

from __future__ import annotations

import pytest

from quantui.molecule import Molecule


def _mol(entry_id: str) -> Molecule:
    from quantui import molecule_library as ml

    e = next(x for x in ml.iter_entries() if x["id"] == entry_id)
    return Molecule(
        atoms=e["atoms"],
        coordinates=e["coordinates"],
        charge=e["charge"],
        multiplicity=e["multiplicity"],
    )


def _water() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )


@pytest.fixture(autouse=True)
def _need_both_backends():
    import quantui.visualization_py3dmol as viz

    if not (viz.PLOTLYMOL_AVAILABLE and viz.PY3DMOL_AVAILABLE):
        pytest.skip("both plotlymol and py3dmol backends required")


class TestMetalViewerFallback:
    def test_plotlymol_backend_falls_back_for_metal(self):
        from quantui.visualization_py3dmol import visualize_molecule

        # Would raise ValueError without the fallback; must return a py3Dmol view
        # (identified by its _make_html method), not a plotly Figure.
        view = visualize_molecule(_mol("inorganic-cisplatin"), backend="plotlymol")
        assert callable(getattr(view, "_make_html", None))

    def test_render_html_shows_structure_not_error(self):
        from quantui.visualization_py3dmol import render_molecule_html

        html = render_molecule_html(
            _mol("inorganic-ferrocene"), backend="plotlymol", width=400, height=300
        )
        assert "Visualization failed" not in html
        assert len(html) > 1000  # a real viewer payload, not a stub error box

    def test_organic_still_uses_plotlymol(self):
        # Regression guard: the fallback must not divert organic molecules, which
        # PlotlyMol renders fine (returns a plotly Figure, no _make_html).
        from quantui.visualization_py3dmol import visualize_molecule

        fig = visualize_molecule(_water(), backend="plotlymol")
        assert getattr(fig, "_make_html", None) is None

    def test_reraises_when_no_py3dmol_fallback(self, monkeypatch):
        import quantui.visualization_py3dmol as viz

        monkeypatch.setattr(viz, "PY3DMOL_AVAILABLE", False)
        # RDKit's valence perception raises ValueError on the metal; with no
        # py3Dmol to fall back to, that must propagate rather than be swallowed.
        with pytest.raises(ValueError):
            viz.visualize_molecule(_mol("inorganic-cisplatin"), backend="plotlymol")


class TestCoordinationBonds:
    """M-METAL MET.6: the py3Dmol viewer draws dashed metal↔donor bonds so a
    coordination metal is never a lone dot."""

    def test_metal_html_has_dashed_cylinders(self):
        pytest.importorskip("py3Dmol")
        from quantui.visualization_py3dmol import render_molecule_html

        html = render_molecule_html(
            _mol("inorganic-cisplatin"), backend="py3dmol", width=300, height=250
        )
        assert "addCylinder" in html  # coordination bonds drawn
        assert "dashed" in html

    def test_ferrocene_hides_default_metal_sticks(self):
        pytest.importorskip("py3Dmol")
        from quantui.visualization_py3dmol import render_molecule_html

        html = render_molecule_html(
            _mol("inorganic-ferrocene"), backend="py3dmol", width=300, height=250
        )
        assert '"hidden": true' in html
        assert html.count("addCylinder") == 10

    def test_organic_html_has_no_cylinders(self):
        pytest.importorskip("py3Dmol")
        from quantui.visualization_py3dmol import render_molecule_html

        html = render_molecule_html(_water(), backend="py3dmol", width=300, height=250)
        assert "addCylinder" not in html
