"""
Tests for quantui.orbital_visualization

Covers OrbitalInfo construction, load helpers, the matplotlib energy-level
diagram, the summary HTML, cube-file parsing, and orbital isosurface plotting.
Cube-file functions are tested with a synthetic minimal cube string.
"""

import textwrap

import numpy as np
import pytest

from quantui.orbital_visualization import (
    HARTREE_TO_EV,
    load_orbital_info,
    orbital_info_from_arrays,
    orbital_summary_html,
    parse_cube_file,
    plot_cube_isosurface,
    plot_orbital_diagram,
    render_orbital_isosurface_py3dmol,
)

try:
    import pyscf  # noqa: F401

    _PYSCF_AVAILABLE = True
except ImportError:
    _PYSCF_AVAILABLE = False

_pyscf_only = pytest.mark.skipif(not _PYSCF_AVAILABLE, reason="PySCF not available")

try:
    import py3Dmol  # noqa: F401

    _PY3DMOL_AVAILABLE = True
except ImportError:
    _PY3DMOL_AVAILABLE = False

_py3dmol_only = pytest.mark.skipif(
    not _PY3DMOL_AVAILABLE, reason="py3Dmol not available"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_mo_data(tmp_path):
    """Create a minimal results.npz with 4 MOs (2 occ, 2 virt)."""
    mo_energy = np.array([-0.6, -0.3, 0.2, 0.8])  # Hartrees
    mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
    np.savez(
        tmp_path / "results.npz",
        mo_energy=mo_energy,
        mo_occ=mo_occ,
        energy=-75.0,
        converged=True,
    )
    return tmp_path / "results.npz"


@pytest.fixture()
def uhf_mo_data(tmp_path):
    """Create results.npz with UHF-style (2, n_mo) arrays."""
    alpha = np.array([-0.7, -0.4, 0.1, 0.5])
    beta = np.array([-0.65, -0.35, 0.15, 0.55])
    mo_energy = np.stack([alpha, beta])  # shape (2, 4)
    mo_occ = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    np.savez(
        tmp_path / "results.npz",
        mo_energy=mo_energy,
        mo_occ=mo_occ,
        energy=-74.5,
        converged=True,
    )
    return tmp_path / "results.npz"


@pytest.fixture()
def minimal_cube_file(tmp_path):
    """Write a trivially small cube file (1 atom, 2x2x2 grid)."""
    content = textwrap.dedent(
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
    p = tmp_path / "test.cube"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# OrbitalInfo construction
# ---------------------------------------------------------------------------


class TestLoadOrbitalInfo:

    def test_basic_load(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data, formula="H2O")
        assert info.n_occupied == 2
        assert info.n_virtual == 2
        assert info.formula == "H2O"

    def test_homo_lumo_values(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data)
        expected_homo = -0.3 * HARTREE_TO_EV
        expected_lumo = 0.2 * HARTREE_TO_EV
        assert abs(info.homo_energy_ev - expected_homo) < 1e-6
        assert abs(info.lumo_energy_ev - expected_lumo) < 1e-6

    def test_gap_positive(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data)
        assert info.homo_lumo_gap_ev > 0

    def test_uhf_uses_alpha(self, uhf_mo_data):
        info = load_orbital_info(uhf_mo_data)
        # Alpha HOMO is at index 1 = -0.4 Ha
        expected_homo = -0.4 * HARTREE_TO_EV
        assert abs(info.homo_energy_ev - expected_homo) < 1e-6

    def test_formula_fallback_to_stem(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data)
        assert info.formula == "results"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_orbital_info(tmp_path / "nonexistent.npz")

    def test_external_mo_occ(self, tmp_path):
        """Pass mo_occ explicitly, overriding what's in the file."""
        mo_energy = np.array([-1.0, -0.5, 0.3, 1.0])
        np.savez(
            tmp_path / "results.npz", mo_energy=mo_energy, energy=-50.0, converged=True
        )
        info = load_orbital_info(
            tmp_path / "results.npz",
            mo_occ=np.array([2.0, 0.0, 0.0, 0.0]),
            formula="test",
        )
        assert info.n_occupied == 1


class TestOrbitalInfoFromArrays:

    def test_basic(self):
        info = orbital_info_from_arrays(
            mo_energy=np.array([-0.5, -0.2, 0.1, 0.6]),
            mo_occ=np.array([2.0, 2.0, 0.0, 0.0]),
            formula="CH4",
        )
        assert info.n_occupied == 2
        assert info.formula == "CH4"

    def test_gap_calculation(self):
        info = orbital_info_from_arrays(
            mo_energy=np.array([-0.5, 0.5]),
            mo_occ=np.array([2.0, 0.0]),
        )
        expected_gap = 1.0 * HARTREE_TO_EV
        assert abs(info.homo_lumo_gap_ev - expected_gap) < 1e-6

    def test_all_occupied_raises(self):
        with pytest.raises(ValueError, match="Cannot determine HOMO/LUMO"):
            orbital_info_from_arrays(
                mo_energy=np.array([-0.5, -0.2]),
                mo_occ=np.array([2.0, 2.0]),
            )

    def test_none_occupied_raises(self):
        with pytest.raises(ValueError, match="Cannot determine HOMO/LUMO"):
            orbital_info_from_arrays(
                mo_energy=np.array([0.5, 1.0]),
                mo_occ=np.array([0.0, 0.0]),
            )


# ---------------------------------------------------------------------------
# Matplotlib diagram
# ---------------------------------------------------------------------------


class TestPlotOrbitalDiagram:

    def test_returns_figure(self, simple_mo_data):
        import matplotlib

        matplotlib.use("Agg")
        info = load_orbital_info(simple_mo_data, formula="H2O")
        fig = plot_orbital_diagram(info)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_custom_title(self, simple_mo_data):
        import matplotlib

        matplotlib.use("Agg")
        info = load_orbital_info(simple_mo_data, formula="H2O")
        fig = plot_orbital_diagram(info, title="Custom Title")
        # Title should be set on the axes
        ax = fig.axes[0]
        assert ax.get_title() == "Custom Title"

    def test_max_orbitals_limits_lines(self, tmp_path):
        """With 20 MOs and max_orbitals=6, only ~6 levels are drawn."""
        import matplotlib

        matplotlib.use("Agg")
        mo_e = np.linspace(-2, 2, 20)
        mo_occ = np.zeros(20)
        mo_occ[:10] = 2.0
        np.savez(
            tmp_path / "results.npz",
            mo_energy=mo_e,
            mo_occ=mo_occ,
            energy=-100.0,
            converged=True,
        )
        info = load_orbital_info(tmp_path / "results.npz", formula="big")
        fig = plot_orbital_diagram(info, max_orbitals=6)
        # Should have plotted 6 lines (each is a Line2D object)
        ax = fig.axes[0]
        lines = [
            c
            for c in ax.get_children()
            if hasattr(c, "get_xdata") and len(c.get_xdata()) == 2
        ]
        assert len(lines) == 6


# ---------------------------------------------------------------------------
# Summary HTML
# ---------------------------------------------------------------------------


class TestOrbitalSummaryHtml:

    def test_contains_formula(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data, formula="H2O")
        html = orbital_summary_html(info)
        assert "H2O" in html

    def test_contains_occupied_count(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data, formula="H2O")
        html = orbital_summary_html(info)
        assert "Occupied MOs: 2" in html

    def test_contains_gap(self, simple_mo_data):
        info = load_orbital_info(simple_mo_data, formula="H2O")
        html = orbital_summary_html(info)
        assert "HOMO–LUMO gap" in html


# ---------------------------------------------------------------------------
# Cube-file parser
# ---------------------------------------------------------------------------


class TestParseCubeFile:

    def test_parse_basic(self, minimal_cube_file):
        cube = parse_cube_file(minimal_cube_file)
        assert cube["nx"] == 2
        assert cube["ny"] == 2
        assert cube["nz"] == 2
        assert cube["data"].shape == (2, 2, 2)

    def test_atom_count(self, minimal_cube_file):
        cube = parse_cube_file(minimal_cube_file)
        assert len(cube["atoms"]) == 1

    def test_origin(self, minimal_cube_file):
        cube = parse_cube_file(minimal_cube_file)
        np.testing.assert_allclose(cube["origin"], [0.0, 0.0, 0.0])

    def test_data_values(self, minimal_cube_file):
        cube = parse_cube_file(minimal_cube_file)
        flat = cube["data"].flatten()
        np.testing.assert_allclose(flat, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


# ---------------------------------------------------------------------------
# Cube-file isosurface (plotly) — M6.2 acceptance criteria
# ---------------------------------------------------------------------------


class TestPlotCubeIsosurface:

    def test_returns_figure(self, minimal_cube_file):
        import plotly.graph_objects as go

        fig = plot_cube_isosurface(minimal_cube_file)
        assert isinstance(fig, go.Figure)

    def test_has_single_isosurface_trace(self, minimal_cube_file):
        # Both lobes are drawn by one go.Isosurface trace (surface_count=2) to
        # halve the figure payload (BUG-ISO-OOM).
        fig = plot_cube_isosurface(minimal_cube_file)
        iso = [t for t in fig.data if t.type == "isosurface"]
        assert len(iso) == 1
        assert iso[0].surface.count == 2  # both lobes via one trace

    def test_large_grid_is_downsampled(self):
        # The cube is written at a fixed resolution regardless of molecule size;
        # a full 60^3 = 216k-point grid flattened into the figure could exhaust
        # browser memory. The render path must stride it under the cap.
        from unittest.mock import patch

        from quantui import orbital_visualization as ov

        n = 60
        fake_cube = {
            "nx": n,
            "ny": n,
            "nz": n,
            "data": np.zeros((n, n, n)),
            "origin": np.zeros(3),
            "axes": np.eye(3) * 0.3,
            "atoms": [],
        }
        with patch.object(ov, "parse_cube_file", return_value=fake_cube):
            fig = ov.plot_cube_isosurface("dummy.cube", isovalue=0.02)
        iso = [t for t in fig.data if t.type == "isosurface"][0]
        assert len(iso.x) <= ov._MAX_ISOSURFACE_POINTS
        assert len(iso.x) < n * n * n  # actually downsampled

    def test_custom_title(self, minimal_cube_file):
        fig = plot_cube_isosurface(minimal_cube_file, title="HOMO Isosurface")
        assert fig.layout.title.text == "HOMO Isosurface"

    def test_scene_has_axis_labels(self, minimal_cube_file):
        fig = plot_cube_isosurface(minimal_cube_file)
        assert "Bohr" in fig.layout.scene.xaxis.title.text

    def test_show_molecule_adds_overlay_traces(self, minimal_cube_file):
        # The isosurface is one trace now; show_molecule adds ≥1 scatter3d
        # overlay (atoms, plus bonds when there are ≥2 atoms).
        fig = plot_cube_isosurface(minimal_cube_file, show_molecule=True)
        scatter = [t for t in fig.data if t.type == "scatter3d"]
        assert len(scatter) >= 1

    def test_show_grid_false_hides_scene_grid(self, minimal_cube_file):
        fig = plot_cube_isosurface(minimal_cube_file, show_grid=False)
        assert fig.layout.scene.xaxis.showgrid is False
        assert fig.layout.scene.yaxis.showgrid is False
        assert fig.layout.scene.zaxis.showgrid is False

    def test_default_view_window_is_larger(self, minimal_cube_file):
        fig = plot_cube_isosurface(minimal_cube_file)
        assert fig.layout.width == 760
        assert fig.layout.height == 620

    def test_paper_background_transparent(self, minimal_cube_file):
        fig = plot_cube_isosurface(minimal_cube_file)
        assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"

    def test_title_color_override(self, minimal_cube_file):
        fig = plot_cube_isosurface(
            minimal_cube_file,
            title="LUMO Isosurface",
            title_color="#123456",
        )
        assert fig.layout.title.font.color == "#123456"


@_py3dmol_only
class TestRenderOrbitalIsosurfacePy3Dmol:
    """py3Dmol orbital isosurface renderer (M-ORBVIZ).

    Renders both ± lobes via in-browser cube isosurfacing and returns
    self-contained HTML through py3Dmol's _make_html — the same offline-safe
    embedding the molecule viewer uses (no CDN).
    """

    def test_returns_html_string(self, minimal_cube_file):
        html = render_orbital_isosurface_py3dmol(minimal_cube_file)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_adds_two_volumetric_lobes(self, minimal_cube_file):
        # Both the positive and negative isosurface calls must be embedded.
        html = render_orbital_isosurface_py3dmol(minimal_cube_file, isovalue=0.02)
        assert html.count("addVolumetricData") == 2
        assert "0.02" in html  # +isovalue
        assert "-0.02" in html  # -isovalue

    def test_embeds_offline_no_cdn(self, minimal_cube_file):
        # py3Dmol's _make_html must not pull 3Dmol.js from a remote CDN —
        # offline Voilà would silently fail to render otherwise.
        html = render_orbital_isosurface_py3dmol(minimal_cube_file)
        assert 'src="http' not in html
        assert "src='http" not in html

    def test_full_resolution_no_downsample(self, minimal_cube_file):
        # py3Dmol surfaces the cube client-side, so the full cube text is
        # handed over verbatim (no Python-side stride/downsample).
        html = render_orbital_isosurface_py3dmol(minimal_cube_file)
        assert "addModel" in html


# ---------------------------------------------------------------------------
# generate_cube_from_arrays — M6.2 acceptance criteria
# ---------------------------------------------------------------------------


class TestGenerateCubeFromArrays:
    def test_raises_importerror_without_pyscf(self, tmp_path, monkeypatch):
        import sys

        from quantui.orbital_visualization import generate_cube_from_arrays

        monkeypatch.setitem(sys.modules, "pyscf", None)
        monkeypatch.setitem(sys.modules, "pyscf.tools", None)
        with pytest.raises(ImportError):
            generate_cube_from_arrays(
                [], "sto-3g", np.zeros((2, 2)), 0, tmp_path / "x.cube"
            )

    @_pyscf_only
    def test_returns_path(self, tmp_path):
        from pyscf import gto, scf

        from quantui.orbital_visualization import generate_cube_from_arrays

        mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        out_path = tmp_path / "homo.cube"
        result = generate_cube_from_arrays(
            mol_atom=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.74]]],
            mol_basis="sto-3g",
            mo_coeff=mf.mo_coeff,
            orbital_index=int(np.where(mf.mo_occ > 0)[0][-1]),
            output_path=out_path,
            nx=10,
            ny=10,
            nz=10,
        )
        assert result == out_path
        assert out_path.exists()

    @_pyscf_only
    def test_cube_file_parseable(self, tmp_path):
        from pyscf import gto, scf

        from quantui.orbital_visualization import generate_cube_from_arrays

        mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        out_path = tmp_path / "test.cube"
        generate_cube_from_arrays(
            mol_atom=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.74]]],
            mol_basis="sto-3g",
            mo_coeff=mf.mo_coeff,
            orbital_index=0,
            output_path=out_path,
            nx=10,
            ny=10,
            nz=10,
        )
        cube = parse_cube_file(out_path)
        assert cube["data"].shape == (10, 10, 10)

    @_pyscf_only
    def test_uhf_mo_coeff_uses_alpha_spin(self, tmp_path):
        from pyscf import gto, scf

        from quantui.orbital_visualization import generate_cube_from_arrays

        mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, charge=0, verbose=0)
        mf = scf.UHF(mol)
        mf.verbose = 0
        mf.kernel()
        out_path = tmp_path / "uhf.cube"
        result = generate_cube_from_arrays(
            mol_atom=[["H", [0.0, 0.0, 0.0]]],
            mol_basis="sto-3g",
            mo_coeff=mf.mo_coeff,  # shape (2, n_ao, n_mo)
            orbital_index=0,
            output_path=out_path,
            nx=8,
            ny=8,
            nz=8,
            spin=1,
        )
        assert result.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
