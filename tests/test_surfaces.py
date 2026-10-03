"""Density, spin-density and ESP surfaces; UHF beta orbitals (M-SURFACES)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantui.orbital_visualization import (
    DEFAULT_SURFACE_ISOVALUE,
    density_matrices,
    esp_legend_html,
    esp_surface_range,
)


class TestDensityMatrices:
    def test_restricted_closed_shell(self):
        c = np.eye(2)
        dm_a, dm_b = density_matrices(c, [2.0, 0.0])
        np.testing.assert_allclose(dm_a, [[1, 0], [0, 0]])
        np.testing.assert_allclose(dm_b, dm_a)

    def test_restricted_open_shell(self):
        # ROHF: the singly occupied orbital is alpha only.
        c = np.eye(3)
        dm_a, dm_b = density_matrices(c, [2.0, 1.0, 0.0])
        assert np.trace(dm_a) == 2 and np.trace(dm_b) == 1

    def test_unrestricted(self):
        c = np.stack([np.eye(2), np.eye(2)])
        dm_a, dm_b = density_matrices(c, [[1.0, 1.0], [1.0, 0.0]])
        assert np.trace(dm_a) == 2 and np.trace(dm_b) == 1

    def test_unrestricted_needs_per_spin_occupations(self):
        with pytest.raises(ValueError):
            density_matrices(np.stack([np.eye(2), np.eye(2)]), [2.0, 0.0])


class TestEspRange:
    def test_uses_points_near_the_isovalue(self):
        rho = np.array([0.5, 0.002, 0.0021, 0.0019, 1e-6])
        pot = np.array([9.0, -0.04, 0.03, 0.02, 5.0])
        rng = esp_surface_range(rho, pot, 0.002)
        # Only the three near-surface points count; the nucleus-side 9.0
        # and the far-field 5.0 do not.
        assert 0.02 < rng <= 0.04

    def test_no_surface_points(self):
        assert esp_surface_range(np.array([1.0]), np.array([0.1]), 0.002) is None

    def test_legend_mentions_both_units(self):
        html = esp_legend_html(0.05)
        assert "0.050 a.u." in html and "kcal/mol" in html
        assert "#ff0000" in html and "#0000ff" in html

    def test_density_defaults_are_the_molecular_surface(self):
        assert DEFAULT_SURFACE_ISOVALUE["esp"] == 0.002
        assert DEFAULT_SURFACE_ISOVALUE["orbital"] == 0.02


def _water_scf(basis="sto-3g"):
    from pyscf import gto, scf

    atoms = [
        ("O", [0.0, 0.0, 0.1173]),
        ("H", [0.0, 0.7572, -0.4692]),
        ("H", [0.0, -0.7572, -0.4692]),
    ]
    mol = gto.M(atom=atoms, basis=basis, verbose=0)
    return atoms, scf.RHF(mol).run()


@pytest.mark.slow
class TestSurfaceCubes:
    @pytest.fixture(autouse=True)
    def _pyscf(self):
        pytest.importorskip("pyscf")

    def test_density_cube_integrates_to_the_electron_count(self, tmp_path):
        from pyscf.tools import cubegen

        from quantui.orbital_visualization import generate_surface_cubes

        atoms, mf = _water_scf()
        out = generate_surface_cubes(
            "density",
            atoms,
            "sto-3g",
            mf.mo_coeff,
            mf.mo_occ,
            tmp_path / "rho.cube",
            nx=60,
            ny=60,
            nz=60,
        )
        cube = cubegen.Cube(mf.mol, 60, 60, 60, margin=5.0)
        rho = cube.read(str(out["cube"]))
        n_elec = float(rho.sum()) * abs(np.linalg.det(cube.box)) / cube.get_ngrids()
        assert n_elec == pytest.approx(10.0, rel=0.03)
        assert "electron density" in out["cube"].read_text().splitlines()[0]
        assert out["esp_cube"] is None

    def test_spin_density_of_a_radical_integrates_to_one(self, tmp_path):
        from pyscf import gto, scf

        from quantui.orbital_visualization import generate_surface_cubes

        atoms = [("O", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.97])]
        mol = gto.M(atom=atoms, basis="sto-3g", spin=1, verbose=0)
        mf = scf.UHF(mol).run()
        out = generate_surface_cubes(
            "spin",
            atoms,
            "sto-3g",
            mf.mo_coeff,
            mf.mo_occ,
            tmp_path / "spin.cube",
            spin=1,
            nx=60,
            ny=60,
            nz=60,
        )
        from quantui.orbital_visualization import parse_cube_file

        cube = parse_cube_file(out["cube"])
        voxel = abs(float(np.linalg.det(np.asarray(cube["axes"], dtype=float))))
        n_unpaired = float(np.asarray(cube["data"]).sum()) * voxel
        assert n_unpaired == pytest.approx(1.0, rel=0.05)

    def test_esp_map_of_water(self, tmp_path):
        from pyscf.tools import cubegen

        from quantui.orbital_visualization import generate_surface_cubes

        atoms, mf = _water_scf()
        out = generate_surface_cubes(
            "esp",
            atoms,
            "sto-3g",
            mf.mo_coeff,
            mf.mo_occ,
            tmp_path / "rho.cube",
            nx=40,
            ny=40,
            nz=40,
        )
        assert out["esp_cube"].exists()
        assert 0.005 < out["esp_range"] < 0.5
        cube = cubegen.Cube(mf.mol, 40, 40, 40, margin=5.0)
        rho = cube.read(str(out["cube"])).ravel()
        pot = cube.read(str(out["esp_cube"])).ravel()
        coords = cube.get_coords()
        near = (rho > 0.0014) & (rho < 0.0028)
        # On the molecular surface: negative beyond the O (lone pairs, the
        # side away from the H atoms), positive beyond the H atoms.
        o_side = near & (coords[:, 2] > 1.5)
        h_side = near & (np.abs(coords[:, 1]) > 2.0) & (coords[:, 2] < -1.0)
        assert pot[o_side].mean() < 0 < pot[h_side].mean()

    def test_viewer_html_for_esp_mode(self, tmp_path):
        from quantui.orbital_visualization import (
            generate_surface_cubes,
            render_surface_py3dmol,
        )

        atoms, mf = _water_scf()
        out = generate_surface_cubes(
            "esp",
            atoms,
            "sto-3g",
            mf.mo_coeff,
            mf.mo_occ,
            tmp_path / "rho.cube",
            nx=20,
            ny=20,
            nz=20,
        )
        html = render_surface_py3dmol(
            out["cube"], mode="esp", esp_cube_path=out["esp_cube"], esp_range=0.04
        )
        assert 'var MODE="esp"' in html
        assert "volscheme" in html and '"rwb"' in html
        assert "electrostatic potential cube" in html  # the potential is embedded
        density_html = render_surface_py3dmol(out["cube"], mode="density")
        assert 'var MODE="density"' in density_html
        assert "VOL=null" in density_html


class TestAppSurfaceDispatch:
    def _app(self, tmp_path, *, unrestricted=False):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        app._last_result_dir = tmp_path
        app._last_orb_info = MagicMock()
        app._last_orb_info.n_occupied = 1
        app._last_orb_info.mo_energies_ev = [-10.0, 2.0]
        app._last_orb_info.formula = "H2"
        if unrestricted:
            app._last_orb_mo_coeff = np.stack([np.eye(2), 2 * np.eye(2)])
            app._last_orb_mo_occ = np.array([[1.0, 1.0], [1.0, 0.0]])
        else:
            app._last_orb_mo_coeff = np.eye(2)
            app._last_orb_mo_occ = np.array([2.0, 0.0])
        app._last_orb_mol_atom = [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.74]]]
        app._last_orb_mol_basis = "sto-3g"
        app._resolve_backend = lambda task: "py3dmol"
        return app

    def test_surface_change_resets_isovalue_and_controls(self, tmp_path):
        app = self._app(tmp_path)
        app._iso_surface_dd.value = "esp"
        assert app._iso_isovalue_slider.value == pytest.approx(0.002, rel=0.05)
        assert app._iso_esp_range_slider.layout.display == ""
        assert app._orb_toggle.layout.display == "none"
        app._iso_surface_dd.value = "orbital"
        assert app._iso_isovalue_slider.value == pytest.approx(0.02, rel=0.05)
        assert app._orb_toggle.layout.display == ""

    def test_esp_mode_generates_two_cubes_and_sets_range(self, tmp_path):
        app = self._app(tmp_path)
        app._iso_surface_dd.value = "esp"

        def _fake(kind, _atom, _basis, _c, _o, out_path, *, esp_output_path, **_kw):
            Path(out_path).write_text("cube")
            Path(esp_output_path).write_text("cube")
            return {"cube": out_path, "esp_cube": esp_output_path, "esp_range": 0.031}

        with (
            patch(
                "quantui.orbital_visualization.generate_surface_cubes",
                side_effect=_fake,
            ) as gen,
            patch(
                "quantui.orbital_visualization.render_surface_py3dmol",
                return_value="<div>esp</div>",
            ) as render,
        ):
            app._render_orbital_isosurface("ESP map")

        assert gen.call_args.args[0] == "esp"
        assert render.call_args.kwargs["mode"] == "esp"
        assert app._last_cube_kind == "esp"
        assert app._last_esp_cube_path.exists()
        assert app._iso_esp_range_slider.value == pytest.approx(0.031, rel=0.05)
        assert "kcal/mol" in app._iso_esp_legend.value

    def test_spin_density_refused_for_closed_shell(self, tmp_path):
        app = self._app(tmp_path)
        app._iso_surface_dd.value = "spin"
        with patch("quantui.orbital_visualization.generate_surface_cubes") as gen:
            app._render_orbital_isosurface("Spin density")
        gen.assert_not_called()

    def test_beta_orbital_uses_beta_coefficients(self, tmp_path):
        app = self._app(tmp_path, unrestricted=True)
        app._orb_spin_toggle.value = "beta"
        seen: dict = {}

        def _fake(_atom, _basis, coeff, idx, out_path, **_kw):
            seen["coeff"] = np.asarray(coeff)
            seen["idx"] = idx
            Path(out_path).write_text("cube")
            return out_path

        with (
            patch(
                "quantui.orbital_visualization.generate_cube_from_arrays",
                side_effect=_fake,
            ),
            patch(
                "quantui.orbital_visualization.render_orbital_isosurface_py3dmol",
                return_value="<div>beta</div>",
            ),
        ):
            app._render_orbital_isosurface("HOMO")

        # Beta channel: 2*I here; beta HOMO is orbital 0 (one beta electron).
        np.testing.assert_allclose(seen["coeff"], 2 * np.eye(2))
        assert seen["idx"] == 0
        assert "β" in app._last_cube_orbital
        assert app._last_cube_path.name.endswith("_beta.cube")
