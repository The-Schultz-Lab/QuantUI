"""M-SURFACES SURF.5: automated half of the surfaces test matrix.

Gaps left by ``test_surfaces.py`` / ``test_orbital_gallery.py``, all on real
PySCF data (RHF/UHF/ROHF, STO-3G):

* cube grids keep the requested (non-cubic) shape, and the density and ESP
  cubes of one surface share a grid;
* the density integrates to the electron count for an open-shell result, and
  the spin density integrates to N_α − N_β (1 for a doublet, 2 for a triplet,
  0 for a closed shell); an ROHF spin density is |SOMO|², never negative;
* ESP sign convention: positive at the nuclei, and positive over the whole
  molecular surface of a cation (H3O+);
* β orbitals of a real UHF result: the β HOMO/LUMO are counted from the β
  occupations and drawn from the β coefficients;
* orbital gallery: HOMO−2…LUMO+2 by default; clipping for a 2-orbital basis;
  separate α and β windows for UHF;
* History replay of a saved UHF result: the Isosurface panel comes back with
  both spin channels, the α/β toggle, a spin density and β orbitals.

The Voilà visual half of SURF.5 is a local manual check, not covered here.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from quantui.molecule import Molecule
from quantui.orbital_visualization import gallery_orbital_indices

try:
    import pyscf  # noqa: F401

    _HAS_PYSCF = sys.platform != "win32"
except ImportError:  # pragma: no cover - depends on the environment
    _HAS_PYSCF = False

needs_pyscf = pytest.mark.skipif(
    not _HAS_PYSCF, reason="PySCF not available or not supported on native Windows"
)

# Neutral OH radical (doublet): 9 electrons, N_α = 5, N_β = 4; STO-3G has 6 MOs.
OH_ATOMS = [("O", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.97])]
# Triplet methylene: 8 electrons, N_α = 5, N_β = 3.
CH2_ATOMS = [
    ("C", [0.0, 0.0, 0.1]),
    ("H", [0.0, 0.99, -0.5]),
    ("H", [0.0, -0.99, -0.5]),
]
WATER_ATOMS = [
    ("O", [0.0, 0.0, 0.1173]),
    ("H", [0.0, 0.7572, -0.4692]),
    ("H", [0.0, -0.7572, -0.4692]),
]
# Pyramidal hydronium, O–H 0.98 Å, H–O–H ≈ 111°.
H3O_ATOMS = [
    ("O", [0.0, 0.0, 0.0]),
    ("H", [0.933, 0.0, -0.3]),
    ("H", [-0.4665, 0.808, -0.3]),
    ("H", [-0.4665, -0.808, -0.3]),
]


def _scf(atoms, kind="RHF", charge=0, spin=0):
    from pyscf import gto, scf

    mol = gto.M(atom=atoms, basis="sto-3g", charge=charge, spin=spin, verbose=0)
    return getattr(scf, kind)(mol).run()


def _integral(cube_path) -> tuple[tuple, float]:
    """(grid shape, ∫ value dV) from a cube file's own voxel vectors."""
    from quantui.orbital_visualization import parse_cube_file

    cube = parse_cube_file(cube_path)
    voxel = abs(float(np.linalg.det(np.asarray(cube["axes"], dtype=float))))
    data = np.asarray(cube["data"])
    return data.shape, float(data.sum()) * voxel


def _session(mol: Molecule, method: str):
    from quantui.session_calc import run_in_session

    return run_in_session(mol, method, "STO-3G", progress_stream=io.StringIO())


def _oh_molecule() -> Molecule:
    return Molecule(["O", "H"], [[0, 0, 0], [0, 0, 0.97]], multiplicity=2)


# ---------------------------------------------------------------------------
# Cube generation helpers
# ---------------------------------------------------------------------------


@needs_pyscf
class TestSurfaceCubeGrids:
    def test_requested_grid_shape_is_kept_for_both_esp_cubes(self, tmp_path):
        from quantui.orbital_visualization import (
            generate_surface_cubes,
            parse_cube_file,
        )

        mf = _scf(WATER_ATOMS)
        out = generate_surface_cubes(
            "esp",
            WATER_ATOMS,
            "sto-3g",
            mf.mo_coeff,
            mf.mo_occ,
            tmp_path / "rho.cube",
            nx=14,
            ny=16,
            nz=18,
        )
        rho = parse_cube_file(out["cube"])
        pot = parse_cube_file(out["esp_cube"])
        for cube in (rho, pot):
            assert (cube["nx"], cube["ny"], cube["nz"]) == (14, 16, 18)
            assert np.asarray(cube["data"]).shape == (14, 16, 18)
            assert len(cube["atoms"]) == 3
        # The viewer samples the potential at the density surface's
        # vertices, so the two cubes must be the same grid.
        np.testing.assert_allclose(rho["origin"], pot["origin"])
        np.testing.assert_allclose(rho["axes"], pot["axes"])
        assert "electrostatic potential" in out["esp_cube"].read_text()[:300]

    def test_unrestricted_density_integrates_to_the_electron_count(self, tmp_path):
        from quantui.orbital_visualization import generate_surface_cubes

        mf = _scf(OH_ATOMS, "UHF", spin=1)
        out = generate_surface_cubes(
            "density",
            OH_ATOMS,
            "sto-3g",
            mf.mo_coeff,
            mf.mo_occ,
            tmp_path / "rho.cube",
            spin=1,
        )
        shape, n_elec = _integral(out["cube"])
        assert shape == (60, 60, 60)
        assert n_elec == pytest.approx(9.0, rel=0.03)  # 8 (O) + 1 (H)


@needs_pyscf
class TestSpinDensity:
    def test_triplet_integrates_to_two_unpaired_electrons(self, tmp_path):
        # N_α − N_β = 5 − 3 = 2 for triplet CH2: the spin density is not
        # hard-wired to a doublet.
        from quantui.orbital_visualization import generate_surface_cubes

        mf = _scf(CH2_ATOMS, "UHF", spin=2)
        n_a, n_b = (float(o.sum()) for o in mf.mo_occ)
        assert (n_a, n_b) == (5.0, 3.0)
        out = generate_surface_cubes(
            "spin",
            CH2_ATOMS,
            "sto-3g",
            mf.mo_coeff,
            mf.mo_occ,
            tmp_path / "spin.cube",
            spin=2,
        )
        _, n_unpaired = _integral(out["cube"])
        assert n_unpaired == pytest.approx(n_a - n_b, rel=0.03)
        assert "spin density" in out["cube"].read_text().splitlines()[0]

    def test_rohf_spin_density_is_the_somo_squared(self, tmp_path):
        # ROHF (one 1-D occupation vector, 2/1/0): α − β is exactly the
        # singly occupied orbital squared — one electron, nowhere negative
        # (no spin polarization, unlike UHF).
        from quantui.orbital_visualization import (
            generate_surface_cubes,
            parse_cube_file,
        )

        mf = _scf(OH_ATOMS, "ROHF", spin=1)
        assert np.asarray(mf.mo_occ).ndim == 1
        out = generate_surface_cubes(
            "spin",
            OH_ATOMS,
            "sto-3g",
            mf.mo_coeff,
            mf.mo_occ,
            tmp_path / "spin.cube",
            spin=1,
        )
        _, n_unpaired = _integral(out["cube"])
        assert n_unpaired == pytest.approx(1.0, rel=0.03)
        data = np.asarray(parse_cube_file(out["cube"])["data"])
        assert data.min() > -1e-10

    def test_closed_shell_spin_density_vanishes(self, tmp_path):
        from quantui.orbital_visualization import (
            generate_surface_cubes,
            parse_cube_file,
        )

        mf = _scf(WATER_ATOMS)
        out = generate_surface_cubes(
            "spin",
            WATER_ATOMS,
            "sto-3g",
            mf.mo_coeff,
            mf.mo_occ,
            tmp_path / "spin.cube",
            nx=20,
            ny=20,
            nz=20,
        )
        data = np.asarray(parse_cube_file(out["cube"])["data"])
        assert np.abs(data).max() < 1e-10


@needs_pyscf
class TestEspSignConvention:
    """Red (negative) where electrons dominate, blue (positive) where nuclei do."""

    @staticmethod
    def _esp(atoms, mf, tmp_path, charge=0):
        from pyscf import gto
        from pyscf.tools import cubegen

        from quantui.orbital_visualization import generate_surface_cubes

        out = generate_surface_cubes(
            "esp",
            atoms,
            "sto-3g",
            mf.mo_coeff,
            mf.mo_occ,
            tmp_path / "rho.cube",
            charge=charge,
            nx=40,
            ny=40,
            nz=40,
        )
        mol = gto.M(atom=atoms, basis="sto-3g", charge=charge, verbose=0)
        cube = cubegen.Cube(mol, 40, 40, 40, margin=5.0)
        rho = cube.read(str(out["cube"])).ravel()
        pot = cube.read(str(out["esp_cube"])).ravel()
        return mol, cube.get_coords(), rho, pot, out

    def test_potential_is_positive_next_to_every_nucleus(self, tmp_path):
        mf = _scf(WATER_ATOMS)
        mol, coords, _rho, pot, _ = self._esp(WATER_ATOMS, mf, tmp_path)
        for i, nucleus in enumerate(mol.atom_coords()):  # Bohr, like coords
            nearest = int(np.argmin(np.linalg.norm(coords - nucleus, axis=1)))
            assert pot[nearest] > 0, f"atom {i} ({mol.atom_symbol(i)})"

    def test_cation_surface_is_positive_everywhere(self, tmp_path):
        # A cation's molecular surface has no negative region: even above
        # the O lone pair of H3O+ the potential is net positive.
        mf = _scf(H3O_ATOMS, charge=1)
        _mol, _coords, rho, pot, out = self._esp(H3O_ATOMS, mf, tmp_path, charge=1)
        near = (rho > 0.0014) & (rho < 0.0028)
        assert near.sum() > 50
        assert pot[near].min() > 0
        assert out["esp_range"] > 0


# ---------------------------------------------------------------------------
# β orbitals of a real UHF result (Isosurface panel render path)
# ---------------------------------------------------------------------------


def _app_with(result, mol, tmp_path):
    from quantui.app import QuantUIApp
    from quantui.app_visualization import show_orbital_diagram

    app = QuantUIApp()
    app._set_molecule(mol, "test")
    show_orbital_diagram(app, result)
    app._last_result_dir = tmp_path
    app._resolve_backend = lambda task: "py3dmol"
    return app


@needs_pyscf
class TestUhfBetaOrbitals:
    @pytest.fixture
    def oh_app(self, tmp_path):
        r = _session(_oh_molecule(), "UHF")
        return r, _app_with(r, _oh_molecule(), tmp_path)

    def _render(self, app, label):
        from quantui import orbital_visualization as ov

        with (
            patch.object(
                ov, "generate_cube_from_arrays", wraps=ov.generate_cube_from_arrays
            ) as gen,
            patch.object(
                ov, "render_orbital_isosurface_py3dmol", return_value="<div>ok</div>"
            ),
        ):
            app._iso_resolution_dd.value = "coarse"
            app._render_orbital_isosurface(label)
        assert gen.call_count == 1
        return gen.call_args

    def test_spin_toggle_only_for_unrestricted(self, oh_app, tmp_path):
        _r, app = oh_app
        assert app._orb_spin_toggle.layout.display == ""
        water = Molecule(["O", "H", "H"], [list(c) for _, c in WATER_ATOMS])
        rhf_app = _app_with(_session(water, "RHF"), water, tmp_path)
        assert rhf_app._orb_spin_toggle.layout.display == "none"

    @pytest.mark.parametrize(
        "spin, label, expected_idx",
        [
            # 5 α electrons: α HOMO = MO 4, α LUMO = MO 5.
            ("alpha", "HOMO", 4),
            ("alpha", "LUMO", 5),
            # 4 β electrons: β HOMO = MO 3, β LUMO = MO 4 (the π orbital
            # that holds the unpaired α electron).
            ("beta", "HOMO", 3),
            ("beta", "LUMO", 4),
            ("beta", "HOMO-1", 2),
        ],
    )
    def test_index_and_coefficients_follow_the_spin_channel(
        self, oh_app, spin, label, expected_idx
    ):
        r, app = oh_app
        app._orb_spin_toggle.value = spin
        call = self._render(app, label)
        _atoms, _basis, coeff, idx, cube_path = call.args[:5]
        assert idx == expected_idx
        coeff = np.asarray(coeff)
        channel = 1 if spin == "beta" else 0
        if coeff.ndim == 3:
            # The generator itself takes the alpha block of a 3-D array.
            assert channel == 0
            coeff = coeff[0]
        np.testing.assert_allclose(coeff, np.asarray(r.mo_coeff)[channel])
        # The UHF cube is built with the radical's spin, not a closed shell.
        assert call.kwargs["spin"] == 1 and call.kwargs["charge"] == 0
        assert Path(cube_path).exists()
        assert Path(cube_path).name.endswith("_beta.cube") == (spin == "beta")
        assert ("β" in app._last_cube_orbital) == (spin == "beta")


# ---------------------------------------------------------------------------
# Orbital gallery
# ---------------------------------------------------------------------------


def _gallery_tiles(html: str) -> list[tuple[str, int]]:
    """[(caption, 1-based MO number shown in the caption), …] in tile order."""
    captions = re.findall(
        r'font-weight:600;margin:2px 0 0;text-align:center">([^<]+)<', html
    )
    details = re.findall(r'min-height:14px">([^<]*)<', html)
    assert len(captions) == len(details)
    return [
        (c, int(re.search(r"MO (\d+)$", d).group(1))) for c, d in zip(captions, details)
    ]


class TestGalleryDefaults:
    def test_default_span_is_homo_minus_2_to_lumo_plus_2(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        assert app._orb_gallery_span_dd.value == 3
        # Water/STO-3G: 5 occupied of 7.
        labels = [lbl for lbl, _ in gallery_orbital_indices(5, 7)]
        assert labels == ["HOMO−2", "HOMO−1", "HOMO", "LUMO", "LUMO+1"]
        labels = [lbl for lbl, _ in gallery_orbital_indices(5, 13)]
        assert labels == ["HOMO−2", "HOMO−1", "HOMO", "LUMO", "LUMO+1", "LUMO+2"]

    @pytest.mark.parametrize(
        "n_occ, n_total, each_side, expected",
        [
            # He/STO-3G: one orbital, occupied, no virtuals.
            (1, 1, 3, [("HOMO", 0)]),
            # Every orbital occupied: the window stops at the HOMO.
            (3, 3, 2, [("HOMO−1", 1), ("HOMO", 2)]),
            # A wider window than the basis has never invents indices.
            (2, 4, 4, [("HOMO−1", 0), ("HOMO", 1), ("LUMO", 2), ("LUMO+1", 3)]),
        ],
    )
    def test_window_is_clipped_to_the_basis(self, n_occ, n_total, each_side, expected):
        assert gallery_orbital_indices(n_occ, n_total, each_side) == expected


@needs_pyscf
class TestGalleryOnRealResults:
    @pytest.fixture(autouse=True)
    def _results_in_tmp(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))

    def test_two_orbital_basis_gives_two_tiles(self, tmp_path):
        from quantui.app_visualization import build_gallery_for_app

        h2 = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])
        app = _app_with(_session(h2, "RHF"), h2, tmp_path)
        tiles = _gallery_tiles(build_gallery_for_app(app, 3))
        # D∞h labels: bonding σg, antibonding σu.
        assert tiles == [("HOMO · 1σg", 1), ("LUMO · 1σu", 2)]

    def test_uhf_alpha_and_beta_windows_differ(self, tmp_path):
        from quantui.app_visualization import build_gallery_for_app

        mol = _oh_molecule()
        app = _app_with(_session(mol, "UHF"), mol, tmp_path)
        alpha = _gallery_tiles(build_gallery_for_app(app, 3))
        # α: 5 of 6 occupied → one virtual.
        assert [i for _, i in alpha] == [3, 4, 5, 6]
        assert [c.split(" · ")[0] for c, _ in alpha] == [
            "HOMO−2",
            "HOMO−1",
            "HOMO",
            "LUMO",
        ]
        app._orb_spin_toggle.value = "beta"
        beta = _gallery_tiles(build_gallery_for_app(app, 3))
        # β: 4 of 6 occupied → HOMO is MO 4, two virtuals.
        assert [i for _, i in beta] == [2, 3, 4, 5, 6]
        assert [c for c, _ in beta] == [
            "HOMO−2 (β)",
            "HOMO−1 (β)",
            "HOMO (β)",
            "LUMO (β)",
            "LUMO+1 (β)",
        ]


# ---------------------------------------------------------------------------
# History replay of a saved UHF result
# ---------------------------------------------------------------------------


@needs_pyscf
@pytest.mark.slow
class TestUhfHistoryReplay:
    @pytest.fixture
    def replayed(self, tmp_path, monkeypatch):
        from quantui.app import QuantUIApp
        from quantui.results_storage import save_orbitals, save_result

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        mol = _oh_molecule()
        r = _session(mol, "UHF")
        saved = save_result(
            r, results_dir=tmp_path, calc_type="single_point", spectra={}, molecule=mol
        )
        save_orbitals(saved, r)
        app = QuantUIApp()
        app._resolve_backend = lambda task: "py3dmol"
        app._history_load_analysis(saved)
        return r, app

    def test_isosurface_panel_restores_both_spin_channels(self, replayed):
        r, app = replayed
        assert "Isosurface" in app._ana_available
        coeff = np.asarray(app._last_orb_mo_coeff)
        assert coeff.shape == (2, 6, 6)
        np.testing.assert_allclose(coeff, np.asarray(r.mo_coeff))
        np.testing.assert_allclose(
            np.asarray(app._last_orb_mo_occ), np.asarray(r.mo_occ)
        )
        assert app._orb_spin_toggle.layout.display == ""

    def test_replayed_spin_density_integrates_to_one(self, replayed):
        _r, app = replayed
        app._iso_resolution_dd.value = "medium"
        app._iso_surface_dd.value = "spin"
        with patch(
            "quantui.orbital_visualization.render_surface_py3dmol",
            return_value="<div>spin</div>",
        ):
            app._render_orbital_isosurface("Spin density")
        assert app._last_cube_kind == "spin"
        _, n_unpaired = _integral(app._last_cube_path)
        assert n_unpaired == pytest.approx(1.0, rel=0.03)

    def test_replayed_beta_homo_uses_beta_coefficients(self, replayed):
        from quantui import orbital_visualization as ov

        r, app = replayed
        app._iso_resolution_dd.value = "coarse"
        app._orb_spin_toggle.value = "beta"
        with (
            patch.object(
                ov, "generate_cube_from_arrays", wraps=ov.generate_cube_from_arrays
            ) as gen,
            patch.object(
                ov, "render_orbital_isosurface_py3dmol", return_value="<div>β</div>"
            ),
        ):
            app._render_orbital_isosurface("HOMO")
        _atoms, _basis, coeff, idx = gen.call_args.args[:4]
        assert idx == 3
        np.testing.assert_allclose(np.asarray(coeff), np.asarray(r.mo_coeff)[1])
        assert app._last_cube_orbital == "HOMO (β)"
        assert app._last_cube_path.exists()
