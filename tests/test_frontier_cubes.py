"""HOMO/LUMO cubes saved with a result, for the PySCF-free Viewer (VW.7)."""

from __future__ import annotations

import gzip
import json
from unittest.mock import MagicMock, patch

import pytest

from quantui.molecule import Molecule
from quantui.orbital_visualization import (
    FRONTIER_CUBES_DIR,
    FRONTIER_CUBES_ENV,
    frontier_cube_labels,
    load_frontier_cube,
    parse_cube_file,
)
from quantui.results_storage import save_orbitals

pytest.importorskip("pyscf")

WATER = Molecule(
    ["O", "H", "H"], [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]]
)
OH = Molecule(["O", "H"], [[0, 0, 0], [0, 0, 0.97]], multiplicity=2)


def _sp(mol, method):
    from quantui.session_calc import run_in_session

    return run_in_session(mol, method, "STO-3G")


def test_closed_shell_saves_homo_and_lumo(tmp_path):
    save_orbitals(tmp_path, _sp(WATER, "RHF"))
    assert frontier_cube_labels(tmp_path) == ["HOMO", "LUMO"]
    index = json.loads((tmp_path / FRONTIER_CUBES_DIR / "index.json").read_text())
    assert index["grid"] == 60 and [e["index"] for e in index["orbitals"]] == [4, 5]
    raw = (tmp_path / FRONTIER_CUBES_DIR / "mo_4.cube.gz").read_bytes()
    assert gzip.decompress(raw).startswith(b"QuantUI orbital cube")
    dest = tmp_path / "homo.cube"
    assert load_frontier_cube(tmp_path, 4, beta=False, dest=dest)
    cube = parse_cube_file(dest)
    assert cube["data"].shape == (60, 60, 60)
    assert not load_frontier_cube(tmp_path, 3, beta=False, dest=dest)  # HOMO-1


def test_open_shell_saves_both_spins(tmp_path):
    save_orbitals(tmp_path, _sp(OH, "UHF"))
    labels = frontier_cube_labels(tmp_path)
    assert labels == ["HOMO (a)", "LUMO (a)", "HOMO (b)", "LUMO (b)"]
    beta_homo = [
        e["index"]
        for e in json.loads((tmp_path / FRONTIER_CUBES_DIR / "index.json").read_text())[
            "orbitals"
        ]
        if e["label"] == "HOMO (b)"
    ][0]
    dest = tmp_path / "b.cube"
    assert load_frontier_cube(tmp_path, beta_homo, beta=True, dest=dest)
    assert not load_frontier_cube(tmp_path, beta_homo, beta=False, dest=dest)


def test_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv(FRONTIER_CUBES_ENV, "0")
    save_orbitals(tmp_path, _sp(WATER, "RHF"))
    assert (tmp_path / "orbitals.npz").exists()
    assert not (tmp_path / FRONTIER_CUBES_DIR).exists()


class TestViewerWithoutPySCF:
    @pytest.fixture
    def app(self, tmp_path):
        from quantui.app import QuantUIApp

        result = _sp(WATER, "RHF")
        save_orbitals(tmp_path, result)
        app = QuantUIApp(viewer=True, results_dir=tmp_path)
        app._last_result_dir = tmp_path
        app._last_orb_info = MagicMock(
            n_occupied=5, mo_energies_ev=[0.0] * 7, formula="H2O"
        )
        app._last_orb_mo_coeff = result.mo_coeff
        app._last_orb_mol_atom = result.pyscf_mol_atom
        app._last_orb_mol_basis = result.pyscf_mol_basis
        app._last_orb_mo_occ = result.mo_occ
        app._resolve_backend = lambda task: "plotlymol"
        return app

    def _render(self, app, label):
        drawn = []

        def _plot(cube_path, *a, **k):
            drawn.append(parse_cube_file(cube_path))
            return MagicMock()

        def _no_pyscf(*a, **k):
            raise ImportError("PySCF is required for cube file generation")

        with (
            patch(
                "quantui.orbital_visualization.generate_cube_from_arrays",
                side_effect=_no_pyscf,
            ),
            patch(
                "quantui.orbital_visualization.plot_cube_isosurface", side_effect=_plot
            ),
            patch("plotly.io.to_html", return_value="<div>iso</div>"),
        ):
            app._render_orbital_isosurface(label)
        return drawn

    def test_saved_homo_is_drawn(self, app):
        drawn = self._render(app, "HOMO")
        assert len(drawn) == 1 and drawn[0]["data"].shape == (60, 60, 60)

    def test_other_orbitals_say_what_is_available(self, app):
        shown = []
        app._set_html_output = lambda out, html: shown.append(html)
        app._queue_main_thread_callback = lambda fn, *a, **k: fn(*a, **k)
        assert self._render(app, "HOMO-1") == []
        assert any("HOMO, LUMO" in h and "full QuantUI" in h for h in shown)
