"""Tests for SLURM dispatch helpers (seed geometry + pre-opt flags)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from quantui.backends.dispatch import (
    build_calculation_request,
    load_seed_molecule,
    seed_path_from_app,
)
from quantui.molecule import Molecule
from quantui.results_storage import save_trajectory


def _fake_app(**overrides):
    mol = Molecule(
        atoms=["H", "H"],
        coordinates=[[0, 0, 0], [0, 0, 0.74]],
        charge=0,
        multiplicity=1,
    )
    defaults = dict(
        _molecule=mol,
        method_dd=SimpleNamespace(value="RHF"),
        basis_dd=SimpleNamespace(value="STO-3G"),
        mult_si=SimpleNamespace(value=1),
        calc_type_dd=SimpleNamespace(value="Single Point"),
        solvent_dd=SimpleNamespace(value=""),
        solvent_cb=SimpleNamespace(value=False),
        fmax_fi=SimpleNamespace(value=0.05),
        max_steps_si=SimpleNamespace(value=100),
        nstates_si=SimpleNamespace(value=5),
        _seed_dd=SimpleNamespace(value=""),
        _freq_preopt_cb=SimpleNamespace(value=False),
        _reorg_mode_dd=SimpleNamespace(value="both"),
        _scan_type_dd=SimpleNamespace(value="Bond"),
        _scan_atom1=SimpleNamespace(value=1),
        _scan_atom2=SimpleNamespace(value=2),
        _scan_atom3=SimpleNamespace(value=3),
        _scan_atom4=SimpleNamespace(value=4),
        _scan_start=SimpleNamespace(value=0.5),
        _scan_stop=SimpleNamespace(value=2.0),
        _scan_steps=SimpleNamespace(value=10),
        _scan_grid_dd=SimpleNamespace(value="linear"),
        _scan_seed_dd=SimpleNamespace(value=""),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _write_seed_trajectory(tmp_path, *, final_bond=0.80):
    seed_dir = tmp_path / "seed-result"
    seed_dir.mkdir()
    initial = Molecule(
        atoms=["H", "H"],
        coordinates=[[0, 0, 0], [0, 0, 0.74]],
        charge=0,
        multiplicity=1,
    )
    final = Molecule(
        atoms=["H", "H"],
        coordinates=[[0, 0, 0], [0, 0, final_bond]],
        charge=0,
        multiplicity=1,
    )
    save_trajectory(seed_dir, [initial, final], [-1.10, -1.12])
    return seed_dir


class TestSeedGeometryDispatch:
    def test_seed_path_from_app_empty(self):
        assert seed_path_from_app(_fake_app()) is None

    def test_load_seed_molecule_returns_final_frame(self, tmp_path):
        seed_dir = _write_seed_trajectory(tmp_path, final_bond=0.82)
        mol = load_seed_molecule(seed_dir)
        assert mol.coordinates[1][2] == pytest.approx(0.82)

    def test_frequency_request_uses_seed_geometry(self, tmp_path):
        seed_dir = _write_seed_trajectory(tmp_path, final_bond=0.81)
        app = _fake_app(
            calc_type_dd=SimpleNamespace(value="Frequency"),
            _seed_dd=SimpleNamespace(value=str(seed_dir)),
        )
        req = build_calculation_request(app, request_id="seed1")
        assert req.molecule["coordinates"][1][2] == pytest.approx(0.81)
        assert req.run_context["seed_result_dir"] == str(seed_dir)
        assert req.options.get("preopt_before_run") is not True

    def test_frequency_preopt_flag_when_checkbox_set(self):
        app = _fake_app(
            calc_type_dd=SimpleNamespace(value="Frequency"),
            _freq_preopt_cb=SimpleNamespace(value=True),
        )
        req = build_calculation_request(app, request_id="preopt1")
        assert req.options.get("preopt_before_run") is True

    def test_frequency_seed_suppresses_preopt_flag(self, tmp_path):
        seed_dir = _write_seed_trajectory(tmp_path)
        app = _fake_app(
            calc_type_dd=SimpleNamespace(value="Frequency"),
            _seed_dd=SimpleNamespace(value=str(seed_dir)),
            _freq_preopt_cb=SimpleNamespace(value=True),
        )
        req = build_calculation_request(app, request_id="both1")
        assert req.run_context["seed_label"] == seed_dir.name
        assert "preopt_before_run" not in req.options

    def test_geometry_opt_request_uses_seed_starting_geometry(self, tmp_path):
        seed_dir = _write_seed_trajectory(tmp_path, final_bond=0.79)
        app = _fake_app(
            calc_type_dd=SimpleNamespace(value="Geometry Opt"),
            _seed_dd=SimpleNamespace(value=str(seed_dir)),
        )
        req = build_calculation_request(app, request_id="geo-seed")
        assert req.calc_type == "geometry_opt"
        assert req.molecule["coordinates"][1][2] == pytest.approx(0.79)

    def test_tddft_preopt_flag(self):
        app = _fake_app(
            calc_type_dd=SimpleNamespace(value="UV-Vis (TD-DFT)"),
            _freq_preopt_cb=SimpleNamespace(value=True),
        )
        req = build_calculation_request(app, request_id="tddft-pre")
        assert req.options.get("preopt_before_run") is True
        assert req.options.get("nstates") == 5

    def test_single_point_ignores_seed(self, tmp_path):
        seed_dir = _write_seed_trajectory(tmp_path, final_bond=0.50)
        app = _fake_app(_seed_dd=SimpleNamespace(value=str(seed_dir)))
        req = build_calculation_request(app, request_id="sp")
        assert req.molecule["coordinates"][1][2] == pytest.approx(0.74)

    def test_pes_scan_request_includes_grid_and_opt_options(self):
        app = _fake_app(
            calc_type_dd=SimpleNamespace(value="PES Scan"),
            _scan_grid_dd=SimpleNamespace(value="log"),
            fmax_fi=SimpleNamespace(value=0.03),
            max_steps_si=SimpleNamespace(value=80),
        )
        req = build_calculation_request(app, request_id="pes1")
        assert req.calc_type == "pes_scan"
        assert req.options["grid"] == "log"
        assert req.options["fmax"] == pytest.approx(0.03)
        assert req.options["max_opt_steps"] == 80

    def test_pes_scan_uses_scan_seed_dropdown(self, tmp_path):
        seed_dir = _write_seed_trajectory(tmp_path, final_bond=0.83)
        app = _fake_app(
            calc_type_dd=SimpleNamespace(value="PES Scan"),
            _scan_seed_dd=SimpleNamespace(value=str(seed_dir)),
        )
        req = build_calculation_request(app, request_id="pes-seed")
        assert req.molecule["coordinates"][1][2] == pytest.approx(0.83)
        assert req.run_context["seed_result_dir"] == str(seed_dir)

    def test_pes_scan_preopt_flag_when_checkbox_set(self):
        app = _fake_app(
            calc_type_dd=SimpleNamespace(value="PES Scan"),
            _freq_preopt_cb=SimpleNamespace(value=True),
        )
        req = build_calculation_request(app, request_id="pes-preopt")
        assert req.options.get("preopt_before_run") is True
