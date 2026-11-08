"""CHEM.3 — Mayer and Wiberg (NAO-basis) bond orders.

Formula tests on small molecules (PySCF-gated, fast), persistence round
trip, the 3D bond-label overlay and the Populations-panel table.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from quantui.molecule import Molecule
from quantui.results_storage import load_result, save_result

try:
    import pyscf as _pyscf  # noqa: F401

    _PYSCF_AVAILABLE = True
except ImportError:  # pragma: no cover — Windows / no pyscf extra
    _PYSCF_AVAILABLE = False

pyscf_only = pytest.mark.skipif(not _PYSCF_AVAILABLE, reason="PySCF not installed")

_WATER = "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0"
_N2 = "N 0 0 0; N 0 0 1.098"
_ETHYLENE = (
    "C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; H 0 -0.9289 1.2321; "
    "H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321"
)


def _benzene():
    atoms = []
    for k in range(6):
        a = math.pi / 3 * k
        atoms.append(("C", (1.397 * math.cos(a), 1.397 * math.sin(a), 0.0)))
    for k in range(6):
        a = math.pi / 3 * k
        atoms.append(("H", (2.481 * math.cos(a), 2.481 * math.sin(a), 0.0)))
    return atoms


def _scf(atom, basis, *, spin=0, charge=0, kind="rhf"):
    from pyscf import gto, scf

    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, verbose=0)
    mf = {"rhf": scf.RHF, "uhf": scf.UHF, "rohf": scf.ROHF}[kind](mol)
    mf.kernel()
    assert mf.converged
    return mf


def _orders(mf):
    from quantui.population_analysis import mayer_bond_orders, wiberg_bond_indices

    dm = mf.make_rdm1()
    return mayer_bond_orders(mf.mol, dm), wiberg_bond_indices(mf.mol, dm)


# ── Formulas ────────────────────────────────────────────────────────────────


@pyscf_only
class TestFormulas:
    @pytest.mark.parametrize("basis", ["sto-3g", "6-31g*"])
    def test_h2_single_bond_is_exactly_one(self, basis):
        m, w = _orders(_scf("H 0 0 0; H 0 0 0.74", basis))
        assert m[0, 1] == pytest.approx(1.0, abs=1e-6)
        assert w[0, 1] == pytest.approx(1.0, abs=1e-6)

    def test_n2_triple_bond(self):
        m, w = _orders(_scf(_N2, "sto-3g"))
        assert m[0, 1] == pytest.approx(3.0, abs=1e-3)
        assert w[0, 1] == pytest.approx(3.0, abs=1e-3)
        # Larger basis: NAO-basis Wiberg stays ~3, Mayer drops (basis dependence).
        m, w = _orders(_scf(_N2, "6-31g*"))
        assert w[0, 1] == pytest.approx(3.03, abs=0.01)
        assert m[0, 1] == pytest.approx(2.83, abs=0.01)

    def test_ethylene_double_bond(self):
        m, w = _orders(_scf(_ETHYLENE, "sto-3g"))
        assert m[0, 1] == pytest.approx(2.01, abs=0.01)
        assert w[0, 1] == pytest.approx(2.02, abs=0.01)
        assert m[0, 2] == pytest.approx(0.98, abs=0.01)  # C–H

    def test_benzene_aromatic_bond_and_pair_filter(self):
        from quantui.population_analysis import bond_order_pairs

        m, w = _orders(_scf(_benzene(), "sto-3g"))
        assert m[0, 1] == pytest.approx(1.43, abs=0.01)
        assert w[0, 1] == pytest.approx(1.44, abs=0.01)
        pairs = {(p["i"], p["j"]): p for p in bond_order_pairs(m, w)}
        assert (0, 1) in pairs  # ortho C–C
        assert (0, 3) in pairs  # para C···C ~0.12
        assert (0, 2) not in pairs  # meta C···C ~0.00
        assert (0, 6) in pairs  # C–H

    def test_matrices_symmetric_with_zero_diagonal(self):
        import numpy as np

        m, w = _orders(_scf(_WATER, "6-31g"))
        assert np.allclose(m, m.T) and np.allclose(w, w.T)
        assert np.allclose(np.diag(m), 0.0) and np.allclose(np.diag(w), 0.0)

    def test_vectorised_mayer_matches_explicit_double_sum(self):
        mf = _scf(_WATER, "6-31g")
        m, _ = _orders(mf)
        ps = mf.make_rdm1() @ mf.get_ovlp()
        sl = mf.mol.aoslice_by_atom()
        a0, a1 = sl[0][2:]
        b0, b1 = sl[1][2:]
        ref = sum(
            ps[mu, nu] * ps[nu, mu] for mu in range(a0, a1) for nu in range(b0, b1)
        )
        assert m[0, 1] == pytest.approx(ref, abs=1e-12)

    def test_open_shell_form_reduces_to_closed_shell(self):
        """Feeding P/2, P/2 as α, β must give the closed-shell numbers."""
        import numpy as np

        from quantui.population_analysis import (
            mayer_bond_orders,
            nao_coefficients,
            wiberg_bond_indices,
        )

        mf = _scf(_WATER, "6-31g")
        dm = mf.make_rdm1()
        split = np.array([dm / 2, dm / 2])
        c = nao_coefficients(mf.mol, dm)
        assert np.allclose(
            mayer_bond_orders(mf.mol, dm), mayer_bond_orders(mf.mol, split)
        )
        assert np.allclose(
            wiberg_bond_indices(mf.mol, dm, c_nao=c),
            wiberg_bond_indices(mf.mol, split, c_nao=c),
        )

    def test_mayer_equals_wiberg_in_orthonormal_basis(self):
        """With S = 1 the Mayer formula is the Wiberg formula."""
        import numpy as np

        from quantui.population_analysis import mayer_bond_orders, wiberg_bond_indices

        mf = _scf(_WATER, "6-31g")
        eye = np.eye(mf.mol.nao_nr())
        dm = mf.make_rdm1()
        assert np.allclose(
            mayer_bond_orders(mf.mol, dm, s=eye),
            wiberg_bond_indices(mf.mol, dm, s=eye, c_nao=eye),
        )

    def test_uhf_one_electron_bond_is_half(self):
        m, w = _orders(
            _scf("H 0 0 0; H 0 0 1.06", "6-31g", spin=1, charge=1, kind="uhf")
        )
        assert m[0, 1] == pytest.approx(0.5, abs=1e-6)
        assert w[0, 1] == pytest.approx(0.5, abs=1e-6)

    @pytest.mark.parametrize("kind", ["uhf", "rohf"])
    def test_triplet_o2_double_bond(self, kind):
        m, w = _orders(_scf("O 0 0 0; O 0 0 1.208", "sto-3g", spin=2, kind=kind))
        assert m[0, 1] == pytest.approx(2.0, abs=0.01)
        assert w[0, 1] == pytest.approx(2.0, abs=0.01)


class TestPairs:
    def test_threshold_sort_and_missing_wiberg(self):
        import numpy as np

        from quantui.population_analysis import bond_order_pairs

        m = np.array([[0, 0.95, 0.05], [0.95, 0, 1.9], [0.05, 1.9, 0]])
        pairs = bond_order_pairs(m)
        assert [(p["i"], p["j"]) for p in pairs] == [(1, 2), (0, 1)]
        assert pairs[0]["wiberg"] is None
        w = np.array([[0, 0.9, 0.2], [0.9, 0, 2.0], [0.2, 2.0, 0]])
        pairs = bond_order_pairs(m, w)
        assert (0, 2) in [(p["i"], p["j"]) for p in pairs]  # Wiberg ≥ 0.10


@pyscf_only
class TestScfPopulationAnalysis:
    def test_returns_charges_and_bonds(self):
        from quantui.population_analysis import scf_population_analysis

        out = scf_population_analysis(_scf(_WATER, "sto-3g"))
        assert len(out["npa_charges"]) == 3
        oh = [p for p in out["bond_orders"] if (p["i"], p["j"]) == (0, 1)][0]
        assert oh["mayer"] == pytest.approx(0.954, abs=0.005)
        assert oh["wiberg"] == pytest.approx(0.958, abs=0.005)

    def test_nao_failure_keeps_mayer(self, monkeypatch):
        import quantui.population_analysis as pa

        def _boom(*_a, **_k):
            raise RuntimeError("nao failed")

        monkeypatch.setattr(pa, "nao_coefficients", _boom)
        out = pa.scf_population_analysis(_scf(_WATER, "sto-3g"))
        assert out["npa_charges"] is None
        assert out["bond_orders"]
        assert all(p["wiberg"] is None for p in out["bond_orders"])

    def test_ground_state_analysis_and_run_in_session(self):
        from quantui.session_calc import run_in_session

        water = Molecule(
            ["O", "H", "H"],
            [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
        )
        res = run_in_session(water, method="RHF", basis="STO-3G", verbose=0)
        pairs = {(p["i"], p["j"]) for p in res.bond_orders}
        assert pairs == {(0, 1), (0, 2)}


# ── Persistence ──────────────────────────────────────────────────────────────

_BONDS = [
    {"i": 0, "j": 1, "mayer": 0.954, "wiberg": 0.958},
    {"i": 0, "j": 2, "mayer": 0.954, "wiberg": 0.958},
]


def _sp_result(**overrides):
    base = dict(
        formula="H2O",
        method="RHF",
        basis="STO-3G",
        energy_hartree=-75.0,
        homo_lumo_gap_ev=10.5,
        converged=True,
        n_iterations=8,
        atom_symbols=["O", "H", "H"],
        mulliken_charges=[-0.366, 0.183, 0.183],
        npa_charges=[-0.40, 0.20, 0.20],
        bond_orders=[dict(b) for b in _BONDS],
        dipole_moment_debye=1.7,
        dipole_vector_debye=[0.0, 1.7, 0.0],
        mo_energy_hartree=None,
        mo_occ=None,
        mo_coeff=None,
        pyscf_mol_atom=None,
        pyscf_mol_basis=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestPersistence:
    def test_result_json_round_trip(self, tmp_path):
        data = load_result(
            save_result(_sp_result(), results_dir=tmp_path, calc_type="single_point")
        )
        assert data["bond_orders"] == _BONDS

    def test_absent_and_malformed(self, tmp_path):
        res = _sp_result()
        del res.bond_orders
        data = load_result(
            save_result(res, results_dir=tmp_path, calc_type="single_point")
        )
        assert data["bond_orders"] is None
        import numpy as np

        res = _sp_result(
            bond_orders=[
                {"i": np.int64(0), "j": 1, "mayer": np.float64(0.9), "wiberg": None},
                {"i": 0},  # malformed → dropped, save still succeeds
            ]
        )
        data = load_result(
            save_result(res, results_dir=tmp_path, calc_type="single_point")
        )
        assert data["bond_orders"] == [{"i": 0, "j": 1, "mayer": 0.9, "wiberg": None}]

    def test_batch_payload_and_ingest(self):
        from quantui.backends.slurm_ingest import _basic_result
        from quantui.backends.worker_payload import (
            _ground_state_fields,
            session_result_payload,
        )

        assert session_result_payload(_sp_result())["bond_orders"] == _BONDS
        assert _ground_state_fields(_sp_result())["bond_orders"] == _BONDS
        record = SimpleNamespace(request_obj=SimpleNamespace(method="RHF", basis="X"))
        assert _basic_result({"bond_orders": _BONDS}, record).bond_orders == _BONDS
        assert _basic_result({}, record).bond_orders is None


# ── 3D overlay labels ────────────────────────────────────────────────────────


class TestOverlayLabels:
    coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]

    def test_midpoint_labels_for_chosen_index(self):
        from quantui.populations_overlay import bond_order_labels

        labels = bond_order_labels(_BONDS, self.coords, "wiberg")
        assert [lab["text"] for lab in labels] == ["0.96", "0.96"]
        assert labels[0]["position"] == pytest.approx(
            {"x": 0.3785, "y": 0.293, "z": 0.0}
        )
        assert bond_order_labels(_BONDS, self.coords, "off") == []

    def test_weak_pairs_and_bad_indices_skipped(self):
        from quantui.populations_overlay import bond_order_labels

        weak = [{"i": 1, "j": 2, "mayer": 0.12, "wiberg": 0.11}]
        bad = [{"i": 0, "j": 9, "mayer": 1.0, "wiberg": 1.0}]
        assert bond_order_labels(weak + bad, self.coords, "mayer") == []

    def test_payload_and_js_bridge(self):
        from quantui.populations_overlay import (
            build_overlay_payload,
            inject_populations_js,
        )

        kw = dict(
            charges=None,
            color_enabled=False,
            dipole_vector=None,
            dipole_enabled=False,
            symbols=["O", "H", "H"],
            coordinates=self.coords,
            bond_orders=_BONDS,
        )
        assert build_overlay_payload(**kw)["labels"] is None
        assert len(build_overlay_payload(**kw, bond_label_index="mayer")["labels"]) == 2
        js = inject_populations_js('<div id="3dmolviewer_abc"></div>', None)
        assert "addLabel" in js and "removeLabel" in js


# ── Populations panel ────────────────────────────────────────────────────────


def _water_molecule() -> Molecule:
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    from quantui.app import QuantUIApp

    a = QuantUIApp()
    a._set_molecule(_water_molecule())
    return a


def _ctx(result, **kw):
    base = dict(
        calc_type="single_point",
        live_result=result,
        result_dir=None,
        spectra_data={},
        source="live",
        formula="H2O",
        method=getattr(result, "method", "RHF"),
        basis="STO-3G",
        label="H2O",
        timestamp="",
        molecule=_water_molecule(),
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestPanel:
    def test_table_and_labels_dropdown(self, app):
        from quantui.app_analysis import pop_mulliken
        from quantui.populations_overlay import overlay_payload

        assert pop_mulliken(app, _ctx(_sp_result())) is True
        table = app._bond_order_table.value
        assert "O1–H2" in table and "O1–H3" in table
        assert "0.954" in table and "0.958" in table
        assert "Wiberg (NAO)" in table
        assert app._bond_label_dd.disabled is False
        assert overlay_payload(app, _water_molecule())["labels"] is None
        app._bond_label_dd.value = "mayer"
        labels = overlay_payload(app, _water_molecule())["labels"]
        assert [lab["text"] for lab in labels] == ["0.95", "0.95"]

    def test_missing_wiberg_shows_dash(self, app):
        from quantui.app_analysis import pop_mulliken

        bonds = [{"i": 0, "j": 1, "mayer": 0.95, "wiberg": None}]
        pop_mulliken(app, _ctx(_sp_result(bond_orders=bonds)))
        assert "—" in app._bond_order_table.value

    def test_unavailable_engine_disables_labels(self, app):
        from quantui.app_analysis import pop_mulliken

        app._bond_label_dd.value = "wiberg"
        res = _sp_result(bond_orders=None, engine_id="pyfock")
        assert pop_mulliken(app, _ctx(res)) is True
        assert app._bond_label_dd.disabled is True
        assert app._bond_label_dd.value == "off"
        assert "not available for the pyfock engine" in app._bond_order_table.value

    def test_history_round_trip_and_old_result(self, tmp_path, app):
        saved = save_result(
            _sp_result(), results_dir=tmp_path, calc_type="single_point"
        )
        app._apply_analysis_context(app._build_history_context(saved))
        assert app._last_bond_orders == _BONDS
        assert "O1–H2" in app._bond_order_table.value

        old = _sp_result()
        del old.bond_orders
        saved = save_result(old, results_dir=tmp_path, calc_type="single_point")
        app._apply_analysis_context(app._build_history_context(saved))
        assert app._last_bond_orders is None
        assert "not saved with this History result" in app._bond_order_table.value

    def test_out_of_range_pairs_rejected(self, app):
        from quantui.app_analysis import pop_mulliken

        bad = [{"i": 0, "j": 7, "mayer": 1.0, "wiberg": 1.0}]
        pop_mulliken(app, _ctx(_sp_result(bond_orders=bad)))
        assert app._last_bond_orders is None


def test_help_topic_explains_mayer_and_wiberg():
    from quantui.help_content import HELP_TOPICS

    body = HELP_TOPICS["mulliken"]["body"]
    assert "Mayer bond order" in body and "Wiberg bond index" in body
    assert "NAO" in body
