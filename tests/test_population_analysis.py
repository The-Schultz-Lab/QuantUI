"""CHEM.2 — NPA (natural population analysis) charges.

Formula tests on small molecules (PySCF-gated, all well under a second),
the persistence round trip (result.json, batch payload, Slurm ingest) and
the Populations-panel Mulliken/NPA toggle.
"""

from __future__ import annotations

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


def _rhf(atom: str, basis: str, *, charge: int = 0):
    from pyscf import gto, scf

    mol = gto.M(atom=atom, basis=basis, charge=charge, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    assert mf.converged
    return mf


def _uhf(atom: str, basis: str, *, spin: int, charge: int = 0):
    from pyscf import gto, scf

    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    assert mf.converged
    return mf


# ── Formula / science ────────────────────────────────────────────────────────


@pyscf_only
class TestNpaCharges:
    def test_water_6_31gs_values_and_sum(self):
        from quantui.population_analysis import scf_npa_charges

        q = scf_npa_charges(_rhf(_WATER, "6-31g*"))
        assert q is not None and len(q) == 3
        # O strongly negative, the two H equal and positive, sum = 0.
        assert q[0] == pytest.approx(-0.960, abs=0.01)
        assert q[1] == pytest.approx(q[2], abs=1e-8)
        assert q[1] == pytest.approx(0.480, abs=0.01)
        assert sum(q) == pytest.approx(0.0, abs=1e-8)

    def test_matches_pyscf_reference_recipe(self):
        """Same numbers as PySCF's own NAO-population example: Mulliken
        analysis with an identity overlap in the orth_ao(mf, 'nao') basis."""
        import numpy as np
        from pyscf import lo

        from quantui.population_analysis import scf_npa_charges

        mf = _rhf(_WATER, "6-31g*")
        c = lo.orth_ao(mf, "nao")
        mo = np.linalg.solve(c, mf.mo_coeff)
        dm_nao = mf.make_rdm1(mo, mf.mo_occ)
        _, ref = mf.mulliken_pop(mf.mol, dm_nao, np.eye(c.shape[0]), verbose=0)
        assert scf_npa_charges(mf) == pytest.approx(list(ref), abs=1e-8)

    def test_npa_far_less_basis_dependent_than_mulliken(self):
        """The teaching point in the help text: water O charge with 6-31G vs
        cc-pVDZ. Mulliken swings by ~0.5 e, NPA by a few hundredths."""
        from quantui.population_analysis import scf_npa_charges

        a = _rhf(_WATER, "6-31g")
        b = _rhf(_WATER, "cc-pvdz")
        npa_shift = abs(scf_npa_charges(a)[0] - scf_npa_charges(b)[0])
        mull_shift = abs(
            a.mulliken_pop(verbose=0)[1][0] - b.mulliken_pop(verbose=0)[1][0]
        )
        assert npa_shift < 0.05
        assert mull_shift > 0.4

    def test_cation_charges_sum_to_molecular_charge(self):
        from quantui.population_analysis import scf_npa_charges

        h3o = "O 0 0 0.1; H 0.94 0 -0.25; H -0.47 0.814 -0.25; H -0.47 -0.814 -0.25"
        q = scf_npa_charges(_rhf(h3o, "6-31g", charge=1))
        assert sum(q) == pytest.approx(1.0, abs=1e-8)

    def test_uhf_uses_total_density(self):
        import numpy as np

        from quantui.population_analysis import npa_charges, scf_npa_charges

        # Methyl radical (doublet), UHF.
        ch3 = "C 0 0 0; H 1.079 0 0; H -0.5395 0.9344 0; H -0.5395 -0.9344 0"
        mf = _uhf(ch3, "6-31g", spin=1)
        q = scf_npa_charges(mf)
        assert sum(q) == pytest.approx(0.0, abs=1e-8)
        dm = np.asarray(mf.make_rdm1())
        assert npa_charges(mf.mol, dm[0] + dm[1]) == pytest.approx(q, abs=1e-10)
        # Three equivalent H, C negative.
        assert q[1] == pytest.approx(q[2], abs=1e-4)
        assert q[1] == pytest.approx(q[3], abs=1e-4)
        assert q[0] < 0 < q[1]

    def test_h2_plus_splits_charge_evenly(self):
        from quantui.population_analysis import scf_npa_charges

        q = scf_npa_charges(_uhf("H 0 0 0; H 0 0 1.06", "sto-3g", spin=1, charge=1))
        assert q == pytest.approx([0.5, 0.5], abs=1e-8)

    def test_uks_object_supported(self):
        from pyscf import dft, gto

        from quantui.population_analysis import scf_npa_charges

        mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="6-31g", spin=1, verbose=0)
        mf = dft.UKS(mol)
        mf.xc = "b3lyp"
        mf.kernel()
        q = scf_npa_charges(mf)
        assert sum(q) == pytest.approx(0.0, abs=1e-8)
        assert q[0] < 0 < q[1]

    def test_no_mol_returns_none(self):
        from quantui.population_analysis import scf_npa_charges

        assert scf_npa_charges(SimpleNamespace()) is None


@pyscf_only
class TestGroundStateAnalysis:
    def _molecule(self):
        return Molecule(
            ["O", "H", "H"],
            [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
        )

    def test_npa_next_to_mulliken(self):
        from quantui.session_calc import ground_state_analysis

        mf = _rhf(_WATER, "sto-3g")
        out = ground_state_analysis(mf, self._molecule(), "STO-3G")
        assert out["mulliken_charges"] is not None
        assert out["npa_charges"] is not None
        assert sum(out["npa_charges"]) == pytest.approx(0.0, abs=1e-8)
        assert out["npa_charges"][0] == pytest.approx(-0.40, abs=0.01)

    def test_failure_never_breaks_the_calculation(self, monkeypatch):
        import quantui.population_analysis as pa
        from quantui.session_calc import ground_state_analysis

        def _boom(_mf):
            raise RuntimeError("nao exploded")

        monkeypatch.setattr(pa, "scf_npa_charges", _boom)
        mf = _rhf(_WATER, "sto-3g")
        out = ground_state_analysis(mf, self._molecule(), "STO-3G")
        assert out["npa_charges"] is None
        assert out["mulliken_charges"] is not None

    def test_run_in_session_sets_field(self):
        from quantui.session_calc import run_in_session

        res = run_in_session(self._molecule(), method="RHF", basis="STO-3G", verbose=0)
        assert res.npa_charges is not None
        assert len(res.npa_charges) == 3
        assert sum(res.npa_charges) == pytest.approx(0.0, abs=1e-6)


# ── Persistence ──────────────────────────────────────────────────────────────


def _sp_result(**overrides):
    base = dict(
        formula="H2O",
        method="RHF",
        basis="6-31G*",
        energy_hartree=-76.0,
        homo_lumo_gap_ev=10.5,
        converged=True,
        n_iterations=8,
        atom_symbols=["O", "H", "H"],
        mulliken_charges=[-0.897, 0.449, 0.449],
        npa_charges=[-0.960, 0.480, 0.480],
        dipole_moment_debye=2.2,
        dipole_vector_debye=[0.0, 2.2, 0.0],
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
        saved = save_result(
            _sp_result(), results_dir=tmp_path, calc_type="single_point"
        )
        data = load_result(saved)
        assert data["npa_charges"] == pytest.approx([-0.960, 0.480, 0.480])

    def test_absent_is_none_not_error(self, tmp_path):
        res = _sp_result()
        del res.npa_charges
        data = load_result(
            save_result(res, results_dir=tmp_path, calc_type="single_point")
        )
        assert data["npa_charges"] is None

    def test_numpy_values_are_json_safe(self, tmp_path):
        import numpy as np

        res = _sp_result(npa_charges=np.array([-0.96, 0.48, 0.48]))
        data = load_result(
            save_result(res, results_dir=tmp_path, calc_type="single_point")
        )
        assert data["npa_charges"] == pytest.approx([-0.96, 0.48, 0.48])

    def test_batch_payloads_carry_npa(self):
        from quantui.backends.worker_payload import (
            _ground_state_fields,
            session_result_payload,
        )

        assert session_result_payload(_sp_result())["npa_charges"] == [
            -0.960,
            0.480,
            0.480,
        ]
        assert _ground_state_fields(_sp_result())["npa_charges"][0] == -0.960
        minimal = SimpleNamespace(
            energy_hartree=-76.0,
            homo_lumo_gap_ev=None,
            converged=True,
            n_iterations=1,
            method="RHF",
            basis="STO-3G",
            formula="H2O",
        )
        assert session_result_payload(minimal)["npa_charges"] is None

    def test_slurm_ingest_reconstructs_npa(self):
        from quantui.backends.slurm_ingest import _basic_result

        record = SimpleNamespace(
            request_obj=SimpleNamespace(method="RHF", basis="6-31G*")
        )
        res = _basic_result({"npa_charges": [-0.96, 0.48, 0.48]}, record)
        assert res.npa_charges == [-0.96, 0.48, 0.48]
        assert _basic_result({}, record).npa_charges is None


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
        basis="6-31G*",
        label="H2O",
        timestamp="",
        molecule=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestPanelNpa:
    def test_table_lists_both_schemes_and_toggle_enabled(self, app):
        from quantui.app_analysis import pop_mulliken

        assert pop_mulliken(app, _ctx(_sp_result())) is True
        table = app._mulliken_table.value
        assert "Mulliken (e)" in table and "NPA (e)" in table
        assert "-0.9600" in table and "-0.8970" in table
        assert app._pop_scheme_tb.disabled is False
        assert app._pop_scheme_note.value == ""
        assert app._last_npa_charges == pytest.approx([-0.96, 0.48, 0.48])

    def test_toggle_switches_summary_chart_and_colours(self, app):
        from quantui.app_analysis import pop_mulliken
        from quantui.populations_overlay import (
            charge_colors,
            displayed_charges,
            overlay_payload,
        )

        pop_mulliken(app, _ctx(_sp_result()))
        assert "Sum of Mulliken charges" in app._mulliken_summary.value
        mull_colors = overlay_payload(app, _water_molecule())["colors"]

        app._pop_scheme_tb.value = "npa"  # observer re-renders
        assert "Sum of NPA charges" in app._mulliken_summary.value
        assert displayed_charges(app) == pytest.approx([-0.96, 0.48, 0.48])
        assert app._last_mulliken_fig.layout.yaxis.title.text == "NPA charge (e)"
        npa_colors = overlay_payload(app, _water_molecule())["colors"]
        assert npa_colors == charge_colors([-0.96, 0.48, 0.48])
        assert mull_colors == charge_colors([-0.897, 0.449, 0.449])

    def test_missing_npa_disables_toggle_with_reason(self, app):
        from quantui.app_analysis import pop_mulliken

        app._pop_scheme_tb.value = "npa"
        assert pop_mulliken(app, _ctx(_sp_result(npa_charges=None))) is True
        assert app._pop_scheme_tb.disabled is True
        assert app._pop_scheme_tb.value == "mulliken"
        assert "could not be computed" in app._pop_scheme_note.value
        assert "NPA (e)" not in app._mulliken_table.value
        assert "Charge (e)" in app._mulliken_table.value

    def test_non_pyscf_engine_says_not_available(self, app):
        from quantui.app_analysis import pop_mulliken

        res = _sp_result(npa_charges=None, engine_id="pyfock")
        assert pop_mulliken(app, _ctx(res)) is True
        assert "not available for the pyfock engine" in app._pop_scheme_note.value

    def test_post_hf_labels_reference_density(self, app):
        from quantui.app_analysis import pop_mulliken

        pop_mulliken(app, _ctx(_sp_result(method="MP2")))
        assert "HF reference density" in app._mulliken_summary.value
        pop_mulliken(app, _ctx(_sp_result(method="RHF")))
        assert "HF reference density" not in app._mulliken_summary.value

    def test_history_round_trip(self, tmp_path, app):
        saved = save_result(
            _sp_result(), results_dir=tmp_path, calc_type="single_point"
        )
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Populations" in app._ana_available
        assert app._last_npa_charges == pytest.approx([-0.96, 0.48, 0.48])
        assert "NPA (e)" in app._mulliken_table.value

    def test_history_without_npa_explains(self, tmp_path, app):
        res = _sp_result()
        del res.npa_charges
        saved = save_result(res, results_dir=tmp_path, calc_type="single_point")
        app._apply_analysis_context(app._build_history_context(saved))
        assert "Populations" in app._ana_available  # Mulliken still shown
        assert app._last_npa_charges is None
        assert "not saved with this History result" in app._pop_scheme_note.value

    def test_context_switch_clears_npa(self, app):
        from quantui.app import _AnalysisContext
        from quantui.app_analysis import pop_mulliken

        pop_mulliken(app, _ctx(_sp_result()))
        assert app._last_npa_charges is not None
        app._apply_analysis_context(
            _AnalysisContext(
                calc_type="single_point",
                formula="H2O",
                method="RHF",
                basis="6-31G*",
                live_result=_sp_result(mulliken_charges=None, npa_charges=None),
                molecule=_water_molecule(),
            )
        )
        assert app._last_npa_charges is None


def test_help_topic_explains_npa_vs_mulliken():
    from quantui.help_content import HELP_TOPICS

    body = HELP_TOPICS["mulliken"]["body"]
    assert "natural atomic orbitals" in body
    assert "not</i> produce natural" in body or "does <i>not</i>" in body
    assert "basis-set dependent" in body
