"""Transition-state guess from reactant + product (M-TS TS.4, QST2-style)."""

from __future__ import annotations

import io
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quantui.molecule import Molecule
from quantui.ts_guess import (
    CLASH_DISTANCE_ANG,
    DEFAULT_N_IMAGES,
    TSGuessResult,
    align_to_reference,
    build_ts_guess,
    interpolate_path,
    path_fractions,
    select_guess,
    validate_endpoints,
)

# HCN -> HNC, atoms in the order C, N, H. Both slightly bent on the same
# side: two exactly collinear endpoints would keep every image on the line.
HCN = Molecule(["C", "N", "H"], [[0, 0, 0], [0, 0, 1.153], [0.18, 0, -1.05]])
HNC = Molecule(["C", "N", "H"], [[0, 0, 0], [0, 0, 1.169], [0.17, 0, 2.15]])
HNC_XYZ = "C 0.0 0.0 0.0\nN 0.0 0.0 1.169\nH 0.17 0.0 2.15"


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _energies(values):
    """energy_fn returning *values* in call order (images are evaluated in
    order, reactant first)."""
    it = iter(values)
    return lambda _mol: next(it)


class TestValidation:
    def test_ok(self):
        validate_endpoints(HCN, HNC)

    def test_atom_count(self):
        p = Molecule(["C", "N"], [[0, 0, 0], [0, 0, 1.1]], validate_spin=False)
        with pytest.raises(ValueError, match="3 atoms but the product has 2"):
            validate_endpoints(HCN, p)

    def test_different_elements(self):
        p = Molecule(
            ["C", "O", "H"], [[0, 0, 0], [0, 0, 1.2], [0, 0, 2.2]], validate_spin=False
        )
        with pytest.raises(ValueError, match="contain different atoms"):
            validate_endpoints(HCN, p)

    def test_different_order(self):
        p = Molecule(["C", "H", "N"], [[0, 0, 0], [0, 0, 2.2], [0, 0, 1.17]])
        with pytest.raises(ValueError, match=r"different order .*atom 2, 3"):
            validate_endpoints(HCN, p)

    def test_charge_and_multiplicity(self):
        p = Molecule(HNC.atoms, HNC.coordinates, charge=1, validate_spin=False)
        with pytest.raises(ValueError, match="charge"):
            validate_endpoints(HCN, p)
        p = Molecule(HNC.atoms, HNC.coordinates, multiplicity=3)
        with pytest.raises(ValueError, match="multiplicity"):
            validate_endpoints(HCN, p)

    def test_interpolate_validates_first(self):
        p = Molecule(["C", "H", "N"], [[0, 0, 0], [0, 0, 2.2], [0, 0, 1.17]])
        with pytest.raises(ValueError, match="different order"):
            interpolate_path(HCN, p)

    def test_same_structure_is_refused(self):
        moved = np.asarray(HCN.coordinates) @ _rotation([1, 2, 3], 0.7).T + [4, -1, 2]
        with pytest.raises(ValueError, match="same structure"):
            interpolate_path(HCN, Molecule(HCN.atoms, moved.tolist()))

    @pytest.mark.parametrize("n", [0, 16])
    def test_image_count_bounds(self, n):
        with pytest.raises(ValueError, match="Number of images"):
            interpolate_path(HCN, HNC, n)


class TestAlignment:
    def test_rotated_translated_copy_superimposes(self):
        ref = np.array([[0, 0, 0], [1.1, 0, 0], [0, 1.3, 0.2], [0.4, 0.3, 1.0]])
        mob = ref @ _rotation([0.3, -1, 0.5], 2.1).T + [3.0, -2.0, 0.5]
        moved, rmsd = align_to_reference(ref, mob)
        assert rmsd == pytest.approx(0.0, abs=1e-8)
        assert moved == pytest.approx(ref, abs=1e-8)

    def test_never_reflects(self):
        # A chiral arrangement and its mirror image cannot be superimposed by
        # a proper rotation.
        ref = np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.2, 0], [0, 0, 1.4]])
        mirror = ref * [1, 1, -1]
        _moved, rmsd = align_to_reference(ref, mirror)
        assert rmsd > 0.1


class TestInterpolation:
    def test_images_and_endpoints(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Product rotated and shifted: the alignment removes that motion.
        moved = np.asarray(HNC.coordinates) @ _rotation([0, 1, 1], 1.0).T + [2, 2, 2]
        product = Molecule(HNC.atoms, moved.tolist())
        images, rmsd, used, warnings = interpolate_path(HCN, product, 5)
        assert used == "idpp" and len(images) == 7 and warnings == []
        np.testing.assert_allclose(images[0].coordinates, HCN.coordinates)
        aligned, rmsd_ref = align_to_reference(HCN.coordinates, HNC.coordinates)
        assert rmsd == pytest.approx(rmsd_ref)
        np.testing.assert_allclose(images[-1].coordinates, aligned, atol=1e-8)
        assert all(m.atoms == HCN.atoms for m in images)
        # ASE's IDPP defaults would write idpp.traj / idpp.log here.
        assert list(tmp_path.iterdir()) == []

    def test_idpp_bridges_instead_of_passing_through(self):
        # IDPP pulls the middle image's distances toward the average of the
        # endpoints': H about 1.6 Å from both C and N, i.e. a triangle with
        # H well off the C-N line (a straight-line midpoint has H 0.17 Å
        # off it, nearly on top of the C-N bond).
        images, *_ = interpolate_path(HCN, HNC, 3)
        c, n, h = (np.asarray(x) for x in images[2].coordinates)
        assert np.linalg.norm(h - c) == pytest.approx(1.61, abs=0.05)
        assert np.linalg.norm(h - n) == pytest.approx(1.61, abs=0.05)
        axis = (n - c) / np.linalg.norm(n - c)
        off_line = np.linalg.norm((h - c) - np.dot(h - c, axis) * axis)
        assert off_line > 1.2

    def test_clash_warning_for_atoms_passing_through(self):
        # Collinear endpoints: H must pass through C and N on a straight line.
        r = Molecule(["C", "N", "H"], [[0, 0, 0], [0, 0, 1.15], [0, 0, -1.06]])
        p = Molecule(["C", "N", "H"], [[0, 0, 0], [0, 0, 1.17], [0, 0, 2.16]])
        _images, _rmsd, used, warnings = interpolate_path(r, p, 7, method="linear")
        assert used == "linear"
        assert any("within" in w and "order" in w for w in warnings)
        assert CLASH_DISTANCE_ANG == 0.5

    def test_idpp_failure_falls_back_to_linear(self, monkeypatch):
        from quantui import ts_guess

        def _boom():
            def _f(*_a, **_k):
                raise RuntimeError("no")

            return _f

        monkeypatch.setattr(ts_guess, "_idpp_interpolate_fn", _boom)
        images, _rmsd, used, warnings = interpolate_path(HCN, HNC, 3)
        assert used == "linear" and "IDPP interpolation failed" in warnings[0]
        mid = (
            np.asarray(images[0].coordinates) + np.asarray(images[-1].coordinates)
        ) / 2
        np.testing.assert_allclose(images[2].coordinates, mid, atol=1e-12)

    def test_path_fractions(self):
        mols = [
            Molecule(["H"], [[x, 0, 0]], validate_spin=False)
            for x in (0.0, 1.0, 1.5, 3.0)
        ]
        assert path_fractions(mols) == pytest.approx([0, 1 / 3, 0.5, 1])
        same = [Molecule(["H"], [[0, 0, 0]], validate_spin=False)] * 3
        assert path_fractions(same) == pytest.approx([0, 0.5, 1])


class TestSelection:
    def test_highest_interior(self):
        assert select_guess([-1.0, -0.8, -0.5, -0.7, -1.1]) == 2

    def test_endpoints_never_chosen(self):
        assert select_guess([5.0, -0.8, -0.5, -0.7, 9.0]) == 2

    def test_failed_images_skipped(self):
        assert select_guess([-1.0, float("nan"), -0.9, None, -1.1]) == 2

    def test_no_interior_energy(self):
        with pytest.raises(ValueError, match="no transition-state guess"):
            select_guess([-1.0, float("nan"), -1.0])


class TestBuildGuess:
    def test_profile_and_choice_with_mock_energies(self):
        log = io.StringIO()
        e = [-1.0, -0.99, -0.97, -0.95, -0.96, -0.98, -1.01]
        g = build_ts_guess(
            HCN, HNC, "RHF", "STO-3G", n_images=5,
            progress_stream=log, energy_fn=_energies(e),
        )  # fmt: skip
        assert isinstance(g, TSGuessResult)
        assert g.n_images == 5 and g.guess_index == 3
        assert g.guess is g.images[3] and g.guess_energy_hartree == -0.95
        assert g.barrier_estimate_kcal == pytest.approx(0.05 * 627.509474)
        assert g.warnings == []
        out = log.getvalue()
        assert "IDPP interpolation" in out and "highest image: TS guess" in out
        assert "image 3" in out

        block = g.to_spectra()
        json.dumps(block)  # stored in result.json
        assert block["guess_index"] == 3 and block["n_images"] == 5
        assert block["energies_hartree"] == e
        assert block["path_fraction"][0] == 0 and block["path_fraction"][-1] == 1

    def test_default_image_count(self):
        g = build_ts_guess(
            HCN, HNC, progress_stream=io.StringIO(), energy_fn=lambda m: 0.0
        )
        assert g.n_images == DEFAULT_N_IMAGES == 7 and len(g.images) == 9

    def test_no_barrier_on_path_is_flagged(self):
        g = build_ts_guess(
            HCN, HNC, n_images=3, progress_stream=io.StringIO(),
            energy_fn=_energies([-1.0, -0.9, -0.8, -0.7, -0.6]),
        )  # fmt: skip
        assert g.guess_index == 3
        assert any("product is the highest point" in w for w in g.warnings)

    def test_failed_single_point_is_skipped(self):
        calls = {"n": 0}

        def _fn(_mol):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("SCF did not converge")
            return [-1.0, -0.96, None, -0.95, -1.0][calls["n"] - 1]

        g = build_ts_guess(
            HCN, HNC, n_images=3, progress_stream=io.StringIO(), energy_fn=_fn
        )
        assert math.isnan(g.energies_hartree[2]) and g.guess_index == 3
        assert any("failed at image(s) 2" in w for w in g.warnings)
        assert g.to_spectra()["energies_hartree"][2] is None

    def test_all_interior_failed(self):
        nan = float("nan")
        with pytest.raises(ValueError, match="no transition-state guess"):
            build_ts_guess(
                HCN, HNC, n_images=2, progress_stream=io.StringIO(),
                energy_fn=_energies([-1.0, nan, nan, -1.0]),
            )  # fmt: skip

    def test_cancel(self):
        from quantui.cancellation import CalcCancelled

        with pytest.raises(CalcCancelled):
            build_ts_guess(
                HCN, HNC, progress_stream=io.StringIO(),
                energy_fn=lambda m: 0.0, cancel_check=lambda: True,
            )  # fmt: skip


class TestSearchPrecheck:
    def _never(self, _mol):
        raise AssertionError("no single point should run")

    def test_post_hf_refused_before_single_points(self):
        from quantui.ts_guess import run_ts_search_from_endpoints

        with pytest.raises(ValueError, match="analytic gradients"):
            run_ts_search_from_endpoints(HCN, HNC, "MP2", energy_fn=self._never)

    def test_missing_sella_refused_before_single_points(self, monkeypatch):
        from quantui.ts_guess import run_ts_search_from_endpoints

        monkeypatch.setitem(sys.modules, "sella", None)
        with pytest.raises(ImportError, match=r"quantui\[ts\]"):
            run_ts_search_from_endpoints(HCN, HNC, "RHF", energy_fn=self._never)


class TestCardAndPlot:
    BLOCK = {
        "interpolation": "idpp",
        "n_images": 3,
        "energies_hartree": [-1.0, -0.95, -0.9, None, -1.02],
        "path_fraction": [0.0, 0.2, 0.5, 0.8, 1.0],
        "guess_index": 2,
        "barrier_estimate_kcal": 62.75,
        "warnings": ["a <warning>"],
    }

    def test_card_row(self):
        from quantui.app_formatters import ts_guess_rows

        assert ts_guess_rows(None) == ""
        row = ts_guess_rows(self.BLOCK)
        assert "Starting guess" in row and "2 of 3" in row and "62.8" in row
        assert "a &lt;warning&gt;" in row

    def test_profile_panel_from_history_block(self):
        from quantui.app import QuantUIApp, _AnalysisContext

        app = QuantUIApp()
        ctx = _AnalysisContext(
            calc_type="transition_state", formula="CHN", method="RHF",
            basis="STO-3G", spectra_data={"ts_guess": self.BLOCK},
            source="history",
        )  # fmt: skip
        assert app._pop_ts_guess_profile(ctx) is True
        res = app._last_pes_result
        assert res.highlight_index == 2 and res.link_trajectory is False
        assert res.energies_relative_kcal[0] == 0.0
        assert math.isnan(res.energies_hartree[3])
        fig = app._last_pes_fig
        names = [t.name for t in fig.data]
        assert "TS guess (highest image)" in names and "Minimum" not in names
        star = fig.data[names.index("TS guess (highest image)")]
        assert list(star.x) == [0.5]

    def test_profile_panel_absent_for_plain_search(self):
        from quantui.app import QuantUIApp, _AnalysisContext

        app = QuantUIApp()
        ctx = _AnalysisContext(
            calc_type="transition_state", formula="CHN", method="RHF",
            basis="STO-3G", live_result=SimpleNamespace(guess=None),
        )  # fmt: skip
        assert app._pop_ts_guess_profile(ctx) is False
        assert "Product structure" in app._ana_unavail_msgs["PES Scan"].value


def test_help_topic_explains_the_option():
    from quantui.help_content import HELP_TOPICS

    body = HELP_TOPICS["transition_state"]["body"]
    assert "Starting from reactant + product" in body and "IDPP" in body
    assert "in the same order" in body and "<b>guess</b>" in body
    assert "one imaginary frequency" in body.split("<b>Limits.</b>")[1]


class TestAppInput:
    def test_product_widgets(self):
        from quantui.app import QuantUIApp
        from quantui.app_runflow import TS_PRODUCT_XYZ

        app = QuantUIApp()
        app.calc_type_dd.value = "Transition State"
        flat = []
        for kid in app.calc_extra_opts.children:
            flat.append(kid)
            flat.extend(getattr(kid, "children", ()))
        for w in ("_ts_product_dd", "_ts_product_xyz", "_ts_n_images_si"):
            assert getattr(app, w) in flat, w
        assert app._ts_product_dd.value == ""
        assert app._ts_product_xyz.layout.display == "none"
        assert app._ts_n_images_si.layout.display == "none"
        assert app._ts_n_images_si.value == DEFAULT_N_IMAGES
        app._ts_product_dd.value = TS_PRODUCT_XYZ
        assert app._ts_product_xyz.layout.display == ""
        assert app._ts_n_images_si.layout.display == ""
        assert "highest-energy image" in app._ts_product_note.value
        app._ts_product_dd.value = ""
        assert app._ts_product_xyz.layout.display == "none"

    def test_resolve_product(self):
        from quantui.app import QuantUIApp
        from quantui.app_runflow import TS_PRODUCT_XYZ, resolve_ts_product

        app = QuantUIApp()
        assert resolve_ts_product(app, HCN) is None
        app._ts_product_dd.value = TS_PRODUCT_XYZ
        with pytest.raises(ValueError, match="box is empty"):
            resolve_ts_product(app, HCN)
        app._ts_product_xyz.value = HNC_XYZ
        log = io.StringIO()
        p = resolve_ts_product(app, HCN, log=log)
        assert p.atoms == ["C", "N", "H"] and p.coordinates[2] == [0.17, 0.0, 2.15]
        assert (p.charge, p.multiplicity) == (HCN.charge, HCN.multiplicity)
        assert "pasted XYZ" in log.getvalue()

    def test_history_products_listed(self, tmp_path, monkeypatch):
        from quantui import save_result
        from quantui.app import QuantUIApp
        from quantui.results_storage import save_trajectory

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))

        def _opt(mol):
            res = SimpleNamespace(
                formula=mol.get_formula(), method="RHF", basis="STO-3G",
                energy_hartree=-91.6, homo_lumo_gap_ev=None, converged=True,
                n_iterations=1,
            )  # fmt: skip
            d = save_result(res, calc_type="geometry_opt", molecule=mol)
            save_trajectory(d, [mol], [-91.6])
            return d

        d_reactant = _opt(HCN)
        d_product = _opt(HNC)
        reordered = Molecule(["H", "C", "N"], [[0, 0, 2.15], [0, 0, 0], [0, 0, 1.17]])
        d_other = _opt(reordered)
        app = QuantUIApp()
        # Loading the reactant while already on Transition State refreshes
        # the list (it depends on the reactant's atoms and geometry).
        app.calc_type_dd.value = "Transition State"
        app._set_molecule(HCN, "reactant")
        values = [v for _l, v in app._ts_product_dd.options]
        assert str(d_product) in values
        assert str(d_reactant) not in values  # the current geometry itself
        assert str(d_other) not in values  # atoms in another order

        from quantui.app_runflow import resolve_ts_product

        app._ts_product_dd.value = str(d_product)
        p = resolve_ts_product(app, HCN)
        np.testing.assert_allclose(p.coordinates, HNC.coordinates)


@pytest.mark.slow
class TestEndToEnd:
    """Real PySCF + Sella: the interpolated guess leads to a true TS."""

    def test_hcn_hnc_from_endpoints(self):
        pytest.importorskip("sella")
        pytest.importorskip("pyscf")
        from quantui.ts_guess import run_ts_search_from_endpoints

        log = io.StringIO()
        r = run_ts_search_from_endpoints(
            HCN, HNC, "RHF", "STO-3G", n_images=5, progress_stream=log
        )
        g = r.guess
        assert 0 < g.guess_index < len(g.images) - 1 and g.warnings == []
        e = g.energies_hartree
        # Reactant (HCN) below product (HNC); the path climbs in between.
        assert e[0] < e[-1] < g.guess_energy_hartree
        assert r.search_converged and r.n_imaginary == 1 and r.is_transition_state
        assert r.imaginary_cm1[0] < -800  # H migrating between C and N
        # The saddle point lies between the minima and below the unrelaxed
        # image: the guess overestimates the barrier.
        assert max(e[0], e[-1]) < r.energy_hartree < g.guess_energy_hartree
        # The search started at the guess.
        np.testing.assert_allclose(r.trajectory[0].coordinates, g.guess.coordinates)
        out = log.getvalue()
        assert out.index("highest image: TS guess") < out.index("Sella")

    def test_app_run_save_and_history(self, tmp_path, monkeypatch):
        pytest.importorskip("sella")
        pytest.importorskip("pyscf")
        from quantui import load_result
        from quantui.app import QuantUIApp
        from quantui.app_formatters import format_past_result
        from quantui.app_runflow import TS_PRODUCT_XYZ

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        app = QuantUIApp()
        app._set_molecule(HCN, "reactant")
        app.calc_type_dd.value = "Transition State"
        app._ts_product_dd.value = TS_PRODUCT_XYZ
        app._ts_product_xyz.value = HNC_XYZ
        app._ts_n_images_si.value = 3
        app._do_run()
        assert app._last_calc_type == "transition_state"
        assert {"Vibrational", "PES Scan", "Trajectory"} <= set(app._ana_available)

        data = load_result(app._last_result_dir)
        assert data["spectra"]["transition_state"]["n_imaginary"] == 1
        block = data["spectra"]["ts_guess"]
        assert block["n_images"] == 3 and len(block["energies_hartree"]) == 5
        card = format_past_result(data, app._last_result_dir)
        assert "Starting guess" in card and "✓ Transition state" in card

        fresh = QuantUIApp()
        fresh._apply_analysis_context(
            fresh._build_history_context(app._last_result_dir)
        )
        assert {"Vibrational", "PES Scan"} <= set(fresh._ana_available)
        assert fresh._last_pes_result.highlight_index == block["guess_index"]
