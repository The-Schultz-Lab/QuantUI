"""Tests for the interactive pre-optimization preview (M-PREOPT PREOPT.2/.3).

Covers the trajectory-capturing backend, the animation renderer, and the
preview / keep / revert handlers. Platform-independent (RDKit, no PySCF).
"""

from __future__ import annotations

import pytest

from quantui.molecule import Molecule


def _water() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )


def _stretched_water() -> Molecule:
    # Mildly stretched O–H (~1.05 Å) — bond-perceivable, so the FF engages.
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.81, 0.67, 0.0], [-0.81, 0.67, 0.0]],
    )


def _embed_smiles(smiles: str):
    """RDKit-embed a SMILES into a Molecule, or None if RDKit/embedding fails."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(m, randomSeed=1) != 0:
        return None
    conf = m.GetConformer()
    return Molecule(
        atoms=[a.GetSymbol() for a in m.GetAtoms()],
        coordinates=[
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(m.GetNumAtoms())
        ],
    )


def _metal_complex() -> Molecule:
    """A bundled coordination complex (cisplatin) — RDKit can't perceive its
    metal bonds, so the classical FF has no model for it (M-METAL MET.4)."""
    from quantui import molecule_library as ml

    e = next(x for x in ml.iter_entries() if x["id"] == "inorganic-cisplatin")
    return Molecule(
        atoms=e["atoms"],
        coordinates=e["coordinates"],
        charge=e["charge"],
        multiplicity=e["multiplicity"],
    )


# ── Backend: preoptimize_with_trajectory ────────────────────────────────────


class TestPreoptTrajectory:
    def test_returns_molecule_rmsd_and_frames(self):
        from quantui.preopt import _RDKIT_AVAILABLE, preoptimize_with_trajectory

        if not _RDKIT_AVAILABLE:
            pytest.skip("rdkit not installed")
        mol, rmsd, frames = preoptimize_with_trajectory(_water())
        assert isinstance(mol, Molecule)
        assert isinstance(rmsd, float) and rmsd >= 0.0
        assert isinstance(frames, list) and len(frames) >= 1
        # Every frame is N atoms × 3 coords.
        for fr in frames:
            assert len(fr) == 3
            assert all(len(xyz) == 3 for xyz in fr)
        # First frame is the input geometry.
        assert frames[0][0] == pytest.approx([0.0, 0.0, 0.0])

    def test_distorted_input_produces_multi_frame_relaxation(self):
        from quantui.preopt import _RDKIT_AVAILABLE, preoptimize_with_trajectory

        if not _RDKIT_AVAILABLE:
            pytest.skip("rdkit not installed")
        mol, rmsd, frames = preoptimize_with_trajectory(_stretched_water())
        assert rmsd > 0.0
        assert len(frames) >= 2  # actually relaxed over multiple steps

    def test_trajectory_ends_at_kept_geometry(self):
        # The last frame must equal the geometry "Keep" adopts (the returned
        # molecule) — otherwise the animation lies about what you're keeping.
        from quantui.preopt import _RDKIT_AVAILABLE, preoptimize_with_trajectory

        if not _RDKIT_AVAILABLE:
            pytest.skip("rdkit not installed")
        import numpy as np

        mol, rmsd, frames = preoptimize_with_trajectory(_stretched_water())
        assert np.allclose(frames[-1], mol.coordinates, atol=1e-6)

    def test_relaxation_is_gradual_no_dominating_jump(self):
        # Regression: capturing single Minimize(maxIts=1) steps restarted RDKit's
        # BFGS each call, so ~80% of the motion landed in one final bulk frame —
        # the animation looked static then snapped. The fresh-from-input even-
        # RMSD capture must spread the motion across frames instead. Uses a
        # medium flexible molecule (a tiny one like water genuinely relaxes in a
        # couple of iterations, so its motion can't be subdivided).
        import numpy as np

        from quantui.preopt import _RDKIT_AVAILABLE, preoptimize_with_trajectory

        if not _RDKIT_AVAILABLE:
            pytest.skip("rdkit not installed")
        mol = _embed_smiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # ibuprofen
        if mol is None:
            pytest.skip("could not embed test molecule")
        _, rmsd, frames = preoptimize_with_trajectory(mol)
        if rmsd < 0.05 or len(frames) < 4:
            pytest.skip("too little motion to assess smoothness")
        fr = [np.asarray(f, dtype=float) for f in frames]
        steps = [
            float(np.sqrt(np.mean(np.sum((fr[i + 1] - fr[i]) ** 2, axis=1))))
            for i in range(len(fr) - 1)
        ]
        total = float(np.sqrt(np.mean(np.sum((fr[-1] - fr[0]) ** 2, axis=1))))
        # No single inter-frame step may carry more than half the total motion
        # (the regressed capture put ~80% in one frame).
        assert max(steps) <= 0.5 * total

    def test_no_backend_is_non_destructive_single_frame(self, monkeypatch):
        import quantui.preopt as preopt_mod

        monkeypatch.setattr(preopt_mod, "_RDKIT_AVAILABLE", False)
        monkeypatch.setattr(preopt_mod, "_XTB_AVAILABLE", False)
        original = _water()
        mol, rmsd, frames = preopt_mod.preoptimize_with_trajectory(original)
        assert rmsd == 0.0
        assert mol.coordinates == original.coordinates
        assert len(frames) == 1  # just the input geometry


# ── Even-RMSD frame selection (platform-independent) ────────────────────────


class TestEvenRmsdSelection:
    def test_back_loaded_path_selects_evenly(self):
        # A 1-atom "trajectory" that barely moves early then rushes late (exactly
        # the RDKit BFGS profile). Even-RMSD selection should sample the active
        # late region densely so playback steps are roughly equal.

        from quantui.preopt import _select_even_rmsd_frames

        n = 120
        xs = [10.0 * (1.0 - (i / (n - 1)) ** 3) for i in range(n)]  # late motion
        waypoints = [[[x, 0.0, 0.0]] for x in xs]
        frames = _select_even_rmsd_frames(waypoints, 11)

        assert frames[0][0] == pytest.approx([10.0, 0.0, 0.0])  # input first
        assert frames[-1][0] == pytest.approx([0.0, 0.0, 0.0])  # relaxed last
        gaps = [
            abs(frames[i + 1][0][0] - frames[i][0][0]) for i in range(len(frames) - 1)
        ]
        mean_gap = sum(gaps) / len(gaps)
        # Even spacing: no gap is wildly larger than the mean (vs the raw
        # iteration-spaced sampling, which would crowd the static early region).
        assert max(gaps) <= 1.8 * mean_gap
        # Strictly monotone toward the final geometry.
        assert all(gaps[i] > 0 for i in range(len(gaps)))

    def test_no_motion_collapses_to_single_frame(self):
        from quantui.preopt import _select_even_rmsd_frames

        same = [[0.0, 0.0, 0.0]]
        waypoints = [list(map(list, same)) for _ in range(10)]
        frames = _select_even_rmsd_frames(waypoints, 20)
        assert len(frames) == 1


# ── Renderer: build_preopt_preview_html ─────────────────────────────────────


class TestPreviewRenderer:
    def test_html_is_offline_and_multi_frame(self):
        pytest.importorskip("py3Dmol")
        from quantui.app_visualization import build_preopt_preview_html

        atoms = ["O", "H", "H"]
        frames = [
            [[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]],
            [[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
        ]
        html = build_preopt_preview_html(atoms, frames)
        assert "cdn.jsdelivr.net" not in html  # offline-safe (vendored 3Dmol)
        assert "addModelsAsFrames" in html

    def test_multi_frame_has_interactive_stepper(self):
        pytest.importorskip("py3Dmol")
        from quantui.app_visualization import build_preopt_preview_html

        atoms = ["O", "H", "H"]
        # Three frames → real relaxation → stepper controls are wired.
        frames = [
            [[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]],
            [[0, 0, 0], [0.98, 0, 0], [0, 0.98, 0]],
            [[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
        ]
        html = build_preopt_preview_html(atoms, frames)
        # Frame navigation is driven client-side via setFrame on the already-
        # loaded multi-frame view (no per-frame HTML rebuild).
        assert "setFrame" in html
        assert 'type="range"' in html  # scrub slider
        assert "Show input" in html  # input <-> relaxed A/B flip
        # Slider spans all frames (0 .. n-1).
        assert 'max="2"' in html

    def test_single_frame_renders_without_controls(self):
        pytest.importorskip("py3Dmol")
        from quantui.app_visualization import build_preopt_preview_html

        html = build_preopt_preview_html(["H"], [[[0, 0, 0]]])
        assert "cdn.jsdelivr.net" not in html
        # Nothing to step through → no stepper controls.
        assert "setFrame" not in html


# ── Handlers: preview / keep / revert ───────────────────────────────────────


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    from quantui.app import QuantUIApp

    return QuantUIApp()


class TestPreviewHandlers:
    def test_preview_without_molecule_prompts(self, app):
        app._molecule = None
        app._on_preopt_preview()
        assert "molecule" in app.preopt_preview_status.value.lower()

    def test_preview_done_reveals_keep_revert(self, app):
        pytest.importorskip("py3Dmol")
        from quantui.app_runflow import _preopt_preview_done

        relaxed = _water()
        frames = [[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]]
        _preopt_preview_done(app, relaxed, 0.123, frames)

        assert app._preopt_relaxed_mol is relaxed
        assert app.preopt_preview_box.layout.display == ""
        assert app.preopt_accept_btn.disabled is False
        assert app.preopt_reset_btn.disabled is False
        assert "0.123" in app.preopt_preview_status.value

    def test_negligible_change_suppresses_animation_and_buttons(self, app):
        # User report: a preview that shows a molecule sitting still, plus a
        # Keep/Revert choice between two effectively identical geometries, reads
        # as "something happened, now judge it" when the honest answer is "your
        # geometry was already fine".
        from quantui.app_runflow import _PREOPT_NEGLIGIBLE_RMSD_A, _preopt_preview_done

        relaxed = _water()
        frames = [[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]]
        _preopt_preview_done(app, relaxed, _PREOPT_NEGLIGIBLE_RMSD_A / 2, frames)

        assert app.preopt_preview_output.layout.display == "none"
        assert app._preopt_actions_box.layout.display == "none"
        assert app.preopt_accept_btn.disabled is True
        assert app.preopt_reset_btn.disabled is True
        # Nothing to accept, so Keep must not be armed even if it were clicked.
        assert app._preopt_relaxed_mol is None

    def test_negligible_change_still_explains_itself(self, app):
        from quantui.app_runflow import _preopt_preview_done

        _preopt_preview_done(app, _water(), 0.004, [[[0, 0, 0]]])

        status = app.preopt_preview_status.value.lower()
        assert app.preopt_preview_box.layout.display == ""  # message is visible
        assert "no meaningful change" in status
        assert "0.004" in app.preopt_preview_status.value  # the actual number
        assert "as-is" in status  # says what will happen next

    def test_threshold_boundary_suppresses(self, app):
        # Exactly at the threshold counts as negligible (<=), so the boundary
        # cannot produce a preview of an imperceptible motion.
        from quantui.app_runflow import _PREOPT_NEGLIGIBLE_RMSD_A, _preopt_preview_done

        _preopt_preview_done(app, _water(), _PREOPT_NEGLIGIBLE_RMSD_A, [[[0, 0, 0]]])
        assert app._preopt_actions_box.layout.display == "none"

    def test_just_above_threshold_shows_the_preview(self, app):
        pytest.importorskip("py3Dmol")
        from quantui.app_runflow import _PREOPT_NEGLIGIBLE_RMSD_A, _preopt_preview_done

        relaxed = _water()
        frames = [[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]]
        _preopt_preview_done(app, relaxed, _PREOPT_NEGLIGIBLE_RMSD_A + 0.01, frames)

        assert app._preopt_actions_box.layout.display == ""
        assert app.preopt_accept_btn.disabled is False
        assert app._preopt_relaxed_mol is relaxed

    def test_meaningful_preview_restores_panes_hidden_by_a_prior_run(self, app):
        # A negligible preview hides the panes; the next meaningful one must put
        # them back, or Keep/Revert would be permanently invisible.
        pytest.importorskip("py3Dmol")
        from quantui.app_runflow import _preopt_preview_done

        _preopt_preview_done(app, _water(), 0.001, [[[0, 0, 0]]])
        assert app._preopt_actions_box.layout.display == "none"

        frames = [[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]]
        _preopt_preview_done(app, _water(), 0.4, frames)
        assert app._preopt_actions_box.layout.display == ""
        assert app.preopt_preview_output.layout.display == ""

    def test_accept_sets_molecule_and_hides_preview(self, app):
        # Pre-opt is Preview-only: Keep makes the relaxed geometry the active
        # molecule (which the run then uses as-is). There is no checkbox.
        relaxed = _water()
        app._preopt_relaxed_mol = relaxed
        app.preopt_preview_box.layout.display = ""

        app._on_preopt_accept()

        assert app._molecule is relaxed
        assert not hasattr(app, "preopt_cb")  # classical-preopt checkbox removed
        assert app._preopt_relaxed_mol is None
        assert app.preopt_preview_box.layout.display == "none"

    def test_revert_discards_without_changing_molecule(self, app):
        original = _water()
        app._set_molecule(original, "orig")
        relaxed = _stretched_water()
        app._preopt_relaxed_mol = relaxed
        app.preopt_preview_box.layout.display = ""

        app._on_preopt_reset()

        assert app._molecule is original  # unchanged
        assert app._preopt_relaxed_mol is None
        assert app.preopt_preview_box.layout.display == "none"


class TestMetalPreoptHonesty:
    """M-METAL MET.4: a 0 Å pre-opt on a metal complex is a *failure* to build a
    force-field model, not a benign "your geometry is already reasonable"."""

    def test_preopt_support_none_for_organic(self):
        from quantui.preopt import _RDKIT_AVAILABLE, preopt_support

        if not _RDKIT_AVAILABLE:
            pytest.skip("rdkit not installed")
        assert preopt_support(_water()) is None

    def test_preopt_support_reason_for_metal_without_xtb(self, monkeypatch):
        # With the GFN-FF (xtb) backend absent, a metal has no pre-opt backend,
        # so preopt_support returns a reason that points to xtb / the DFT opt.
        import quantui.preopt as preopt_mod

        if not preopt_mod._RDKIT_AVAILABLE:
            pytest.skip("rdkit not installed")
        monkeypatch.setattr(preopt_mod, "_XTB_AVAILABLE", False)
        reason = preopt_mod.preopt_support(_metal_complex())
        assert reason is not None and "xtb" in reason.lower()

    def test_metal_negligible_change_reports_honestly_without_xtb(
        self, app, monkeypatch
    ):
        # With no GFN-FF backend, a metal FF no-op at RMSD 0.0 must NOT claim the
        # geometry is "already reasonable"; it must point to xtb / the DFT opt.
        import quantui.preopt as preopt_mod
        from quantui.app_runflow import _preopt_preview_done

        if not preopt_mod._RDKIT_AVAILABLE:
            pytest.skip("rdkit not installed")
        monkeypatch.setattr(preopt_mod, "_XTB_AVAILABLE", False)
        metal = _metal_complex()
        _preopt_preview_done(app, metal, 0.0, [[list(c) for c in metal.coordinates]])

        status = app.preopt_preview_status.value.lower()
        assert "geometry optimization" in status
        assert "isn't available" in status or "not available" in status
        assert "already reasonable" not in status
        assert app._preopt_relaxed_mol is None
        assert app._preopt_actions_box.layout.display == "none"

    def test_organic_negligible_change_still_says_already_reasonable(self, app):
        # The honest-metal path must not regress the organic no-op wording.
        from quantui.app_runflow import _preopt_preview_done
        from quantui.preopt import _RDKIT_AVAILABLE

        if not _RDKIT_AVAILABLE:
            pytest.skip("rdkit not installed")
        _preopt_preview_done(app, _water(), 0.01, [[[0, 0, 0]]])
        status = app.preopt_preview_status.value.lower()
        assert "already reasonable" in status
        assert "no meaningful change" in status


class TestStaleRunStatus:
    """Regression: 'Pre-optimized geometry accepted.' lingered next to Run
    after switching molecules / reverting a later preview."""

    def test_loading_molecule_clears_stale_run_status(self, app):
        app._calc_running = False
        app.run_status.value = "Pre-optimized geometry accepted."
        app._set_molecule(_water(), "new mol")
        assert app.run_status.value == ""

    def test_loading_molecule_midrun_keeps_run_status(self, app):
        # The mid-run pre-opt sets the molecule via _set_molecule; it must NOT
        # wipe the live "Pre-optimizing…" status.
        app._calc_running = True
        app.run_status.value = "Pre-optimizing..."
        app._set_molecule(_water(), "preopt mid-run")
        assert app.run_status.value == "Pre-optimizing..."

    def test_revert_clears_stale_accepted_status(self, app):
        app._set_molecule(_water(), "orig")
        app.run_status.value = "Pre-optimized geometry accepted."
        app._on_preopt_reset()
        assert app.run_status.value == ""
