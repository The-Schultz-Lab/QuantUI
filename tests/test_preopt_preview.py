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

    def test_rdkit_absent_is_non_destructive_single_frame(self, monkeypatch):
        import quantui.preopt as preopt_mod

        monkeypatch.setattr(preopt_mod, "_RDKIT_AVAILABLE", False)
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

    def test_accept_sets_molecule_and_unchecks_autopreopt(self, app):
        relaxed = _water()
        app._preopt_relaxed_mol = relaxed
        app.preopt_cb.value = True
        app.preopt_preview_box.layout.display = ""

        app._on_preopt_accept()

        assert app._molecule is relaxed
        assert app.preopt_cb.value is False  # decoupled: no redundant re-opt
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
