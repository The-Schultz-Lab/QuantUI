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

    def test_rdkit_absent_is_non_destructive_single_frame(self, monkeypatch):
        import quantui.preopt as preopt_mod

        monkeypatch.setattr(preopt_mod, "_RDKIT_AVAILABLE", False)
        original = _water()
        mol, rmsd, frames = preopt_mod.preoptimize_with_trajectory(original)
        assert rmsd == 0.0
        assert mol.coordinates == original.coordinates
        assert len(frames) == 1  # just the input geometry


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
