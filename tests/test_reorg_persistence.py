"""Reorganization-energy results survive History (M-REORG REORG.1/.2/.4/.6).

Reported by a student, 2026-08-05: *"after reorganization energy calcs are
reloaded from history, the actual reorg energy outputs and stuff aren't
displayed in the results tab like they are prior to loading from history."*

It presented as a display bug and was not one. ``save_result`` wrote a fixed
schema of scalar fields with no ``channels`` key, so the λ breakdown was never
persisted — the live card rendered from an object still in memory, and the
History card had nothing behind it. Fixing the formatter alone could not have
worked.

The structural fix is that **both cards now render from the same plain-dict
payload**, so "renders live" implies "renders after reload" rather than
depending on two implementations agreeing. These tests exercise the actual
save → load → render path rather than inspecting either formatter, because
inspecting a formatter is exactly what would have missed this.

Platform-independent: no PySCF, no browser.
"""

from __future__ import annotations

import pytest

from quantui.app_formatters import format_past_result, format_reorg_result
from quantui.molecule import Molecule
from quantui.reorganization_energy import (
    ReorganizationEnergyResult,
    ReorgChannelResult,
    geometry_rmsd,
    max_atom_displacement,
)
from quantui.results_storage import load_result, save_result


@pytest.fixture
def neutral() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )


@pytest.fixture
def cation() -> Molecule:
    # Deliberately displaced from the neutral: the relaxation is the physical
    # content of λ, so a fixture where nothing moved would test nothing.
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [1.03, 0.0, 0.0], [-0.27, 1.00, 0.0]],
        charge=1,
        multiplicity=2,
    )


@pytest.fixture
def result(neutral, cation) -> ReorganizationEnergyResult:
    ch = ReorgChannelResult(
        kind="hole",
        ion_charge=1,
        ion_multiplicity=2,
        e_neutral_at_neutral=-76.4,
        e_ion_at_ion=-76.0,
        e_ion_at_neutral=-75.98,
        e_neutral_at_ion=-76.39,
        lambda1_hartree=0.02,
        lambda2_hartree=0.01,
        lambda_hartree=0.03,
        converged=True,
        ion_molecule=cation,
    )
    return ReorganizationEnergyResult(
        formula="H2O",
        method="B3LYP",
        basis="6-31G*",
        mode="hole",
        molecule=neutral,
        neutral_charge=0,
        neutral_multiplicity=1,
        neutral_energy_hartree=-76.4,
        channels=[ch],
    )


@pytest.fixture
def saved(result, tmp_path, monkeypatch):
    """A genuinely round-tripped result: saved to disk, then loaded back."""
    monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
    d = save_result(result, pyscf_log="", calc_type="reorganization_energy")
    return d, load_result(d)


class TestTheChannelDataSurvivesHistory:
    """The reported bug, tested through the real path."""

    def test_channels_are_persisted_at_all(self, saved):
        _, data = saved
        assert data.get("reorg_channels"), (
            "the λ payload is not being written — this is the bug, and no "
            "amount of formatter work can compensate"
        )

    def test_every_energy_of_the_four_point_scheme_round_trips(self, saved, result):
        _, data = saved
        got = data["reorg_channels"][0]
        src = result.channels[0]
        for field in (
            "e_neutral_at_neutral",
            "e_ion_at_ion",
            "e_ion_at_neutral",
            "e_neutral_at_ion",
            "lambda1_hartree",
            "lambda2_hartree",
            "lambda_hartree",
        ):
            assert got[field] == pytest.approx(getattr(src, field)), field

    def test_the_history_card_shows_lambda(self, saved):
        d, data = saved
        html = format_past_result(data, d)
        assert "λ" in html
        assert "0.8163" in html or "eV" in html

    def test_the_history_card_does_not_claim_data_is_missing(self, saved):
        d, data = saved
        assert "Re-run this calculation" not in format_past_result(data, d)

    def test_both_cards_render_the_same_channel_content(self, saved, result):
        """The structural guarantee. Two renderers drifting IS the bug."""
        d, data = saved
        live = format_reorg_result(result)
        history = format_past_result(data, d)
        for marker in ("λ₁ ion relaxation", "λ₂ neutral relaxation", "Geometry RMSD"):
            assert marker in live, f"live card lost {marker}"
            assert marker in history, f"history card lost {marker}"


class TestGeometriesAreRetained:
    def test_the_ion_geometry_is_saved(self, saved, cation):
        _, data = saved
        geom = data["reorg_channels"][0]["ion_geometry"]
        assert geom["atoms"] == list(cation.atoms)
        assert geom["coordinates"][1] == pytest.approx(list(cation.coordinates[1]))

    def test_the_neutral_geometry_travels_with_the_payload(self, saved):
        # The top-level schema stores atom_symbols but not coordinates, so
        # without this the history card cannot reach R_neutral — and with no
        # R_neutral there is no relaxation to compute.
        _, data = saved
        assert data["reorg_channels"][0].get("neutral_geometry")


class TestRelaxationMetrics:
    def test_rmsd_is_positive_when_the_ion_relaxed(self, neutral, cation):
        assert geometry_rmsd(neutral, cation) > 0

    def test_rmsd_is_zero_for_identical_geometries(self, neutral):
        assert geometry_rmsd(neutral, neutral) == pytest.approx(0.0)

    def test_mismatched_geometries_return_none_rather_than_raising(self, neutral):
        other = Molecule(atoms=["H", "H"], coordinates=[[0, 0, 0], [0, 0, 0.74]])
        assert geometry_rmsd(neutral, other) is None
        assert max_atom_displacement(neutral, other) is None

    def test_the_largest_shift_identifies_the_right_atom(self, neutral, cation):
        # RMSD averages the relaxation away; a single atom moving far in an
        # otherwise rigid molecule is the interesting case.
        idx, dist = max_atom_displacement(neutral, cation)
        assert idx == 2  # the O-H that moved furthest in the fixture
        assert dist > geometry_rmsd(neutral, cation)

    def test_relaxation_appears_on_the_card(self, saved):
        d, data = saved
        html = format_past_result(data, d)
        assert "Geometry RMSD" in html
        assert "Largest atom shift" in html


class TestOldResultsSayWhatToDo:
    """Requested explicitly: results saved before λ persistence cannot be
    recovered — λ is two optimizations and four SCF energies — so the card must
    say so and name the remedy rather than rendering an empty section."""

    def test_a_result_without_channels_gets_an_explanation(self, saved):
        d, data = saved
        stripped = {k: v for k, v in data.items() if k != "reorg_channels"}
        html = format_past_result(stripped, d)
        assert "Re-run this calculation" in html
        assert "not saved" in html

    def test_detection_is_by_absence_not_by_version(self, saved):
        # A version or timestamp cutoff would misfire on anything re-saved or
        # imported from elsewhere. The payload's presence is the ground truth.
        d, data = saved
        stripped = dict(data)
        stripped["reorg_channels"] = None
        stripped["_schema_version"] = 999
        assert "Re-run this calculation" in format_past_result(stripped, d)

    def test_other_calc_types_never_see_the_notice(self, saved):
        d, data = saved
        other = dict(data)
        other["calc_type"] = "single_point"
        other.pop("reorg_channels", None)
        assert "Re-run this calculation" not in format_past_result(other, d)


class TestTheDistinctGeometries:
    """REORG.3. The Marcus scheme is four energies on TWO geometries per
    channel — E_ion(R_neutral) shares its geometry with E_neutral(R_neutral),
    and likewise for R_ion. A four-step control would show each geometry twice
    and imply all four differ."""

    @staticmethod
    def _channels():
        return [
            {
                "kind": "hole",
                "ion_charge": 1,
                "ion_geometry": {
                    "atoms": ["O", "H", "H"],
                    "coordinates": [[0, 0, 0], [1.03, 0, 0], [-0.27, 1.0, 0]],
                },
            },
            {
                "kind": "electron",
                "ion_charge": -1,
                "ion_geometry": {
                    "atoms": ["O", "H", "H"],
                    "coordinates": [[0, 0, 0], [0.92, 0, 0], [-0.22, 0.89, 0]],
                },
            },
        ]

    @staticmethod
    def _neutral():
        return {
            "atoms": ["O", "H", "H"],
            "coordinates": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        }

    def test_both_channels_give_three_geometries_not_four(self):
        from quantui.reorganization_energy import reorg_geometries

        geoms = reorg_geometries(self._channels(), self._neutral())
        assert len(geoms) == 3, "R_neutral is shared and must appear once"

    def test_one_channel_gives_two(self):
        from quantui.reorganization_energy import reorg_geometries

        assert len(reorg_geometries(self._channels()[:1], self._neutral())) == 2

    def test_each_geometry_names_the_energies_evaluated_on_it(self):
        # This is what connects the picture back to λ; without it the viewer is
        # three structures with no stated relationship to the number.
        from quantui.reorganization_energy import reorg_geometries

        geoms = reorg_geometries(self._channels(), self._neutral())
        assert "E_neutral(R_neutral)" in geoms[0]["note"]
        assert "E_hole(R_neutral)" in geoms[0]["note"]
        assert "E_hole(R_hole)" in geoms[1]["note"]

    def test_a_missing_ion_geometry_is_skipped_not_faked(self):
        from quantui.reorganization_energy import reorg_geometries

        chans = self._channels()
        chans[1].pop("ion_geometry")
        assert len(reorg_geometries(chans, self._neutral())) == 2

    def test_the_stepper_is_not_animated(self):
        # λ is a comparison between states, not a trajectory through them;
        # looping would imply a path that was never computed.
        from quantui.app_visualization import build_reorg_geometry_viewer_html
        from quantui.reorganization_energy import reorg_geometries

        html = build_reorg_geometry_viewer_html(
            reorg_geometries(self._channels(), self._neutral())
        )
        assert "LOOP=0" in html.replace(" ", "")

    def test_the_overlay_does_not_superimpose(self):
        """Both geometries come from optimizations seeded from the same
        structure with the same atom ordering, so the displacement is physical.
        Aligning (Kabsch) would rotate away part of what λ measures."""
        from quantui.app_visualization import build_reorg_overlay_html
        from quantui.reorganization_energy import reorg_geometries

        geoms = reorg_geometries(self._channels(), self._neutral())
        html = build_reorg_overlay_html(geoms[0], geoms[1])
        assert html.count("addModel(") == 2
        # "align" alone is useless as a signal — CSS text-align appears in the
        # viewer chrome. Check for an alignment ALGORITHM instead.
        assert "kabsch" not in html.lower()
        assert "superimpose(" not in html.lower()
        # ...and that the viewer tells the user, since a superimposed overlay
        # would look plausible and be quietly wrong.
        assert "not superimposed" in html

    def test_the_overlay_leads_with_displacement_arrows(self):
        """Reported 2026-08-05: two solid ball-and-stick models overlaid were
        an unreadable blob, because λ relaxations are small and the structures
        nearly coincide. Arrows have direction and length — which is what a
        displacement IS — so they carry the signal and the structures became
        thin wireframes providing context."""
        from quantui.app_visualization import build_reorg_overlay_html
        from quantui.reorganization_energy import reorg_geometries

        geoms = reorg_geometries(self._channels(), self._neutral())
        html = build_reorg_overlay_html(geoms[0], geoms[1])
        assert "addArrow(" in html
        assert '"radius": 0.05' in html, "structures must be thin, not bulky"

    def test_atoms_that_did_not_move_get_no_arrow(self):
        # A zero-length arrow renders as a dot and reads as noise.
        from quantui.app_visualization import build_reorg_overlay_html

        g = {
            "label": "R_neutral",
            "atoms": ["O", "H"],
            "coordinates": [[0, 0, 0], [0.96, 0, 0]],
        }
        html = build_reorg_overlay_html(g, g)
        assert html.count("addArrow(") == 0
        assert "no atom moved" in html

    def test_exaggeration_scales_arrows_not_structures(self):
        """The structures must always show TRUE positions — only the arrows are
        amplified, and the legend says by how much. Scaling the geometry would
        put a molecule on screen that was never computed."""
        from quantui.app_visualization import build_reorg_overlay_html
        from quantui.reorganization_energy import reorg_geometries

        geoms = reorg_geometries(self._channels(), self._neutral())
        plain = build_reorg_overlay_html(geoms[0], geoms[1], exaggerate=1.0)
        scaled = build_reorg_overlay_html(geoms[0], geoms[1], exaggerate=5.0)
        # Same atom coordinates in both — only the arrow endpoints differ.
        coord = f"{geoms[1]['coordinates'][1][0]:.6f}"
        assert coord in plain and coord in scaled
        assert "&times;5" in scaled and "&times;5" not in plain

    def test_the_overlay_colours_by_structure_not_by_element(self):
        # In an overlay the question is "which structure is this atom from";
        # element colouring makes the two indistinguishable where they overlap.
        from quantui.app_visualization import build_reorg_overlay_html
        from quantui.reorganization_energy import reorg_geometries

        geoms = reorg_geometries(self._channels(), self._neutral())
        html = build_reorg_overlay_html(geoms[0], geoms[1])
        # Grey reference, coloured relaxed structure, red arrows — a hierarchy
        # rather than two equal-weight solids.
        assert "#94a3b8" in html  # reference: context
        assert "#2166ac" in html  # relaxed structure
        assert "#b2182b" in html  # displacement arrows: the signal


class TestTheAnalysisTabIsWiredUp:
    """REORG.7 — reorganization_energy had NO _PANEL_REGISTRY entry, so the
    Analysis tab populated nothing at all for these runs."""

    def test_the_calc_type_is_registered(self):
        from quantui.app import QuantUIApp

        assert "reorganization_energy" in QuantUIApp._PANEL_REGISTRY

    def test_energies_runs_before_isosurface(self):
        # The same dependency that broke geometry_opt on 2026-08-04:
        # _pop_energies loads the orbital state _pop_isosurface checks. Now
        # pinned in a second place, because the trap is per-calc-type.
        from quantui.app import QuantUIApp

        names = [n for n, _, _ in QuantUIApp._PANEL_REGISTRY["reorganization_energy"]]
        assert names.index("Energies") < names.index("Isosurface")

    def test_geometries_is_the_default_panel(self):
        # FIRST auto_select=True wins — not last.
        from quantui.app import QuantUIApp

        auto = [
            n
            for n, _, sel in QuantUIApp._PANEL_REGISTRY["reorganization_energy"]
            if sel
        ]
        assert auto[0] == "Geometries"

    def test_the_panel_has_a_shell_to_render_into(self):
        from quantui.app import QuantUIApp

        assert any(n == "Geometries" for n, _, _ in QuantUIApp._PANEL_META)

    def test_every_registered_panel_is_actually_in_the_analysis_tab(self):
        """Registering a panel is not enough — it must also be a CHILD of the
        Analysis VBox or it can never render.

        Found the hard way (2026-08-05): the Geometries accordion existed, the
        registry knew about it, every unit test passed, and the tab showed
        nothing. Checked for every panel rather than just the new one, since
        the gap is structural and the next panel would hit it too.
        """
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        rendered = {
            c.get_title(0)
            for c in app.analysis_tab_panel.children
            if hasattr(c, "get_title")
        }
        for name, attr, _ in QuantUIApp._PANEL_META:
            acc = getattr(app, attr, None)
            assert acc is not None, f"{name}: no accordion attribute {attr}"
            assert acc in app.analysis_tab_panel.children, (
                f"{name} is registered but is not a child of the Analysis tab, "
                "so it can never appear"
            )
        assert rendered, "no accordions rendered at all"

    def test_live_and_history_read_the_same_payload(self):
        # The original bug was two paths reading different things. This one
        # must never grow a second reader.
        import inspect

        from quantui.app_analysis import _reorg_payload

        src = inspect.getsource(_reorg_payload)
        assert "_reorg_channels_payload" in src  # live path builds the saved shape
        assert 'data.get("reorg_channels")' in src  # history path reads it


class TestLambdaComparisonAcrossHistory:
    """Screening candidates by λ is the workflow reorganization energy exists
    for — a single λ is hard to judge without others beside it."""

    @staticmethod
    def _entry(name, lam_hole, lam_electron=None, with_geom=True):
        neutral = {
            "atoms": ["O", "H", "H"],
            "coordinates": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        }

        def _ch(kind, lam, disp):
            e = {
                "kind": kind,
                "lambda_hartree": lam,
                "ion_charge": 1 if kind == "hole" else -1,
            }
            if with_geom:
                e["neutral_geometry"] = neutral
                e["ion_geometry"] = {
                    "atoms": ["O", "H", "H"],
                    "coordinates": [
                        [0, 0, 0],
                        [0.96 + disp, 0, 0],
                        [-0.24, 0.93, 0],
                    ],
                }
            return e

        chans = [_ch("hole", lam_hole, 0.07)]
        if lam_electron is not None:
            chans.append(_ch("electron", lam_electron, 0.03))
        return (
            name,
            {
                "formula": name,
                "method": "B3LYP",
                "basis": "6-31G*",
                "calc_type": "reorganization_energy",
                "reorg_channels": chans,
            },
        )

    def test_each_channel_gets_its_own_column(self):
        # λ is per-channel, so it cannot be folded into the general
        # one-row-per-result comparison without inventing a combined number
        # that has no physical meaning.
        from quantui.app_formatters import reorg_comparison_html

        html = reorg_comparison_html(
            [self._entry("H2O", 0.03, 0.02), self._entry("C6H6", 0.012, 0.009)]
        )
        assert "λ hole" in html and "λ electron" in html

    def test_values_are_shown_in_ev(self):
        from quantui.app_formatters import reorg_comparison_html

        html = reorg_comparison_html([self._entry("H2O", 0.03)])
        assert "0.8163" in html  # 0.03 Ha in eV

    def test_a_result_without_lambda_is_listed_not_dropped(self):
        # Silently omitting it would look like it was never selected.
        from quantui.app_formatters import reorg_comparison_html

        old = ("Old", {"formula": "Old", "calc_type": "reorganization_energy"})
        html = reorg_comparison_html([self._entry("H2O", 0.03), old])
        assert "Old" in html
        assert "re-run" in html.lower()

    def test_nothing_renders_when_no_result_has_channels(self):
        from quantui.app_formatters import reorg_comparison_html

        assert reorg_comparison_html([]) == ""
        assert reorg_comparison_html([("x", {"formula": "x"})]) == ""

    def test_the_table_says_which_direction_is_better(self):
        # A bare number invites the wrong reading; λ is one of the few
        # quantities where lower is unambiguously the goal.
        from quantui.app_formatters import reorg_comparison_html

        assert "Lower λ" in reorg_comparison_html([self._entry("H2O", 0.03)])


class TestGeometryViewersFollowTheTheme:
    def test_the_theme_rerender_covers_the_reorg_geometries(self):
        # Same bake-in as every other py3Dmol viewer: bgcolor is painted into
        # the scene at render time, so nothing re-reads it on a theme toggle.
        import inspect

        from quantui.app_visualization import rerender_3d_scenes_for_theme

        assert "render_reorg_geometries" in inspect.getsource(
            rerender_3d_scenes_for_theme
        )
