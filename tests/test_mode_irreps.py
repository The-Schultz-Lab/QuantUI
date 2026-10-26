"""Vibrational-mode symmetry labels (M-SPECTRA2 SPC2.4).

The fixture holds RHF/STO-3G normal modes from PySCF's analytic Hessian at
exactly symmetric (unoptimized) geometries; labelling still needs PySCF for
point-group detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pyscf")

from quantui import symmetry  # noqa: E402
from quantui.symmetry import html_gamma, html_irrep, label_mode_irreps  # noqa: E402

MODES = json.loads(
    (Path(__file__).parent / "data" / "normal_modes_hf_sto3g.json").read_text()
)


def _label(name, **kw):
    m = MODES[name]
    return label_mode_irreps(m["atoms"], m["freqs"], m["disps"], **kw)


# Textbook Γvib for each fixture molecule.
GAMMA = {
    "H2O": ("C2v", "2A1 + B2"),
    "NH3": ("C3v", "2A1 + 2E"),
    "CO2": ("Dooh", "Σg⁺ + Σu⁺ + Πu"),
    "HCN": ("Coov", "2Σ⁺ + Π"),
    "CH4": ("Td", "A1 + E + 2T2"),
    "BF3": ("D3h", "A1′ + 2E′ + A2″"),
    "C2H4": ("D2h", "3Ag + B2g + 2B3g + Au + 2B1u + 2B2u + B3u"),
    "C2H6": ("D3d", "3A1g + 3Eg + A1u + 2A2u + 3Eu"),
    "CH3Cl": ("C3v", "3A1 + 3E"),
    "SF6": ("Oh", "A1g + Eg + T2g + 2T1u + T2u"),
    "C6H6": (
        "D6h",
        "2A1g + A2g + 2B2g + E1g + 4E2g + A2u + 2B1u + 2B2u + 3E1u + 2E2u",
    ),
}


class TestGammaVib:
    @pytest.mark.parametrize("name", sorted(GAMMA))
    def test_textbook_gamma(self, name):
        group, gamma = GAMMA[name]
        res = _label(name)
        assert res is not None
        assert (res.group, res.gamma) == (group, gamma)
        assert not res.is_subgroup
        assert all(lbl for lbl in res.labels)

    def test_ammonia_modes(self):
        # CHEM.1 success criterion: NH3 → symmetry-labelled modes, 2A1 + 2E.
        res = _label("NH3")
        assert res.labels == ["a1", "e", "e", "a1", "e", "e"]
        assert res.caption() == "Mode symmetry in C3v: Γvib = 2A1 + 2E"

    def test_water_modes(self):
        # Bend and symmetric stretch a1, antisymmetric stretch b2.
        assert _label("H2O").labels == ["a1", "a1", "b2"]

    def test_linear_modes_use_sigma_pi(self):
        assert _label("CO2").labels == ["πu", "πu", "σg⁺", "σu⁺"]

    def test_accidental_near_degeneracy_is_split(self):
        # Ethane's A1g and A2u C-H stretches sit 0.3 cm⁻¹ apart.
        m = MODES["C2H6"]
        res = _label("C2H6")
        i = next(k for k, f in enumerate(m["freqs"]) if f > 3527)
        assert res.labels[i : i + 2] == ["a1g", "a2u"]

    def test_lone_mode_peeled_off_degenerate_pair(self):
        # Benzene: an E2g pair and a B2u mode within 2.3 cm⁻¹.
        res = _label("C6H6")
        m = MODES["C6H6"]
        i = next(k for k, f in enumerate(m["freqs"]) if f > 1361)
        assert res.labels[i : i + 3] == ["e2g", "e2g", "b2u"]

    @pytest.mark.parametrize("name", ["NH3", "CH4", "SF6", "C6H6", "CO2"])
    def test_orientation_independent(self, name):
        import numpy as np

        rot, _ = np.linalg.qr(
            np.array([[0.3, -1.1, 0.7], [1.2, 0.4, -0.5], [0.2, 0.9, 1.3]])
        )
        m = MODES[name]
        atoms = [
            [a, (np.array(x) @ rot.T + [0.3, -1.2, 2.0]).tolist()]
            for a, x in m["atoms"]
        ]
        disps = (np.array(m["disps"]) @ rot.T).tolist()
        res = label_mode_irreps(atoms, m["freqs"], disps)
        assert res is not None
        assert res.labels == _label(name).labels

    def test_unrecoverable_group_keeps_subgroup_labels(self, monkeypatch):
        monkeypatch.delitem(symmetry._RECOVERABLE, ("C3v", "Cs"))
        res = _label("NH3")
        assert res.group == "Cs" and res.top_group == "C3v" and res.is_subgroup
        assert res.labels[1] == "a′+a″"
        assert res.gamma == "2A′ + 2(A′+A″)"
        assert "(subgroup of C3v)" in res.caption()


class TestRefusals:
    def test_near_zero_modes_are_unlabelled(self):
        m = MODES["H2O"]
        disps = [m["disps"][0]] + m["disps"]
        res = label_mode_irreps(m["atoms"], [3.0] + m["freqs"], disps)
        assert res.labels == [None, "a1", "a1", "b2"]

    def test_shape_mismatch(self):
        m = MODES["H2O"]
        assert label_mode_irreps(m["atoms"], m["freqs"][:2], m["disps"]) is None

    def test_impure_mode(self):
        m = MODES["H2O"]
        disps = [list(map(list, d)) for d in m["disps"]]
        disps[2] = [
            [a + 0.6 * b for a, b in zip(x, y)] for x, y in zip(disps[2], disps[0])
        ]
        assert label_mode_irreps(m["atoms"], m["freqs"], disps) is None

    def test_c1_geometry(self):
        atoms = [
            ["C", [0, 0, 0]],
            ["H", [0, 0, 1.09]],
            ["F", [1.3, 0, -0.4]],
            ["Cl", [-0.8, 1.4, -0.5]],
            ["Br", [-0.9, -1.5, -0.6]],
        ]
        disps = [[[0.1, 0.0, 0.0]] * 5]
        assert label_mode_irreps(atoms, [500.0], disps) is None


class TestHtml:
    def test_irrep_subscripts(self):
        assert html_irrep("a1g") == "a<sub>1g</sub>"
        assert html_irrep("e") == "e"
        assert html_irrep("σg⁺") == "σ<sub>g</sub>⁺"
        assert html_irrep("a2″") == "a<sub>2</sub>″"
        assert html_irrep("b1+b2") == "b<sub>1</sub>+b<sub>2</sub>"

    def test_gamma_subscripts(self):
        assert html_gamma("2A1 + 2E") == "2A<sub>1</sub> + 2E"
        assert html_gamma("2(A′+A″)") == "2(A′+A″)"


class TestUi:
    @staticmethod
    def _nh3_result(**extra):
        m = MODES["NH3"]
        return SimpleNamespace(
            frequencies_cm1=m["freqs"],
            ir_intensities=[1.0] * 6,
            raman_activities=[],
            displacements=m["disps"],
            **extra,
        )

    def test_freq_mode_irreps_uses_live_geometry_or_molecule(self):
        from quantui.app_visualization import freq_mode_irreps
        from quantui.molecule import Molecule

        m = MODES["NH3"]
        live = self._nh3_result(pyscf_mol_atom=m["atoms"])
        assert freq_mode_irreps(live).gamma == "2A1 + 2E"
        mol = Molecule([a for a, _ in m["atoms"]], [xyz for _, xyz in m["atoms"]])
        assert freq_mode_irreps(self._nh3_result(), mol).gamma == "2A1 + 2E"
        assert freq_mode_irreps(self._nh3_result()) is None

    def test_mode_table_has_symmetry_column(self):
        from quantui.app_visualization import freq_mode_irreps, vib_mode_table_html

        res = self._nh3_result(pyscf_mol_atom=MODES["NH3"]["atoms"])
        html = vib_mode_table_html(res, freq_mode_irreps(res))
        assert ">Sym.</th>" in html
        assert "<td style='padding:2px 10px;text-align:left'>a<sub>1</sub></td>" in html
        assert "Γ<sub>vib</sub> = 2A<sub>1</sub> + 2E" in html
        assert "Sym." not in vib_mode_table_html(res)

    def test_ir_and_raman_hover_show_symmetry(self):
        from quantui.ir_plot import plot_ir_spectrum
        from quantui.raman_plot import plot_raman_spectrum

        freqs = [-200.0, 1000.0, 2000.0]
        labels = ["b2", "a1", None]
        fig = plot_ir_spectrum(freqs, [1.0, 2.0, 3.0], labels=labels)
        peaks = fig.data[-1]
        assert list(peaks.customdata) == ["<br>Symmetry: a<sub>1</sub>", ""]
        assert "%{customdata}" in peaks.hovertemplate
        fig = plot_raman_spectrum(freqs, [1.0, 2.0, 3.0], labels=labels)
        assert "%{customdata}" in fig.data[-1].hovertemplate
        assert plot_ir_spectrum(freqs, [1.0] * 3).data[-1].customdata is None

    def test_mode_dropdown_names_the_irrep(self, tmp_path, monkeypatch):
        from quantui.app import QuantUIApp
        from quantui.molecule import Molecule

        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "s.json"))
        app = QuantUIApp()
        m = MODES["NH3"]
        mol = Molecule([a for a, _ in m["atoms"]], [xyz for _, xyz in m["atoms"]])
        app._show_vib_animation(self._nh3_result(), mol)
        labels = [lbl for lbl, _ in app.vib_mode_dd.options]
        assert labels[0] == "Mode 1: 1090.5 cm⁻¹ · a1"
        assert labels[1].endswith("· e")

    def test_history_stub_carries_modes_and_geometry(self):
        from quantui.app_analysis import _mode_symmetry_inputs

        m = MODES["NH3"]
        ctx = SimpleNamespace(
            spectra_data={
                "ir": {"displacements": m["disps"]},
                "molecule": {
                    "atoms": [a for a, _ in m["atoms"]],
                    "coords": [xyz for _, xyz in m["atoms"]],
                },
            }
        )
        kw = _mode_symmetry_inputs(ctx)
        assert kw["displacements"] is m["disps"]
        assert kw["pyscf_mol_atom"][0][0] == "N"
        assert _mode_symmetry_inputs(SimpleNamespace(spectra_data={})) == {}
