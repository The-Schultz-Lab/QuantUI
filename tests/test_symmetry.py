"""Point groups and orbital symmetry labels (M-CHEM CHEM.1, DEC-023)."""

from __future__ import annotations

import pytest

from quantui.symmetry import (
    PointGroup,
    group_order,
    html_group,
    pretty_group,
)

WATER = (
    ["O", "H", "H"],
    [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]],
)
# 4-decimal coordinates: off exact C3v by ~1e-4 Å.
NH3 = (
    ["N", "H", "H", "H"],
    [
        [0, 0, 0.1],
        [0.9377, 0, -0.27],
        [-0.4689, 0.8121, -0.27],
        [-0.4689, -0.8121, -0.27],
    ],
)
CO2 = (["O", "C", "O"], [[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]])
CHFCLBR = (
    ["C", "H", "F", "Cl", "Br"],
    [[0, 0, 0], [0, 0, 1.09], [1.3, 0, -0.4], [-0.8, 1.4, -0.5], [-0.9, -1.5, -0.6]],
)


class TestNames:
    def test_pretty_linear_groups(self):
        assert pretty_group("Dooh") == "D∞h"
        assert pretty_group("Coov") == "C∞v"
        assert pretty_group("C2v") == "C2v"

    def test_html_subscript(self):
        assert html_group("C2v") == "C<sub>2v</sub>"
        assert html_group("Dooh") == "D<sub>∞h</sub>"

    @pytest.mark.parametrize(
        "name, order",
        [
            ("C1", 1),
            ("Cs", 2),
            ("C2v", 4),
            ("C3v", 6),
            ("D3d", 12),
            ("D6h", 24),
            ("Td", 24),
            ("Oh", 48),
            ("S4", 4),
        ],
    )
    def test_group_order(self, name, order):
        assert group_order(name) == order

    def test_infinite_groups_outrank_finite(self):
        assert group_order("Dooh") > group_order("Ih") > group_order("Oh")

    def test_summary_mentions_tolerance_only_when_needed(self):
        exact = PointGroup("C2v", 0.01, exact_group="C2v")
        assert exact.summary() == "C2v"
        loose = PointGroup("C3v", 0.01, exact_group="Cs")
        assert "within 0.01 Å" in loose.summary()
        nearly = PointGroup("C2h", 0.01, exact_group="C2h", loose_group="D3d")
        assert "nearly D3d" in nearly.summary()
        assert "D<sub>3d</sub>" in nearly.summary_html()


class TestDetection:
    @pytest.fixture(autouse=True)
    def _pyscf(self):
        pytest.importorskip("pyscf")

    @pytest.mark.parametrize(
        "geom, expected",
        [(WATER, "C2v"), (NH3, "C3v"), (CO2, "Dooh"), (CHFCLBR, "C1")],
    )
    def test_teaching_molecules(self, geom, expected):
        from quantui.symmetry import detect_point_group

        pg = detect_point_group(*geom)
        assert pg is not None
        assert pg.group == expected

    def test_rounded_coordinates_still_find_full_symmetry(self):
        # PySCF's strict default tolerance sees Cs here; the teaching
        # tolerance must find C3v and say it used a tolerance.
        from quantui.symmetry import detect_point_group

        pg = detect_point_group(*NH3)
        assert pg.exact_group == "Cs"
        assert pg.group == "C3v"
        assert "within 0.01" in pg.summary()

    def test_detection_does_not_leak_the_tolerance(self):
        from pyscf.symm import geom

        from quantui.symmetry import detect_point_group

        before = geom.TOLERANCE
        detect_point_group(*NH3)
        assert geom.TOLERANCE == before

    def test_molecule_helper(self):
        from quantui.molecule import Molecule
        from quantui.symmetry import point_group_of_molecule

        mol = Molecule(atoms=WATER[0], coordinates=WATER[1])
        assert point_group_of_molecule(mol).group == "C2v"

    def test_empty_molecule(self):
        from quantui.symmetry import detect_point_group

        assert detect_point_group([], []) is None


class TestOrbitalLabels:
    @pytest.fixture(autouse=True)
    def _pyscf(self):
        pytest.importorskip("pyscf")

    @staticmethod
    def _scf(geom, basis="sto-3g", spin=0):
        from pyscf import gto, scf

        mol_atom = [(a, list(map(float, c))) for a, c in zip(*geom)]
        mol = gto.M(atom=mol_atom, basis=basis, spin=spin, verbose=0)
        mf = (scf.RHF(mol) if spin == 0 else scf.UHF(mol)).run()
        return mol_atom, mf

    def test_water_textbook_labels(self):
        from quantui.symmetry import label_mo_irreps

        mol_atom, mf = self._scf(WATER, "6-31g*")
        ir = label_mo_irreps(mol_atom, "6-31g*", mf.mo_coeff, mo_energy=mf.mo_energy)
        assert ir.group == "C2v" and not ir.is_subgroup
        # 1a1 2a1 1b2 3a1 1b1 — the HOMO is the out-of-plane 1b1 lone pair.
        assert ir.labels[:5] == ["1a1", "2a1", "1b2", "3a1", "1b1"]

    def test_labels_independent_of_orientation(self):
        import numpy as np

        from quantui.symmetry import label_mo_irreps

        theta = 0.7
        rot = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
        coords = (np.array(WATER[1]) @ rot.T + [0.3, -0.2, 0.5]).tolist()
        mol_atom, mf = self._scf((WATER[0], coords))
        ir = label_mo_irreps(mol_atom, "sto-3g", mf.mo_coeff)
        assert ir.labels[:5] == ["1a1", "2a1", "1b2", "3a1", "1b1"]

    def test_ammonia_gets_c3v_labels(self):
        from quantui.symmetry import label_mo_irreps

        mol_atom, mf = self._scf(NH3)
        ir = label_mo_irreps(mol_atom, "sto-3g", mf.mo_coeff, mo_energy=mf.mo_energy)
        assert ir.group == "C3v"
        # Occupied: 1a1 (N 1s), 2a1, 1e (degenerate pair), 3a1 (lone pair).
        assert ir.labels[:5] == ["1a1", "2a1", "1e", "1e", "3a1"]

    def test_ammonia_without_energies_keeps_subgroup_labels(self):
        from quantui.symmetry import label_mo_irreps

        mol_atom, mf = self._scf(NH3)
        ir = label_mo_irreps(mol_atom, "sto-3g", mf.mo_coeff)
        assert ir.group == "Cs" and ir.top_group == "C3v"
        assert "subgroup of C3v" in ir.caption()

    def test_linear_molecule_uses_sigma_pi(self):
        from quantui.symmetry import label_mo_irreps

        mol_atom, mf = self._scf(CO2)
        ir = label_mo_irreps(mol_atom, "sto-3g", mf.mo_coeff, mo_energy=mf.mo_energy)
        homo = int((mf.mo_occ > 0).sum()) - 1
        # CO2's HOMO is the doubly degenerate 1πg pair.
        assert ir.labels[homo] == "1πg"
        assert ir.labels[homo - 1] == "1πg"
        assert "1πu" in ir.labels

    def test_open_shell_alpha_channel(self):
        from quantui.symmetry import label_mo_irreps

        mol_atom, mf = self._scf((["O", "H"], [[0, 0, 0], [0, 0, 0.97]]), spin=1)
        ir = label_mo_irreps(mol_atom, "sto-3g", mf.mo_coeff[0])
        assert ir is not None
        assert "1π" in ir.labels

    def test_no_symmetry_returns_none(self):
        from quantui.symmetry import label_mo_irreps

        mol_atom, mf = self._scf(CHFCLBR)
        assert label_mo_irreps(mol_atom, "sto-3g", mf.mo_coeff) is None

    def test_mismatched_basis_returns_none(self):
        from quantui.symmetry import label_mo_irreps

        mol_atom, mf = self._scf(WATER)
        assert label_mo_irreps(mol_atom, "6-31g*", mf.mo_coeff) is None


class TestDiagramAndCards:
    @pytest.fixture(autouse=True)
    def _pyscf(self):
        pytest.importorskip("pyscf")

    def test_diagram_hover_and_caption_carry_labels(self):
        pytest.importorskip("plotly")
        import numpy as np

        from quantui.orbital_visualization import (
            orbital_info_from_arrays,
            plot_orbital_diagram_plotly,
        )

        info = orbital_info_from_arrays(
            np.array([-20.0, -1.3, -0.7, -0.5, -0.4, 0.6, 0.8]),
            np.array([2, 2, 2, 2, 2, 0, 0]),
            formula="H2O",
        )
        info.irreps = ["1a1", "2a1", "1b2", "3a1", "1b1", "4a1", "2b2"]
        info.irrep_caption = "Orbital symmetry labels in C2v"
        fig = plot_orbital_diagram_plotly(info)
        hovers = " ".join(str(tr.hovertemplate) for tr in fig.data)
        assert "(1b1) — HOMO" in hovers
        assert "C2v" in fig.layout.title.text

    def test_result_card_point_group_row(self):
        from types import SimpleNamespace

        from quantui.app_formatters import _result_extra_rows

        r = SimpleNamespace(
            pyscf_mol_atom=[(a, c) for a, c in zip(*WATER)], solvent=None
        )
        html = _result_extra_rows(lambda k, d=None: getattr(r, k, d))
        assert "Point group" in html
        assert "C<sub>2v</sub>" in html
