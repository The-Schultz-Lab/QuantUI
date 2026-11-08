"""M-CHEM CHEM.5: point groups and MO irreps, the automated half.

Gaps left by ``test_symmetry.py``:

* point groups of the classic teaching set from ideal geometries built here
  (CH4 Td, benzene D6h, staggered vs eclipsed ethane D3d / D3h, BF3 D3h,
  ethylene D2h, SF6 Oh, HCN C∞v) and two chiral molecules — CHFClBr (C1)
  and H2O2 (C2: chiral does not mean C1);
* the tolerance ladder on a distorted water: within the 0.01 Å teaching
  tolerance it is still C2v (and says so), a little further it is Cs with a
  "nearly C2v" hint (0.05 Å), far off it is plain Cs;
* every RHF/STO-3G water MO label, virtuals included, in textbook order;
* History replay: the saved result's point group on the History card and
  the MO labels on the replayed orbital diagram.

The Voilà visual half of CHEM.5 is a local manual check, not covered here.
"""

from __future__ import annotations

import io
import math
import sys

import pytest

try:
    import pyscf  # noqa: F401

    _HAS_PYSCF = sys.platform != "win32"
except ImportError:  # pragma: no cover - depends on the environment
    _HAS_PYSCF = False

pytestmark = pytest.mark.skipif(
    not _HAS_PYSCF, reason="PySCF not available or not supported on native Windows"
)

WATER = (
    ["O", "H", "H"],
    [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
)


def _ring(radius, n, z=0.0, phase=0.0):
    return [
        [
            radius * math.cos(phase + 2 * math.pi * k / n),
            radius * math.sin(phase + 2 * math.pi * k / n),
            z,
        ]
        for k in range(n)
    ]


def _methane():
    a = 1.09 / math.sqrt(3)
    h = [[a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]]
    return ["C"] + ["H"] * 4, [[0.0, 0.0, 0.0]] + h


def _benzene():
    return ["C"] * 6 + ["H"] * 6, _ring(1.39, 6) + _ring(2.48, 6)


def _ethane(staggered=True):
    # C–C 1.53 Å along z; C–H 1.09 Å at a tetrahedral-ish 111°.
    cc = 0.765
    r = 1.09 * math.sin(math.radians(180 - 111))
    h = 1.09 * math.cos(math.radians(180 - 111))
    top = _ring(r, 3, z=cc + h)
    bottom = _ring(r, 3, z=-cc - h, phase=math.pi / 3 if staggered else 0.0)
    return (
        ["C", "C"] + ["H"] * 6,
        [[0.0, 0.0, cc], [0.0, 0.0, -cc]] + top + bottom,
    )


def _bf3():
    return ["B"] + ["F"] * 3, [[0.0, 0.0, 0.0]] + _ring(1.31, 3)


def _ethylene():
    return ["C", "C", "H", "H", "H", "H"], [
        [0.0, 0.0, 0.667],
        [0.0, 0.0, -0.667],
        [0.0, 0.923, 1.237],
        [0.0, -0.923, 1.237],
        [0.0, 0.923, -1.237],
        [0.0, -0.923, -1.237],
    ]


def _sf6():
    d = 1.56
    ligands = [
        [d, 0, 0],
        [-d, 0, 0],
        [0, d, 0],
        [0, -d, 0],
        [0, 0, d],
        [0, 0, -d],
    ]
    return ["S"] + ["F"] * 6, [[0.0, 0.0, 0.0]] + ligands


def _hcn():
    return ["H", "C", "N"], [[0.0, 0.0, -1.066], [0.0, 0.0, 0.0], [0.0, 0.0, 1.156]]


def _ammonia():
    # Ideal C3v (the rounded NH3 in test_symmetry.py is the tolerance case).
    r = 1.012 * math.sin(math.radians(68.0))
    h = -1.012 * math.cos(math.radians(68.0))
    return ["N"] + ["H"] * 3, [[0.0, 0.0, 0.0]] + _ring(r, 3, z=h)


def _hydrogen_peroxide():
    # C2 axis along x maps (x, y, z) → (x, −y, −z): O1↔O2, H1↔H2. The
    # H–O–O–H dihedral is ≈ 97°, so there is no mirror plane: chiral, yet C2.
    return ["O", "O", "H", "H"], [
        [0.0, 0.725, 0.0],
        [0.0, -0.725, 0.0],
        [0.6, 0.975, 0.68],
        [0.6, -0.975, -0.68],
    ]


def _chfclbr():
    return ["C", "H", "F", "Cl", "Br"], [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.09],
        [1.3, 0.0, -0.4],
        [-0.8, 1.4, -0.5],
        [-0.9, -1.5, -0.6],
    ]


CLASSIC = {
    "H2O": (WATER, "C2v"),
    "NH3": (_ammonia(), "C3v"),
    "CH4": (_methane(), "Td"),
    "C6H6": (_benzene(), "D6h"),
    "CO2": ((["O", "C", "O"], [[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]]), "Dooh"),
    "HCN": (_hcn(), "Coov"),
    "C2H6 staggered": (_ethane(staggered=True), "D3d"),
    "C2H6 eclipsed": (_ethane(staggered=False), "D3h"),
    "BF3": (_bf3(), "D3h"),
    "C2H4": (_ethylene(), "D2h"),
    "SF6": (_sf6(), "Oh"),
    "H2O2": (_hydrogen_peroxide(), "C2"),
    "CHFClBr": (_chfclbr(), "C1"),
}


class TestClassicPointGroups:
    @pytest.mark.parametrize("name", list(CLASSIC))
    def test_ideal_geometry(self, name):
        from quantui.symmetry import detect_point_group

        geom, expected = CLASSIC[name]
        pg = detect_point_group(*geom)
        assert pg is not None
        # Ideal geometries need no tolerance and have no "nearly" hint.
        assert (pg.group, pg.exact_group, pg.loose_group) == (expected, expected, None)
        assert pg.summary() == {"Dooh": "D∞h", "Coov": "C∞v"}.get(expected, expected)

    def test_chirality_is_not_the_same_as_c1(self):
        # Both chiral (no mirror plane, no improper axis), but H2O2 keeps a
        # proper C2 axis.
        from quantui.symmetry import detect_point_group, group_order

        h2o2 = detect_point_group(*_hydrogen_peroxide())
        chfclbr = detect_point_group(*_chfclbr())
        assert group_order(h2o2.group) == 2 and h2o2.group != "Cs"
        assert group_order(chfclbr.group) == 1

    def test_detection_does_not_depend_on_orientation_or_origin(self):
        import numpy as np

        from quantui.symmetry import detect_point_group

        rot, _ = np.linalg.qr(
            np.array([[0.3, -1.1, 0.7], [1.2, 0.4, -0.5], [0.2, 0.9, 1.3]])
        )
        for name in ("CH4", "C6H6", "C2H6 staggered", "SF6"):
            (atoms, coords), expected = CLASSIC[name]
            moved = (np.array(coords) @ rot.T + [1.5, -2.0, 0.7]).tolist()
            assert detect_point_group(atoms, moved).group == expected, name


def _water_with_long_oh(delta):
    """Water with one O–H direction stretched by *delta* Å along y."""
    atoms, coords = WATER
    coords = [list(c) for c in coords]
    coords[1][1] += delta
    return atoms, coords


class TestToleranceLadder:
    """0.01 Å teaching tolerance, 0.05 Å "nearly" hint (DEFAULT/LOOSE)."""

    def test_documented_tolerances(self):
        from quantui.symmetry import (
            DEFAULT_TOLERANCE_ANGSTROM,
            LOOSE_TOLERANCE_ANGSTROM,
        )

        assert DEFAULT_TOLERANCE_ANGSTROM == 0.01
        assert LOOSE_TOLERANCE_ANGSTROM == 0.05

    def test_small_distortion_is_still_c2v_and_says_so(self):
        from quantui.symmetry import detect_point_group

        pg = detect_point_group(*_water_with_long_oh(0.003))
        assert pg.exact_group == "Cs"  # strict PySCF sees the broken mirror
        assert pg.group == "C2v"
        assert pg.loose_group is None
        assert pg.summary() == "C2v (within 0.01 Å)"

    def test_moderate_distortion_is_cs_but_nearly_c2v(self):
        from quantui.symmetry import detect_point_group

        pg = detect_point_group(*_water_with_long_oh(0.012))
        assert (pg.group, pg.exact_group, pg.loose_group) == ("Cs", "Cs", "C2v")
        assert pg.summary() == "Cs; nearly C2v (within 0.05 Å)"
        assert "nearly C<sub>2v</sub>" in pg.summary_html()

    def test_large_distortion_is_plain_cs(self):
        from quantui.symmetry import detect_point_group

        pg = detect_point_group(*_water_with_long_oh(0.1))
        assert (pg.group, pg.loose_group) == ("Cs", None)
        assert pg.summary() == "Cs"

    def test_explicit_tighter_tolerance(self):
        # The tolerance argument is honoured: at PySCF-strict-like 1e-4 Å the
        # 0.003 Å distortion is no longer forgiven.
        from quantui.symmetry import detect_point_group

        pg = detect_point_group(*_water_with_long_oh(0.003), tolerance_angstrom=1e-4)
        assert pg.group == "Cs"


class TestWaterMoIrreps:
    def test_all_sto3g_orbitals_in_textbook_order(self):
        from pyscf import gto, scf

        from quantui.symmetry import label_mo_irreps

        mol_atom = list(zip(*WATER))
        mf = scf.RHF(gto.M(atom=mol_atom, basis="sto-3g", verbose=0)).run()
        ir = label_mo_irreps(mol_atom, "sto-3g", mf.mo_coeff, mo_energy=mf.mo_energy)
        assert ir.group == "C2v" and not ir.is_subgroup
        # Occupied 1a1 (O 1s) 2a1 1b2 3a1 1b1 (HOMO, out-of-plane lone pair);
        # virtual 4a1 (LUMO) and 2b2 — the antibonding partners of 3a1/1b2.
        assert ir.labels == ["1a1", "2a1", "1b2", "3a1", "1b1", "4a1", "2b2"]
        assert ir.caption() == "Orbital symmetry labels in C2v"


class TestHistoryReplay:
    @pytest.fixture
    def water_mol(self):
        from quantui.molecule import Molecule

        return Molecule(*WATER)

    def test_frequency_history_card_shows_the_point_group(self, tmp_path, water_mol):
        from types import SimpleNamespace

        from quantui.app_formatters import format_past_result
        from quantui.results_storage import load_result, save_result

        r = SimpleNamespace(
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-74.96,
            converged=True,
            n_iterations=8,
            frequencies_cm1=[2170.0, 4140.0, 4391.0],
            ir_intensities=[1.0, 1.0, 1.0],
            zpve_hartree=0.0244,
        )
        spectra = {
            "ir": {"frequencies_cm1": r.frequencies_cm1},
            "molecule": {
                "atoms": list(water_mol.atoms),
                "coords": [list(map(float, c)) for c in water_mol.coordinates],
                "charge": 0,
                "multiplicity": 1,
            },
        }
        saved = save_result(
            r, results_dir=tmp_path, calc_type="frequency", spectra=spectra
        )
        html = format_past_result(load_result(saved), result_dir=saved)
        assert "Point group" in html
        assert "C<sub>2v</sub>" in html

    @pytest.fixture
    def saved_sp(self, tmp_path, water_mol):
        from quantui.results_storage import save_orbitals, save_result
        from quantui.session_calc import run_in_session

        r = run_in_session(water_mol, "RHF", "STO-3G", progress_stream=io.StringIO())
        saved = save_result(
            r,
            results_dir=tmp_path,
            calc_type="single_point",
            spectra={},
            molecule=water_mol,
        )
        save_orbitals(saved, r)
        return r, saved

    def test_live_single_point_card_shows_the_point_group(self, saved_sp):
        from quantui.app_formatters import format_result

        r, _saved = saved_sp
        html = format_result(r)
        assert "Point group" in html and "C<sub>2v</sub>" in html

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "app_formatters._point_group_html reads only pyscf_mol_atom (live) "
            "or spectra['molecule'] (frequency). A saved single-point "
            "result.json carries its geometry under 'geometry' (atoms/"
            "coordinates), which is never read, so the History card drops the "
            "Point group row the live card shows."
        ),
    )
    def test_single_point_history_card_shows_the_point_group(self, saved_sp):
        from quantui.app_formatters import format_past_result
        from quantui.results_storage import load_result

        _r, saved = saved_sp
        data = load_result(saved)
        assert data["geometry"]["atoms"] == ["O", "H", "H"]  # it is on disk
        html = format_past_result(data, result_dir=saved)
        assert "Point group" in html and "C<sub>2v</sub>" in html

    def test_replayed_orbital_diagram_carries_mo_irreps(self, saved_sp):
        from quantui.app import QuantUIApp

        _r, saved = saved_sp
        app = QuantUIApp()
        app._history_load_analysis(saved)
        assert "Energies" in app._ana_available
        info = app._last_orb_info
        assert info.irreps == ["1a1", "2a1", "1b2", "3a1", "1b1", "4a1", "2b2"]
        assert info.irrep_caption == "Orbital symmetry labels in C2v"
