"""Thermochemistry rotational symmetry number from QuantUI's point group
(audit SCF-4 / GEOM-7).

PySCF's detector rounds inertia moments, so the group it reports is not
monotonic in the tolerance: the bundled 2-decimal benzene came back D2h at
0.01 Å (sigma 4 instead of 12, S too high by R ln 3 = 9.13 J/(mol K)) while
the UI's own detector said "nearly D6h", and ±0.0005 Å noise dropped NH3 /
CH4 / benzene to C1 / C3 / C2h. Detection now takes the highest-order group
over a tolerance ladder, admits a looser-rung group only when every atom is
really within 0.01 Å of its symmetric image, and thermo takes sigma from
that same group.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("pyscf")

from quantui.symmetry import (
    detect_point_group,
    rotational_symmetry_number,
)  # noqa: E402

R = 8.314462618

NH3 = (
    ["N", "H", "H", "H"],
    [
        [0.0, 0.0, 0.116489],
        [0.0, 0.939731, -0.271808],
        [0.813831, -0.469865, -0.271808],
        [-0.813831, -0.469865, -0.271808],
    ],
)
CH4 = (
    ["C", "H", "H", "H", "H"],
    [
        [0.0, 0.0, 0.0],
        [0.629118, 0.629118, 0.629118],
        [-0.629118, -0.629118, 0.629118],
        [0.629118, -0.629118, -0.629118],
        [-0.629118, 0.629118, -0.629118],
    ],
)
BENZENE = (
    ["C"] * 6 + ["H"] * 6,
    [
        [0.0, 1.395248, 0.0],
        [1.20832, 0.697624, 0.0],
        [1.20832, -0.697624, 0.0],
        [0.0, -1.395248, 0.0],
        [-1.20832, -0.697624, 0.0],
        [-1.20832, 0.697624, 0.0],
        [0.0, 2.48236, 0.0],
        [2.149787, 1.24118, 0.0],
        [2.149787, -1.24118, 0.0],
        [0.0, -2.48236, 0.0],
        [-2.149787, -1.24118, 0.0],
        [-2.149787, 1.24118, 0.0],
    ],
)


def _library_benzene():
    from quantui import molecule_library as ml

    e = next(x for x in ml.iter_entries() if x["id"] == "C6H6")
    return e["atoms"], e["coordinates"]


def _noisy(geom, seed, amp=5e-4):
    rng = np.random.default_rng(seed)
    c = np.asarray(geom[1]) + rng.uniform(-amp, amp, (len(geom[0]), 3))
    return geom[0], c.tolist()


class TestDetectPointGroupLadder:
    def test_bundled_benzene_is_d6h(self):
        pg = detect_point_group(*_library_benzene())
        assert pg.group == "D6h"
        assert pg.loose_group is None

    @pytest.mark.parametrize(
        "geom, group", [(NH3, "C3v"), (CH4, "Td"), (BENZENE, "D6h")]
    )
    @pytest.mark.parametrize("seed", range(6))
    def test_sub_milliangstrom_noise_keeps_the_group(self, geom, group, seed):
        assert detect_point_group(*_noisy(geom, seed)).group == group

    def test_real_distortion_is_not_upgraded(self):
        # One O-H stretched by 0.03 Å: really Cs, not C2v within 0.01 Å.
        atoms = ["O", "H", "H"]
        coords = [[0.0, 0.0, 0.0], [0.99, 0.0, 0.0], [-0.24, 0.93, 0.0]]
        pg = detect_point_group(atoms, coords)
        assert pg.group == "Cs"

    def test_benzene_ring_distortion_is_not_d6h(self):
        atoms, coords = BENZENE
        c = np.asarray(coords)
        for i in range(6):  # alternate C atoms in/out along the radius by 0.03 Å
            c[i] *= 1.0 + (0.03 if i % 2 else -0.03) / np.linalg.norm(c[i])
        pg = detect_point_group(atoms, c.tolist())
        assert pg.group == "D3h"
        assert pg.loose_group == "D6h"


class TestRotationalSymmetryNumber:
    @pytest.mark.parametrize(
        "group, sigma",
        [
            ("C1", 1),
            ("Cs", 1),
            ("Ci", 1),
            ("Coov", 1),
            ("Dooh", 2),
            ("C2v", 2),
            ("C3v", 3),
            ("C2h", 2),
            ("D2h", 4),
            ("D3d", 6),
            ("D6h", 12),
            ("Td", 12),
            ("Oh", 24),
            ("Ih", 60),
            ("S4", 2),
            ("S6", 3),
        ],
    )
    def test_table(self, group, sigma):
        assert rotational_symmetry_number(group) == sigma


class TestThermoSigma:
    def test_bundled_benzene_thermo_uses_sigma_12(self):
        from pyscf import gto
        from pyscf.hessian import thermo as pyscf_thermo

        from quantui.freq_calc import _cm1_to_thermo_au, compute_thermochemistry
        from quantui.symmetry import detection_tolerance

        atoms, coords = _library_benzene()
        freqs = [400.0 + 100.0 * i for i in range(30)]
        td = compute_thermochemistry(
            atoms, coords, energy_hartree=-230.0, frequencies_cm1=freqs
        )
        assert td.symmetry_number == 12
        assert td.symmetry_group == "D6h"

        # PySCF alone (even at QuantUI's 0.01 Å) says sigma 4 for this
        # geometry; ours is lower in S by exactly R ln(12/4).
        mol = gto.M(atom=list(zip(atoms, coords)), basis="sto-3g", verbose=0)

        class _Model:
            pass

        model = _Model()
        model.mol, model.e_tot = mol, -230.0
        with detection_tolerance():
            ref = pyscf_thermo.thermo(model, _cm1_to_thermo_au(freqs), 298.15, 101325)
        assert ref["sym_number"][0] == 4
        from quantui.freq_calc import _HARTREE_TO_JMOL

        s_ref = ref["S_tot"][0] * _HARTREE_TO_JMOL
        assert s_ref - td.S_jmol == pytest.approx(R * math.log(3), rel=1e-4)
        assert td.G_hartree == pytest.approx(
            td.H_hartree - 298.15 * td.S_jmol / _HARTREE_TO_JMOL, abs=1e-10
        )

    def test_thermo_table_names_the_group(self):
        from quantui.app_visualization import thermo_table_html
        from quantui.freq_calc import ThermoData

        td = ThermoData(
            zpve_hartree=0.1,
            H_hartree=-230.0,
            S_jmol=260.0,
            G_hartree=-230.03,
            symmetry_number=12,
            symmetry_group="D6h",
        )
        html = thermo_table_html(td, -230.1)
        assert "D<sub>6h</sub>" in html
