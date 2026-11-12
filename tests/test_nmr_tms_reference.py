"""TMS comes out at delta = 0 with QuantUI's own reference table (audit SCF-1).

The table used to hold literature-style constants that did not match this
GIAO code: TMS itself came out at delta(13C) = -7.7 ppm at B3LYP/6-31G* and
-52 ppm at RHF/STO-3G, so every reported shift was off by that much. The
table is now computed by QuantUI's run_nmr_calc on TMS optimized at each
level; this test redoes the cheapest row.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("pyscf")
pytest.importorskip("pyscf.prop.nmr")

from quantui.molecule import Molecule  # noqa: E402

# TMS optimized with QuantUI at RHF/STO-3G (Td, fmax 0.01 eV/A).
TMS_ATOMS = [
    "Si",
    "C",
    "H",
    "H",
    "H",
    "C",
    "H",
    "H",
    "H",
    "C",
    "H",
    "H",
    "H",
    "C",
    "H",
    "H",
    "H",
]
TMS_RHF_STO3G = [
    [-0.0, -0.0, -0.0],
    [1.076, 1.0043, 1.1431],
    [1.7162, 1.6798, 0.5909],
    [0.4817, 1.6021, 1.8221],
    [1.7147, 0.3681, 1.742],
    [1.076, -1.0043, -1.1431],
    [1.7162, -1.6798, -0.5909],
    [0.4817, -1.6021, -1.8221],
    [1.7147, -0.3681, -1.742],
    [-1.076, 1.1431, -1.0043],
    [-1.7162, 0.5909, -1.6798],
    [-1.7147, 1.742, -0.3681],
    [-0.4817, 1.8221, -1.6021],
    [-1.076, -1.1431, 1.0043],
    [-1.7162, -0.5909, 1.6798],
    [-1.7147, -1.742, 0.3681],
    [-0.4817, -1.8221, 1.6021],
]


def test_tms_shift_is_zero_at_rhf_sto3g():
    from quantui.nmr_calc import run_nmr_calc

    tms = Molecule(TMS_ATOMS, TMS_RHF_STO3G)
    result = run_nmr_calc(
        tms, method="RHF", basis="STO-3G", progress_stream=io.StringIO()
    )
    assert result.is_fallback_reference is False
    shifts = result.chemical_shifts_ppm
    h = [shifts[i] for i, a in enumerate(TMS_ATOMS) if a == "H"]
    c = [shifts[i] for i, a in enumerate(TMS_ATOMS) if a == "C"]
    assert max(abs(x) for x in h) < 0.1
    assert max(abs(x) for x in c) < 0.5


def test_every_row_has_h_and_c_and_the_default_is_b3lyp_631gs():
    from quantui.config import NMR_DEFAULT_REFERENCE, NMR_REFERENCE_SHIELDINGS

    for key, row in NMR_REFERENCE_SHIELDINGS.items():
        assert set(row) >= {"H", "C"}, key
        assert 28.0 < row["H"] < 36.0 and 170.0 < row["C"] < 260.0, key
    assert NMR_DEFAULT_REFERENCE == NMR_REFERENCE_SHIELDINGS["B3LYP/6-31G*"]
