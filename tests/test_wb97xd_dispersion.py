"""wB97X-D includes its own dispersion term (audit SCF-2).

LibXC's ``hyb_gga_xc_wb97x_d`` is only the exchange-correlation part, and
PySCF's dispersion module does not implement the functional's
Chai-Head-Gordon atom-pair term, so "wB97X-D" ran with no dispersion at all.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys

import numpy as np
import pytest

from quantui.molecule import Molecule

pyscf = pytest.importorskip("pyscf")

WATER_DIMER = Molecule(
    ["O", "H", "H", "O", "H", "H"],
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.587],
        [0.0, -0.757, 0.587],
        [0.0, 0.0, 2.9],
        [0.9, 0.0, 3.2],
        [-0.2, 0.0, 3.85],
    ],
)


def _mol(m: Molecule, basis: str = "sto-3g"):
    from pyscf import gto

    return gto.M(
        atom=m.to_pyscf_format(), basis=basis, spin=m.multiplicity - 1, verbose=0
    )


def test_derivatives_match_finite_differences():
    from quantui.chg_dispersion import chg_dispersion

    mol = _mol(WATER_DIMER)
    grad, hess = chg_dispersion(mol, 1), chg_dispersion(mol, 2)
    x0, eps = mol.atom_coords(), 1e-4
    for a in range(mol.natm):
        for k in range(3):
            xp, xm = x0.copy(), x0.copy()
            xp[a, k] += eps
            xm[a, k] -= eps
            mp = mol.set_geom_(xp, unit="Bohr", inplace=False)
            mm = mol.set_geom_(xm, unit="Bohr", inplace=False)
            fd = (chg_dispersion(mp) - chg_dispersion(mm)) / (2 * eps)
            assert grad[a, k] == pytest.approx(fd, abs=1e-9)
            fd_h = (chg_dispersion(mp, 1) - chg_dispersion(mm, 1)) / (2 * eps)
            assert np.allclose(hess[:, a, :, k], fd_h, atol=1e-9)


def test_pair_term_value():
    # One H...H pair at 3 Angstrom, by hand from the published form.
    from quantui.chg_dispersion import chg_dispersion

    mol = _mol(Molecule(["H", "H"], [[0, 0, 0], [0, 0, 3.0]], multiplicity=3))
    c6 = 0.14 * (10 / 0.52917721092) ** 6 / 2625499.639
    r = 3.0 / 0.52917721092
    rr = 2 * 1.001 / 0.52917721092
    expected = -c6 / (r**6 * (1 + 6.0 * (r / rr) ** -12))
    assert chg_dispersion(mol) == pytest.approx(expected, rel=1e-12)


def test_single_point_energy_includes_the_term():
    from quantui.chg_dispersion import chg_dispersion
    from quantui.session_calc import run_in_session

    res = run_in_session(WATER_DIMER, method="wB97X-D", basis="STO-3G", verbose=0)
    assert res.dispersion_applied is True

    from pyscf import dft

    mf = dft.RKS(_mol(WATER_DIMER))
    mf.xc = "hyb_gga_xc_wb97x_d"
    mf.verbose = 0
    xc_only = mf.kernel()
    e_disp = chg_dispersion(_mol(WATER_DIMER))
    assert e_disp < -1e-4
    assert res.energy_hartree - xc_only == pytest.approx(e_disp, abs=1e-7)


def test_gradient_and_hessian_include_the_term():
    from pyscf import dft

    from quantui.chg_dispersion import chg_dispersion
    from quantui.session_calc import maybe_apply_d3

    mol = _mol(Molecule(["O", "H", "H"], WATER_DIMER.coordinates[:3]))

    def build(with_disp):
        mf = dft.RKS(mol)
        mf.xc = "hyb_gga_xc_wb97x_d"
        mf.verbose = 0
        if with_disp:
            mf, applied = maybe_apply_d3(mf, "wB97X-D")
            assert applied is True
        mf.kernel()
        return mf

    plain, full = build(False), build(True)
    g = full.nuc_grad_method().kernel() - plain.nuc_grad_method().kernel()
    assert np.allclose(g, chg_dispersion(mol, 1), atol=1e-8)
    h = full.Hessian().kernel() - plain.Hessian().kernel()
    assert np.allclose(h, chg_dispersion(mol, 2), atol=1e-7)


def test_elements_without_parameters_are_refused():
    from quantui.chg_dispersion import chg_dispersion

    mol = _mol(
        Molecule(["Pt", "H"], [[0, 0, 0], [0, 0, 1.6]], multiplicity=2), "def2-svp"
    )
    with pytest.raises(ValueError, match="H-Xe"):
        chg_dispersion(mol)


def test_exported_script_reproduces_the_app_energy(tmp_path):
    from quantui.calculator import PySCFCalculation
    from quantui.session_calc import run_in_session

    script = tmp_path / "wb97xd.py"
    PySCFCalculation(WATER_DIMER, "wB97X-D", "STO-3G").generate_calculation_script(
        script
    )
    out = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    m = re.search(r"Total energy:\s+(-?\d+\.\d+) Ha", out.stdout)
    assert m, out.stdout[-2000:]
    app = run_in_session(
        WATER_DIMER,
        method="wB97X-D",
        basis="STO-3G",
        verbose=0,
        progress_stream=io.StringIO(),
    )
    assert float(m.group(1)) == pytest.approx(app.energy_hartree, abs=1e-6)
