"""Golden-value regression tests for TD-DFT excitations and NMR shieldings.

Audit INFRA-2 (2026-11-10): the TD-DFT and NMR tests only checked shapes and
loose ranges (``len(...) <= 2``, ``-5 < δ < 15``). ``nmr_calc`` monkey-patches
pyscf-properties internals and ``pyscf-properties`` is unpinned, so a wrong
patch or an upstream change could shift every shielding and still pass CI.
IR, Raman, thermo, bond orders and ⟨S²⟩ already had reference values; these
two did not.

Values were computed with QuantUI's own entry points (PySCF 2.x, 2026-11-12)
at STO-3G on fixed geometries. The TD-DFT/TDHF energies and oscillator
strengths match plain ``mf.TDDFT()`` / ``mf.TDHF()`` to all printed digits.
The shieldings come from QuantUI's patched GIAO path (unpatched
pyscf-properties raises on this PySCF), so they pin current behaviour.

Tolerances: 1e-3 eV for excitation energies, 5e-4 for oscillator strengths,
0.05 ppm for isotropic shieldings — loose enough for grid/BLAS noise, far
tighter than any real regression.
"""

from __future__ import annotations

import io

import pytest

from quantui.molecule import Molecule

pytest.importorskip("pyscf")

EV_TOL = 1e-3
F_TOL = 5e-4
PPM_TOL = 0.05


def _water() -> Molecule:
    return Molecule(
        ["O", "H", "H"], [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]]
    )


def _methane() -> Molecule:
    d = 0.629
    return Molecule(
        ["C", "H", "H", "H", "H"],
        [[0, 0, 0], [d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]],
    )


# H2O / STO-3G, nstates=3. B3LYP -> full TD-DFT (mf.TDDFT); RHF -> TDHF/RPA.
TDDFT_GOLDEN = {
    "B3LYP": {
        "energies_ev": [11.658437, 13.960484, 14.511851],
        "f": [0.002334, 0.0, 0.064021],
    },
    "RHF": {
        "energies_ev": [13.144333, 15.122597, 16.663399],
        "f": [0.003256, 0.0, 0.066565],
    },
}

# Isotropic GIAO shieldings (ppm), atom order as in the molecule.
NMR_GOLDEN = {
    ("H2O", "RHF"): [365.613, 33.671, 33.671],
    ("H2O", "B3LYP"): [340.6806, 33.9798, 33.9798],
    ("CH4", "RHF"): [234.9298, 32.6631, 32.6631, 32.6631, 32.6631],
    ("CH4", "B3LYP"): [224.8538, 32.4004, 32.4004, 32.4004, 32.4004],
}


@pytest.mark.parametrize("method", sorted(TDDFT_GOLDEN))
def test_h2o_sto3g_excitations_match_golden(method):
    from quantui.tddft_calc import run_tddft_calc

    result = run_tddft_calc(
        _water(),
        method=method,
        basis="STO-3G",
        nstates=3,
        progress_stream=io.StringIO(),
    )
    golden = TDDFT_GOLDEN[method]

    assert result.converged
    assert result.excitation_energies_ev == pytest.approx(
        golden["energies_ev"], abs=EV_TOL
    )
    assert result.oscillator_strengths == pytest.approx(golden["f"], abs=F_TOL)


@pytest.mark.parametrize("molecule,method", sorted(NMR_GOLDEN))
def test_sto3g_isotropic_shieldings_match_golden(molecule, method):
    pytest.importorskip("pyscf.prop.nmr")
    from quantui.nmr_calc import run_nmr_calc

    mol = {"H2O": _water, "CH4": _methane}[molecule]()
    result = run_nmr_calc(
        mol, method=method, basis="STO-3G", progress_stream=io.StringIO()
    )

    assert result.converged
    assert result.shielding_iso_ppm == pytest.approx(
        NMR_GOLDEN[(molecule, method)], abs=PPM_TOL
    )
