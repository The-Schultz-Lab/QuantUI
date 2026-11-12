"""ωB97X-D's empirical dispersion term (Chai & Head-Gordon 2008).

ωB97X-D is the ωB97X range-separated hybrid *plus* an atom-pair dispersion
correction with Chai–Head-Gordon ("CHG") damping:

    E_disp = - sum_{i<j} C6_ij / ( R_ij^6 * (1 + a * (R_ij / Rr_ij)^-12) )

with a = 6.0, no global scaling (s6 = 1), C6_ij = sqrt(C6_i * C6_j) and
Rr_ij = R0_i + R0_j, using Grimme's 2006 (DFT-D2) atomic C6 and R0 values
[J.-D. Chai, M. Head-Gordon, Phys. Chem. Chem. Phys. 10, 6615 (2008);
S. Grimme, J. Comput. Chem. 27, 1787 (2006)].

LibXC's ``hyb_gga_xc_wb97x_d`` holds only the exchange-correlation part, and
PySCF's own dispersion module does not implement this term (it rejects
``wb97x-d``), so without this module "wB97X-D" ran as ωB97X-D's XC
functional with no dispersion at all.

:func:`install` hooks the term into PySCF's existing dispersion plumbing:
an SCF object with ``mf.disp = CHG_DISP`` gets it in its total energy
(``mf.scf_summary['dispersion']``), analytic gradients and analytic
Hessians, exactly where PySCF adds its own D3/D4 corrections. Every other
SCF object is untouched.

This file is deliberately self-contained (numpy + pyscf only): the exported
standalone script embeds its source so the script reproduces the app.
"""

from __future__ import annotations

import numpy as np

CHG_DISP = "chg-d2"

# Grimme (2006) DFT-D2: element -> (C6 / J nm^6 mol^-1, R0 / Angstrom).
# Same values as ASE's ase.calculators.vdwcorrection.vdWDB_Grimme06jcc.
_D2 = {
    "H": (0.14, 1.001),
    "He": (0.08, 1.012),
    "Li": (1.61, 0.825),
    "Be": (1.61, 1.408),
    "B": (3.13, 1.485),
    "C": (1.75, 1.452),
    "N": (1.23, 1.397),
    "O": (0.70, 1.342),
    "F": (0.75, 1.287),
    "Ne": (0.63, 1.243),
    "Na": (5.71, 1.144),
    "Mg": (5.71, 1.364),
    "Al": (10.79, 1.639),
    "Si": (9.23, 1.716),
    "P": (7.84, 1.705),
    "S": (5.57, 1.683),
    "Cl": (5.07, 1.639),
    "Ar": (4.61, 1.595),
    "K": (10.80, 1.485),
    "Ca": (10.80, 1.474),
    "Ga": (16.99, 1.650),
    "Ge": (17.10, 1.727),
    "As": (16.37, 1.760),
    "Se": (12.64, 1.771),
    "Br": (12.47, 1.749),
    "Kr": (12.01, 1.727),
    "Rb": (24.67, 1.628),
    "Sr": (24.67, 1.606),
    "In": (37.32, 1.672),
    "Sn": (38.71, 1.804),
    "Sb": (38.44, 1.881),
    "Te": (31.74, 1.892),
    "I": (31.50, 1.892),
    "Xe": (29.99, 1.881),
}
for _el in ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"):
    _D2[_el] = (10.80, 1.562)
for _el in ("Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"):
    _D2[_el] = (24.67, 1.639)

_DAMP_A = 6.0
_BOHR_ANG = 0.52917721092
# J nm^6 mol^-1 -> Hartree Bohr^6
_C6_TO_AU = (10.0 / _BOHR_ANG) ** 6 / 2625499.639


def _params(symbols):
    missing = sorted({s for s in symbols if s not in _D2})
    if missing:
        raise ValueError(
            "wB97X-D dispersion (Grimme D2 parameters) is defined for H-Xe "
            f"only; no parameters for {', '.join(missing)}. Choose another "
            "functional for this molecule."
        )
    c6 = np.array([_D2[s][0] for s in symbols]) * _C6_TO_AU
    r0 = np.array([_D2[s][1] for s in symbols]) / _BOHR_ANG
    return c6, r0


def _symbols(mol):
    from pyscf.data.elements import _std_symbol

    return [_std_symbol(mol.atom_pure_symbol(i)) for i in range(mol.natm)]


def chg_dispersion(mol, deriv=0):
    """CHG dispersion for a PySCF ``Mole``: energy, gradient or Hessian.

    ``deriv=0`` returns the energy (Hartree); ``1`` the gradient
    ``(natm, 3)`` in Hartree/Bohr; ``2`` the Hessian ``(natm, natm, 3, 3)``
    in Hartree/Bohr^2 (PySCF's Hessian layout).
    """
    coords = np.asarray(mol.atom_coords(), dtype=float)  # Bohr
    c6, r0 = _params(_symbols(mol))
    n = len(coords)
    energy = 0.0
    grad = np.zeros((n, 3))
    hess = np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(i + 1, n):
            d = coords[i] - coords[j]
            r = float(np.sqrt(d @ d))
            cij = float(np.sqrt(c6[i] * c6[j]))
            b = _DAMP_A * float(r0[i] + r0[j]) ** 12
            # E = -C6 / D(r),  D = r^6 + b r^-6  (= r^6 (1 + a (r/Rr)^-12))
            dd0 = r**6 + b * r**-6
            energy -= cij / dd0
            if deriv == 0:
                continue
            dd1 = 6.0 * r**5 - 6.0 * b * r**-7
            de = cij * dd1 / dd0**2  # dE/dr
            u = d / r
            grad[i] += de * u
            grad[j] -= de * u
            if deriv == 1:
                continue
            dd2 = 30.0 * r**4 + 42.0 * b * r**-8
            d2e = cij * (dd2 * dd0 - 2.0 * dd1**2) / dd0**3  # d2E/dr2
            uu = np.outer(u, u)
            block = d2e * uu + (de / r) * (np.eye(3) - uu)
            hess[i, i] += block
            hess[j, j] += block
            hess[i, j] -= block
            hess[j, i] -= block
    if deriv == 0:
        return energy
    if deriv == 1:
        return grad
    return hess


def _uses_chg(mf):
    return getattr(mf, "disp", None) == CHG_DISP


def install():
    """Hook :func:`chg_dispersion` into PySCF (idempotent).

    Wraps the four dispersion entry points PySCF's SCF, gradient and Hessian
    drivers call (``do_disp`` / ``get_dispersion``); objects whose
    ``disp`` is not :data:`CHG_DISP` go through PySCF's own code unchanged.
    """
    from pyscf import grad, hessian, scf
    from pyscf.grad import dispersion as _gd  # noqa: F401 — sets PySCF's hook first
    from pyscf.hessian import dispersion as _hd  # noqa: F401
    from pyscf.hessian import rhf as _hrhf

    scf_cls = scf.hf.SCF
    if getattr(scf_cls, "_quantui_chg_installed", False):
        return
    grad_cls = grad.rhf.GradientsBase
    hess_cls = _hrhf.HessianBase
    _ = hessian  # imported for the side effect of registering hessian modules

    orig_do_disp = scf_cls.do_disp
    orig_scf_disp = scf_cls.get_dispersion
    orig_grad_disp = grad_cls.get_dispersion
    orig_hess_disp = hess_cls.get_dispersion

    def do_disp(mf, *args, **kwargs):
        if _uses_chg(mf):
            return True
        return orig_do_disp(mf, *args, **kwargs)

    def scf_get_dispersion(mf, *args, **kwargs):
        if _uses_chg(mf):
            return chg_dispersion(mf.mol, 0)
        return orig_scf_disp(mf, *args, **kwargs)

    def grad_get_dispersion(mf_grad, *args, **kwargs):
        if _uses_chg(mf_grad.base):
            return chg_dispersion(mf_grad.mol, 1)
        return orig_grad_disp(mf_grad, *args, **kwargs)

    def hess_get_dispersion(hessobj, *args, **kwargs):
        if _uses_chg(hessobj.base):
            return chg_dispersion(hessobj.mol, 2)
        return orig_hess_disp(hessobj, *args, **kwargs)

    scf_cls.do_disp = do_disp
    scf_cls.get_dispersion = scf_get_dispersion
    grad_cls.get_dispersion = grad_get_dispersion
    hess_cls.get_dispersion = hess_get_dispersion
    scf_cls._quantui_chg_installed = True
