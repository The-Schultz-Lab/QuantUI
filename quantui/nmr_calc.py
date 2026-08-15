"""
NMR chemical shift prediction using PySCF GIAO.

Computes isotropic NMR shielding tensors via GIAO (Gauge-Including
Atomic Orbitals) and converts to ¹H/¹³C chemical shifts relative to
TMS using tabulated reference constants from config.py.

Typical usage::

    from quantui.nmr_calc import run_nmr_calc
    result = run_nmr_calc(molecule, method="B3LYP", basis="6-31G*")
    for atom_idx, delta_ppm in result.h_shifts():
        print(f"H-{atom_idx+1}: {delta_ppm:.2f} ppm")
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .molecule import Molecule

logger = logging.getLogger(__name__)


@dataclass
class NMRResult:
    """Structured output from an NMR shielding calculation."""

    atom_symbols: List[str]
    shielding_iso_ppm: List[float]
    chemical_shifts_ppm: Dict[int, float]  # atom_index → δ (ppm), ¹H and ¹³C only
    method: str
    basis: str
    formula: str
    reference_compound: str = "TMS"
    converged: bool = True
    # Fix (2026-07-14): which config.NMR_REFERENCE_SHIELDINGS entry
    # was actually applied, and whether it's an exact match for method/basis
    # or a fallback. NMR_REFERENCE_SHIELDINGS only tabulates a handful of
    # method/basis combinations; any other combination previously fell back
    # to the B3LYP/6-31G* constants with no record of it anywhere, so
    # chemical shifts could be silently offset by several ppm with no way
    # for the student to know the reference wasn't calibrated for their
    # method/basis.
    reference_key: str = ""
    is_fallback_reference: bool = False

    def h_shifts(self) -> List[Tuple[int, float]]:
        """(atom_index, δ ppm) pairs for all H atoms in molecule order."""
        return [
            (i, d)
            for i, d in sorted(self.chemical_shifts_ppm.items())
            if self.atom_symbols[i] == "H"
        ]

    def c_shifts(self) -> List[Tuple[int, float]]:
        """(atom_index, δ ppm) pairs for all C atoms in molecule order."""
        return [
            (i, d)
            for i, d in sorted(self.chemical_shifts_ppm.items())
            if self.atom_symbols[i] == "C"
        ]


def resolve_nmr_reference(
    method: str, basis: str
) -> Tuple[Dict[str, float], str, bool]:
    """Resolve TMS reference shielding constants for ``method``/``basis``.

    Looks up ``config.NMR_REFERENCE_SHIELDINGS`` case-insensitively (keys
    there are declared as e.g. ``"B3LYP/6-31G*"``). Returns
    ``(ref_map, matched_key, is_fallback)``:

    - ``ref_map``: the ``{"H": ..., "C": ...}`` shielding constants to use.
    - ``matched_key``: the table key that was actually applied.
    - ``is_fallback``: ``True`` when no entry exists for this exact
      method/basis and ``config.NMR_DEFAULT_REFERENCE`` (B3LYP/6-31G*) was
      substituted instead — chemical shifts computed with a substituted
      reference can be off by several ppm relative to properly calibrated
      constants for the requested level of theory.
    """
    from . import config as _config

    requested_key = f"{method}/{basis}"
    requested_upper = requested_key.upper()
    for table_key, ref_map in _config.NMR_REFERENCE_SHIELDINGS.items():
        if table_key.upper() == requested_upper:
            return ref_map, table_key, False
    return _config.NMR_DEFAULT_REFERENCE, "B3LYP/6-31G*", True


def run_nmr_calc(
    molecule: Molecule,
    method: str = "B3LYP",
    basis: str = "6-31G*",
    progress_stream=None,
) -> NMRResult:
    """Run NMR shielding calculation and return ¹H/¹³C chemical shifts.

    Uses PySCF GIAO (Gauge-Including Atomic Orbitals) formalism.
    Chemical shifts are reported relative to TMS using reference constants
    from :data:`~quantui.config.NMR_REFERENCE_SHIELDINGS`.

    Args:
        molecule: Validated :class:`~quantui.molecule.Molecule` object.
        method: SCF or DFT method. Recommended: B3LYP.
        basis: Basis set. Recommended: 6-31G* or better.
        progress_stream: Optional writable text stream for PySCF output.

    Returns:
        :class:`NMRResult` with per-atom shieldings and ¹H/¹³C shifts.

    Raises:
        ImportError: If PySCF is not installed.
        RuntimeError: If the SCF or GIAO-NMR calculation fails.
    """
    # Post-HF methods (MP2/CCSD/CCSD(T)) have no special-casing below —
    # without this guard, method='CCSD' silently falls into the DFT
    # branch (sets mf.xc = "CCSD") and fails deep inside PySCF with a
    # cryptic "LibXCFunctional: name 'CCSD' not found" instead of a clear
    # message. GIAO-NMR shielding is not defined for these methods here.
    from . import config as _config

    if method.strip().upper() in _config.POST_HF_METHODS:
        raise ValueError(
            f"'{method}' is a post-HF method and cannot be used for NMR "
            "shielding — use RHF, UHF, or a DFT functional instead."
        )

    try:
        from pyscf import dft, gto, scf
    except ImportError as exc:
        raise ImportError(
            "PySCF is not installed — cannot run NMR calculations.\n"
            "Note: PySCF is Linux / macOS / WSL only."
        ) from exc

    stream = progress_stream if progress_stream is not None else sys.stdout

    # See quantui/c_stderr.py — captures fd-2 stderr
    # from libcint / BLAS / LAPACK / GIAO / NMR-CPHF C code and relays to
    # ``stream`` on exit. POSIX-only; no-op on Windows.
    from quantui.c_stderr import capture_c_stderr

    with capture_c_stderr(stream):
        return _run_nmr_calc_body(
            molecule=molecule,
            method=method,
            basis=basis,
            progress_stream=progress_stream,
            _dft=dft,
            _gto=gto,
            _scf=scf,
            stream=stream,
        )


# Fix (2026-07-14): bump this whenever the patch bodies below
# change — it doubles as the idempotency sentinel's value, so a bumped
# version forces re-patching instead of silently keeping stale closures
# from an older QuantUI version installed earlier in the process.
_NMR_COMPAT_PATCH_VERSION = 1
_NMR_PATCH_VERSION_ATTR = "_quantui_nmr_compat_patch_version"


def _ensure_nmr_compat_patches_applied() -> None:
    """Idempotently patch pyscf.prop.nmr for QuantUI-specific compatibility fixes.

    Both patches below used to be applied unconditionally on every NMR
    calculation, re-defining the same closures and reassigning the same
    module attributes on every call even though nothing about the
    installed pyscf/pyscf-properties changes between calls. Each patch
    is now a one-time, idempotent operation per process: a sentinel
    attribute (versioned via ``_NMR_COMPAT_PATCH_VERSION``) on the
    currently-installed function is checked first, so repeated NMR runs
    are no-ops here.
    """
    # pyscf-properties 0.1.0 gen_vind hardcodes reshape(3, nmo, nocc).
    # pyscf 2.x krylov reduces the batch below 3 via linear-dependency masking,
    # causing "cannot reshape array of size N into shape (3,nmo,nocc)".
    # Patch gen_vind to use reshape(-1, nmo, nocc) so any batch size works.
    try:
        from functools import reduce as _reduce_nmr

        import numpy as _np
        import pyscf.prop.nmr.rhf as _prop_nmr_rhf
        from pyscf import lib as _pyscf_lib_nmr

        if (
            getattr(_prop_nmr_rhf.gen_vind, _NMR_PATCH_VERSION_ATTR, 0)
            < _NMR_COMPAT_PATCH_VERSION
        ):

            def _fixed_gen_vind(mf_arg, mo_coeff, mo_occ):
                vresp = mf_arg.gen_response(singlet=True, hermi=2)
                occidx = mo_occ > 0
                orbo = mo_coeff[:, occidx]
                nocc = orbo.shape[1]
                _nao, nmo = mo_coeff.shape

                def vind(mo1):
                    _mo1 = _np.asarray(mo1).reshape(-1, nmo, nocc)
                    dm1 = _np.asarray(
                        [
                            _reduce_nmr(_np.dot, (mo_coeff, x * 2, orbo.T.conj()))
                            for x in _mo1
                        ]
                    )
                    dm1 = dm1 - dm1.transpose(0, 2, 1).conj()
                    v1mo = _pyscf_lib_nmr.einsum(
                        "xpq,pi,qj->xij", vresp(dm1), mo_coeff.conj(), orbo
                    )
                    return v1mo.ravel()

                return vind

            setattr(_fixed_gen_vind, _NMR_PATCH_VERSION_ATTR, _NMR_COMPAT_PATCH_VERSION)
            _prop_nmr_rhf.gen_vind = _fixed_gen_vind
    except (ImportError, AttributeError) as exc:  # noqa: BLE001 — optional probe
        logger.debug("pyscf.prop.nmr.rhf.gen_vind patch not applied: %s", exc)

    # pyscf-properties 0.1.0 get_vxc_giao computes
    #   blksize = min(int(X*BLKSIZE)*BLKSIZE, ngrids)
    # which equals ngrids when ngrids < X*BLKSIZE, and ngrids may not be
    # divisible by BLKSIZE.  pyscf 2.x block_loop asserts blksize%BLKSIZE==0.
    # Patch get_vxc_giao to round blksize down to the nearest BLKSIZE multiple.
    try:
        import numpy as _np_rks
        import pyscf.prop.nmr.rks as _prop_nmr_rks
        from pyscf.dft import numint as _numint_rks

        if (
            getattr(_prop_nmr_rks.get_vxc_giao, _NMR_PATCH_VERSION_ATTR, 0)
            < _NMR_COMPAT_PATCH_VERSION
        ):

            def _fixed_get_vxc_giao(
                ni, mol, grids, xc_code, dms, max_memory=2000, verbose=None
            ):
                xctype = ni._xc_type(xc_code)
                make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi=1)
                ngrids = len(grids.weights)
                _BLKSIZE = _numint_rks.BLKSIZE
                _raw_blk = int(max_memory / 12 * 1e6 / 8 / nao / _BLKSIZE) * _BLKSIZE
                blksize = max(_BLKSIZE, (min(_raw_blk, ngrids) // _BLKSIZE) * _BLKSIZE)
                shls_slice = (0, mol.nbas)
                ao_loc = mol.ao_loc_nr()

                vmat = _np_rks.zeros((3, nao, nao))
                if xctype == "LDA":
                    buf = _np_rks.empty((4, blksize, nao))
                    ao_deriv = 0
                    for ao, mask, weight, coords in ni.block_loop(
                        mol, grids, nao, ao_deriv, max_memory, blksize=blksize, buf=buf
                    ):
                        rho = make_rho(0, ao, mask, "LDA")
                        vxc = ni.eval_xc(xc_code, rho, 0, deriv=1)[1]
                        vrho = vxc[0]
                        aow = _np_rks.einsum("pi,p->pi", ao, weight * vrho)
                        giao = mol.eval_gto(
                            "GTOval_ig", coords, comp=3, non0tab=mask, out=buf[1:]
                        )
                        vmat[0] += _numint_rks._dot_ao_ao(
                            mol, aow, giao[0], mask, shls_slice, ao_loc
                        )
                        vmat[1] += _numint_rks._dot_ao_ao(
                            mol, aow, giao[1], mask, shls_slice, ao_loc
                        )
                        vmat[2] += _numint_rks._dot_ao_ao(
                            mol, aow, giao[2], mask, shls_slice, ao_loc
                        )
                        rho = vxc = vrho = aow = None
                elif xctype == "GGA":
                    buf = _np_rks.empty((10, blksize, nao))
                    ao_deriv = 1
                    for ao, mask, weight, coords in ni.block_loop(
                        mol, grids, nao, ao_deriv, max_memory, blksize=blksize, buf=buf
                    ):
                        rho = make_rho(0, ao, mask, "GGA")
                        vxc = ni.eval_xc(xc_code, rho, 0, deriv=1)[1]
                        vrho, vsigma = vxc[:2]
                        wv = _np_rks.empty_like(rho)
                        wv[0] = weight * vrho
                        wv[1:] = rho[1:] * (weight * vsigma * 2)
                        aow = _np_rks.einsum("npi,np->pi", ao[:4], wv)
                        giao = mol.eval_gto(
                            "GTOval_ig", coords, 3, non0tab=mask, out=buf[4:]
                        )
                        vmat[0] += _numint_rks._dot_ao_ao(
                            mol, aow, giao[0], mask, shls_slice, ao_loc
                        )
                        vmat[1] += _numint_rks._dot_ao_ao(
                            mol, aow, giao[1], mask, shls_slice, ao_loc
                        )
                        vmat[2] += _numint_rks._dot_ao_ao(
                            mol, aow, giao[2], mask, shls_slice, ao_loc
                        )
                        giao = mol.eval_gto(
                            "GTOval_ipig", coords, 9, non0tab=mask, out=buf[1:]
                        )
                        _prop_nmr_rks._gga_sum_(
                            vmat, mol, ao, giao, wv, mask, shls_slice, ao_loc
                        )
                        rho = vxc = vrho = vsigma = wv = aow = None
                elif xctype == "MGGA":
                    raise NotImplementedError("meta-GGA")

                return vmat - vmat.transpose(0, 2, 1)

            setattr(
                _fixed_get_vxc_giao, _NMR_PATCH_VERSION_ATTR, _NMR_COMPAT_PATCH_VERSION
            )
            _prop_nmr_rks.get_vxc_giao = _fixed_get_vxc_giao
    except (ImportError, AttributeError) as exc:  # noqa: BLE001 — optional probe
        logger.debug("pyscf.prop.nmr.rks.get_vxc_giao patch not applied: %s", exc)


def _run_nmr_calc_body(
    *,
    molecule: Molecule,
    method: str,
    basis: str,
    progress_stream: Any,
    _dft: Any,
    _gto: Any,
    _scf: Any,
    stream: Any,
) -> NMRResult:
    """Inner body of :func:`run_nmr_calc` (split out for stderr-capture wrap)."""
    dft, gto, scf = _dft, _gto, _scf

    import numpy as _np

    from . import config as _config
    from .session_calc import maybe_apply_d3, resolve_xc

    mol = gto.Mole()
    mol.atom = molecule.to_pyscf_format()
    mol.basis = basis
    mol.charge = molecule.charge
    mol.spin = molecule.multiplicity - 1
    mol.verbose = 4
    mol.stdout = stream
    mol.build()

    method_upper = method.upper()
    if method_upper == "RHF":
        mf = scf.RHF(mol)
    elif method_upper == "UHF":
        mf = scf.UHF(mol)
    else:
        # Route through resolve_xc + maybe_apply_d3 so
        # wB97X-D / PBE-D3 work for NMR calcs (was using raw _XC_ALIAS
        # lookup before, which would fail for wB97X-D after the alias
        # change to "wb97x" + external D3).
        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf.xc = resolve_xc(method)
        mf = maybe_apply_d3(mf, method, progress_stream=stream)

    # Density fitting (RI), opt-in (M-DF). Off by default. See DF.5: DF shifts
    # absolute shieldings, but chemical shifts are differences so the error
    # largely cancels — this is flagged for explicit validation before DF is
    # ever defaulted on for NMR.
    from .density_fitting import try_density_fit as _try_density_fit

    mf, _ = _try_density_fit(mf)

    # Cooperative cancel between SCF cycles.
    from .cancellation import attach_scf_cancel_callback, cancel_check_from_stream
    from .log_utils import emit_status

    attach_scf_cancel_callback(mf, cancel_check_from_stream(stream))

    emit_status(stream, "Running SCF…")
    try:
        mf.kernel()
    except Exception as exc:
        raise RuntimeError(
            f"SCF failed for {molecule.get_formula()} ({method}/{basis}): {exc}"
        ) from exc

    converged = bool(getattr(mf, "converged", False))

    # pyscf.nmr does not exist in released pyscf; use pyscf.prop.nmr (pyscf-properties).
    _pyscf_nmr: Any = None
    try:
        import pyscf.prop.nmr

        _pyscf_nmr = pyscf.prop.nmr
    except ImportError as exc:
        raise ImportError(
            "PySCF NMR module not found. "
            "Install pyscf-properties: pip install pyscf-properties"
        ) from exc

    _ensure_nmr_compat_patches_applied()

    emit_status(stream, "Computing NMR shielding tensors (GIAO)…")
    try:
        if method_upper == "RHF":
            nmr_obj = _pyscf_nmr.RHF(mf)
        elif method_upper == "UHF":
            nmr_obj = _pyscf_nmr.UHF(mf)
        else:
            nmr_obj = _pyscf_nmr.RKS(mf) if mol.spin == 0 else _pyscf_nmr.UKS(mf)
        tensors = nmr_obj.kernel()
    except Exception as exc:
        raise RuntimeError(
            f"NMR shielding failed for {molecule.get_formula()}: {exc}"
        ) from exc

    shielding_iso: List[float] = []
    for tensor in tensors:
        arr = _np.array(tensor)
        if arr.ndim == 2:
            shielding_iso.append(float(_np.trace(arr) / 3.0))
        else:
            shielding_iso.append(float(arr))

    ref_map, matched_ref_key, is_fallback_ref = resolve_nmr_reference(method, basis)
    ref_H = float(ref_map.get("H", _config.NMR_DEFAULT_REFERENCE["H"]))
    ref_C = float(ref_map.get("C", _config.NMR_DEFAULT_REFERENCE["C"]))

    atoms = list(molecule.atoms)
    chemical_shifts: Dict[int, float] = {}
    for i, (atom, sigma) in enumerate(zip(atoms, shielding_iso)):
        if atom == "H":
            chemical_shifts[i] = round(ref_H - sigma, 2)
        elif atom == "C":
            chemical_shifts[i] = round(ref_C - sigma, 2)

    return NMRResult(
        atom_symbols=atoms,
        shielding_iso_ppm=shielding_iso,
        chemical_shifts_ppm=chemical_shifts,
        method=method,
        basis=basis,
        formula=molecule.get_formula(),
        converged=converged,
        reference_key=matched_ref_key,
        is_fallback_reference=is_fallback_ref,
    )
