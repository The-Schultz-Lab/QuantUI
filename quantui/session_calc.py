"""
In-session quantum chemistry calculation using PySCF directly.

Runs SCF calculations in the current Jupyter kernel and returns structured
data. PySCF's verbose output is routed through mol.stdout so the notebook
can display live SCF iterations in a widget.

Platform notes
--------------
PySCF is **Linux / macOS / WSL only** — not available on native Windows.
This module imports PySCF lazily inside :func:`run_in_session` so it can
be imported safely on any platform without raising at import time.

Typical notebook usage
----------------------
>>> from quantui import run_in_session, SessionResult
>>> result = run_in_session(molecule, method="RHF", basis="6-31G")
>>> print(result.summary())
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import IO, Any, List, Optional

from .molecule import Molecule

logger = logging.getLogger(__name__)

# NIST 2018 CODATA — consistent with PySCF's internal constant
HARTREE_TO_EV: float = 27.211386245988


# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class SessionResult:
    """
    Structured output from a completed in-session quantum chemistry calculation.

    Attributes:
        energy_hartree: Total SCF energy in Hartrees.
        homo_lumo_gap_ev: HOMO-LUMO gap in electronvolts, or ``None`` if the
            gap cannot be determined (e.g. open-shell UHF with complex orbital
            occupations, or too few occupied orbitals).
        converged: ``True`` if the SCF iterations reached the convergence
            threshold; ``False`` if the maximum iteration count was hit.
        n_iterations: Number of SCF macro-iterations completed.  May be
            ``-1`` if the underlying calculator does not expose this.
        method: Calculation method used (e.g. ``'RHF'``, ``'UHF'``).
        basis: Basis set used (e.g. ``'6-31G'``, ``'STO-3G'``).
        formula: Hill-notation molecular formula of the input molecule.
    """

    energy_hartree: float
    homo_lumo_gap_ev: Optional[float]
    converged: bool
    n_iterations: int
    method: str
    basis: str
    formula: str
    atom_symbols: Optional[List[str]] = None
    mulliken_charges: Optional[List[float]] = None
    dipole_moment_debye: Optional[float] = None
    mp2_correlation_hartree: Optional[float] = None
    # CCSD post-HF correlation energy (Hartree), populated when method is
    # ``"CCSD"`` or ``"CCSD(T)"``. ``None`` for HF/DFT/MP2 paths. The
    # ``energy_hartree`` field already includes this correlation when set
    # (matches the existing ``mp2_correlation_hartree`` convention).
    ccsd_correlation_hartree: Optional[float] = None
    # CCSD(T) perturbative-triples correction (Hartree), populated only when
    # method is ``"CCSD(T)"``. ``None`` for plain CCSD. Again, included in
    # ``energy_hartree`` when set.
    ccsd_t_correction_hartree: Optional[float] = None
    # GPU offload status (M-GPU / GPU.2). ``gpu_used`` is True only when the
    # SCF object was successfully migrated to gpu4pyscf for this run.
    # ``gpu_name`` carries the CUDA device name when ``gpu_used`` is True so
    # the result card can show *which* GPU ran the calc.
    gpu_used: bool = False
    gpu_name: Optional[str] = None
    solvent: Optional[str] = None
    mo_energy_hartree: Optional[Any] = None  # np.ndarray (n_mo,) or (2, n_mo) UHF
    mo_occ: Optional[Any] = None  # np.ndarray (n_mo,) or (2, n_mo) UHF
    mo_coeff: Optional[Any] = None  # np.ndarray (n_ao, n_mo) or (2, n_ao, n_mo) UHF
    pyscf_mol_atom: Optional[Any] = None  # list of (symbol, [x,y,z]) tuples (Angstrom)
    pyscf_mol_basis: Optional[str] = None  # basis set string for cube generation

    @property
    def energy_ev(self) -> float:
        """Total energy converted to electronvolts."""
        return self.energy_hartree * HARTREE_TO_EV

    def summary(self) -> str:
        """Return a multi-line human-readable result summary suitable for printing."""
        lines = [
            "=" * 60,
            "Calculation Results",
            "=" * 60,
            f"  Molecule      : {self.formula}",
            f"  Method/Basis  : {self.method}/{self.basis}",
            f"  SCF converged : {'Yes' if self.converged else '❌ NO — treat results with caution'}",
            f"  Iterations    : {self.n_iterations}",
            f"  Total energy  : {self.energy_hartree:.8f} Ha",
        ]
        if self.homo_lumo_gap_ev is not None:
            lines.append(f"  HOMO-LUMO gap : {self.homo_lumo_gap_ev:.4f} eV")
        lines += [
            "=" * 60,
            (
                "✅ Calculation completed successfully!"
                if self.converged
                else "⚠️  SCF did not converge — try a different starting geometry, basis, or method."
            ),
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================================
# Main function
# ============================================================================


# Maps QuantUI display names → PySCF xc strings where they differ.
_XC_ALIAS: dict = {
    "M06-L": "m06l",
    "wB97X-D": "wb97x-d",
    "CAM-B3LYP": "camb3lyp",
    "PBE-D3": "pbe",  # base functional; D3 applied separately
}
# Methods that require Grimme D3 dispersion correction via pyscf.dftd3.
_NEEDS_D3: frozenset = frozenset({"PBE-D3"})


def run_in_session(
    molecule: Molecule,
    method: str = "RHF",
    basis: str = "6-31G",
    verbose: int = 4,
    progress_stream: Optional[IO[str]] = None,
    solvent: Optional[str] = None,
) -> SessionResult:
    """
    Run a quantum chemistry calculation in the current kernel using PySCF.

    Returns a :class:`SessionResult` with structured data — energy,
    HOMO-LUMO gap, convergence status — rather than requiring callers to
    parse PySCF stdout.

    PySCF's verbose log is routed to *progress_stream* (or ``sys.stdout``
    if not provided) so live SCF iteration output can still be displayed in a
    Jupyter widget by passing a stream-backed output widget.

    Args:
        molecule: Validated :class:`~quantui.molecule.Molecule` object.
        method: SCF method — ``'RHF'`` for closed-shell molecules or
            ``'UHF'`` for open-shell / radical species.  Default: ``'RHF'``.
        basis: Basis set name as recognised by PySCF (e.g. ``'STO-3G'``,
            ``'6-31G'``, ``'6-31G*'``, ``'cc-pVDZ'``).  Default: ``'6-31G'``.
        verbose: PySCF verbosity level (0 = silent … 9 = very detailed).
            Level 3 prints per-iteration SCF energies; level 4 adds
            convergence diagnostics.  Default: 3.
        progress_stream: Optional writable text stream.  All PySCF output
            during the calculation is written here.  Pass a widget-backed
            stream (e.g. ``_WidgetStream``) in the notebook for live display;
            leave ``None`` to write to ``sys.stdout``.

    Returns:
        :class:`SessionResult` containing energy, HOMO-LUMO gap, convergence
        information, and metadata.

    Raises:
        ImportError: If PySCF is not installed.
        ValueError: If an unsupported method is requested.
        RuntimeError: If PySCF raises an unexpected exception during the
            calculation (original exception is chained).
    """
    # --- Dependency check ---
    try:
        from pyscf import dft, gto, scf
    except ImportError as exc:
        raise ImportError(
            "PySCF is not installed — cannot run in-session calculations.\n"
            "  conda install -c conda-forge pyscf\n"
            "Note: PySCF is Linux / macOS / WSL only."
        ) from exc

    stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout

    # M-STDERR / STDERR.1: capture C-level (fd-2) stderr from libcint / BLAS
    # / LAPACK and relay it to ``stream`` on exit. Without this wrapper, the
    # bytes surface as red text above the cell output in Voilà / Jupyter.
    # POSIX-only; no-op on Windows. See quantui/c_stderr.py for design.
    from quantui.c_stderr import capture_c_stderr

    with capture_c_stderr(stream):
        return _run_session_calc_body(
            molecule=molecule,
            method=method,
            basis=basis,
            verbose=verbose,
            progress_stream=progress_stream,
            solvent=solvent,
            _dft=dft,
            _gto=gto,
            _scf=scf,
            stream=stream,
        )


def _run_session_calc_body(
    *,
    molecule: Molecule,
    method: str,
    basis: str,
    verbose: int,
    progress_stream: Optional[IO[str]],
    solvent: Optional[str],
    _dft: Any,
    _gto: Any,
    _scf: Any,
    stream: IO[str],
) -> SessionResult:
    """Inner body of :func:`run_session_calc` — see public docstring.

    Split out so the public entry can wrap the C-heavy work in the
    ``capture_c_stderr`` context manager without re-indenting ~150 lines.
    Imports of ``pyscf`` are passed through so the dependency check stays
    in the public entry (where its ImportError can reach the user via
    Python's normal stderr).
    """
    dft, gto, scf = _dft, _gto, _scf

    # --- Validate method ---
    from . import config as _config

    if method.upper() not in [m.upper() for m in _config.SUPPORTED_METHODS]:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Supported: {', '.join(_config.SUPPORTED_METHODS)}"
        )

    # --- Build PySCF Mole object ---
    mol = gto.Mole()
    mol.atom = molecule.to_pyscf_format()
    mol.basis = basis
    mol.charge = molecule.charge
    mol.spin = molecule.multiplicity - 1
    mol.verbose = verbose
    mol.stdout = stream
    mol.build()

    # --- Select SCF method ---
    method_upper = method.upper()
    # Normalise to the key used in _XC_ALIAS / _NEEDS_D3 (preserve original case)
    _method_key = next((k for k in _XC_ALIAS if k.upper() == method_upper), method)

    if method_upper == "RHF":
        mf = scf.RHF(mol)
    elif method_upper == "UHF":
        mf = scf.UHF(mol)
    elif method_upper == "MP2":
        mf = scf.RHF(mol)  # MP2 runs on top of RHF
    elif method_upper in ("CCSD", "CCSD(T)"):
        # Coupled cluster builds on an RHF reference (M8.1). The correlation
        # energy (and optional perturbative-triples correction) is added
        # post-SCF below.
        mf = scf.RHF(mol)
    else:
        # DFT: resolve alias then auto-select RKS / UKS
        xc_string = _XC_ALIAS.get(_method_key, method)
        if mol.spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = xc_string
        # Apply D3 dispersion correction where needed
        if _method_key in _NEEDS_D3:
            try:
                from pyscf import dftd3 as _dftd3

                mf = _dftd3.dftd3(mf)
            except ImportError:
                if progress_stream is not None:
                    progress_stream.write(
                        f"\n⚠  pyscf.dftd3 not available — running {method} "
                        "without D3 correction.\n"
                    )

    # --- Wrap with implicit solvent (PCM) if requested ---
    if solvent is not None:
        from . import config as _cfg

        _eps = _cfg.SOLVENT_OPTIONS.get(solvent)
        if _eps is not None:
            try:
                from pyscf.solvent import PCM as _PCM

                mf = _PCM(mf)
                mf.with_solvent.eps = _eps
            except (
                Exception
            ) as exc:  # noqa: BLE001 — optional probe (PySCF version drift)
                logger.debug(
                    "PCM solvent unavailable, falling back to gas phase: %s", exc
                )
                if progress_stream is not None:
                    progress_stream.write(
                        "\n⚠  PCM solvent unavailable — running in gas phase.\n"
                    )

    # --- Try GPU offload (M-GPU / GPU.1) ---
    # Migrate the SCF object to gpu4pyscf when (a) the package is installed,
    # (b) a CUDA device is available, and (c) the method is supported.
    # Failures fall back to CPU silently — the calc still runs. The
    # ``gpu_used`` + ``gpu_name`` fields on the SessionResult carry the
    # outcome so the UI can show which device produced the numbers.
    from .gpu_offload import try_to_gpu as _try_to_gpu

    mf, gpu_used, gpu_name = _try_to_gpu(mf, method_upper)
    if gpu_used and progress_stream is not None:
        try:
            progress_stream.write(f"\n🚀  GPU offload active — running on {gpu_name}\n")
        except Exception:  # noqa: BLE001 — cleanup (progress stream may be closed)
            pass

    # --- Run SCF ---
    try:
        energy_hartree = float(mf.kernel())
    except Exception as exc:
        raise RuntimeError(
            f"PySCF calculation failed for {molecule.get_formula()} "
            f"({method}/{basis}): {exc}"
        ) from exc

    # --- MP2 correlation energy (post-HF) ---
    mp2_correlation_hartree: Optional[float] = None
    if method_upper == "MP2":
        try:
            from pyscf import mp as _mp

            _mp2 = _mp.MP2(mf)
            _e_corr, _ = _mp2.kernel()
            mp2_correlation_hartree = float(_e_corr)
            energy_hartree += float(_e_corr)
        except Exception as exc:
            raise RuntimeError(
                f"MP2 correction failed for {molecule.get_formula()}: {exc}"
            ) from exc

    # --- Coupled cluster correlation (M8.1) ---
    # CCSD adds singles + doubles excitations on top of the RHF reference;
    # CCSD(T) adds a perturbative-triples correction on top of CCSD. Both
    # report their corrections as separate result fields so the UI can
    # show the HF reference + correlation breakdown (mirrors the MP2 path).
    ccsd_correlation_hartree: Optional[float] = None
    ccsd_t_correction_hartree: Optional[float] = None
    if method_upper in ("CCSD", "CCSD(T)"):
        try:
            from pyscf import cc as _cc

            _ccsd_obj = _cc.CCSD(mf)
            _e_corr_ccsd, _t1, _t2 = _ccsd_obj.kernel()
            ccsd_correlation_hartree = float(_e_corr_ccsd)
            energy_hartree += float(_e_corr_ccsd)
        except Exception as exc:
            raise RuntimeError(
                f"CCSD correction failed for {molecule.get_formula()}: {exc}"
            ) from exc
        if method_upper == "CCSD(T)":
            try:
                _e_t = _ccsd_obj.ccsd_t()
                ccsd_t_correction_hartree = float(_e_t)
                energy_hartree += float(_e_t)
            except Exception as exc:
                raise RuntimeError(
                    f"CCSD(T) triples correction failed "
                    f"for {molecule.get_formula()}: {exc}"
                ) from exc

    # --- Extract results from the mean-field object ---
    converged = bool(getattr(mf, "converged", False))
    n_iterations = int(getattr(mf, "cycles", -1))

    homo_lumo_gap_ev: Optional[float] = None
    try:
        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy
        import numpy as _np

        if isinstance(mo_energy, (list, _np.ndarray)) and hasattr(
            mo_energy[0], "__len__"
        ):
            # UHF: mo_energy is (2, n_mo) — use alpha spin for the gap estimate
            mo_energy_ref = mo_energy[0]
            mo_occ_ref = mo_occ[0]
        else:
            mo_energy_ref = mo_energy
            mo_occ_ref = mo_occ

        n_occ = int((_np.array(mo_occ_ref) > 0).sum())
        if 0 < n_occ < len(mo_energy_ref):
            homo_lumo_gap_ev = float(
                (mo_energy_ref[n_occ] - mo_energy_ref[n_occ - 1]) * HARTREE_TO_EV
            )
    except Exception as exc:
        logger.debug("HOMO-LUMO gap extraction failed (non-fatal): %s", exc)

    mulliken_charges: Optional[List[float]] = None
    dipole_moment_debye: Optional[float] = None
    if method_upper != "UHF":
        try:
            _, chg = mf.mulliken_pop(verbose=0)
            mulliken_charges = [float(c) for c in chg]
        except Exception as exc:
            logger.debug("Mulliken population extraction failed: %s", exc)
        try:
            import numpy as _np2

            dip = mf.dip_moment(verbose=0)
            dipole_moment_debye = float(_np2.linalg.norm(dip))
        except Exception as exc:
            logger.debug("Dipole moment extraction failed: %s", exc)

    # MO arrays for orbital visualization (non-fatal if extraction fails).
    #
    # GPU-offload note (BUG fix, 2026-05-25): when ``gpu4pyscf`` migrated
    # ``mf`` to the GPU, ``mf.mo_energy`` / ``mo_coeff`` / ``mo_occ`` are
    # CuPy arrays. ``numpy.array(cupy_array)`` raises ``TypeError`` (numpy
    # refuses implicit device transfers), so the bare ``except`` swallowed
    # it and we silently shipped a ``SessionResult`` with all MO fields
    # ``None``. That in turn made ``save_orbitals`` no-op (it short-
    # circuits when both ``mo_e`` and ``mo_occ`` are None), and history
    # replay of GPU-run geo-opts / single-points showed "Not available"
    # in the Energies + Isosurface panels. ``_to_numpy_array`` below
    # detects CuPy arrays and copies them to host via ``cupy.asnumpy``.
    _mo_energy_ha_arr: Optional[Any] = None
    _mo_occ_arr: Optional[Any] = None
    _mo_coeff_arr: Optional[Any] = None
    _pyscf_mol_atom: Optional[Any] = None
    _pyscf_mol_basis: Optional[str] = None

    def _to_numpy_array(arr: Any) -> Any:
        """Convert ``arr`` to a NumPy array, transferring from GPU if needed."""
        if arr is None:
            return None
        # CuPy arrays have a ``.get()`` method (synchronous device→host copy).
        # Probe for it rather than importing cupy, so the CPU-only path
        # doesn't pull cupy onto the import graph.
        get = getattr(arr, "get", None)
        if callable(get) and type(arr).__module__.startswith("cupy"):
            return _np.asarray(get())
        return _np.asarray(arr)

    try:
        _mo_energy_ha_arr = _to_numpy_array(mf.mo_energy)
        _mo_occ_arr = _to_numpy_array(mf.mo_occ)
        _mo_coeff_arr = _to_numpy_array(mf.mo_coeff)
        _pyscf_mol_atom = [
            (atom, list(map(float, coords)))
            for atom, coords in zip(molecule.atoms, molecule.coordinates)
        ]
        _pyscf_mol_basis = basis
    except Exception as exc:
        # Bug-A class (session 55): a silent failure here ships a
        # SessionResult with mo_coeff=None, which makes save_orbitals
        # no-op and breaks Energies + Isosurface panels on history
        # replay. Surface to the event log so a future regression is
        # visible in `quantui log tail` immediately.
        logger.warning(
            "MO array extraction failed for %s (%s/%s): %s",
            molecule.get_formula(),
            method,
            basis,
            exc,
        )
        try:
            from . import calc_log as _clog

            _clog.log_event(
                "mo_array_extract_failed",
                f"{method}/{basis} on {molecule.get_formula()}",
                error=str(exc)[:300],
                gpu_used=gpu_used,
            )
        except Exception:  # noqa: BLE001 — telemetry self-guard
            pass

    formula = molecule.get_formula()
    logger.info(
        "Session calculation: %s %s/%s  E=%.8f Ha  converged=%s  iters=%d",
        formula,
        method,
        basis,
        energy_hartree,
        converged,
        n_iterations,
    )

    return SessionResult(
        energy_hartree=energy_hartree,
        homo_lumo_gap_ev=homo_lumo_gap_ev,
        converged=converged,
        n_iterations=n_iterations,
        method=method,
        basis=basis,
        formula=formula,
        atom_symbols=list(molecule.atoms),
        mulliken_charges=mulliken_charges,
        dipole_moment_debye=dipole_moment_debye,
        mp2_correlation_hartree=mp2_correlation_hartree,
        ccsd_correlation_hartree=ccsd_correlation_hartree,
        ccsd_t_correction_hartree=ccsd_t_correction_hartree,
        gpu_used=gpu_used,
        gpu_name=gpu_name,
        solvent=solvent,
        mo_energy_hartree=_mo_energy_ha_arr,
        mo_occ=_mo_occ_arr,
        mo_coeff=_mo_coeff_arr,
        pyscf_mol_atom=_pyscf_mol_atom,
        pyscf_mol_basis=_pyscf_mol_basis,
    )
