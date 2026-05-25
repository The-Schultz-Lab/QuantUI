"""
Vibrational frequency analysis using PySCF's analytical Hessian.

Runs an SCF calculation and then computes the analytical Hessian to
obtain vibrational frequencies, zero-point vibrational energy (ZPVE),
and (where available) IR intensities.

Platform notes
--------------
Requires PySCF — Linux / macOS / WSL only.

Educational value
-----------------
* Students see which vibrational modes are IR-active and which are not.
* ZPVE correction shows how quantum mechanical zero-point motion contributes
  to molecular stability.
* Imaginary frequencies flag a transition state or saddle point — the
  geometry should be optimised first.

Typical usage
-------------
>>> from quantui.freq_calc import run_freq_calc
>>> result = run_freq_calc(molecule, method="RHF", basis="STO-3G")
>>> print(result.frequencies_cm1[:6])  # first 6 vibrational modes
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import IO, Any, List, Optional

from .molecule import Molecule
from .session_calc import HARTREE_TO_EV

logger = logging.getLogger(__name__)

# 1 cm^-1 = h·c·100 / E_h  (NIST 2018 CODATA)
_CM1_TO_HARTREE: float = 4.556335252912e-6

# Exact: 1 Hartree = HARTREE_TO_EV * e * N_A joules/mol
_HARTREE_TO_JMOL: float = 2625499.6  # J/mol per Hartree (NIST 2018 CODATA)


# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class ThermoData:
    """Thermochemical data from the harmonic approximation at 298.15 K / 1 atm.

    All energies are in Hartrees; entropy is in J/(mol·K).
    H and G include the SCF electronic energy.
    """

    zpve_hartree: float
    H_hartree: float
    S_jmol: float
    G_hartree: float
    temperature_k: float = 298.15


@dataclass
class FreqResult:
    """Structured output from a vibrational frequency analysis.

    Attributes:
        energy_hartree: SCF energy at the input geometry in Hartrees.
        homo_lumo_gap_ev: HOMO-LUMO gap in eV, or ``None``.
        converged: ``True`` if the SCF converged.
        n_iterations: Number of SCF macro-iterations.
        method: Calculation method (e.g. ``'RHF'``, ``'B3LYP'``).
        basis: Basis set (e.g. ``'STO-3G'``).
        formula: Hill-notation molecular formula.
        frequencies_cm1: Vibrational frequencies in cm⁻¹.  Negative values
            indicate imaginary frequencies (transition-state modes).
        ir_intensities: IR intensities in km/mol per mode.  Empty list if
            the IR calculation is not available.
        zpve_hartree: Zero-point vibrational energy in Hartrees, computed as
            ½·Σ(ν_i) for all positive-frequency modes.
    """

    energy_hartree: float
    homo_lumo_gap_ev: Optional[float]
    converged: bool
    n_iterations: int
    method: str
    basis: str
    formula: str
    frequencies_cm1: List[float] = field(default_factory=list)
    ir_intensities: List[float] = field(default_factory=list)
    zpve_hartree: float = 0.0
    thermo: Optional[ThermoData] = None
    displacements: Optional[List] = None
    """Normalized displacement vectors from PySCF harmonic analysis.

    Shape: ``(n_modes, n_atoms, 3)`` stored as a nested Python list.
    ``None`` if the Hessian calculation failed or PySCF version does not
    provide ``norm_mode``.
    """
    mo_energy_hartree: Optional[List] = None
    mo_occ: Optional[List] = None
    pyscf_mol_atom: Optional[List] = None
    pyscf_mol_basis: Optional[str] = None

    @property
    def energy_ev(self) -> float:
        """SCF energy in electronvolts."""
        return self.energy_hartree * HARTREE_TO_EV

    def n_real_modes(self) -> int:
        """Number of real (positive-frequency) vibrational modes."""
        return sum(1 for f in self.frequencies_cm1 if f > 0)

    def n_imaginary_modes(self) -> int:
        """Number of imaginary (negative-frequency) modes."""
        return sum(1 for f in self.frequencies_cm1 if f < 0)


# ============================================================================
# Main function
# ============================================================================


def run_freq_calc(
    molecule: Molecule,
    method: str = "RHF",
    basis: str = "STO-3G",
    progress_stream: Optional[IO[str]] = None,
) -> FreqResult:
    """Run SCF + analytical Hessian to obtain vibrational frequencies.

    The function first converges the SCF energy, then computes the analytical
    Hessian and performs a normal-mode analysis to extract frequencies and
    (optionally) IR intensities.

    For physically meaningful frequencies, the input geometry should be at
    (or near) a local energy minimum.  Frequencies from an unoptimised
    geometry will be large and potentially imaginary.

    Args:
        molecule: Validated :class:`~quantui.molecule.Molecule`.  Should be
            an optimised geometry for best results.
        method: SCF method — ``'RHF'``, ``'UHF'``, or a DFT functional
            name (e.g. ``'B3LYP'``).  Default: ``'RHF'``.
        basis: Basis set name.  Default: ``'STO-3G'``.
        progress_stream: Optional writable text stream for live PySCF output.

    Returns:
        :class:`FreqResult` with frequencies, ZPVE, and SCF properties.

    Raises:
        ImportError: If PySCF is not installed.
        RuntimeError: If the SCF calculation fails.  If the Hessian
            computation fails, frequencies are omitted and a warning is
            written to progress_stream — no exception is raised.
    """
    try:
        from pyscf import dft, gto, scf
        from pyscf.hessian import thermo as pyscf_thermo
    except ImportError as exc:
        raise ImportError(
            "PySCF is not installed — cannot run frequency analysis.\n"
            "PySCF requires Linux, macOS, or WSL."
        ) from exc

    stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout

    # M-STDERR / STDERR.1: see quantui/c_stderr.py — captures fd-2 stderr
    # from libcint / BLAS / LAPACK / Hessian C code and relays to ``stream``
    # on exit. POSIX-only; no-op on Windows.
    from quantui.c_stderr import capture_c_stderr

    with capture_c_stderr(stream):
        return _run_freq_calc_body(
            molecule=molecule,
            method=method,
            basis=basis,
            progress_stream=progress_stream,
            _dft=dft,
            _gto=gto,
            _scf=scf,
            _pyscf_thermo=pyscf_thermo,
            stream=stream,
        )


def _run_freq_calc_body(
    *,
    molecule: Molecule,
    method: str,
    basis: str,
    progress_stream: Optional[IO[str]],
    _dft: Any,
    _gto: Any,
    _scf: Any,
    _pyscf_thermo: Any,
    stream: IO[str],
) -> FreqResult:
    """Inner body of :func:`run_freq_calc` (split out for STDERR.1 wrap)."""
    dft, gto, scf, pyscf_thermo = _dft, _gto, _scf, _pyscf_thermo

    def _status(msg: str) -> None:
        """Emit a status marker line consumable by QuantUI's log capture."""
        try:
            stream.write(f"\n[QuantUI_STATUS] {msg}\n")
        except Exception:
            pass

    # ── Build Mole object ────────────────────────────────────────────────────
    mol = gto.Mole()
    mol.atom = molecule.to_pyscf_format()
    mol.basis = basis
    mol.charge = molecule.charge
    mol.spin = molecule.multiplicity - 1
    mol.verbose = 4
    mol.stdout = stream
    mol.build()

    # ── SCF ──────────────────────────────────────────────────────────────────
    method_upper = method.upper()
    if method_upper == "RHF":
        mf = scf.RHF(mol)
    elif method_upper == "UHF":
        mf = scf.UHF(mol)
    else:
        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf.xc = method

    try:
        energy_hartree = float(mf.kernel())
    except Exception as exc:
        raise RuntimeError(
            f"SCF failed for {molecule.get_formula()} ({method}/{basis}): {exc}"
        ) from exc

    _status("SCF converged. Computing analytical Hessian...")

    converged = bool(getattr(mf, "converged", False))
    n_iterations = int(getattr(mf, "cycles", -1))

    # ── HOMO-LUMO gap (non-fatal) ────────────────────────────────────────────
    homo_lumo_gap_ev: Optional[float] = None
    try:
        import numpy as _np

        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy
        if isinstance(mo_energy, (list, _np.ndarray)) and hasattr(
            mo_energy[0], "__len__"
        ):
            mo_e_ref, mo_occ_ref = mo_energy[0], mo_occ[0]
        else:
            mo_e_ref, mo_occ_ref = mo_energy, mo_occ
        n_occ = int((_np.array(mo_occ_ref) > 0).sum())
        if 0 < n_occ < len(mo_e_ref):
            homo_lumo_gap_ev = float(
                (mo_e_ref[n_occ] - mo_e_ref[n_occ - 1]) * HARTREE_TO_EV
            )
    except Exception:
        pass

    # ── MO data for orbital energy diagram (best-effort) ─────────────────────
    mo_energy_hartree: Optional[List] = None
    mo_occ_list: Optional[List] = None
    pyscf_mol_atom: Optional[List] = None
    try:
        import numpy as _np_mo

        _moe = mf.mo_energy
        _moo = mf.mo_occ
        if isinstance(_moe, (list, _np_mo.ndarray)) and hasattr(_moe[0], "__len__"):
            _moe, _moo = _moe[0], _moo[0]
        mo_energy_hartree = _np_mo.asarray(_moe, dtype=float).tolist()
        mo_occ_list = _np_mo.asarray(_moo, dtype=float).tolist()
        pyscf_mol_atom = [(str(s), list(map(float, c))) for s, c in mol._atom]
    except Exception:
        pass

    # ── Hessian + frequency analysis ─────────────────────────────────────────
    frequencies_cm1: List[float] = []
    ir_intensities: List[float] = []
    zpve_hartree: float = 0.0
    displacements: Optional[List] = None
    thermo_data: Optional[ThermoData] = None

    try:
        hess_obj = mf.Hessian()
        hess_obj.verbose = mol.verbose
        hess_obj.stdout = stream

        h = hess_obj.kernel()

        _status("Analytical Hessian complete. Running harmonic analysis...")

        freq_info = pyscf_thermo.harmonic_analysis(mol, h)

        # freq_wavenumber entries may be complex numbers when PySCF uses a
        # complex square-root convention for imaginary modes.  Map them to
        # signed real values: negative = imaginary frequency.
        raw_freqs = freq_info["freq_wavenumber"]
        frequencies_cm1 = []
        for f in raw_freqs:
            if hasattr(f, "imag") and abs(f.imag) > abs(f.real):
                frequencies_cm1.append(float(-abs(f.imag)))
            else:
                frequencies_cm1.append(float(f.real if hasattr(f, "real") else f))

        # ZPVE = ½ · Σ ν_i (positive modes only), converted cm⁻¹ → Hartree
        zpve_hartree = sum(0.5 * f * _CM1_TO_HARTREE for f in frequencies_cm1 if f > 0)

        # Normalized displacement vectors: shape (n_modes, n_atoms, 3).
        # Stored as a nested Python list for JSON-friendliness and to avoid
        # a hard numpy dependency in the dataclass.
        try:
            import numpy as _np

            norm_mode = freq_info.get("norm_mode")
            if norm_mode is not None:
                # norm_mode has shape (n_modes, n_atoms*3) or (n_modes, n_atoms, 3);
                # reshape to (n_modes, n_atoms, 3) if needed.
                nm = _np.array(norm_mode, dtype=float)
                n_modes_out = nm.shape[0]
                n_atoms = len(molecule.atoms)
                if nm.ndim == 2:
                    nm = nm.reshape(n_modes_out, n_atoms, 3)
                displacements = nm.tolist()
        except Exception:
            displacements = None

        # Numerical IR intensities via finite-difference dipole derivatives.
        # pyscf.prop.infrared is absent from released pyscf/pyscf-properties;
        # we compute ∂μ/∂R by displacing each atom ±DELTA, then project onto
        # the harmonic normal modes.
        # Reference: Porezag & Pederson, Phys. Rev. B 54, 7830 (1996).
        if displacements is not None and frequencies_cm1:
            try:
                import numpy as _np_ir

                _DELTA = 0.01  # Bohr
                _BOHR_TO_ANG = 0.52917721092
                _KM_MOL_FAC = 42.255  # (D/Å)²/amu → km/mol

                _n_ir = mol.natm
                _ir_total_solves = _n_ir * 3 * 2
                _ir_done_solves = 0
                _coords0 = mol.atom_coords().copy()
                _dm0 = mf.make_rdm1()
                _dpdx = _np_ir.zeros((_n_ir * 3, 3))
                _xc = getattr(mf, "xc", None)

                _status(
                    "Numerical IR intensities: "
                    f"{_ir_done_solves}/{_ir_total_solves} extra SCF solves complete "
                    f"({_ir_total_solves - _ir_done_solves} remaining)"
                )

                # Inner-SCF helper: builds the right RHF/UHF/RKS/UKS object
                # for the current ``mol`` geometry, attempts gpu4pyscf
                # offload (M-GPU extension to the IR-intensity loop —
                # without this wrap, the per-displacement SCFs run on CPU
                # even when the outer SCF was GPU-offloaded), and returns
                # the dipole moment as a numpy array. Used for both +Δ and
                # -Δ steps so the +/-/half-loop logic stays compact.
                from quantui.gpu_offload import try_to_gpu as _try_to_gpu_inner

                def _displaced_scf_dipole() -> _np_ir.ndarray:
                    if _xc is not None:
                        _mf_d = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
                        _mf_d.xc = _xc
                    else:
                        _mf_d = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
                    _mf_d.verbose = 0
                    _mf_d.stdout = stream
                    # ``method_upper="RHF"`` is a label — try_to_gpu only
                    # uses it to skip CCSD(T). For RHF/UHF/DFT the wrapper
                    # attempts ``mf.to_gpu()`` and falls back to CPU on any
                    # failure, so this is safe to call unconditionally.
                    _mf_d, _used_gpu, _gpu_name = _try_to_gpu_inner(_mf_d, "RHF")
                    _mf_d.kernel(dm0=_dm0)
                    return _np_ir.array(_mf_d.dip_moment(verbose=0))

                # Opt-in parallel path (Pass B). When (a) the user has
                # set ``QUANTUI_FREQ_PARALLEL=1``, (b) no GPU is available,
                # (c) the host has >= 4 cores, and (d) the molecule has >= 2
                # atoms, we fan the per-displacement SCFs out across a
                # ProcessPoolExecutor. The decision is centralised in
                # ``freq_ir_workers.parallel_enabled_for_run`` so tests
                # can pin the contract.
                from quantui import freq_ir_workers as _ir_par
                from quantui.gpu_offload import is_gpu_available

                _gpu_ok, _ = is_gpu_available()
                _cpu_count = os.cpu_count() or 1
                _use_parallel = _ir_par.parallel_enabled_for_run(
                    cpu_count=_cpu_count,
                    displacement_count=_ir_total_solves,
                    gpu_available=_gpu_ok,
                )

                _mol_v = mol.verbose
                mol.verbose = 0
                try:
                    if _use_parallel:
                        # Stash dm0 once on disk so workers can map-load it
                        # via initargs (avoids per-task pickling).
                        import concurrent.futures as _cf
                        import multiprocessing as _mp
                        import pickle as _pickle
                        import tempfile as _tempfile

                        _n_workers = _ir_par.pick_worker_count(
                            _cpu_count, _ir_total_solves
                        )
                        _threads_each = _ir_par.threads_per_worker(
                            _cpu_count, _n_workers
                        )

                        # Build all 6N task arguments first; pickling-safe
                        # flat lists per-displacement.
                        _tasks: list[tuple[int, int, int, list[float]]] = []
                        for _I in range(_n_ir):
                            for _ax in range(3):
                                _cp = _coords0.copy()
                                _cp[_I, _ax] += _DELTA
                                _tasks.append((_I, _ax, +1, _cp.flatten().tolist()))
                                _cm = _coords0.copy()
                                _cm[_I, _ax] -= _DELTA
                                _tasks.append((_I, _ax, -1, _cm.flatten().tolist()))

                        _dm0_handle = _tempfile.NamedTemporaryFile(
                            delete=False, suffix=".dm0.pkl"
                        )
                        try:
                            _pickle.dump(_dm0, _dm0_handle)
                            _dm0_handle.close()

                            # Pyscf-format atom string for worker rebuild.
                            _atom_str = molecule.to_pyscf_format()
                            _spin = molecule.multiplicity - 1
                            _charge = molecule.charge
                            _ctx = _mp.get_context("spawn")
                            with _cf.ProcessPoolExecutor(
                                max_workers=_n_workers,
                                mp_context=_ctx,
                                initializer=_ir_par.init_worker,
                                initargs=(
                                    _atom_str,
                                    basis,
                                    _charge,
                                    _spin,
                                    _xc,
                                    _dm0_handle.name,
                                    _threads_each,
                                ),
                            ) as _pool:
                                # Submit all and store futures keyed by task
                                # index so we can assemble +/- per (I, ax).
                                _futs = {
                                    _pool.submit(
                                        _ir_par.run_displaced_scf, _task[3]
                                    ): _task
                                    for _task in _tasks
                                }
                                # Accumulate results into a temporary map
                                # ``(I, ax, sign) -> dipole_array``.
                                _dipoles: dict = {}
                                for _fut in _cf.as_completed(_futs):
                                    _I, _ax, _sign, _coords_done = _futs[_fut]
                                    _dipoles[(_I, _ax, _sign)] = _fut.result()
                                    _ir_done_solves += 1
                                    _status(
                                        "Numerical IR intensities (parallel ×"
                                        f"{_n_workers}): "
                                        f"{_ir_done_solves}/{_ir_total_solves} "
                                        "extra SCF solves complete "
                                        f"({_ir_total_solves - _ir_done_solves} "
                                        "remaining)"
                                    )
                        finally:
                            try:
                                os.unlink(_dm0_handle.name)
                            except OSError:
                                pass

                        # Assemble dpdx now that all dipoles are in hand.
                        for _I in range(_n_ir):
                            for _ax in range(3):
                                _mu_p = _dipoles[(_I, _ax, +1)]
                                _mu_m = _dipoles[(_I, _ax, -1)]
                                _dpdx[3 * _I + _ax] = (_mu_p - _mu_m) / (2 * _DELTA)
                    else:
                        for _I in range(_n_ir):
                            for _ax in range(3):
                                # +Δ displacement
                                _cp = _coords0.copy()
                                _cp[_I, _ax] += _DELTA
                                mol.set_geom_(_cp, unit="Bohr")
                                _mu_p = _displaced_scf_dipole()
                                _ir_done_solves += 1
                                _status(
                                    "Numerical IR intensities: "
                                    f"{_ir_done_solves}/{_ir_total_solves} "
                                    "extra SCF solves complete "
                                    f"({_ir_total_solves - _ir_done_solves} "
                                    "remaining)"
                                )

                                # -Δ displacement
                                _cm = _coords0.copy()
                                _cm[_I, _ax] -= _DELTA
                                mol.set_geom_(_cm, unit="Bohr")
                                _mu_m = _displaced_scf_dipole()
                                _ir_done_solves += 1
                                _status(
                                    "Numerical IR intensities: "
                                    f"{_ir_done_solves}/{_ir_total_solves} "
                                    "extra SCF solves complete "
                                    f"({_ir_total_solves - _ir_done_solves} "
                                    "remaining)"
                                )

                                _dpdx[3 * _I + _ax] = (_mu_p - _mu_m) / (2 * _DELTA)
                finally:
                    mol.set_geom_(_coords0, unit="Bohr")
                    mol.verbose = _mol_v

                _dpdx_AA = _dpdx / _BOHR_TO_ANG
                _nm_flat = _np_ir.array(displacements).reshape(len(frequencies_cm1), -1)
                _dpdQ = _nm_flat @ _dpdx_AA
                _ir = (_KM_MOL_FAC * (_dpdQ**2).sum(axis=1)).tolist()
                if len(_ir) == len(frequencies_cm1):
                    ir_intensities = _ir
                _status(
                    "Numerical IR intensities complete. Computing thermochemistry..."
                )
            except Exception as _ir_exc:
                logger.warning("Numerical IR intensities failed: %s", _ir_exc)
                _status(
                    "Numerical IR intensities failed; continuing without IR intensities."
                )

        # Thermochemistry at 298.15 K / 1 atm — best-effort
        try:
            import numpy as _np

            _status("Computing thermochemistry...")

            _freq_au = freq_info.get("freq_au")
            if _freq_au is None:
                _freq_au = _np.array(frequencies_cm1) * _CM1_TO_HARTREE
            else:
                # PySCF may return complex freq_au for imaginary modes; take real parts.
                _freq_au = _np.array(
                    [f.real if hasattr(f, "real") else f for f in _freq_au],
                    dtype=float,
                )

            # PySCF 2.x thermo() may or may not accept the pressure argument.
            try:
                _tout = pyscf_thermo.thermo(mf, _freq_au, 298.15, 101325)
            except TypeError:
                _tout = pyscf_thermo.thermo(mf, _freq_au, 298.15)

            # PySCF 2.x returns (value, unit_string) tuples; earlier versions
            # return plain floats.  _tv() extracts the numeric value either way.
            def _tv(v):
                if isinstance(v, (tuple, list)):
                    return float(v[0])
                if hasattr(v, "item"):
                    return float(v.item())
                return float(v)

            # PySCF 2.x (>=2.6) uses "H_tot"/"S_tot"; earlier versions used "H"/"S".
            _H_raw, _S_raw, _Z_raw = None, None, None
            for _k in ("H_tot", "H", "Htot", "H_0K"):
                if _tout.get(_k) is not None:
                    _H_raw = _tout[_k]
                    break
            for _k in ("S_tot", "S", "Stot"):
                if _tout.get(_k) is not None:
                    _S_raw = _tout[_k]
                    break
            for _k in ("ZPE", "zpve", "ZPE_vib"):
                if _tout.get(_k) is not None:
                    _Z_raw = _tout[_k]
                    break
            if _H_raw is None or _S_raw is None:
                raise KeyError(
                    f"Missing H or S in thermo dict (keys: {sorted(_tout.keys())})"
                )
            _H = _tv(_H_raw)
            _S = _tv(_S_raw)  # J/(mol·K)
            _zpve = _tv(_Z_raw) if _Z_raw is not None else zpve_hartree
            _G = _H - 298.15 * _S / _HARTREE_TO_JMOL
            thermo_data = ThermoData(
                zpve_hartree=_zpve,
                H_hartree=_H,
                S_jmol=_S,
                G_hartree=_G,
            )
            _status("Frequency backend complete.")
        except Exception as _exc:
            logger.warning("Thermochemistry failed: %s", _exc)
            _status("Thermochemistry failed; frequency backend complete.")

    except Exception as exc:
        logger.warning("Hessian/frequency computation failed: %s", exc)
        _status("Hessian/frequency step failed.")
        if progress_stream is not None:
            try:
                progress_stream.write(f"\n⚠ Hessian failed: {exc}\n")
            except Exception:
                pass

    return FreqResult(
        energy_hartree=energy_hartree,
        homo_lumo_gap_ev=homo_lumo_gap_ev,
        converged=converged,
        n_iterations=n_iterations,
        method=method,
        basis=basis,
        formula=molecule.get_formula(),
        frequencies_cm1=frequencies_cm1,
        ir_intensities=ir_intensities,
        zpve_hartree=zpve_hartree,
        thermo=thermo_data,
        displacements=displacements,
        mo_energy_hartree=mo_energy_hartree,
        mo_occ=mo_occ_list,
        pyscf_mol_atom=pyscf_mol_atom,
        pyscf_mol_basis=basis,
    )
