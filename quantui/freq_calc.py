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
from pathlib import Path
from typing import IO, Any, List, Optional, cast

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
        raman_activities: Static Raman activities in Å⁴/amu per mode.
            Empty list if Raman is disabled or unavailable.
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
    raman_activities: List[float] = field(default_factory=list)
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
    density_fit: bool = False

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
# Mode-following geometry perturbation (imaginary / TS modes)
# ============================================================================

# Matches the default vibrational animation amplitude in ``app_visualization``.
VIB_MODE_DISPLAY_AMPLITUDE_ANGSTROM: float = 0.4

# Typical scale when following an imaginary mode off a saddle point.
DEFAULT_MODE_PERTURBATION_FRACTION: float = 0.75

# Seed-dropdown prefix for a saved Frequency result (mode chosen separately).
FREQ_SEED_PREFIX: str = "freq:"


def perturb_along_mode(
    molecule: Molecule,
    displacements: List,
    mode_index: int,
    *,
    fraction: float = DEFAULT_MODE_PERTURBATION_FRACTION,
    amplitude: float = VIB_MODE_DISPLAY_AMPLITUDE_ANGSTROM,
    sign: float = 1.0,
) -> Molecule:
    """Return a copy of *molecule* displaced along one normal mode.

    PySCF ``norm_mode`` vectors are unit-normalized; *amplitude* (Å) scales
    them the same way as the Vibrational panel animation.  *fraction* is the
    user-facing scale (default 75%).  Per-atom displacement is
    ``sign * fraction * amplitude * mode_vector[atom]``.

    Args:
        molecule: Starting geometry (Å).
        displacements: Nested list shaped ``(n_modes, n_atoms, 3)``.
        mode_index: **0-based** index into *displacements* / ``frequencies_cm1``.
        fraction: Scalar multiplier on *amplitude* (0–1 typical).
        amplitude: Base displacement scale in Å (animation default 0.4).
        sign: ``+1`` or ``-1`` to flip mode direction.

    Raises:
        ValueError: On shape mismatch or out-of-range *mode_index*.
    """
    import numpy as np

    if mode_index < 0 or mode_index >= len(displacements):
        raise ValueError(
            f"mode_index {mode_index} out of range for {len(displacements)} modes"
        )

    disp = np.asarray(displacements[mode_index], dtype=float)
    coords = np.asarray(molecule.coordinates, dtype=float)
    if disp.shape != coords.shape:
        raise ValueError(
            f"displacement shape {disp.shape} != coordinates shape {coords.shape}"
        )

    scale = float(sign) * float(fraction) * float(amplitude)
    new_coords = (coords + scale * disp).tolist()
    return Molecule(
        atoms=list(molecule.atoms),
        coordinates=new_coords,
        charge=int(molecule.charge),
        multiplicity=int(molecule.multiplicity),
        validate_spin=molecule._validate_spin,
    )


def load_frequency_mode_seed_data(
    result_dir: Path,
) -> tuple[Molecule, List, List[float]]:
    """Load geometry and mode data from a saved Frequency ``result.json``."""
    from quantui.results_storage import load_result

    data = load_result(result_dir)
    if data.get("calc_type") != "frequency":
        raise ValueError(f"Not a frequency result: {result_dir}")

    spectra = data.get("spectra") or {}
    ir = spectra.get("ir") or {}
    mol_data = spectra.get("molecule") or {}
    displacements = ir.get("displacements")
    frequencies = ir.get("frequencies_cm1") or []

    if not mol_data.get("atoms"):
        raise ValueError(f"No molecule geometry in frequency result: {result_dir}")
    if not displacements:
        raise ValueError(
            f"No normal-mode displacements in frequency result: {result_dir}"
        )

    molecule = Molecule(
        atoms=list(mol_data["atoms"]),
        coordinates=[list(c) for c in mol_data["coords"]],
        charge=int(mol_data.get("charge", 0)),
        multiplicity=int(mol_data.get("multiplicity", 1)),
    )
    return molecule, displacements, [float(f) for f in frequencies]


def molecule_from_freq_mode_seed(
    result_dir: Path,
    mode_number: int,
    *,
    fraction: float = DEFAULT_MODE_PERTURBATION_FRACTION,
    amplitude: float = VIB_MODE_DISPLAY_AMPLITUDE_ANGSTROM,
    sign: float = 1.0,
) -> tuple[Molecule, dict[str, object]]:
    """Build a perturbed molecule from a saved freq result and 1-based mode index."""
    molecule, displacements, frequencies = load_frequency_mode_seed_data(result_dir)
    if mode_number < 1 or mode_number > len(displacements):
        raise ValueError(
            f"mode_number {mode_number} out of range for {len(displacements)} modes"
        )
    mode_index = mode_number - 1
    perturbed = perturb_along_mode(
        molecule,
        displacements,
        mode_index,
        fraction=fraction,
        amplitude=amplitude,
        sign=sign,
    )
    freq_cm1 = frequencies[mode_index] if mode_index < len(frequencies) else None
    meta = {
        "result_dir": str(result_dir),
        "mode_number": mode_number,
        "frequency_cm1": freq_cm1,
        "fraction": fraction,
        "amplitude_angstrom": amplitude,
        "sign": sign,
    }
    return perturbed, meta


def is_freq_mode_seed(path_str: str) -> bool:
    return bool(path_str) and path_str.startswith(FREQ_SEED_PREFIX)


def freq_mode_seed_result_dir(path_str: str) -> Path:
    if not is_freq_mode_seed(path_str):
        raise ValueError(f"Not a frequency mode seed: {path_str!r}")
    return Path(path_str[len(FREQ_SEED_PREFIX) :])


# ============================================================================
# Main function
# ============================================================================


def run_freq_calc(
    molecule: Molecule,
    method: str = "RHF",
    basis: str = "STO-3G",
    progress_stream: Optional[IO[str]] = None,
    scf_rescue: bool = True,
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
        scf_rescue: Whether every SCF here (the reference geometry, plus
            each finite-difference displacement for numerical IR
            intensities) automatically retries through the shared rescue
            helper on non-convergence (M-SCF-ROBUST, see
            :mod:`quantui.scf_robust`). Default ``True``.

    Returns:
        :class:`FreqResult` with frequencies, ZPVE, and SCF properties.

    Raises:
        ImportError: If PySCF is not installed.
        RuntimeError: If the SCF calculation fails.  If the Hessian
            computation fails, frequencies are omitted and a warning is
            written to progress_stream — no exception is raised.
    """
    # Post-HF methods (MP2/CCSD/CCSD(T)) have no special-casing below —
    # without this guard, method='CCSD' silently falls into the DFT
    # branch (sets mf.xc = "CCSD") and fails deep inside PySCF with a
    # cryptic "LibXCFunctional: name 'CCSD' not found" instead of a clear
    # message. No post-HF Hessian is wired up here, so these methods are
    # single-point only (see session_calc.py).
    from . import config as _config

    if method.strip().upper() in _config.POST_HF_METHODS:
        raise ValueError(
            f"'{method}' is a post-HF method and cannot be used for "
            "frequency analysis — QuantUI only has an analytical Hessian "
            "wired up for HF/DFT methods here. Use RHF, UHF, or a DFT "
            "functional instead."
        )

    try:
        from pyscf import dft, gto, scf
        from pyscf.hessian import thermo as pyscf_thermo
    except ImportError as exc:
        raise ImportError(
            "PySCF is not installed — cannot run frequency analysis.\n"
            "PySCF requires Linux, macOS, or WSL."
        ) from exc

    stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout

    # See quantui/c_stderr.py — captures fd-2 stderr
    # from libcint / BLAS / LAPACK / Hessian C code and relays to ``stream``
    # on exit. POSIX-only; no-op on Windows.
    from quantui.c_stderr import capture_c_stderr

    with capture_c_stderr(stream):
        return _run_freq_calc_body(
            molecule=molecule,
            method=method,
            basis=basis,
            progress_stream=progress_stream,
            scf_rescue=scf_rescue,
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
    scf_rescue: bool = True,
    _dft: Any,
    _gto: Any,
    _scf: Any,
    _pyscf_thermo: Any,
    stream: IO[str],
) -> FreqResult:
    """Inner body of :func:`run_freq_calc` (split out for stderr-capture wrap)."""
    dft, gto, scf, pyscf_thermo = _dft, _gto, _scf, _pyscf_thermo

    def _status(msg: str) -> None:
        """Emit a status marker line consumable by QuantUI's log capture."""
        try:
            stream.write(f"\n[QuantUI_STATUS] {msg}\n")
        except Exception:  # noqa: BLE001 — cleanup (stream may be closed)
            pass

    # ── Build Mole object ────────────────────────────────────────────────────
    from .inorganic_guards import ecp_for_basis

    mol = gto.Mole()
    mol.atom = molecule.to_pyscf_format()
    mol.basis = basis
    # Heavy-element ECP (LANL2DZ / def2); empty dict for all-electron bases.
    mol.ecp = ecp_for_basis(basis, molecule.atoms)
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
        # Route through resolve_xc + maybe_apply_d3 so
        # methods like wB97X-D (PySCF rejects "wb97x-d") map to the
        # bare functional + external D3 dispersion.
        from .session_calc import maybe_apply_d3, resolve_xc

        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf.xc = resolve_xc(method)
        mf = maybe_apply_d3(mf, method, progress_stream=stream)

    # Density fitting (RI), opt-in (M-DF). Off by default. Applied to the main
    # SCF; the per-displacement inner SCFs below get the same treatment so the
    # numerical IR-intensity step stays consistent with the reference.
    from .density_fitting import try_density_fit as _try_density_fit

    mf, _density_fit_used = _try_density_fit(mf)

    # Cooperative cancel between SCF cycles (the Hessian block that
    # follows is a single long native call the callback can't interrupt).
    from .cancellation import attach_scf_cancel_callback, cancel_check_from_stream

    attach_scf_cancel_callback(mf, cancel_check_from_stream(stream))

    _status("Running SCF…")
    from .scf_robust import run_scf_with_rescue

    try:
        energy_hartree = float(
            run_scf_with_rescue(mf, rescue=scf_rescue, stream=stream)
        )
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
    except Exception as exc:
        logger.debug("HOMO-LUMO gap extraction failed in freq calc: %s", exc)

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
        # Build from molecule.atoms/coordinates (Angstrom) rather than
        # mol._atom, which PySCF always stores internally in Bohr. Every
        # consumer of pyscf_mol_atom (Molden export, cube generation,
        # session_calc's/optimizer's own construction of this field)
        # assumes Angstrom; using mol._atom here silently shipped Bohr
        # coordinates ~1.89x too large.
        pyscf_mol_atom = [
            (atom, list(map(float, coords)))
            for atom, coords in zip(molecule.atoms, molecule.coordinates)
        ]
    except Exception as exc:
        # Silent failure here ships a FreqResult with no MO data,
        # breaking the Energies panel on history replay. Log to surface
        # in the Log tab.
        logger.warning(
            "MO data extraction failed in freq calc for %s: %s",
            molecule.get_formula(),
            exc,
        )

    # ── Hessian + frequency analysis ─────────────────────────────────────────
    frequencies_cm1: List[float] = []
    ir_intensities: List[float] = []
    raman_activities: List[float] = []
    zpve_hartree: float = 0.0
    displacements: Optional[List] = None
    thermo_data: Optional[ThermoData] = None

    try:
        hess_obj = mf.Hessian()
        # verbose=6 surfaces per-atom integral contractions — the only
        # in-kernel progress signal during an analytical Hessian build.
        # _LogCapture.write greps for per-atom contraction lines.
        hess_obj.verbose = 6
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
        except Exception as exc:
            logger.debug("Normal-mode displacement extraction failed: %s", exc)
            displacements = None

        # Numerical IR intensities via finite-difference dipole derivatives.
        # pyscf.prop.infrared is absent from released pyscf/pyscf-properties;
        # we compute ∂μ/∂R by displacing each atom ±DELTA, then project onto
        # the harmonic normal modes.
        # Reference: Porezag & Pederson, Phys. Rev. B 54, 7830 (1996).
        if displacements is not None and frequencies_cm1:
            import numpy as _np_ir

            _dm0 = mf.make_rdm1()
            _dm0_is_unrestricted = _np_ir.asarray(_dm0).ndim == 3

            try:
                from .config import BOHR_TO_ANGSTROM as _BOHR_TO_ANG

                _DELTA = 0.01  # Bohr
                _KM_MOL_FAC = 42.255  # (D/Å)²/amu → km/mol

                _n_ir = mol.natm
                _ir_total_solves = _n_ir * 3 * 2
                _ir_done_solves = 0
                _coords0 = mol.atom_coords().copy()
                _dpdx = _np_ir.zeros((_n_ir * 3, 3))
                _xc = getattr(mf, "xc", None)
                # Fix (2026-07-14): whether the inner displaced-SCF
                # loop needs an unrestricted (UHF/UKS) object is determined
                # by _dm0's actual shape — (2, nao, nao) for UHF/UKS/ROHF,
                # (nao, nao) for RHF/RKS — NOT by mol.spin == 0. Those two
                # signals only agree when the user's method choice matches
                # the molecule's natural spin state. They diverge when a
                # user explicitly selects UHF for a closed-shell molecule
                # (mol.spin == 0 but the parent mf, and therefore _dm0, is
                # still UHF-shaped): the inner loop used to build RHF from
                # mol.spin == 0, then feed it the UHF-shaped _dm0, which
                # raised a shape-mismatch ValueError inside PySCF and
                # silently dropped IR intensities for the whole calc (caught
                # by the broad except below).
                _status(
                    "Numerical IR intensities: "
                    f"{_ir_done_solves}/{_ir_total_solves} finite-difference displacement SCFs done (6 per atom) "
                    f"({_ir_total_solves - _ir_done_solves} remaining)"
                )

                # Inner-SCF helper: builds the right RHF/UHF/RKS/UKS object
                # for the current ``mol`` geometry, attempts gpu4pyscf
                # offload (GPU extension to the IR-intensity loop —
                # without this wrap, the per-displacement SCFs run on CPU
                # even when the outer SCF was GPU-offloaded), and returns
                # the dipole moment as a numpy array. Used for both +Δ and
                # -Δ steps so the +/-/half-loop logic stays compact.
                from quantui.gpu_offload import try_to_gpu as _try_to_gpu_inner

                def _displaced_scf_dipole() -> _np_ir.ndarray:
                    if _xc is not None:
                        _mf_d = dft.UKS(mol) if _dm0_is_unrestricted else dft.RKS(mol)
                        _mf_d.xc = _xc
                    else:
                        _mf_d = scf.UHF(mol) if _dm0_is_unrestricted else scf.RHF(mol)
                    _mf_d.verbose = 0
                    _mf_d.stdout = stream
                    # Match the main SCF's density-fitting choice (M-DF) before
                    # any GPU offload, so displaced dipoles stay consistent with
                    # the reference energy.
                    _mf_d, _ = _try_density_fit(_mf_d, enabled=_density_fit_used)
                    # ``method_upper="RHF"`` is a label — try_to_gpu only
                    # uses it to skip CCSD(T). For RHF/UHF/DFT the wrapper
                    # attempts ``mf.to_gpu()`` and falls back to CPU on any
                    # failure, so this is safe to call unconditionally.
                    _mf_d, _used_gpu, _gpu_name = _try_to_gpu_inner(_mf_d, "RHF")
                    run_scf_with_rescue(_mf_d, dm0=_dm0, rescue=scf_rescue)
                    # pyscf has no type stubs (ignore_missing_imports), so
                    # dip_moment()'s Any return defeats asarray's overload
                    # resolution too; dip_moment() genuinely returns an
                    # array-like of floats.
                    return cast(
                        _np_ir.ndarray,
                        _np_ir.asarray(_mf_d.dip_moment(verbose=0), dtype=float),
                    )

                # Opt-in parallel path (Pass B). When (a) the user has
                # set ``QUANTUI_FREQ_PARALLEL=1``, (b) the host has >= 4
                # cores, and (c) the molecule has >= 2 atoms, we fan the
                # per-displacement SCFs out across a ProcessPoolExecutor
                # (CPU workers — see freq_ir_workers). Without the env var,
                # displaced SCFs stay serial and may use gpu4pyscf when
                # available. The decision is centralised in
                # ``freq_ir_workers.parallel_enabled_for_run`` so tests
                # can pin the contract.
                from quantui import freq_ir_workers as _ir_par

                _cpu_count = os.cpu_count() or 1
                _use_parallel = _ir_par.parallel_enabled_for_run(
                    cpu_count=_cpu_count,
                    displacement_count=_ir_total_solves,
                )

                _mol_v = mol.verbose
                mol.verbose = 0
                _parallel_failed = False
                try:
                    if _use_parallel:
                        try:
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
                                            "finite-difference displacement SCFs done (6 per atom) "
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
                        except Exception as _par_exc:
                            logger.warning(
                                "Parallel IR-intensity computation failed (%s); falling back to serial.",
                                _par_exc,
                            )
                            _status(
                                "Parallel IR intensities failed; falling back to serial computation."
                            )
                            _parallel_failed = True
                            # Reset so the serial loop's progress messages
                            # below start clean rather than continuing from
                            # wherever the failed parallel attempt left off.
                            _ir_done_solves = 0
                    if not _use_parallel or _parallel_failed:
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
                                    "finite-difference displacement SCFs done (6 per atom) "
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
                                    "finite-difference displacement SCFs done (6 per atom) "
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

            # Static Raman activities: analytical polarizability (pyscf-properties)
            # + the same ±Δ geometry FD loop as IR (see quantui.raman_calc).
            if displacements is not None and frequencies_cm1:
                try:
                    from quantui.raman_calc import compute_raman_activities

                    _raman = compute_raman_activities(
                        mf=mf,
                        mol=mol,
                        scf=scf,
                        dft=dft,
                        displacements=displacements,
                        frequencies_cm1=frequencies_cm1,
                        dm0=_dm0,
                        dm0_is_unrestricted=_dm0_is_unrestricted,
                        density_fit_used=_density_fit_used,
                        stream=stream,
                        status=_status,
                        hessian=h,
                        atom_str=molecule.to_pyscf_format(),
                        scf_rescue=scf_rescue,
                    )
                    if len(_raman) == len(frequencies_cm1):
                        raman_activities = _raman
                except Exception as _ram_exc:
                    logger.warning("Numerical Raman activities failed: %s", _ram_exc)
                    _status(
                        "Numerical Raman activities failed; continuing without Raman."
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
            except Exception:  # noqa: BLE001 — cleanup (stream may be closed)
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
        raman_activities=raman_activities,
        zpve_hartree=zpve_hartree,
        thermo=thermo_data,
        displacements=displacements,
        mo_energy_hartree=mo_energy_hartree,
        mo_occ=mo_occ_list,
        pyscf_mol_atom=pyscf_mol_atom,
        pyscf_mol_basis=basis,
        density_fit=_density_fit_used,
    )
