"""PyFock adapter for the guarded Phase-1 single-point DFT subset."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional

from .base import (
    EngineCapabilities,
    EngineRequest,
    EngineResult,
    EngineUnavailableError,
    QuantumEngine,
    UnsupportedCapabilityError,
)

_PYFOCK_SPEC = importlib.util.find_spec("pyfock")
_AVAILABLE = _PYFOCK_SPEC is not None

# PyFock 0.1.7 documents exact-exchange/hybrid support as future work. Do not
# advertise B3LYP/PBE0 until upstream implements and QuantUI validates them.
_PYFOCK_METHODS = ("PBE",)
_PYFOCK_BASES = ("def2-SVP", "def2-TZVP")
_AUX_BASIS = "def2-universal-jfit"
_HARTREE_TO_EV = 27.211386245988


def is_pyfock_available() -> bool:
    """True when PyFock is importable in the current environment."""
    return _AVAILABLE


def _pyfock_version() -> str:
    if not _AVAILABLE:
        return ""
    try:
        import importlib.metadata as im

        return im.version("pyfock")
    except Exception:  # noqa: BLE001 — best-effort version probe
        return "unknown"


class PyfockEngine:
    """Pure-Python DFT engine for native Windows and teaching single points."""

    @property
    def engine_id(self) -> str:
        return "pyfock"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_id="pyfock",
            display_name="PyFock",
            supported_calc_types=("single_point",),
            supported_methods=_PYFOCK_METHODS,
            supported_basis_sets=_PYFOCK_BASES,
            supports_solvent=False,
            supports_checkpoint_warm_start=False,
            supports_gpu=False,
            supports_post_hf=False,
            supports_orbital_export=False,
            platform_notes=(
                "Phase-1 neutral, closed-shell PBE single points. Density fitting "
                "is always on; hybrids, solvent, checkpoints, and GPU are gated off. "
                "Install with pip install quantui[pyfock]."
            ),
            recommended_auxbasis=_AUX_BASIS,
            version=_pyfock_version(),
        )

    def run(self, request: EngineRequest) -> EngineResult:
        """Run one neutral, closed-shell PBE single point through PyFock."""
        self._validate_request(request)
        stream = request.progress_stream or sys.stdout

        atoms, coordinates = _molecule_arrays(request.molecule)
        pyfock_atoms = [
            [symbol, float(x), float(y), float(z)]
            for symbol, (x, y, z) in zip(atoms, coordinates)
        ]

        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                try:
                    Mol, Basis, DFT = _load_pyfock_api()
                except (ImportError, OSError) as exc:
                    raise EngineUnavailableError(
                        f"PyFock could not be imported: {exc}",
                        user_message=(
                            "PyFock is installed but could not start. Reinstall the "
                            "quantui[pyfock] extra and restart the kernel."
                        ),
                    ) from exc
                print("\n-- PyFock Phase-1 single point -----------------------------")
                print(
                    f"Engine: PyFock {_pyfock_version() or 'unknown'} | "
                    f"{request.method}/{request.basis} | density fitting: on"
                )
                mol = Mol(atoms=pyfock_atoms, charge=0)
                basis = Basis(
                    mol,
                    {"all": Basis.load(mol=mol, basis_name=request.basis)},
                )
                auxbasis = Basis(
                    mol,
                    {"all": Basis.load(mol=mol, basis_name=_AUX_BASIS)},
                )
                dft = DFT(
                    mol,
                    basis,
                    auxbasis,
                    xc="PBE",
                    conv_crit=_positive_float(
                        request.options.get("conv_crit"), default=1.0e-7
                    ),
                    ncores=_positive_int(request.options.get("ncores"), default=1),
                    use_gpu=False,
                )
                dft.max_itr = _positive_int(
                    request.options.get("max_iterations"), default=50
                )
                # Match PySCF/standard def2 basis conventions (5d/7f spherical
                # functions). PyFock defaults to Cartesian 6d/10f, which shifts
                # even the water reference outside the milestone parity bound.
                dft.sao = True
                energy, _density = dft.scf()
        except EngineUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize third-party failures
            return EngineResult(
                request_id=request.request_id,
                engine_id=self.engine_id,
                status="error",
                converged=False,
                method=request.method,
                basis=request.basis,
                formula=_formula(atoms),
                error={
                    "code": "PYFOCK_CALCULATION_FAILED",
                    "message": f"{type(exc).__name__}: {exc}",
                    "user_message": (
                        "PyFock could not complete this calculation. Review the "
                        "live output for its diagnostic details."
                    ),
                    "retryable": False,
                },
            )

        gap_ev = _homo_lumo_gap_ev(
            getattr(dft, "mo_energies", None),
            getattr(dft, "mo_occupations", None),
        )
        warnings = [
            "PyFock Phase 1 uses density fitting with def2-universal-jfit.",
            "Orbital export and Mulliken/dipole analysis are not available yet.",
        ]
        if not bool(getattr(dft, "converged", False)):
            warnings.append("PyFock reached its iteration limit without convergence.")

        result = EngineResult(
            request_id=request.request_id,
            engine_id=self.engine_id,
            status="success",
            converged=bool(getattr(dft, "converged", False)),
            energy_hartree=float(energy),
            n_iterations=int(getattr(dft, "niter", 0)),
            method=request.method,
            basis=request.basis,
            formula=_formula(atoms),
            homo_lumo_gap_ev=gap_ev,
            warnings=warnings,
        )
        result.native_result = result.to_session_result()
        return result

    def _validate_request(self, request: EngineRequest) -> None:
        caps = self.capabilities()
        if request.calc_type not in caps.supported_calc_types:
            raise UnsupportedCapabilityError(
                f"PyFock does not support {request.calc_type!r} in Phase 1.",
                user_message="PyFock Phase 1 supports Single Point calculations only.",
            )
        if request.method.upper() not in caps.supported_methods:
            raise UnsupportedCapabilityError(
                f"PyFock method {request.method!r} is not validated.",
                user_message=(
                    "PyFock Phase 1 currently supports PBE only. Hybrid "
                    "functionals require exact exchange that upstream PyFock "
                    "does not yet provide."
                ),
            )
        if request.basis not in (caps.supported_basis_sets or ()):
            raise UnsupportedCapabilityError(
                f"PyFock basis {request.basis!r} is not validated.",
                user_message=(
                    "PyFock Phase 1 currently supports def2-SVP and def2-TZVP."
                ),
            )
        if request.charge != 0:
            raise UnsupportedCapabilityError(
                "PyFock charged molecules are not enabled in Phase 1.",
                user_message=(
                    "PyFock Phase 1 is limited to neutral molecules while its "
                    "charge convention is validated. Select PySCF for ions."
                ),
            )
        if request.multiplicity != 1:
            raise UnsupportedCapabilityError(
                "PyFock open-shell calculations are not enabled in Phase 1.",
                user_message=(
                    "PyFock Phase 1 is limited to closed-shell singlets. "
                    "Select PySCF for radicals or other spin states."
                ),
            )
        if request.solvent:
            raise UnsupportedCapabilityError(
                "PyFock solvent models are not enabled in Phase 1.",
                user_message="Implicit solvent is not available with PyFock Phase 1.",
            )
        _molecule_arrays(request.molecule)


def _load_pyfock_api():
    from pyfock import DFT, Basis, Mol

    return Mol, Basis, DFT


def _molecule_arrays(molecule: dict):
    atoms = molecule.get("atoms")
    coordinates = molecule.get("coordinates")
    if not isinstance(atoms, list) or not isinstance(coordinates, list):
        raise UnsupportedCapabilityError(
            "Molecule envelope needs atoms and coordinates."
        )
    if not atoms or len(atoms) != len(coordinates):
        raise UnsupportedCapabilityError(
            "Molecule atom/coordinate counts do not match."
        )
    for coord in coordinates:
        if not isinstance(coord, (list, tuple)) or len(coord) != 3:
            raise UnsupportedCapabilityError("Every atom needs three coordinates.")
    return atoms, coordinates


def _positive_int(value, *, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _positive_float(value, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _homo_lumo_gap_ev(energies, occupations) -> Optional[float]:
    if energies is None or occupations is None:
        return None
    try:
        occupied = [index for index, occ in enumerate(occupations) if float(occ) > 1e-8]
        if not occupied or occupied[-1] + 1 >= len(energies):
            return None
        homo = occupied[-1]
        return float(energies[homo + 1] - energies[homo]) * _HARTREE_TO_EV
    except (TypeError, ValueError, IndexError):
        return None


def _formula(atoms) -> str:
    from quantui.connectivity import hill_formula

    return str(hill_formula(list(atoms)))


def build_pyfock_engine() -> Optional[QuantumEngine]:
    if not _AVAILABLE:
        return None
    return PyfockEngine()
