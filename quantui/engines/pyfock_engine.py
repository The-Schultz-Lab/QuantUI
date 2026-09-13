"""PyFock adapter for the guarded neutral, closed-shell DFT subset."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Optional

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
    """Pure-Python DFT engine for native Windows and teaching workflows."""

    @property
    def engine_id(self) -> str:
        return "pyfock"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_id="pyfock",
            display_name="PyFock",
            supported_calc_types=("single_point", "geometry_opt"),
            supported_methods=_PYFOCK_METHODS,
            supported_basis_sets=_PYFOCK_BASES,
            supports_solvent=False,
            supports_checkpoint_warm_start=False,
            supports_gpu=False,
            supports_post_hf=False,
            supports_orbital_export=True,
            platform_notes=(
                "Neutral, closed-shell PBE single points and geometry optimizations. "
                "Density fitting and analytical gradients are used; hybrids, "
                "solvent, checkpoints, and GPU are gated off. "
                "Install with pip install quantui[pyfock]."
            ),
            recommended_auxbasis=_AUX_BASIS,
            version=_pyfock_version(),
        )

    def run(self, request: EngineRequest) -> EngineResult:
        """Run one validated PyFock calculation."""
        self._validate_request(request)
        if request.calc_type == "geometry_opt":
            return self._run_geometry_opt(request)
        return self._run_single_point(request)

    def _run_single_point(self, request: EngineRequest) -> EngineResult:
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
                print("\n-- PyFock single point -------------------------------------")
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
                energy, density = dft.scf()
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
        warnings = ["PyFock uses density fitting with def2-universal-jfit."]
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
        native = result.to_session_result()
        _attach_analysis(native, mol, basis, dft, density, atoms, coordinates, warnings)
        result.native_result = native
        return result

    def _run_geometry_opt(self, request: EngineRequest) -> EngineResult:
        from quantui.molecule import Molecule
        from quantui.optimizer import optimize_geometry

        atoms, coordinates = _molecule_arrays(request.molecule)
        molecule = Molecule(
            atoms=list(atoms),
            coordinates=[list(c) for c in coordinates],
            charge=request.charge,
            multiplicity=request.multiplicity,
        )
        native = optimize_geometry(
            molecule=molecule,
            method=request.method,
            basis=request.basis,
            fmax=_positive_float(request.options.get("fmax"), default=0.05),
            steps=_positive_int(request.options.get("steps"), default=200),
            progress_stream=request.progress_stream,
            expected_steps=request.options.get("expected_steps"),
            engine_id=self.engine_id,
            ncores=_positive_int(request.options.get("ncores"), default=1),
        )
        return EngineResult(
            request_id=request.request_id,
            engine_id=self.engine_id,
            status="success",
            converged=native.converged,
            energy_hartree=native.energy_hartree,
            n_iterations=native.n_steps,
            method=native.method,
            basis=native.basis,
            formula=native.formula,
            native_result=native,
        )

    def _validate_request(self, request: EngineRequest) -> None:
        caps = self.capabilities()
        if request.calc_type not in caps.supported_calc_types:
            raise UnsupportedCapabilityError(
                f"PyFock does not support {request.calc_type!r}.",
                user_message=(
                    "PyFock supports Single Point and Geometry Opt calculations."
                ),
            )
        if request.method.upper() not in caps.supported_methods:
            raise UnsupportedCapabilityError(
                f"PyFock method {request.method!r} is not validated.",
                user_message=(
                    "PyFock currently supports PBE only. Hybrid "
                    "functionals require exact exchange that upstream PyFock "
                    "does not yet provide."
                ),
            )
        if request.basis not in (caps.supported_basis_sets or ()):
            raise UnsupportedCapabilityError(
                f"PyFock basis {request.basis!r} is not validated.",
                user_message=("PyFock currently supports def2-SVP and def2-TZVP."),
            )
        if request.charge != 0:
            raise UnsupportedCapabilityError(
                "PyFock charged molecules are not enabled.",
                user_message=(
                    "PyFock is limited to neutral molecules while its "
                    "charge convention is validated. Select PySCF for ions."
                ),
            )
        if request.multiplicity != 1:
            raise UnsupportedCapabilityError(
                "PyFock open-shell calculations are not enabled.",
                user_message=(
                    "PyFock is limited to closed-shell singlets. "
                    "Select PySCF for radicals or other spin states."
                ),
            )
        if request.solvent:
            raise UnsupportedCapabilityError(
                "PyFock solvent models are not enabled.",
                user_message="Implicit solvent is not available with PyFock.",
            )
        _molecule_arrays(request.molecule)


def _load_pyfock_api():
    from pyfock import DFT, Basis, Mol

    return Mol, Basis, DFT


def _attach_analysis(
    result: Any,
    mol: Any,
    basis: Any,
    dft: Any,
    density: Any,
    atoms: list[str],
    coordinates: list[list[float]],
    warnings: list[str],
) -> None:
    """Attach portable orbital, Mulliken, and dipole fields when available."""
    try:
        import numpy as np

        result.mo_energy_hartree = np.asarray(dft.mo_energies, dtype=float)
        result.mo_occ = np.asarray(dft.mo_occupations, dtype=float)
        result.mo_coeff = np.asarray(dft.mo_coefficients, dtype=float)
        result.pyscf_mol_atom = [
            (symbol, [float(x), float(y), float(z)])
            for symbol, (x, y, z) in zip(atoms, coordinates)
        ]
        result.pyscf_mol_basis = result.basis
    except Exception as exc:  # noqa: BLE001 - analysis is additive
        warnings.append(f"PyFock orbital extraction was unavailable: {exc}")

    try:
        import numpy as np
        from pyfock import Integrals

        dmat = np.asarray(density, dtype=float)
        overlap = np.asarray(Integrals.overlap_mat_symm(basis), dtype=float)
        populations = np.diag(dmat @ overlap)
        gross = np.zeros(len(atoms), dtype=float)
        for ao_index, atom_index in enumerate(basis.bfs_atoms):
            gross[int(atom_index)] += populations[ao_index]
        result.atom_symbols = list(atoms)
        result.mulliken_charges = [
            float(charge - population)
            for charge, population in zip(mol.Zcharges, gross)
        ]

        dipole_matrix = Integrals.dipole_moment_mat_symm(basis)
        dipole_debye = (
            np.asarray(mol.get_dipole_moment(dipole_matrix, dmat), dtype=float)
            * 2.541746473
        )
        result.dipole_vector_debye = [float(v) for v in dipole_debye]
        result.dipole_moment_debye = float(np.linalg.norm(dipole_debye))
    except Exception as exc:  # noqa: BLE001 - analysis is additive
        warnings.append(f"PyFock population analysis was unavailable: {exc}")


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
