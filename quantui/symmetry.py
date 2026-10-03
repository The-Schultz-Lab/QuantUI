"""Molecular point groups and orbital symmetry labels (M-CHEM CHEM.1).

Point-group detection uses PySCF's ``symm.geom.detect_symm``. Its default
tolerance (1e-5 Bohr) is too strict for typed-in or optimized coordinates
(NH3 with 4-decimal coordinates comes back as Cs), so detection runs at a
teaching tolerance of 0.01 Å, and a looser 0.05 Å check reports when the
structure is *nearly* more symmetric. Both tolerances are shown to the user.

Orbital labels: PySCF symmetry-adapts orbitals only in D2h and its
subgroups (plus its own linear-molecule groups). Labels are therefore given
in the computational subgroup PySCF picks (e.g. Cs for NH3, whose full group
is C3v), and :class:`MOIrreps` says which group that is. They are numbered
per irrep in energy order, the textbook way (1a1, 2a1, 1b2, 3a1, 1b1 for
water).

QuantUI never runs its SCFs with ``symmetry=True``. Labels are assigned
afterwards by projecting the converged orbitals onto PySCF's
symmetry-adapted AOs, built for the same geometry and basis (PySCF keeps
the input orientation, so no rotation is needed).
"""

from __future__ import annotations

import logging
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

BOHR_PER_ANGSTROM = 1.0 / 0.52917721092

#: Default detection tolerance (Å): how far atoms may sit from exact
#: symmetry positions and still count as symmetric.
DEFAULT_TOLERANCE_ANGSTROM = 0.01
#: Looser tolerance for the "nearly more symmetric" hint.
LOOSE_TOLERANCE_ANGSTROM = 0.05

# PySCF's tolerance is a module global; serialize every change to it.
# Re-entrant: detection may run inside detection_tolerance().
_TOLERANCE_LOCK = threading.RLock()


@contextmanager
def detection_tolerance(tolerance_angstrom: float = DEFAULT_TOLERANCE_ANGSTROM):
    """Run PySCF symmetry detection at *tolerance_angstrom* inside the block.

    For PySCF code that detects symmetry internally, e.g.
    ``hessian.thermo.rotational_symmetry_number``: at PySCF's strict default
    an optimized NH3 gets symmetry number 1 instead of 3, overstating the
    rotational entropy by R ln 3.
    """
    from pyscf.symm import geom

    with _TOLERANCE_LOCK:
        old = geom.TOLERANCE
        geom.TOLERANCE = tolerance_angstrom * BOHR_PER_ANGSTROM
        try:
            yield
        finally:
            geom.TOLERANCE = old


_PRETTY = {"Dooh": "D∞h", "Coov": "C∞v", "SO3": "SO(3)"}


@dataclass
class PointGroup:
    """Detected point group of a geometry."""

    group: str  # PySCF name at the default tolerance, e.g. "C2v", "Dooh"
    tolerance_angstrom: float
    exact_group: str  # at PySCF's strict default tolerance
    loose_group: Optional[str] = None  # set only when higher than ``group``

    @property
    def display(self) -> str:
        return pretty_group(self.group)

    def summary(self) -> str:
        """One-line plain-text description."""
        text = self.display
        if self.exact_group != self.group:
            text += f" (within {self.tolerance_angstrom:g} Å)"
        if self.loose_group:
            text += (
                f"; nearly {pretty_group(self.loose_group)} "
                f"(within {LOOSE_TOLERANCE_ANGSTROM:g} Å)"
            )
        return text

    def summary_html(self) -> str:
        """Like :meth:`summary`, with subscripts."""
        text = html_group(self.group)
        if self.exact_group != self.group:
            text += f" (within {self.tolerance_angstrom:g} Å)"
        if self.loose_group:
            text += (
                f"; nearly {html_group(self.loose_group)} "
                f"(within {LOOSE_TOLERANCE_ANGSTROM:g} Å)"
            )
        return text


@dataclass
class MOIrreps:
    """Symmetry labels for a set of MOs."""

    labels: List[str]  # numbered, lower case: "1a1", "2a1", "1b2", "1a'"
    group: str  # group the labels belong to (PySCF's computational group)
    top_group: str  # full point group of the geometry

    @property
    def is_subgroup(self) -> bool:
        return self.group != self.top_group

    def caption(self) -> str:
        if self.is_subgroup:
            return (
                f"Orbital symmetry labels in {pretty_group(self.group)} "
                f"(subgroup of {pretty_group(self.top_group)})"
            )
        return f"Orbital symmetry labels in {pretty_group(self.group)}"


def pretty_group(name: str) -> str:
    """``"Dooh"`` → ``"D∞h"``; other Schoenflies names unchanged."""
    return _PRETTY.get(name, name)


def html_group(name: str) -> str:
    """Schoenflies symbol with the subscript as HTML (``C<sub>2v</sub>``)."""
    pretty = pretty_group(name)
    if len(pretty) <= 1 or pretty.startswith("SO"):
        return pretty
    return f"{pretty[0]}<sub>{pretty[1:]}</sub>"


_ORDER_FIXED = {
    "C1": 1,
    "Ci": 2,
    "Cs": 2,
    "T": 12,
    "Td": 24,
    "Th": 24,
    "O": 24,
    "Oh": 48,
    "I": 60,
    "Ih": 120,
    "Coov": 10_000,
    "Dooh": 20_000,
    "SO3": 100_000,
}


def group_order(name: str) -> int:
    """Number of symmetry operations (∞-groups sort above all finite ones)."""
    if name in _ORDER_FIXED:
        return _ORDER_FIXED[name]
    m = re.fullmatch(r"([CDS])(\d+)([vhd]?)", name)
    if not m:
        return 1
    kind, n_str, suffix = m.groups()
    n = int(n_str)
    if kind == "S":
        return n
    if kind == "C":
        return n if not suffix else 2 * n
    return 2 * n if not suffix else 4 * n  # Dn: 2n; Dnh, Dnd: 4n


def _detect(atoms_bohr: list, tol_bohr: Optional[float]) -> str:
    from pyscf.symm import geom

    with _TOLERANCE_LOCK:
        old = geom.TOLERANCE
        try:
            if tol_bohr is not None:
                geom.TOLERANCE = tol_bohr
            return str(geom.detect_symm(atoms_bohr)[0])
        finally:
            geom.TOLERANCE = old


def detect_point_group(
    atoms: Sequence[str],
    coords_angstrom: Sequence[Sequence[float]],
    tolerance_angstrom: float = DEFAULT_TOLERANCE_ANGSTROM,
) -> Optional[PointGroup]:
    """Point group of a geometry, or ``None`` if it cannot be determined.

    ``None`` when PySCF is not installed (e.g. native Windows / PyFock) or
    detection fails; callers simply omit the row.
    """
    if not atoms:
        return None
    try:
        atoms_bohr = [
            (str(sym), [float(v) * BOHR_PER_ANGSTROM for v in xyz])
            for sym, xyz in zip(atoms, coords_angstrom)
        ]
        exact = _detect(atoms_bohr, None)
        group = _detect(atoms_bohr, tolerance_angstrom * BOHR_PER_ANGSTROM)
        loose = _detect(atoms_bohr, LOOSE_TOLERANCE_ANGSTROM * BOHR_PER_ANGSTROM)
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 — symmetry is informational only
        logger.debug("Point-group detection failed: %s", exc)
        return None
    # The tolerance ladder must never report *less* symmetry at a looser
    # tolerance; PySCF's detector rounds inertia moments, so guard it.
    if group_order(exact) > group_order(group):
        group = exact
    return PointGroup(
        group=group,
        tolerance_angstrom=tolerance_angstrom,
        exact_group=exact,
        loose_group=loose if group_order(loose) > group_order(group) else None,
    )


def point_group_of_molecule(molecule: Any) -> Optional[PointGroup]:
    """:func:`detect_point_group` for a :class:`~quantui.molecule.Molecule`."""
    return detect_point_group(
        list(getattr(molecule, "atoms", []) or []),
        list(getattr(molecule, "coordinates", []) or []),
    )


def point_group_of_atom_list(mol_atom: Any) -> Optional[PointGroup]:
    """:func:`detect_point_group` for a ``[(symbol, [x, y, z]), ...]`` list (Å)."""
    try:
        atoms = [str(a[0]) for a in mol_atom]
        coords = [list(map(float, a[1])) for a in mol_atom]
    except Exception:  # noqa: BLE001 — malformed saved data
        return None
    return detect_point_group(atoms, coords)


def _numbered(raw: Sequence[str]) -> List[str]:
    counts: dict = {}
    out = []
    for name in raw:
        counts[name] = counts.get(name, 0) + 1
        out.append(f"{counts[name]}{name.lower()}")
    return out


# PySCF's linear-molecule irreps → the σ/π/δ notation chemists use.
_LINEAR_GREEK = {"A1": "σ", "A2": "σ⁻", "E1": "π", "E2": "δ", "E3": "φ", "E4": "γ"}


def _linear_base(name: str) -> str:
    """``"E1ux"`` → ``"πu"``, ``"A1g"`` → ``"σg"``, ``"E1x"`` → ``"π"``."""
    m = re.fullmatch(r"([AE]\d+)([gu]?)([xy]?)", name)
    if not m:
        return name.lower()
    head, parity, _ = m.groups()
    return _LINEAR_GREEK.get(head, head.lower()) + parity


def _numbered_linear(raw: Sequence[str]) -> List[str]:
    """Number linear-molecule orbitals; an x/y degenerate pair shares a number."""
    counts: dict = {}
    out: List[str] = []
    prev_raw = None
    for name in raw:
        base = _linear_base(name)
        partner = (
            prev_raw is not None
            and name[-1:] in ("x", "y")
            and prev_raw[:-1] == name[:-1]
            and prev_raw[-1:] != name[-1:]
        )
        if partner:
            out.append(out[-1])
            prev_raw = None  # a pair is complete; never chain three
            continue
        counts[base] = counts.get(base, 0) + 1
        out.append(f"{counts[base]}{base}")
        prev_raw = name
    return out


def _c3v_from_cs(raw: Sequence[str], mo_energy: Sequence[float]) -> List[str]:
    """Cs labels of a C3v molecule → C3v labels (a1, a2, e) via degeneracy.

    C3v → Cs correlation: A1 → A', A2 → A'', E → A' + A''. A degenerate
    pair is an e level; a lone A' is a1, a lone A'' is a2. Unambiguous,
    unlike most other non-abelian groups (e.g. Td → D2 cannot separate
    A1 from A2), which keep their subgroup labels.
    """
    tol = 1e-4  # Hartree
    counts: dict = {}
    out: List[str] = []
    i = 0
    n = len(raw)
    while i < n:
        pair = i + 1 < n and abs(mo_energy[i + 1] - mo_energy[i]) < tol
        name = "e" if pair else ("a1" if raw[i] == "A'" else "a2")
        counts[name] = counts.get(name, 0) + 1
        label = f"{counts[name]}{name}"
        out += [label, label] if pair else [label]
        i += 2 if pair else 1
    return out


def label_mo_irreps(
    mol_atom: Any,
    basis: str,
    mo_coeff: Any,
    *,
    tolerance_angstrom: float = DEFAULT_TOLERANCE_ANGSTROM,
    min_purity: float = 0.9,
    mo_energy: Optional[Sequence[float]] = None,
) -> Optional[MOIrreps]:
    """Numbered irrep labels for the columns of *mo_coeff*.

    *mol_atom* is the ``[(symbol, [x, y, z]), ...]`` list in Å that the
    orbitals were computed for, *basis* its basis-set name. For an
    unrestricted result pass one spin channel (2-D *mo_coeff*). With
    *mo_energy* (Hartree, same channel), a C3v molecule gets C3v labels
    rather than its Cs subgroup's.

    Returns ``None`` for C1 geometries, when PySCF is unavailable, when the
    AO basis does not match *mo_coeff*, or when any orbital is less than
    *min_purity* symmetry-pure (a geometry too far from the detected
    symmetry to label honestly).
    """
    try:
        import numpy as np
        from pyscf import gto, symm
        from pyscf.symm import geom

        from .inorganic_guards import ecp_for_basis

        atoms = [str(a[0]) for a in mol_atom]
        coeff = np.asarray(mo_coeff, dtype=float)
        if coeff.ndim != 2:
            return None
        with _TOLERANCE_LOCK:
            old = geom.TOLERANCE
            geom.TOLERANCE = tolerance_angstrom * BOHR_PER_ANGSTROM
            try:
                mol = None
                for spin in (0, 1):
                    try:
                        mol = gto.M(
                            atom=[(a[0], list(map(float, a[1]))) for a in mol_atom],
                            basis=basis,
                            ecp=ecp_for_basis(basis, atoms),
                            spin=spin,
                            symmetry=True,
                            verbose=0,
                        )
                        break
                    except RuntimeError:
                        continue
            finally:
                geom.TOLERANCE = old
        if mol is None or mol.groupname == "C1" or mol.nao != coeff.shape[0]:
            return None
        s = mol.intor_symmetric("int1e_ovlp")
        # Degenerate partners (e.g. the E pair of NH3) come out of an
        # unsymmetrized SCF as arbitrary mixtures; rotate each degenerate
        # space onto symmetry-pure combinations first.
        coeff = symm.symmetrize_space(mol, coeff, s=s, check=False)
        raw = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, coeff, s=s, check=False
        )
        # Purity: weight of each orbital in its assigned irrep's subspace.
        purity = []
        for i, name in enumerate(raw):
            so = mol.symm_orb[list(mol.irrep_name).index(name)]
            proj = so.T @ s @ coeff[:, i]
            sso = so.T @ s @ so
            purity.append(float(proj @ np.linalg.solve(sso, proj)))
        if min(purity) < min_purity:
            logger.debug(
                "MO irreps skipped: min purity %.3f < %.2f", min(purity), min_purity
            )
            return None
        names = [str(n) for n in raw]
        group, top = str(mol.groupname), str(mol.topgroup)
        if group in ("Dooh", "Coov"):
            labels = _numbered_linear(names)
        elif top == "C3v" and group == "Cs" and mo_energy is not None:
            labels = _c3v_from_cs(names, [float(e) for e in mo_energy])
            group = "C3v"
        else:
            labels = _numbered(names)
        return MOIrreps(labels=labels, group=group, top_group=top)
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 — symmetry is informational only
        logger.debug("MO irrep labelling failed: %s", exc)
        return None
