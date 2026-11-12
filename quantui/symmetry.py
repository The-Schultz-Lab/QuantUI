"""Molecular point groups; orbital and vibrational-mode symmetry labels.

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

Vibrational modes: :func:`label_mode_irreps` applies
the subgroup's operations to the normal-mode displacements, labelling each
degenerate set by the trace of its representation. Full-group labels come
back where the subgroup plus degeneracy determines them (C3v, D3h, D3,
D3d, D6h, linear molecules) or from built-in character tables (Td, Oh,
D4h, D4, C4v, D2d); other groups (e.g. D5h) keep subgroup labels, flagged
as such.

QuantUI never runs its SCFs with ``symmetry=True``. Labels are assigned
afterwards by projecting the converged orbitals onto PySCF's
symmetry-adapted AOs, built for the same geometry and basis (PySCF keeps
the input orientation, so no rotation is needed).
"""

from __future__ import annotations

import html as _html
import logging
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    # Escaped: the name can come from a saved result.json.
    pretty = _html.escape(pretty_group(name))
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


def _detect_full(atoms_bohr: list, tol_bohr: Optional[float]) -> Tuple[str, Any, Any]:
    """PySCF ``detect_symm`` at *tol_bohr*: ``(group, origin, axes)``."""
    from pyscf.symm import geom

    with _TOLERANCE_LOCK:
        old = geom.TOLERANCE
        try:
            if tol_bohr is not None:
                geom.TOLERANCE = tol_bohr
            group, orig, axes = geom.detect_symm(atoms_bohr)
            return str(group), orig, axes
        finally:
            geom.TOLERANCE = old


def _detect(atoms_bohr: list, tol_bohr: Optional[float]) -> str:
    return _detect_full(atoms_bohr, tol_bohr)[0]


#: Tolerances (Å) tried between PySCF's strict default and the loose hint.
#: PySCF's detector rounds inertia moments and distances, so the group it
#: finds is not monotonic in the tolerance (a 2-decimal benzene is D2h at
#: 0.001-0.015 Å and D6h from 0.02 Å; noisy NH3/CH4/benzene drop to
#: C1/C3/C2h at single tolerances). Taking the highest order over a ladder
#: removes most of that noise.
_LADDER_ANGSTROM = (
    0.001,
    0.003,
    DEFAULT_TOLERANCE_ANGSTROM,
    0.02,
    0.03,
    LOOSE_TOLERANCE_ANGSTROM,
)

#: Uniform scale factors the ladder is also run at. Scaling the geometry
#: and the tolerance together changes nothing physically but moves PySCF's
#: rounding boundaries; ±0.0005 Å noise on benzene gives C2h at every
#: tolerance unscaled, D6h at most rungs scaled by 1.013.
_LADDER_SCALES = (1.0, 1.013)


def _rot(axis: Any, angle: float) -> Any:
    import numpy as np

    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _mirror(normal: Any) -> Any:
    import numpy as np

    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return np.eye(3) - 2.0 * np.outer(n, n)


def _generators(group: str) -> Optional[List[Any]]:
    """Generators of *group* in PySCF's ``detect_symm`` axis frame.

    PySCF puts the principal axis on z, a C2 axis (D groups) or a mirror
    normal (Cnv, Cs) on x, and the cubic groups' C2/C4 axes on x, y, z.
    ``None`` for a group that is not checked here (I, Ih).
    """
    import numpy as np

    x, _y, z = np.eye(3)
    inv = -np.eye(3)
    if group in ("C1", "SO3", "Coov", "Dooh"):
        return []
    if group == "Ci":
        return [inv]
    if group == "Cs":
        return [_mirror(z)]
    c3 = _rot(np.ones(3), 2.0 * np.pi / 3.0)
    s4 = _mirror(z) @ _rot(z, np.pi / 2.0)
    cubic = {
        "T": [_rot(z, np.pi), _rot(x, np.pi), c3],
        "Td": [_rot(z, np.pi), _rot(x, np.pi), c3, s4],
        "Th": [_rot(z, np.pi), _rot(x, np.pi), c3, inv],
        "O": [_rot(z, np.pi / 2.0), c3],
        "Oh": [_rot(z, np.pi / 2.0), c3, inv],
    }
    if group in cubic:
        return cubic[group]
    m = re.fullmatch(r"([CDS])(\d+)([vhd]?)", group)
    if not m:
        return None
    kind, n_str, suffix = m.groups()
    n = int(n_str)
    if kind == "S":
        return [_mirror(z) @ _rot(z, 2.0 * np.pi / n)]
    gens = [_rot(z, 2.0 * np.pi / n)]
    if kind == "D":
        gens.append(_rot(x, np.pi))
    if suffix == "v":
        gens.append(_mirror(x))
    elif suffix == "h":
        gens.append(_mirror(z))
    elif suffix == "d":
        gens.append(_mirror(z) @ _rot(z, np.pi / n))
    return gens


def _holds_within(
    group: str, atoms_bohr: list, orig: Any, axes: Any, tol_bohr: float
) -> bool:
    """True when every generator of *group* maps each atom onto an atom of
    the same element no more than *tol_bohr* away (a real distance check,
    unlike PySCF's rounded detection)."""
    import numpy as np

    gens = _generators(group)
    if gens is None:
        return False
    syms = [a[0] for a in atoms_bohr]
    coords = (np.array([a[1] for a in atoms_bohr], dtype=float) - orig) @ np.asarray(
        axes
    ).T
    if group in ("Coov", "Dooh"):
        if np.abs(coords[:, :2]).max() > tol_bohr:
            return False
        if group == "Dooh":
            gens = [-np.eye(3)]
    for op in gens:
        moved = coords @ np.asarray(op).T
        for i, sym in enumerate(syms):
            same = coords[[j for j, s in enumerate(syms) if s == sym]]
            if np.linalg.norm(same - moved[i], axis=1).min() > tol_bohr:
                return False
    return True


def rotational_symmetry_number(group: str) -> int:
    """Rotational symmetry number σ of a point group (PySCF names).

    σ is the order of the group's proper-rotation subgroup: 2 for water
    (C2v), 3 for NH3 (C3v), 12 for CH4 (Td) and benzene (D6h), 1 for C1/Cs/
    Ci/C∞v, 2 for D∞h. Same table as PySCF's ``hessian.thermo``.
    """
    if group in ("SO3", "C1", "Ci", "Cs", "Coov"):
        return 1
    if group == "Dooh":
        return 2
    if group in ("T", "Td", "Th"):
        return 12
    if group in ("O", "Oh"):
        return 24
    if group in ("I", "Ih"):
        return 60
    m = re.fullmatch(r"([CDS])(\d+)([vhd]?)", group)
    if not m:
        return 1
    kind, n_str, _ = m.groups()
    n = int(n_str)
    if kind == "C":
        return n
    if kind == "D":
        return 2 * n
    return max(1, n // 2)  # Sn (n even): proper subgroup C(n/2)


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
        exact = _detect_full(atoms_bohr, None)
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 — symmetry is informational only
        logger.debug("Point-group detection failed: %s", exc)
        return None
    # (tolerance Å, (group, origin, axes)); exact first.
    candidates: List[Tuple[float, Tuple[str, Any, Any]]] = [(0.0, exact)]
    ladder = sorted(
        set(_LADDER_ANGSTROM) | {float(tolerance_angstrom), LOOSE_TOLERANCE_ANGSTROM}
    )
    for scale in _LADDER_SCALES:
        scaled = [(sym, [v * scale for v in xyz]) for sym, xyz in atoms_bohr]
        for tol in ladder:
            try:
                name, orig, axes = _detect_full(scaled, tol * BOHR_PER_ANGSTROM * scale)
            except Exception as exc:  # noqa: BLE001 — skip one failed rung
                logger.debug("Point-group detection at %g Å failed: %s", tol, exc)
                continue
            candidates.append((tol, (name, orig / scale, axes)))
    # The group is the highest-order one found at or below the working
    # tolerance: PySCF's detector rounds inertia moments, so a single
    # tolerance can report *less* symmetry than a stricter one (audit GEOM-7).
    group = exact[0]
    for tol, (name, _o, _a) in candidates:
        if tol <= tolerance_angstrom and group_order(name) > group_order(group):
            group = name
    # A higher group found only at a looser rung still counts when every
    # atom really is within the working tolerance of its symmetric position
    # (a 2-decimal benzene is D6h within 0.004 Å, but PySCF's rounding only
    # finds D6h from 0.02 Å; audit SCF-4).
    tol_bohr = tolerance_angstrom * BOHR_PER_ANGSTROM
    for tol, (name, orig, axes) in candidates:
        if (
            tol > tolerance_angstrom
            and group_order(name) > group_order(group)
            and _holds_within(name, atoms_bohr, orig, axes, tol_bohr)
        ):
            group = name
    loose = group
    for tol, (name, _o, _a) in candidates:
        if tol <= LOOSE_TOLERANCE_ANGSTROM and group_order(name) > group_order(loose):
            loose = name
    return PointGroup(
        group=group,
        tolerance_angstrom=tolerance_angstrom,
        exact_group=exact[0],
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


#: Largest energy gap (Hartree) between the A' and A'' halves of a C3v e
#: pair. A geometry symmetric only to within the 0.01 Å detection
#: tolerance splits an e pair by a few tenths of a mHa (NH3, ±0.0006 Å:
#: 0.4 mHa), so 1e-4 Ha was too tight.
_C3V_E_PAIR_TOL = 5e-3


def _c3v_from_cs(raw: Sequence[str], mo_energy: Sequence[float]) -> Optional[List[str]]:
    """Cs labels of a C3v molecule → C3v labels (a1, a2, e) via degeneracy.

    C3v → Cs correlation: A1 → A', A2 → A'', E → A' + A''. An e level is
    an adjacent A'/A'' pair (one of each) within ``_C3V_E_PAIR_TOL``; a
    lone A' is a1. A lone A'' would be a2, but it is far more often half
    of an e pair split by an imperfect geometry, so the mapping is
    refused (``None``: keep the Cs labels) rather than invent an a2.
    Unambiguous otherwise, unlike most other non-abelian groups (e.g.
    Td → D2 cannot separate A1 from A2), which keep their subgroup labels.
    """
    n = len(raw)

    def gap(i: int) -> Optional[float]:
        """|ΔE| of orbitals i, i+1 when they can form an e pair, else None."""
        if i + 1 >= n or {raw[i], raw[i + 1]} != {"A'", 'A"'}:
            return None
        d = abs(mo_energy[i + 1] - mo_energy[i])
        return d if d < _C3V_E_PAIR_TOL else None

    counts: dict = {}
    out: List[str] = []
    i = 0
    while i < n:
        g = gap(i)
        g_next = gap(i + 1)
        # A' A'' A' with the closer pair second: orbital i is the lone a1.
        pair = g is not None and (g_next is None or g <= g_next)
        if not pair and raw[i] != "A'":
            return None
        name = "e" if pair else "a1"
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
        else:
            c3v = None
            if top == "C3v" and group == "Cs" and mo_energy is not None:
                c3v = _c3v_from_cs(names, [float(e) for e in mo_energy])
            if c3v is not None:
                labels, group = c3v, "C3v"
            else:
                labels = _numbered(names)
        return MOIrreps(labels=labels, group=group, top_group=top)
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 — symmetry is informational only
        logger.debug("MO irrep labelling failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Vibrational-mode irreps
# ---------------------------------------------------------------------------

#: Modes this close in frequency (cm⁻¹) are treated as one degenerate set.
#: Wide enough for the splitting a 0.01 Å-symmetric geometry leaves in an
#: E pair, narrow enough that accidental near-degeneracies are rare.
MODE_DEGENERACY_CM1 = 5.0

# Full-group labels recoverable from the computational subgroup once
# degenerate sets are known: (top, subgroup) → (degenerate pairs, singles,
# character-table order). A subgroup pair (a, b) in one set is the full
# group's E; a lone subgroup irrep maps one-to-one. Where that is
# ambiguous (Td → D2 cannot tell A1 from A2, nor T1 from T2) a full table
# below is used, or the subgroup's labels are kept (e.g. D4h → D2h, where
# A1g and B1g both become Ag). Linear molecules run in D2h / C2v with z
# along the axis: σ along it, π an x/y pair.
_RECOVERABLE = {
    ("Dooh", "D2h"): (
        {("B2g", "B3g"): "Πg", ("B2u", "B3u"): "Πu"},
        {"Ag": "Σg⁺", "B1g": "Σg⁻", "B1u": "Σu⁺", "Au": "Σu⁻"},
        ["Σg⁺", "Σg⁻", "Πg", "Σu⁺", "Σu⁻", "Πu"],
    ),
    ("Coov", "C2v"): (
        {("B1", "B2"): "Π"},
        {"A1": "Σ⁺", "A2": "Σ⁻"},
        ["Σ⁺", "Σ⁻", "Π"],
    ),
    # C3v → Cs (σ = a σv): A1 → A′, A2 → A″, E → A′ + A″.
    ("C3v", "Cs"): ({("A′", "A″"): "E"}, {"A′": "A1", "A″": "A2"}, ["A1", "A2", "E"]),
    # D3h → C2v (PySCF puts σh as σ(xz), so B1 is symmetric under it).
    ("D3h", "C2v"): (
        {("A1", "B1"): "E′", ("A2", "B2"): "E″"},
        {"A1": "A1′", "B1": "A2′", "A2": "A1″", "B2": "A2″"},
        ["A1′", "A2′", "E′", "A1″", "A2″", "E″"],
    ),
    ("D3", "C2"): ({("A", "B"): "E"}, {"A": "A1", "B": "A2"}, ["A1", "A2", "E"]),
    ("D3d", "C2h"): (
        {("Ag", "Bg"): "Eg", ("Au", "Bu"): "Eu"},
        {"Ag": "A1g", "Bg": "A2g", "Au": "A1u", "Bu": "A2u"},
        ["A1g", "A2g", "Eg", "A1u", "A2u", "Eu"],
    ),
    # D6h → D2h with z the C6 axis and x a C2' axis (through atoms, the
    # convention for benzene); label_mode_irreps orients x that way.
    ("D6h", "D2h"): (
        {
            ("Ag", "B1g"): "E2g",
            ("B2g", "B3g"): "E1g",
            ("Au", "B1u"): "E2u",
            ("B2u", "B3u"): "E1u",
        },
        {
            "Ag": "A1g",
            "B1g": "A2g",
            "B3g": "B1g",
            "B2g": "B2g",
            "Au": "A1u",
            "B1u": "A2u",
            "B3u": "B1u",
            "B2u": "B2u",
        },
        ["A1g", "A2g", "B1g", "B2g", "E1g", "E2g"]
        + ["A1u", "A2u", "B1u", "B2u", "E1u", "E2u"],
    ),
}


# Cubic groups, whose subgroup labels say little (CH4's E pair is "a+a" in
# D2), get their own character tables: one operation per class, written in
# the frame PySCF uses for the subgroup (x, y, z along the C2 / C4 axes).
_C3_111 = ((0, 0, 1), (1, 0, 0), (0, 1, 0))  # about (1, 1, 1): x → y → z
_C4_Z = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
_S4_Z = ((0, -1, 0), (1, 0, 0), (0, 0, -1))
_C2_Z = ((-1, 0, 0), (0, -1, 0), (0, 0, 1))
_SIGMA_XY_SWAP = ((0, 1, 0), (1, 0, 0), (0, 0, 1))  # σd through z, (1, 1, 0)
_C2_110 = ((0, 1, 0), (1, 0, 0), (0, 0, -1))


def _neg(m: tuple) -> tuple:
    return tuple(tuple(-v for v in row) for row in m)


_E3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_ClassOps = List[Tuple[int, tuple]]
_Irreps = List[Tuple[str, Tuple[int, ...]]]
_FULL_TABLES: Dict[str, Tuple[str, _ClassOps, _Irreps]] = {
    # group: (computational subgroup, [(class size, operation)], [(irrep, χ)])
    "Td": (
        "D2",
        [(1, _E3), (8, _C3_111), (3, _C2_Z), (6, _S4_Z), (6, _SIGMA_XY_SWAP)],
        [
            ("A1", (1, 1, 1, 1, 1)),
            ("A2", (1, 1, 1, -1, -1)),
            ("E", (2, -1, 2, 0, 0)),
            ("T1", (3, 0, -1, 1, -1)),
            ("T2", (3, 0, -1, -1, 1)),
        ],
    ),
    "Oh": (
        "D2h",
        [
            (1, _E3),
            (8, _C3_111),
            (6, _C2_110),
            (6, _C4_Z),
            (3, _C2_Z),
            (1, _neg(_E3)),  # i
            (6, _S4_Z),
            (8, _neg(_C3_111)),  # S6 = i·C3
            (3, ((1, 0, 0), (0, 1, 0), (0, 0, -1))),  # σh
            (6, _neg(_C2_110)),  # σd = i·C2'
        ],
        [
            ("A1g", (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
            ("A2g", (1, 1, -1, -1, 1, 1, -1, 1, 1, -1)),
            ("Eg", (2, -1, 0, 0, 2, 2, 0, -1, 2, 0)),
            ("T1g", (3, 0, -1, 1, -1, 3, 1, 0, -1, -1)),
            ("T2g", (3, 0, 1, -1, -1, 3, -1, 0, -1, 1)),
            ("A1u", (1, 1, 1, 1, 1, -1, -1, -1, -1, -1)),
            ("A2u", (1, 1, -1, -1, 1, -1, 1, -1, -1, 1)),
            ("Eu", (2, -1, 0, 0, 2, -2, 0, 1, -2, 0)),
            ("T1u", (3, 0, -1, 1, -1, -3, -1, 0, 1, 1)),
            ("T2u", (3, 0, 1, -1, -1, -3, 1, 0, 1, -1)),
        ],
    ),
}


_C2_X = ((1, 0, 0), (0, -1, 0), (0, 0, -1))
_SIGMA_XZ = ((1, 0, 0), (0, -1, 0), (0, 0, 1))  # contains z and x
_SIGMA_H = ((1, 0, 0), (0, 1, 0), (0, 0, -1))
# A1, A2, B1, B2, E of the groups with a fourfold axis (D4, C4v, D2d): columns
# E, 2X4, C2, 2(C2' or σv), 2(C2'' or σd).
_FOURFOLD_IRREPS: _Irreps = [
    ("A1", (1, 1, 1, 1, 1)),
    ("A2", (1, 1, 1, -1, -1)),
    ("B1", (1, -1, 1, 1, -1)),
    ("B2", (1, -1, 1, -1, 1)),
    ("E", (2, 0, -2, 0, 0)),
]


def _with_inversion(rows: _Irreps) -> _Irreps:
    """g and u irreps of G × Ci, classes ordered (G, i·G)."""
    return [(n + "g", c + c) for n, c in rows] + [
        (n + "u", c + tuple(-x for x in c)) for n, c in rows
    ]


# Their D2h / C2v / D2 subgroup cannot tell B1 from B2 (or A1 from B1), so
# these get full tables too. B1 vs B2 depends on which C2 axes are C2' (or
# which mirror planes are σv); _fourfold_frame puts x along a C2' axis
# through atoms (or the xz plane, a σv, through atoms), the usual convention
# (square-planar AB4: B1g in-plane stretch, B2g in-plane bend).
_D4_CLASSES: _ClassOps = [
    (1, _E3),
    (2, _C4_Z),
    (1, _C2_Z),
    (2, _C2_X),  # C2'
    (2, _C2_110),  # C2''
]
_FULL_TABLES.update(
    {
        # i·C4 is in the S4 class, i·C2 is σh, i·C2' a σv, i·C2'' a σd.
        "D4h": (
            "D2h",
            _D4_CLASSES + [(w, _neg(m)) for w, m in _D4_CLASSES],
            _with_inversion(_FOURFOLD_IRREPS),
        ),
        "D4": ("D2", _D4_CLASSES, _FOURFOLD_IRREPS),
        "C4v": (
            "C2v",
            [(1, _E3), (2, _C4_Z), (1, _C2_Z), (2, _SIGMA_XZ), (2, _SIGMA_XY_SWAP)],
            _FOURFOLD_IRREPS,
        ),
        "D2d": (
            "D2",
            [(1, _E3), (2, _S4_Z), (1, _C2_Z), (2, _C2_X), (2, _SIGMA_XY_SWAP)],
            _FOURFOLD_IRREPS,
        ),
    }
)


def _fourfold_frame(top: str, coords: Any, axes: Any, match_tol: float) -> Any:
    """Axes in which *top*'s full table applies, or None.

    Tries PySCF's frame and the same frame turned 45° about z. A frame must
    map the molecule onto itself under every class operation; of those, the
    one with an atom on the x axis (D4h, D4: x is a C2' axis) or in the xz
    plane off the z axis (C4v: xz is a σv) follows the usual convention.
    D2d has a single valid frame. With no such atom B1 and B2 cannot be
    named, and the subgroup's labels are kept.
    """
    import numpy as np

    ops = [np.asarray(m, float) for _, m in _FULL_TABLES[top][1]]

    def valid(c: Any) -> bool:
        return all(
            np.linalg.norm(c - row, axis=1).min() <= match_tol
            for r in ops
            for row in c @ r.T
        )

    def follows_convention(c: Any) -> bool:
        if top == "D2d":
            return True
        off_z = np.abs(c[:, 0]) > match_tol
        if top == "C4v":
            return bool(np.any((np.abs(c[:, 1]) < match_tol) & off_z))
        return bool(np.any((np.abs(c[:, 1:]).max(axis=1) < match_tol) & off_z))

    h = np.sqrt(0.5)
    rot = np.array([[h, h, 0.0], [-h, h, 0.0], [0.0, 0.0, 1.0]])
    for frame, c in ((axes, coords), (rot @ axes, coords @ rot.T)):
        if valid(c) and follows_convention(c):
            return frame
    return None


def _primes(name: str) -> str:
    """PySCF's Cs names ``A'`` / ``A"`` → ``A′`` / ``A″``."""
    return name.replace('"', "″").replace("'", "′")


@dataclass
class ModeIrreps:
    """Symmetry labels for a set of normal modes."""

    # One per mode, aligned with ``frequencies_cm1``; ``None`` for the
    # near-zero modes the UI skips. Lower case: "a1", "e", "b2u", "πu".
    # A degenerate set the group cannot name shares a joined label ("b1+b2").
    labels: List[Optional[str]]
    group: str  # group the labels belong to
    top_group: str  # full point group of the geometry
    gamma: str  # Γvib, e.g. "2A1 + 2E"

    @property
    def is_subgroup(self) -> bool:
        return self.group != self.top_group

    def caption(self) -> str:
        where = pretty_group(self.group)
        if self.is_subgroup:
            where += f" (subgroup of {pretty_group(self.top_group)})"
        return f"Mode symmetry in {where}: Γvib = {self.gamma}"

    def caption_html(self) -> str:
        where = html_group(self.group)
        if self.is_subgroup:
            where += f" (subgroup of {html_group(self.top_group)})"
        return f"Mode symmetry in {where}: Γ<sub>vib</sub> = {html_gamma(self.gamma)}"


def mode_irreps_from_saved(saved: Any) -> Optional[ModeIrreps]:
    """A :class:`ModeIrreps` from the dict saved with a Frequency result.

    The labels are computed when the result is saved (PySCF at hand), so a
    reader without PySCF (History on Windows, the Viewer) can show them.
    """
    if isinstance(saved, ModeIrreps):
        return saved
    if not isinstance(saved, dict) or not isinstance(saved.get("labels"), list):
        return None
    try:
        return ModeIrreps(
            labels=[None if lbl is None else str(lbl) for lbl in saved["labels"]],
            group=str(saved["group"]),
            top_group=str(saved.get("top_group") or saved["group"]),
            gamma=str(saved.get("gamma") or ""),
        )
    except (KeyError, TypeError):
        return None


def _mode_label(irrep: str) -> str:
    """``"A1"`` → ``"a1"``, ``"Πu"`` → ``"πu"``, ``"B1+B2"`` → ``"b1+b2"``."""
    return "+".join(p[:1].lower() + p[1:] for p in irrep.split("+"))


def html_irrep(label: str) -> str:
    """Irrep or mode label with HTML subscripts: ``"a1g"`` → ``a<sub>1g</sub>``.

    Primes (Cs) and ⁺/⁻ (Σ) follow the subscript; joined labels
    (``"b1+b2"``) are rendered per part.
    """
    # Labels come from saved results (a shared results folder too), so
    # every piece is escaped; only the <sub> tags added here are markup.
    out = []
    for part in label.split("+"):
        m = re.fullmatch(r"(.)([0-9gu]*)(.*)", part, flags=re.DOTALL)
        if not m:
            return _html.escape(label)
        head, sub, rest = m.groups()
        out.append(
            _html.escape(head)
            + (f"<sub>{sub}</sub>" if sub else "")
            + _html.escape(rest)
        )
    return "+".join(out)


def html_gamma(gamma: str) -> str:
    """``"2A1 + 2E"`` with HTML subscripts."""
    terms = []
    for term in gamma.split(" + "):
        m = re.fullmatch(r"(\d*)(\(?)(.*?)(\)?)", term, flags=re.DOTALL)
        count, lp, body, rp = m.groups() if m else ("", "", term, "")
        terms.append(f"{count}{lp}{html_irrep(body)}{rp}")
    return " + ".join(terms)


def _gamma(terms: List[str], order: Sequence[str]) -> str:
    """``["A1", "E", "A1", "E"]`` → ``"2A1 + 2E"``, in character-table order."""
    counts: dict = {}
    for t in terms:
        counts[t] = counts.get(t, 0) + 1

    def key(t: str):
        head = t.split("+")[0]
        return (order.index(head) if head in order else len(order), t)

    out = []
    for t in sorted(counts, key=key):
        name = f"({t})" if "+" in t else t
        out.append(f"{counts[t] if counts[t] > 1 else ''}{name}")
    return " + ".join(out)


def _name_set(counts: dict, subgroup: str, top: str) -> Optional[List[str]]:
    """Subgroup irrep counts of one degenerate set → full-group irrep names.

    One name per irrep the set contains (a degenerate irrep once), or
    ``None`` when the full group's labels cannot be recovered.
    """
    rule = _RECOVERABLE.get((top, subgroup))
    if rule is None:
        return None
    pairs, singles, _ = rule
    left = dict(counts)
    names: List[str] = []
    for (a, b), name in pairs.items():
        while left.get(a, 0) > 0 and left.get(b, 0) > 0:
            left[a] -= 1
            left[b] -= 1
            names.append(name)
    for irrep, n in left.items():
        if n == 0:
            continue
        if irrep not in singles:
            return None
        names += [singles[irrep]] * n
    return names


def label_mode_irreps(
    mol_atom: Any,
    frequencies_cm1: Sequence[float],
    displacements: Any,
    *,
    tolerance_angstrom: float = DEFAULT_TOLERANCE_ANGSTROM,
    degeneracy_cm1: float = MODE_DEGENERACY_CM1,
    min_purity: float = 0.9,
    min_frequency_cm1: float = 10.0,
) -> Optional[ModeIrreps]:
    """Irrep label for each normal mode (Γvib of the molecule).

    *mol_atom* is the ``[(symbol, [x, y, z]), ...]`` geometry in Å the
    Hessian was computed at; *displacements* the ``(n_modes, n_atoms, 3)``
    normal-mode vectors aligned with *frequencies_cm1*. Modes below
    *min_frequency_cm1* (residual translations/rotations) get ``None``.

    Characters come from applying the computational subgroup's operations
    (D2h and its subgroups, as for :func:`label_mo_irreps`) to each mode.
    A degenerate set (frequencies within *degeneracy_cm1*) is labelled as a
    whole: the trace of its representation is basis-independent, so the
    arbitrary mixing of, e.g., NH3's E pair does not matter. Full-group
    labels are recovered where the subgroup determines them unambiguously
    (C3v: NH3 → 2A1 + 2E; linear molecules: σ/π); otherwise the subgroup's
    labels are used and :attr:`ModeIrreps.is_subgroup` is true.

    Returns ``None`` for C1 geometries, when PySCF is unavailable, when the
    shapes do not match, or when any mode set is less than *min_purity*
    symmetry-pure.
    """
    try:
        import numpy as np
        from pyscf.symm import geom, param

        atoms = [str(a[0]) for a in mol_atom]
        xyz = np.array([list(map(float, a[1])) for a in mol_atom]) * BOHR_PER_ANGSTROM
        disp = np.asarray(displacements, dtype=float)
        freqs = [float(f) for f in frequencies_cm1]
        if disp.shape != (len(freqs), len(atoms), 3) or not atoms:
            return None
        with _TOLERANCE_LOCK:
            old = geom.TOLERANCE
            geom.TOLERANCE = tolerance_angstrom * BOHR_PER_ANGSTROM
            try:
                top, orig, axes = geom.detect_symm(list(zip(atoms, xyz)))
            finally:
                geom.TOLERANCE = old
            # PySCF rounds inertia moments, so a looser tolerance can report
            # less symmetry (an exact square pyramid as C2v); as in
            # detect_point_group, never take less than the strict result.
            strict = geom.detect_symm(list(zip(atoms, xyz)))
            if group_order(str(strict[0])) > group_order(str(top)):
                top, orig, axes = strict
        top = str(top)
        if top in ("Dooh", "Coov"):
            subgroup = "D2h" if top == "Dooh" else "C2v"  # z along the axis
        else:
            subgroup, axes = geom.as_subgroup(top, axes)
        if subgroup not in param.OPERATOR_TABLE or subgroup == "C1":
            return None
        match_tol = max(0.1, 5 * tolerance_angstrom) * BOHR_PER_ANGSTROM
        coords = (xyz - orig) @ axes.T
        recoverable = True
        if top == "D6h":
            # Which C2 axes are C2' (through atoms) decides B1 vs B2.
            def through_atoms(k: int) -> bool:
                off = np.abs(np.delete(coords, k, axis=1)).max(axis=1)
                return bool(
                    np.any((off < match_tol) & (np.abs(coords[:, k]) > match_tol))
                )

            if through_atoms(1) and not through_atoms(0):
                axes = axes[[1, 0, 2]]
                coords = (xyz - orig) @ axes.T
            recoverable = through_atoms(0)
        table_group: Optional[str] = top
        if top in ("D4h", "D4", "C4v", "D2d") and subgroup == _FULL_TABLES[top][0]:
            frame = _fourfold_frame(top, coords, axes, match_tol)
            if frame is None:
                table_group = None
            else:
                axes = frame
                coords = (xyz - orig) @ axes.T
        vecs = disp @ axes.T
        full_table = _FULL_TABLES.get(table_group) if table_group else None
        if full_table is not None and full_table[0] == subgroup:
            classes = [(w, np.asarray(m, float)) for w, m in full_table[1]]
            irreps = [(name, chars) for name, chars in full_table[2]]
        else:
            full_table = None
            op_mats = geom.symm_ops(subgroup)
            classes = []
            for op in param.OPERATOR_TABLE[subgroup]:
                mat = op_mats[op]  # inversion is stored as the scalar -1
                r = np.eye(3) * mat if np.ndim(mat) == 0 else np.asarray(mat, float)
                classes.append((1, r))
            irreps = [
                (_primes(row[0]), row[1:]) for row in param.CHARACTER_TABLE[subgroup]
            ]
        h = sum(w for w, _ in classes)

        # Atom permutation of each operation.
        ops = []
        for weight, r in classes:
            moved = coords @ r.T
            perm = []
            for i, sym in enumerate(atoms):
                dist = np.linalg.norm(coords - moved[i], axis=1)
                dist[[j for j, s in enumerate(atoms) if s != sym]] = np.inf
                j = int(np.argmin(dist))
                if dist[j] > match_tol:
                    return None
                perm.append(j)
            if sorted(perm) != list(range(len(atoms))):
                return None
            ops.append((weight, r, perm))

        def counts_of(idx: List[int]) -> Optional[dict]:
            """Irreps spanned by modes *idx*, or None if not symmetry-pure."""
            d = vecs[idx].reshape(len(idx), -1)
            s = d @ d.T
            chars = []
            for _, r, perm in ops:
                moved = np.zeros_like(vecs[idx])
                moved[:, perm, :] = vecs[idx] @ r.T
                m = d @ moved.reshape(len(idx), -1).T
                # Trace of the representation; basis-independent, so the
                # arbitrary mixing inside a degenerate set does not matter.
                chars.append(float(np.trace(np.linalg.solve(s, m))))
            out = {}
            for name, row in irreps:
                n = sum(w * c * x for (w, _, _), c, x in zip(ops, chars, row)) / h
                k = round(n)
                if abs(n - k) > 1.0 - min_purity or k < 0:
                    return None
                if k:
                    out[name] = k
            dims = {name: row[0] for name, row in irreps}
            total = sum(k * dims[name] for name, k in out.items())
            return out if total == len(idx) else None

        valid = sorted(
            (i for i, f in enumerate(freqs) if abs(f) >= min_frequency_cm1),
            key=lambda i: freqs[i],
        )
        sets: List[List[int]] = []
        for i in valid:
            if sets and freqs[i] - freqs[sets[-1][0]] <= degeneracy_cm1:
                sets[-1].append(i)
            else:
                sets.append([i])

        # Labels straight from the table (abelian group, or a full table);
        # recovered from subgroup labels plus degeneracy; or, failing both,
        # the subgroup's own.
        direct = top == subgroup or full_table is not None
        rule = None if direct or not recoverable else _RECOVERABLE.get((top, subgroup))

        def names_of(c: dict) -> Optional[List[str]]:
            if direct:
                return [n for n, k in c.items() for _ in range(k)]
            return _name_set(c, subgroup, top) if rule is not None else None

        resolved: List[tuple] = []  # (mode indices, irrep counts)
        for idx in sets:
            whole = counts_of(idx)
            if len(idx) > 1:
                # A near-coincidence of modes that symmetry does not make
                # degenerate is labelled mode by mode: e.g. ethane's A1g and
                # A2u C-H stretches, 0.3 cm⁻¹ apart. (A single mode of a
                # degenerate irrep is never pure on its own.)
                singles = [counts_of([i]) for i in idx]
                names = names_of(whole) if whole else None
                no_e = direct or (
                    rule is not None
                    and names is not None
                    and all(n in rule[1].values() for n in names)
                )
                if no_e and all(c is not None for c in singles):
                    resolved += [([i], c) for i, c in zip(idx, singles)]
                    continue
            if whole is None:
                logger.debug("Mode irreps skipped: impure set %s", idx)
                return None
            # A lone mode that coincides with a degenerate set (benzene's
            # B2u 2 cm⁻¹ from an E2g pair) is peeled off when the split
            # names the same irreps as the whole set.
            target = names_of(whole)
            peeled = True
            while peeled and len(idx) > 1 and target is not None:
                peeled = False
                for i in idx:
                    rest = [j for j in idx if j != i]
                    ci, cr = counts_of([i]), counts_of(rest)
                    ni = names_of(ci) if ci is not None else None
                    nr = names_of(cr) if cr is not None else None
                    if ni is None or nr is None or len(ni) != 1:
                        continue
                    if sorted(ni + nr) == sorted(target):
                        resolved.append(([i], ci))
                        idx, whole, target, peeled = rest, cr, nr, True
                        break
            resolved.append((idx, whole))

        names_per = [names_of(c) for _, c in resolved]
        named = all(n is not None for n in names_per)
        labels: List[Optional[str]] = [None] * len(freqs)
        terms: List[str] = []
        for (unit, c), names in zip(resolved, names_per):
            if named and names is not None:
                term = "+".join(names)
                terms += names
            else:
                term = "+".join(n for n, k in c.items() for _ in range(k))
                terms.append(term)
            for i in unit:
                labels[i] = _mode_label(term)
        if named and rule is not None:
            group, order = top, rule[2]
        elif named and full_table is not None:
            group, order = top, [name for name, _ in irreps]
        else:
            group, order = subgroup, [name for name, _ in irreps]
        gamma = _gamma(terms, order)
        return ModeIrreps(labels=labels, group=group, top_group=top, gamma=gamma)
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 — symmetry is informational only
        logger.debug("Mode irrep labelling failed: %s", exc)
        return None
