"""Distance-based (metal-aware) connectivity for coordination complexes.

RDKit's ``DetermineBonds`` models only organic valence and *raises* on a
transition metal ("Atom N has no valences defined"), so the whole
structure → geometry → viewer stack loses the metal. This module provides a
purely geometric alternative used by the M-METAL work: two atoms are treated as
bonded when their separation is within a tolerance of the sum of their covalent
radii. That covers metal↔donor coordination bonds RDKit can't perceive, without
any valence model at all.

Shared primitive for:

* **MET.2** — detect a fetched structure that resolved to a *disconnected* salt
  (cisplatin's name returns 2 NH₃ + 2 HCl + Pt²⁺, not the square-planar complex)
  and warn instead of silently computing a wrong geometry.
* **MET.1 / MET.6** — coordination-aware connectivity for geometry handling and
  for drawing coordination bonds in the viewer.

Pure logic — no RDKit, no PySCF, no widgets. Never raises on ordinary input.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# Covalent radii in Å (Cordero et al., Dalton Trans. 2008, 2832). A broad subset
# covering the organic elements, common heteroatoms/ions, and the transition /
# heavy metals QuantUI's inorganic examples use. Anything missing falls back to
# ``_FALLBACK_RADIUS`` — generous enough not to sever a real bond by accident.
COVALENT_RADII: Dict[str, float] = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.52,
    "Co": 1.50,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "I": 1.39,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Pb": 1.46,
}

_FALLBACK_RADIUS = 0.75

# A bond is inferred when the interatomic distance is within this factor of the
# summed covalent radii. 1.3 is the usual slack for covalent-radii bond
# perception; it keeps all three bundled coordination complexes (cisplatin,
# hexaamminecobalt(III), ferrocene) as a single connected component while still
# separating a genuine ionic salt into fragments.
DEFAULT_TOLERANCE = 1.3


def _radius(symbol: str) -> float:
    return COVALENT_RADII.get(symbol, _FALLBACK_RADIUS)


def covalent_components(
    atoms: Sequence[str],
    coords: Sequence[Sequence[float]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> List[List[int]]:
    """Group atom indices into connected components by covalent-radii distance.

    Two atoms *i*, *j* are bonded when ``dist(i, j) <= tolerance * (r_i + r_j)``.
    Returns a list of components (each a sorted list of atom indices), ordered
    largest first then by first index — deterministic for a given input.
    """
    n = len(atoms)
    if n == 0:
        return []
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        ri = _radius(atoms[i])
        xi, yi, zi = coords[i][0], coords[i][1], coords[i][2]
        for j in range(i + 1, n):
            threshold = tolerance * (ri + _radius(atoms[j]))
            dx = xi - coords[j][0]
            dy = yi - coords[j][1]
            dz = zi - coords[j][2]
            if dx * dx + dy * dy + dz * dz <= threshold * threshold:
                adj[i].append(j)
                adj[j].append(i)

    seen = [False] * n
    components: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        comp: List[int] = []
        while stack:
            u = stack.pop()
            if seen[u]:
                continue
            seen[u] = True
            comp.append(u)
            stack.extend(adj[u])
        components.append(sorted(comp))
    components.sort(key=lambda c: (-len(c), c[0]))
    return components


def _hill_formula(symbols: Sequence[str]) -> str:
    """Formula in Hill order (C, H, then alphabetical); '2' subscripts inline."""
    counts: Dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1

    def fmt(sym: str) -> str:
        c = counts[sym]
        return sym if c == 1 else f"{sym}{c}"

    ordered: List[str] = []
    for special in ("C", "H"):
        if special in counts:
            ordered.append(fmt(special))
    for sym in sorted(k for k in counts if k not in ("C", "H")):
        ordered.append(fmt(sym))
    return "".join(ordered)


def is_disconnected(
    atoms: Sequence[str],
    coords: Sequence[Sequence[float]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """True when the geometry splits into more than one covalent component."""
    return len(covalent_components(atoms, coords, tolerance)) > 1


def describe_disconnection(
    atoms: Sequence[str],
    coords: Sequence[Sequence[float]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> Optional[str]:
    """A teaching-toned warning if the structure is disconnected, else ``None``.

    Names the fragments by formula (grouping identical ones with an ``N×``
    multiplier), so a student sees *why* the loaded structure is suspect — the
    cisplatin-salt case (MET.2), where a name resolves to separate ions rather
    than the coordinated complex.
    """
    components = covalent_components(atoms, coords, tolerance)
    if len(components) <= 1:
        return None

    formula_counts: Dict[str, int] = {}
    order: List[str] = []
    for comp in components:
        f = _hill_formula([atoms[i] for i in comp])
        if f not in formula_counts:
            order.append(f)
        formula_counts[f] = formula_counts.get(f, 0) + 1
    parts = [
        (f"{formula_counts[f]}×{f}" if formula_counts[f] > 1 else f) for f in order
    ]
    fragments = " + ".join(parts)

    return (
        f"This structure is disconnected — it resolved to {len(components)} "
        f"separate fragments ({fragments}), not one bonded molecule. For a metal "
        "complex this usually means the name returned an ionic salt form rather "
        "than the coordinated complex, so the geometry shown is not the real "
        "molecule. Start from a known-good geometry instead — paste one in the "
        "XYZ Input tab, or load a bundled inorganic example."
    )
