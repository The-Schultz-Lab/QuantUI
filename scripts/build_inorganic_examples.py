#!/usr/bin/env python3
"""Generate bundled inorganic / coordination-complex examples — M-METAL MET.9.

Metal complexes cannot ride the SMILES→embed path the organic library uses: it
scatters the metal (that is the M-METAL bug). So the bundled inorganic examples
carry **explicit, idealized coordinates** built here from standard coordination
geometry and literature bond lengths, with the correct charge and multiplicity.

These are **starting geometries**, not reference structures — they are
connected, clash-free, and roughly metric so the DFT geometry optimization has a
sane place to begin. (A geometry-optimization validation pass is a local
follow-up, per the M-METAL cloud/local split.)

Run to (re)write ``quantui/data/manifests/inorganic.json``; the library store is
then rebuilt from all manifests by ``molecule_library.build_from_manifests``.

    python scripts/build_inorganic_examples.py            # write manifest
    python scripts/build_inorganic_examples.py --rebuild  # + rebuild the store
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

Atoms = List[str]
Coords = List[List[float]]

_MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "quantui"
    / "data"
    / "manifests"
    / "inorganic.json"
)


def _orthonormal_frame(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to ``axis`` (and to each other)."""
    axis = axis / np.linalg.norm(axis)
    seed = (
        np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    v = seed - axis * np.dot(seed, axis)
    v /= np.linalg.norm(v)
    w = np.cross(axis, v)
    return v, w


def _ammine_hydrogens(
    n_pos: np.ndarray, metal_pos: np.ndarray, nh: float = 1.02
) -> Coords:
    """Three H of an M–NH3, lone pair toward the metal, H splayed outward.

    Each N–H makes the tetrahedral 70.5° with the outward M→N axis (i.e. 109.5°
    with the metal-facing lone pair), tripod-arranged at 120°.
    """
    u = n_pos - metal_pos
    u = u / np.linalg.norm(u)
    v, w = _orthonormal_frame(u)
    theta = math.radians(70.5)
    out = []
    for k in range(3):
        phi = math.radians(120.0 * k)
        direction = math.cos(theta) * u + math.sin(theta) * (
            math.cos(phi) * v + math.sin(phi) * w
        )
        out.append((n_pos + nh * direction).tolist())
    return out


def cisplatin() -> Tuple[Atoms, Coords, int, int]:
    """cis-[PtCl2(NH3)2] — square planar Pt(II) d8, singlet, neutral."""
    atoms: Atoms = ["Pt"]
    coords: Coords = [[0.0, 0.0, 0.0]]
    metal = np.zeros(3)
    d_ptcl, d_ptn = 2.33, 2.05
    # cis: the two Cl adjacent (90°), the two N adjacent (90°).
    for ang in (45.0, 135.0):
        a = math.radians(ang)
        atoms.append("Cl")
        coords.append([d_ptcl * math.cos(a), d_ptcl * math.sin(a), 0.0])
    for ang in (225.0, 315.0):
        a = math.radians(ang)
        n_pos = np.array([d_ptn * math.cos(a), d_ptn * math.sin(a), 0.0])
        atoms.append("N")
        coords.append(n_pos.tolist())
        for h in _ammine_hydrogens(n_pos, metal):
            atoms.append("H")
            coords.append(h)
    return atoms, coords, 0, 1


def hexaamminecobalt() -> Tuple[Atoms, Coords, int, int]:
    """[Co(NH3)6]3+ — octahedral Co(III) d6 low-spin, singlet, +3."""
    atoms: Atoms = ["Co"]
    coords: Coords = [[0.0, 0.0, 0.0]]
    metal = np.zeros(3)
    d = 1.97
    axes = [
        (d, 0, 0),
        (-d, 0, 0),
        (0, d, 0),
        (0, -d, 0),
        (0, 0, d),
        (0, 0, -d),
    ]
    for ax in axes:
        n_pos = np.array(ax, dtype=float)
        atoms.append("N")
        coords.append(n_pos.tolist())
        for h in _ammine_hydrogens(n_pos, metal):
            atoms.append("H")
            coords.append(h)
    return atoms, coords, 3, 1


def ferrocene() -> Tuple[Atoms, Coords, int, int]:
    """Fe(C5H5)2 — sandwich Fe(II) d6, singlet, neutral (eclipsed start)."""
    atoms: Atoms = ["Fe"]
    coords: Coords = [[0.0, 0.0, 0.0]]
    r_c = 1.21  # ring carbon radius from the C5 axis
    ch = 1.08
    z = 1.66  # Fe → ring-plane distance
    for sign in (1.0, -1.0):
        for k in range(5):
            a = math.radians(72.0 * k)
            cx, cy = r_c * math.cos(a), r_c * math.sin(a)
            atoms.append("C")
            coords.append([cx, cy, sign * z])
            # H radially outward in the ring plane.
            hx, hy = (r_c + ch) * math.cos(a), (r_c + ch) * math.sin(a)
            atoms.append("H")
            coords.append([hx, hy, sign * z])
    return atoms, coords, 0, 1


_BUILDERS = {
    "cisplatin": (
        cisplatin,
        "cis-diamminedichloroplatinum(II) — square-planar Pt(II) chemotherapy " "drug",
        "cisplatin;cis-platin;CDDP;PtCl2(NH3)2",
    ),
    "hexaamminecobalt(III)": (
        hexaamminecobalt,
        "[Co(NH3)6]3+ — classic octahedral Werner complex (low-spin d6)",
        "hexaamminecobalt;cobalt hexammine;Co(NH3)6",
    ),
    "ferrocene": (
        ferrocene,
        "Fe(C5H5)2 — the archetypal metallocene sandwich compound",
        "ferrocene;bis(cyclopentadienyl)iron;Cp2Fe",
    ),
}

# A minimal covalent-radius table (Å) for the sanity connectivity check.
_COV = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "Cl": 1.02,
    "Fe": 1.32,
    "Co": 1.26,
    "Pt": 1.36,
    "Zn": 1.22,
}


def _sanity(atoms: Atoms, coords: Coords) -> List[str]:
    """Return a list of problems (empty = geometry looks sane)."""
    problems: List[str] = []
    pts = np.array(coords)
    n = len(atoms)
    # No atomic clashes.
    for i in range(n):
        for j in range(i + 1, n):
            dij = float(np.linalg.norm(pts[i] - pts[j]))
            if dij < 0.7:
                problems.append(f"clash: {atoms[i]}{i}-{atoms[j]}{j} = {dij:.2f} Å")
    # Metal is bonded to the expected number of donors (within 1.3× radii sum).
    metal_syms = {"Pt", "Co", "Fe", "Zn"}
    for i, sym in enumerate(atoms):
        if sym not in metal_syms:
            continue
        neigh = 0
        for j in range(n):
            if j == i:
                continue
            cutoff = 1.3 * (_COV.get(sym, 1.3) + _COV.get(atoms[j], 0.7))
            if float(np.linalg.norm(pts[i] - pts[j])) <= cutoff:
                neigh += 1
        if neigh == 0:
            problems.append(f"metal {sym}{i} has no neighbours within bonding range")
    return problems


def _formula(atoms: Atoms) -> str:
    from collections import Counter

    c = Counter(atoms)
    order = ["C", "H"] + sorted(k for k in c if k not in ("C", "H"))
    seen = set()
    out = ""
    for el in order:
        if el in c and el not in seen:
            seen.add(el)
            out += el + (str(c[el]) if c[el] > 1 else "")
    return out


def build_manifest() -> list:
    entries = []
    print("# building inorganic examples (idealized starting geometries)\n")
    for name, (fn, desc, syn) in _BUILDERS.items():
        atoms, coords, charge, mult = fn()
        problems = _sanity(atoms, coords)
        status = "OK" if not problems else "PROBLEMS: " + "; ".join(problems)
        print(
            f"  {name:24s} {_formula(atoms):12s} "
            f"charge={charge:+d} mult={mult}  {status}"
        )
        if problems:
            raise SystemExit(f"geometry sanity failed for {name}: {problems}")
        entries.append(
            {
                "id": f"inorganic-{name.replace('(', '').replace(')', '').replace(' ', '-').lower()}",
                "name": name,
                "formula": _formula(atoms),
                "category": "inorganic-complex",
                "charge": charge,
                "multiplicity": mult,
                "source": "quantui-idealized",
                "synonyms": syn,
                "description": desc,
                "atoms": atoms,
                "coordinates": [[round(x, 6) for x in xyz] for xyz in coords],
            }
        )
    return entries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the library store from all manifests after writing.",
    )
    args = ap.parse_args(argv)

    entries = build_manifest()
    _MANIFEST.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"\n# wrote {len(entries)} entries -> {_MANIFEST}")

    if args.rebuild:
        from quantui import molecule_library as ml

        path = ml.build_from_manifests()
        print(f"# rebuilt store -> {path} ({ml.count()} total entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
