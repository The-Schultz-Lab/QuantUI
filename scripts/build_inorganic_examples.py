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


def _aqua_hydrogens(
    o_pos: np.ndarray, metal_pos: np.ndarray, oh: float = 0.96, hoh_deg: float = 104.5
) -> Coords:
    """Two H of an M–OH2: the O lone pair faces the metal, H splayed outward."""
    u = o_pos - metal_pos
    u = u / np.linalg.norm(u)
    v, _w = _orthonormal_frame(u)
    half = math.radians(hoh_deg / 2.0)
    out = []
    for s in (1.0, -1.0):
        direction = math.cos(half) * u + s * math.sin(half) * v
        out.append((o_pos + oh * direction).tolist())
    return out


def _octahedral_dirs() -> List[np.ndarray]:
    return [
        np.array(a, float)
        for a in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    ]


def _tetrahedral_dirs() -> List[np.ndarray]:
    raw = ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    return [np.array(a, float) / math.sqrt(3.0) for a in raw]


def _square_planar_dirs() -> List[np.ndarray]:
    return [np.array(a, float) for a in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))]


def _aqua_ligand(d_mo: float):
    """Place an aqua (H2O) donor along a unit direction from the metal."""

    def place(metal: np.ndarray, u: np.ndarray) -> Tuple[Atoms, Coords]:
        o = metal + u * d_mo
        atoms: Atoms = ["O"]
        coords: Coords = [o.tolist()]
        for h in _aqua_hydrogens(o, metal):
            atoms.append("H")
            coords.append(h)
        return atoms, coords

    return place


def _ammine_ligand(d_mn: float):
    """Place an ammine (NH3) donor along a unit direction from the metal."""

    def place(metal: np.ndarray, u: np.ndarray) -> Tuple[Atoms, Coords]:
        n = metal + u * d_mn
        atoms: Atoms = ["N"]
        coords: Coords = [n.tolist()]
        for h in _ammine_hydrogens(n, metal):
            atoms.append("H")
            coords.append(h)
        return atoms, coords

    return place


def _linear_ligand(near: str, far: str, d_near: float, d_far: float):
    """Place a linear diatomic donor (M–near≡far), e.g. cyanide C≡N or CO."""

    def place(metal: np.ndarray, u: np.ndarray) -> Tuple[Atoms, Coords]:
        p_near = metal + u * d_near
        p_far = p_near + u * d_far
        return [near, far], [p_near.tolist(), p_far.tolist()]

    return place


def _mono_ligand(elem: str, d: float):
    """Place a single-atom donor (M–X), e.g. chloro or oxo."""

    def place(metal: np.ndarray, u: np.ndarray) -> Tuple[Atoms, Coords]:
        return [elem], [(metal + u * d).tolist()]

    return place


def _homoleptic(
    metal_sym: str, dirs: List[np.ndarray], place, charge: int, mult: int
) -> Tuple[Atoms, Coords, int, int]:
    """Assemble a homoleptic complex: one ligand type on every coordination site."""
    atoms: Atoms = [metal_sym]
    coords: Coords = [[0.0, 0.0, 0.0]]
    metal = np.zeros(3)
    for u in dirs:
        u = u / np.linalg.norm(u)
        a, c = place(metal, u)
        atoms.extend(a)
        coords.extend(c)
    return atoms, coords, charge, mult


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
    # ── Octahedral aqua complexes (weak-field H2O → high-spin) ──────────────
    "hexaaquairon(II)": (
        lambda: _homoleptic("Fe", _octahedral_dirs(), _aqua_ligand(2.12), 2, 5),
        "[Fe(H2O)6]2+ — high-spin octahedral aqua complex (d6, 4 unpaired)",
        "hexaaquairon;iron(II) hexaaqua;Fe(H2O)6 2+",
    ),
    "hexaaquachromium(III)": (
        lambda: _homoleptic("Cr", _octahedral_dirs(), _aqua_ligand(1.96), 3, 4),
        "[Cr(H2O)6]3+ — octahedral aqua complex (d3, 3 unpaired)",
        "hexaaquachromium;chromium(III) hexaaqua;Cr(H2O)6 3+",
    ),
    "hexaaquanickel(II)": (
        lambda: _homoleptic("Ni", _octahedral_dirs(), _aqua_ligand(2.05), 2, 3),
        "[Ni(H2O)6]2+ — octahedral aqua complex (d8, 2 unpaired)",
        "hexaaquanickel;nickel(II) hexaaqua;Ni(H2O)6 2+",
    ),
    "hexaaquatitanium(III)": (
        lambda: _homoleptic("Ti", _octahedral_dirs(), _aqua_ligand(2.03), 3, 2),
        "[Ti(H2O)6]3+ — d1 octahedral aqua complex (the classic single-band "
        "UV-Vis example)",
        "hexaaquatitanium;titanium(III) hexaaqua;Ti(H2O)6 3+",
    ),
    # ── Octahedral cyanides (strong-field CN- → low-spin) ───────────────────
    "hexacyanoferrate(III)": (
        lambda: _homoleptic(
            "Fe", _octahedral_dirs(), _linear_ligand("C", "N", 1.93, 1.16), -3, 2
        ),
        "[Fe(CN)6]3- — ferricyanide, low-spin octahedral (d5, 1 unpaired)",
        "ferricyanide;hexacyanoferrate(III);Fe(CN)6 3-",
    ),
    "hexacyanoferrate(II)": (
        lambda: _homoleptic(
            "Fe", _octahedral_dirs(), _linear_ligand("C", "N", 1.92, 1.16), -4, 1
        ),
        "[Fe(CN)6]4- — ferrocyanide, low-spin octahedral (d6, diamagnetic)",
        "ferrocyanide;hexacyanoferrate(II);Fe(CN)6 4-",
    ),
    # ── Tetrahedral complexes ───────────────────────────────────────────────
    "tetracarbonylnickel(0)": (
        lambda: _homoleptic(
            "Ni", _tetrahedral_dirs(), _linear_ligand("C", "O", 1.82, 1.14), 0, 1
        ),
        "Ni(CO)4 — tetrahedral d10 carbonyl (18-electron, diamagnetic)",
        "tetracarbonylnickel;nickel tetracarbonyl;Ni(CO)4",
    ),
    "tetraamminezinc(II)": (
        lambda: _homoleptic("Zn", _tetrahedral_dirs(), _ammine_ligand(2.03), 2, 1),
        "[Zn(NH3)4]2+ — tetrahedral d10 ammine (diamagnetic)",
        "tetraamminezinc;zinc(II) tetraammine;Zn(NH3)4 2+",
    ),
    "tetrachlorocobaltate(II)": (
        lambda: _homoleptic("Co", _tetrahedral_dirs(), _mono_ligand("Cl", 2.28), -2, 4),
        "[CoCl4]2- — tetrahedral high-spin cobalt(II) (d7, 3 unpaired), the "
        "classic blue ion",
        "tetrachlorocobaltate;CoCl4 2-",
    ),
    "permanganate": (
        lambda: _homoleptic("Mn", _tetrahedral_dirs(), _mono_ligand("O", 1.63), -1, 1),
        "[MnO4]- — tetrahedral d0 manganese(VII) oxoanion (deep purple)",
        "permanganate;MnO4;tetraoxomanganate",
    ),
    # ── Square-planar (d8) ──────────────────────────────────────────────────
    "tetrachloroplatinate(II)": (
        lambda: _homoleptic(
            "Pt", _square_planar_dirs(), _mono_ligand("Cl", 2.31), -2, 1
        ),
        "[PtCl4]2- — square-planar platinum(II) (d8, diamagnetic), the cisplatin "
        "precursor",
        "tetrachloroplatinate;PtCl4 2-",
    ),
}


def _sanity(atoms: Atoms, coords: Coords) -> List[str]:
    """Return a list of problems (empty = geometry looks sane).

    Uses the shipped, metal-aware connectivity finder so the checks match how the
    app itself perceives bonds: no clashes, one connected component (not a
    scattered salt), and every metal centre actually coordinated.
    """
    from quantui.connectivity import (
        covalent_components,
        is_metal,
        metal_coordination_bonds,
    )

    problems: List[str] = []
    pts = np.array(coords)
    n = len(atoms)
    # No atomic clashes.
    for i in range(n):
        for j in range(i + 1, n):
            dij = float(np.linalg.norm(pts[i] - pts[j]))
            if dij < 0.7:
                problems.append(f"clash: {atoms[i]}{i}-{atoms[j]}{j} = {dij:.2f} Å")
    # One connected component — never a scattered / disconnected structure.
    comps = covalent_components(atoms, coords)
    if len(comps) != 1:
        problems.append(f"{len(comps)} disconnected fragments (expected 1)")
    # Every metal centre is actually coordinated.
    bonded = {i for bond in metal_coordination_bonds(atoms, coords) for i in bond}
    for i, sym in enumerate(atoms):
        if is_metal(sym) and i not in bonded:
            problems.append(f"metal {sym}{i} has no coordination bonds")
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


def _validate_gfnff(entries: list) -> None:
    """Relax each entry with GFN-FF and report — a real quality gate for the
    idealized geometries (needs xtb; skipped if unavailable).

    Confirms each starting geometry is physically sane: the GFN-FF relaxation
    stays a single connected component and doesn't move far (a large RMSD would
    mean the idealized metrics are off). Spin-independent (GFN-FF is a force
    field), so multiplicity doesn't enter here.
    """
    try:
        from quantui.connectivity import covalent_components
        from quantui.molecule import Molecule
        from quantui.preopt import _XTB_AVAILABLE, preoptimize
    except Exception as exc:  # noqa: BLE001
        print(f"\n# GFN-FF validation unavailable ({exc}); skipped")
        return
    if not _XTB_AVAILABLE:
        print("\n# GFN-FF validation skipped (xtb not installed)")
        return

    print("\n# GFN-FF validation (relax idealized geometry; expect small RMSD)\n")
    for e in entries:
        mol = Molecule(
            atoms=e["atoms"],
            coordinates=e["coordinates"],
            charge=e["charge"],
            multiplicity=e["multiplicity"],
        )
        relaxed, rmsd = preoptimize(mol)
        comps = len(covalent_components(relaxed.atoms, relaxed.coordinates))
        flag = "OK" if comps == 1 and rmsd < 0.6 else "REVIEW"
        print(f"  {e['name']:26s} RMSD={rmsd:5.3f} Å  components={comps}  {flag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the library store from all manifests after writing.",
    )
    ap.add_argument(
        "--validate-gfnff",
        action="store_true",
        help="Relax each geometry with GFN-FF (xtb) and report as a quality gate.",
    )
    args = ap.parse_args(argv)

    entries = build_manifest()
    _MANIFEST.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"\n# wrote {len(entries)} entries -> {_MANIFEST}")

    if args.validate_gfnff:
        _validate_gfnff(entries)

    if args.rebuild:
        from quantui import molecule_library as ml

        path = ml.build_from_manifests()
        print(f"# rebuilt store -> {path} ({ml.count()} total entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
