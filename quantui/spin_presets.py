"""Oxidation-state → d-count → spin-state multiplicity suggestions (M-METAL MET.5).

**Suggests, never sets.** A transition metal's spin multiplicity is *not* fixed
by its oxidation state alone: for an octahedral d⁴–d⁷ centre the ligand field
decides **high-spin vs low-spin** (strong-field ligands like CN⁻/CO/en pair the
electrons → low-spin; weak-field ligands like H₂O/halides → high-spin; NH₃ and
other intermediate-field ligands can go either way). This module turns a metal +
oxidation state into the d-electron count and returns *both* physically
reasonable spin states with a plain-language explanation, so the student picks
the one matching their complex rather than being handed a single (possibly
wrong) number.

**Transparency over convenience.** Nothing is pre-selected for the student, and
every assumption or enforcement is surfaced as an explicit ``caveat`` string
(tetrahedral treated as high-spin; an unusual oxidation state; the square-planar
restriction) rather than applied silently.

Scope (per the classroom's metals): first-row transition metals (Sc–Zn) and the
common 4d/5d centres (Ru, Rh, Pd, Pt, …). Geometries: octahedral (default,
high/low-spin), tetrahedral (assumed high-spin), and square-planar — which is
**restricted to d⁸** (the only case with a standard, unambiguous spin state,
e.g. Pt(II) in cisplatin); a non-d⁸ square-planar request is refused with a
clear message rather than given a shaky number. Charge is deliberately *not*
inferred — the overall complex charge depends on the ligand charges, which the
metal centre alone doesn't determine; the student supplies that.

Pure logic — no PySCF, no widgets. Raises ``ValueError`` for a metal /
oxidation state / geometry outside the supported set, so a caller can fall back
cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# Group number (new IUPAC 3–12) = neutral-atom valence (s + d) electron count,
# so the d-electron count of an ion is group − oxidation_state. Covers the
# first-row TMs and the common 4d/5d centres the course uses.
_GROUP: Dict[str, int] = {
    # 3d
    "Sc": 3,
    "Ti": 4,
    "V": 5,
    "Cr": 6,
    "Mn": 7,
    "Fe": 8,
    "Co": 9,
    "Ni": 10,
    "Cu": 11,
    "Zn": 12,
    # 4d
    "Y": 3,
    "Zr": 4,
    "Nb": 5,
    "Mo": 6,
    "Tc": 7,
    "Ru": 8,
    "Rh": 9,
    "Pd": 10,
    "Ag": 11,
    "Cd": 12,
    # 5d
    "Hf": 4,
    "Ta": 5,
    "W": 6,
    "Re": 7,
    "Os": 8,
    "Ir": 9,
    "Pt": 10,
    "Au": 11,
    "Hg": 12,
}

GEOMETRIES = ("octahedral", "tetrahedral", "square_planar")

# Well-established oxidation states for the in-scope metals (teaching level). A
# request outside this set isn't refused — it's computed and *flagged* ("unusual
# … double-check"), per the transparency-over-enforcement principle.
_COMMON_OXIDATION_STATES: Dict[str, frozenset] = {
    "Sc": frozenset({3}),
    "Ti": frozenset({3, 4}),
    "V": frozenset({2, 3, 4, 5}),
    "Cr": frozenset({2, 3, 6}),
    "Mn": frozenset({2, 3, 4, 6, 7}),
    "Fe": frozenset({2, 3}),
    "Co": frozenset({2, 3}),
    "Ni": frozenset({2, 3}),
    "Cu": frozenset({1, 2}),
    "Zn": frozenset({2}),
    "Y": frozenset({3}),
    "Zr": frozenset({4}),
    "Nb": frozenset({5}),
    "Mo": frozenset({4, 6}),
    "Tc": frozenset({4, 7}),
    "Ru": frozenset({2, 3, 4}),
    "Rh": frozenset({3}),
    "Pd": frozenset({2, 4}),
    "Ag": frozenset({1}),
    "Cd": frozenset({2}),
    "Hf": frozenset({4}),
    "Ta": frozenset({5}),
    "W": frozenset({4, 6}),
    "Re": frozenset({4, 7}),
    "Os": frozenset({4, 6, 8}),
    "Ir": frozenset({3, 4}),
    "Pt": frozenset({2, 4}),
    "Au": frozenset({1, 3}),
    "Hg": frozenset({1, 2}),
}


@dataclass(frozen=True)
class SpinState:
    """One candidate spin state for a d^n centre."""

    label: str  # "high-spin", "low-spin", or "" when unambiguous
    n_unpaired: int
    multiplicity: int  # n_unpaired + 1


@dataclass(frozen=True)
class SpinSuggestion:
    """The full suggestion for a metal centre — d-count + candidate spin states."""

    element: str
    oxidation_state: int
    geometry: str
    d_count: int
    states: List[SpinState]
    explanation: str
    caveats: List[str]  # explicit flags the UI must surface (never applied silently)

    @property
    def is_ambiguous(self) -> bool:
        """True when more than one spin state is offered (high- vs low-spin)."""
        return len(self.states) > 1


def supported_metals() -> List[str]:
    """Metals this module can suggest for (sorted by atomic group then symbol)."""
    return sorted(_GROUP, key=lambda el: (_GROUP[el], el))


def d_electron_count(element: str, oxidation_state: int) -> int:
    """d-electron count of ``element`` in the given oxidation state (group − ox)."""
    if element not in _GROUP:
        raise ValueError(f"{element!r} is not a supported transition metal")
    d = _GROUP[element] - oxidation_state
    if d < 0 or d > 10:
        raise ValueError(
            f"{element}({oxidation_state:+d}) gives d{d}, outside the d0–d10 range"
        )
    return d


# Unpaired-electron counts by d-count. Octahedral splits high-spin vs low-spin
# for d4–d7; d0–d3 and d8–d10 are unambiguous. Tetrahedral is effectively always
# high-spin (Δ_t is small — no low-spin tetrahedral complexes in practice).
_OCTAHEDRAL_HS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0}
_OCTAHEDRAL_LS = {4: 2, 5: 1, 6: 0, 7: 1}
_TETRAHEDRAL = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0}


def _states_for_geometry(d: int, geometry: str) -> List[SpinState]:
    if geometry == "octahedral":
        hs = _OCTAHEDRAL_HS[d]
        if d in _OCTAHEDRAL_LS:
            ls = _OCTAHEDRAL_LS[d]
            return [
                SpinState("high-spin", hs, hs + 1),
                SpinState("low-spin", ls, ls + 1),
            ]
        return [SpinState("", hs, hs + 1)]
    if geometry == "tetrahedral":
        u = _TETRAHEDRAL[d]
        return [SpinState("", u, u + 1)]
    if geometry == "square_planar":
        # Square-planar is only standard for d8 (Ni(II)/Pd(II)/Pt(II)):
        # diamagnetic, all electrons paired. Other d-counts have no textbook
        # square-planar spin state, so refuse rather than invent one.
        if d == 8:
            return [SpinState("", 0, 1)]
        raise ValueError(
            f"Square-planar spin states are only standard for d8; "
            f"this centre is d{d}. Choose octahedral or tetrahedral, or set the "
            f"multiplicity manually."
        )
    raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")


def _explain(
    element: str, ox: int, d: int, geometry: str, states: List[SpinState]
) -> str:
    head = (
        f"{element}({ox:+d}) is a d{d} centre "
        f"({element} is group {_GROUP[element]}; d-count = group − oxidation state)."
    )
    if len(states) > 1:
        hs, ls = states[0], states[1]
        body = (
            f" In an {geometry} field this is ambiguous: strong-field ligands "
            f"(e.g. CN⁻, CO, en) give low-spin — {ls.n_unpaired} unpaired, "
            f"multiplicity {ls.multiplicity}; weak-field ligands (e.g. H₂O, "
            f"halides) give high-spin — {hs.n_unpaired} unpaired, multiplicity "
            f"{hs.multiplicity}. Intermediate-field ligands like NH₃ can go "
            f"either way depending on the metal — pick the state matching your "
            f"complex."
        )
    elif geometry == "square_planar" and d == 8:
        body = (
            " A square-planar d8 centre (e.g. Pt(II), Pd(II)) is diamagnetic — "
            "all electrons paired, multiplicity 1."
        )
    else:
        s = states[0]
        body = (
            f" This d-count has a single spin state in an {geometry} field: "
            f"{s.n_unpaired} unpaired, multiplicity {s.multiplicity}."
        )
    tail = (
        " This sets the multiplicity only — the overall charge depends on your "
        "ligands, so set that from the complex."
    )
    return head + body + tail


def _caveats_for(
    element: str, oxidation_state: int, d: int, geometry: str
) -> List[str]:
    """Explicit flags to surface — assumptions/enforcements, never applied silently."""
    caveats: List[str] = []
    if geometry == "tetrahedral":
        caveats.append(
            "Tetrahedral fields are assumed high-spin (low-spin tetrahedral "
            "complexes are essentially unknown)."
        )
    if geometry == "octahedral" and d == 8:
        caveats.append(
            "A d8 centre is often square-planar instead (diamagnetic, "
            "multiplicity 1) — confirm the geometry."
        )
    common = _COMMON_OXIDATION_STATES.get(element)
    if common is not None and oxidation_state not in common:
        common_str = ", ".join(f"{o:+d}" for o in sorted(common))
        caveats.append(
            f"{oxidation_state:+d} is an unusual oxidation state for {element} "
            f"(common: {common_str}) — double-check it."
        )
    return caveats


def suggest_spin_states(
    element: str, oxidation_state: int, geometry: str = "octahedral"
) -> SpinSuggestion:
    """Suggest candidate spin multiplicities for a metal centre.

    Returns a :class:`SpinSuggestion` with the d-count, one or two
    :class:`SpinState` candidates (two when an octahedral field leaves
    high-/low-spin ambiguous), and a list of ``caveats`` naming every assumption
    or enforcement in play. Raises ``ValueError`` for an unsupported metal, an
    out-of-range d-count, an unknown geometry, or a non-d⁸ square-planar request.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
    d = d_electron_count(element, oxidation_state)
    states = _states_for_geometry(d, geometry)
    return SpinSuggestion(
        element=element,
        oxidation_state=oxidation_state,
        geometry=geometry,
        d_count=d,
        states=states,
        explanation=_explain(element, oxidation_state, d, geometry, states),
        caveats=_caveats_for(element, oxidation_state, d, geometry),
    )
