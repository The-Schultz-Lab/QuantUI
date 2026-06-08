"""Unified structure-resolver chain with offline fallback (M-STRUCT STRUCT.4).

Resolution order (first hit wins):

1. **Local RDKit** for SMILES / InChI input — offline, no network.
2. **Bundled library** exact hit (formula key) — offline, instant.
3. **PubChem** (hardened client, STRUCT.2).
4. **NCI CACTUS** resolver (STRUCT.3).
5. **Bundled-library fuzzy fallback** (name/description substring) — the
   last-resort offline answer so the search box is never a dead end, even with
   no network.

Every resolver returns a normalized :class:`ResolvedStructure` so callers
(and the future disambiguation UI, STRUCT.5) treat all sources uniformly. The
``source`` field records which resolver answered, so the UI can be honest about
provenance.

The bundled-library steps currently search ``config.MOLECULE_LIBRARY`` (the 20
presets). When STRUCT.6/.7/.8 move the library into an indexed package-data
store, only :func:`_library_exact` / :func:`_library_fuzzy` change — the chain
is unaffected.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import cactus, config
from .pubchem import (
    MoleculeNotFoundError,
    PubChemAPIError,
    classify_query,
    fetch_structure,
    search_pubchem_candidates,
)

logger = logging.getLogger(__name__)

# Elements that count as "heavy" exclusions when tallying heavy atoms.
_HYDROGEN = "H"


@dataclass
class ResolvedStructure:
    """A structure resolved by the provider chain, normalized across sources."""

    xyz: str
    source: str  # "rdkit-smiles" | "rdkit-inchi" | "library" | "pubchem" | "cactus" | "library-offline-fallback"
    formula: str = "?"
    num_atoms: int = 0
    num_heavy_atoms: int = 0
    charge: int = 0
    multiplicity: int = 1
    molecular_weight: Optional[float] = None
    conformer_origin: str = ""
    identifiers: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_offline(self) -> bool:
        return self.source in (
            "library",
            "library-offline-fallback",
        ) or self.source.startswith("rdkit-")


def _from_metadata(xyz: str, meta: Dict[str, Any], *, source: str) -> ResolvedStructure:
    """Build a ResolvedStructure from a ``(xyz, metadata)`` resolver result."""
    identifiers = {
        k: meta[k]
        for k in ("pubchem_cid", "smiles", "canonical_smiles", "inchi", "query_type")
        if k in meta
    }
    return ResolvedStructure(
        xyz=xyz,
        source=source,
        formula=meta.get("formula", "?"),
        num_atoms=int(meta.get("num_atoms", 0) or 0),
        num_heavy_atoms=int(meta.get("num_heavy_atoms", 0) or 0),
        charge=int(meta.get("charge", 0) or 0),
        molecular_weight=meta.get("molecular_weight"),
        conformer_origin=meta.get("conformer_origin", meta.get("source", "")),
        identifiers=identifiers,
    )


def _entry_to_xyz(name: str, entry: Dict[str, Any]) -> str:
    """Format a ``MOLECULE_LIBRARY`` entry as an XYZ string."""
    atoms = entry["atoms"]
    coords = entry["coordinates"]
    desc = entry.get("description", "")
    lines = [str(len(atoms)), f"{name}: {desc}".strip()]
    for sym, (x, y, z) in zip(atoms, coords):
        lines.append(f"{sym:3s} {float(x):12.6f} {float(y):12.6f} {float(z):12.6f}")
    return "\n".join(lines)


def _entry_to_resolved(
    name: str, entry: Dict[str, Any], *, source: str
) -> ResolvedStructure:
    atoms = entry["atoms"]
    return ResolvedStructure(
        xyz=_entry_to_xyz(name, entry),
        source=source,
        formula=name,
        num_atoms=len(atoms),
        num_heavy_atoms=sum(1 for a in atoms if a != _HYDROGEN),
        charge=int(entry.get("charge", 0)),
        multiplicity=int(entry.get("multiplicity", 1)),
        conformer_origin="library",
        identifiers={"library_key": name},
    )


def _library_exact(query: str) -> Optional[ResolvedStructure]:
    """Exact (case-insensitive) match on a bundled-library formula key."""
    q = query.strip().lower()
    for key, entry in config.MOLECULE_LIBRARY.items():
        if q == key.lower():
            logger.info(f"Resolved '{query}' from bundled library (exact key '{key}')")
            return _entry_to_resolved(key, entry, source="library")
    return None


def _library_fuzzy(query: str) -> Optional[ResolvedStructure]:
    """Loose offline fallback: substring match over keys + descriptions.

    Used only when the network resolvers are unreachable or all miss, so a
    looser match is acceptable (and clearly labelled as a fallback to the user).
    """
    q = query.strip().lower()
    if not q:
        return None
    for key, entry in config.MOLECULE_LIBRARY.items():
        haystack = f"{key} {entry.get('description', '')}".lower()
        if q == key.lower() or q in haystack:
            logger.info(f"Offline fallback matched '{query}' to library entry '{key}'")
            return _entry_to_resolved(key, entry, source="library-offline-fallback")
    return None


def resolve_structure(
    query: str,
    *,
    conformer_3d: bool = True,
    allow_network: bool = True,
) -> ResolvedStructure:
    """Resolve ``query`` through the provider chain. Raises on total failure.

    Raises:
        MoleculeNotFoundError: nothing in the chain could resolve the query.
        ValueError: empty query, or RDKit failed to parse a SMILES/InChI.
    """
    qtype = classify_query(query)
    logger.info(f"Resolving '{query}' (type={qtype}, network={allow_network})")

    # 1. Local RDKit for SMILES / InChI — no network, no library needed.
    if qtype in ("smiles", "inchi"):
        xyz, meta = fetch_structure(query, conformer_3d=conformer_3d)
        return _from_metadata(xyz, meta, source=meta["source"])

    # 2. Exact bundled-library hit (offline, instant) — e.g. "H2O", "C6H6".
    hit = _library_exact(query)
    if hit is not None:
        return hit

    # 3/4. Network resolvers: PubChem, then CACTUS. A genuine miss
    # (MoleculeNotFoundError) or a transport error (PubChemAPIError) both fall
    # through to the next resolver, then to the offline fallback.
    errors: List[Tuple[str, Exception]] = []
    if allow_network:
        try:
            xyz, meta = fetch_structure(query, conformer_3d=conformer_3d)
            return _from_metadata(xyz, meta, source="pubchem")
        except (MoleculeNotFoundError, PubChemAPIError) as exc:
            logger.info(f"PubChem did not resolve '{query}': {exc}")
            errors.append(("pubchem", exc))

        try:
            xyz, meta = cactus.fetch_from_cactus(query, conformer_3d=conformer_3d)
            return _from_metadata(xyz, meta, source="cactus")
        except (MoleculeNotFoundError, PubChemAPIError) as exc:
            logger.info(f"CACTUS did not resolve '{query}': {exc}")
            errors.append(("cactus", exc))

    # 5. Offline fuzzy fallback against the bundled library.
    hit = _library_fuzzy(query)
    if hit is not None:
        return hit

    tried = ", ".join(name for name, _ in errors) or "offline only"
    raise MoleculeNotFoundError(
        f"Could not resolve '{query}' (tried: {tried}, bundled library)"
    )


def search_candidates(query: str) -> List[Dict[str, Any]]:
    """Return disambiguation candidates for a multi-match name/formula query.

    Only name/formula queries can be ambiguous (SMILES/InChI/CID resolve to a
    single structure, and the local library is exact). Returns ``[]`` for those
    types, and ``[]`` on any network failure so the caller falls back to the
    full single-result chain (PubChem → CACTUS → offline library).
    """
    if classify_query(query) not in ("name", "formula"):
        return []
    try:
        return search_pubchem_candidates(query)
    except (MoleculeNotFoundError, PubChemAPIError):
        return []


def student_friendly_resolve(query: str) -> Tuple[Optional[str], str]:
    """Chain-backed, student-friendly resolve. Drop-in for the UI handler.

    Returns ``(xyz_string_or_None, message)``.
    """
    try:
        result = resolve_structure(query, conformer_3d=True)
    except (MoleculeNotFoundError, ValueError) as exc:
        return None, (
            f"❌ Could not resolve '{query}'.\n"
            f"   {exc}\n"
            f"   Try a different name, a SMILES (e.g. CC(=O)O), a CAS number, "
            f"or check spelling.\n"
            f"   Search manually at: https://pubchem.ncbi.nlm.nih.gov/"
        )
    except Exception as exc:  # pragma: no cover - unexpected
        logger.error(f"Unexpected error resolving '{query}': {exc}", exc_info=True)
        return None, f"❌ Error resolving '{query}': {exc}"

    source_label = {
        "rdkit-smiles": "generated locally from SMILES",
        "rdkit-inchi": "generated locally from InChI",
        "library": "the bundled library (offline)",
        "library-offline-fallback": "the bundled library (offline fallback — "
        "network resolvers were unavailable)",
        "pubchem": "PubChem",
        "cactus": "NCI CACTUS",
    }.get(result.source, result.source)

    embedded = (
        " (2D structure embedded by RDKit)"
        if result.conformer_origin == "rdkit-embedded"
        else ""
    )
    mw = f"{result.molecular_weight:.2f} g/mol" if result.molecular_weight else "—"
    message = (
        f"✓ Resolved '{query}' via {source_label}.\n"
        f"  Formula: {result.formula}\n"
        f"  Atoms: {result.num_atoms} ({result.num_heavy_atoms} heavy)\n"
        f"  Molecular weight: {mw}{embedded}"
    )
    return result.xyz, message
