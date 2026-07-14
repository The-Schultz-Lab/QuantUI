"""
PubChem Integration Module

Provides functions to search and retrieve molecular structures from PubChem
for educational use in quantum chemistry calculations.
"""

import logging
import re
import threading
import time
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

import requests

from . import config

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


logger = logging.getLogger(__name__)

# PubChem API endpoints
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
# Back-compat alias; canonical value lives in config (constraint #5).
PUBCHEM_TIMEOUT = config.PUBCHEM_TIMEOUT_S

# ── HTTP client: client-side throttle + bounded 503 back-off ─────────────────
# A single process-wide limiter keeps us under PUG-REST's ~5 req/s ceiling even
# when several search threads fire at once.
_request_lock = threading.Lock()
_last_request_time = 0.0


def _throttle() -> None:
    """Block just long enough to honor the client-side minimum request gap."""
    global _last_request_time
    with _request_lock:
        wait = config.PUBCHEM_MIN_REQUEST_INTERVAL_S - (
            time.monotonic() - _last_request_time
        )
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()


def _http_get(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> requests.Response:
    """GET with client-side throttle + exponential back-off on 503 throttling.

    Retries only on HTTP 503 (PUG-REST's throttle signal). All other status
    codes are returned to the caller unchanged; network exceptions
    (``Timeout`` / ``ConnectionError`` / ...) propagate so callers can map them
    to :class:`PubChemAPIError` exactly as before.
    """
    timeout = timeout if timeout is not None else config.PUBCHEM_TIMEOUT_S
    # M10 audit fix (2026-07-14): if config.PUBCHEM_MAX_RETRIES were ever 0
    # (or negative), `range(config.PUBCHEM_MAX_RETRIES)` would iterate zero
    # times, leaving `response` at its None initializer and returning None
    # from a function typed to return requests.Response — every caller
    # then hits AttributeError on response.status_code. Always attempt at
    # least once regardless of the configured retry count.
    max_attempts = max(1, config.PUBCHEM_MAX_RETRIES)
    response = None
    for attempt in range(max_attempts):
        _throttle()
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code != 503:
            return response
        # Throttled — back off (capped) and retry, unless this was the last try.
        if attempt < max_attempts - 1:
            backoff = min(
                config.PUBCHEM_BACKOFF_BASE_S * (2**attempt),
                config.PUBCHEM_BACKOFF_MAX_S,
            )
            logger.warning(
                "PubChem throttled (503); retrying in %.1fs (attempt %d/%d)",
                backoff,
                attempt + 1,
                max_attempts,
            )
            time.sleep(backoff)
    # Exhausted retries — hand the last 503 back; caller raises via raise_for_status.
    return response  # type: ignore[return-value]


class PubChemError(Exception):
    """Base exception for PubChem-related errors."""

    pass


class MoleculeNotFoundError(PubChemError):
    """Raised when a molecule cannot be found in PubChem."""

    pass


class PubChemAPIError(PubChemError):
    """Raised when PubChem API request fails."""

    pass


def search_molecule_by_name(name: str) -> int:
    """
    Search for a molecule in PubChem by name and return its CID.

    Args:
        name: Common name or IUPAC name of the molecule

    Returns:
        int: PubChem Compound ID (CID)

    Raises:
        PubChemAPIError: If API request fails
        MoleculeNotFoundError: If molecule not found
    """
    url = f"{PUBCHEM_BASE_URL}/compound/name/{quote(name, safe='')}/cids/JSON"

    try:
        logger.debug(f"Searching PubChem for: {name}")
        response = _http_get(url)

        if response.status_code == 404:
            raise MoleculeNotFoundError(f"Molecule '{name}' not found in PubChem")

        response.raise_for_status()
        data = response.json()

        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            raise MoleculeNotFoundError(f"No CID found for '{name}'")

        cid: int = int(cids[0])  # Take first match
        logger.info(f"Found CID {cid} for '{name}'")
        return cid

    except requests.RequestException as e:
        logger.error(f"PubChem API request failed: {e}")
        raise PubChemAPIError(f"Failed to connect to PubChem: {e}") from e


def search_cid_by_inchikey(inchikey: str) -> int:
    """Resolve a standard InChIKey to a PubChem CID.

    InChIKeys are hashes and cannot be inverted to a structure locally, so this
    is the one identifier type that always requires the network.
    """
    url = f"{PUBCHEM_BASE_URL}/compound/inchikey/{quote(inchikey, safe='')}/cids/JSON"
    try:
        response = _http_get(url)
        if response.status_code == 404:
            raise MoleculeNotFoundError(f"InChIKey '{inchikey}' not found in PubChem")
        response.raise_for_status()
        cids = response.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            raise MoleculeNotFoundError(f"No CID found for InChIKey '{inchikey}'")
        return int(cids[0])
    except requests.RequestException as e:
        logger.error(f"PubChem InChIKey request failed: {e}")
        raise PubChemAPIError(f"Failed to connect to PubChem: {e}") from e


def search_cids_by_name(name: str) -> list:
    """Return ALL PubChem CIDs matching a name (best-match order); [] if none.

    Unlike :func:`search_molecule_by_name` (which returns just the first hit),
    this exposes every match so the UI can disambiguate.
    """
    url = f"{PUBCHEM_BASE_URL}/compound/name/{quote(name, safe='')}/cids/JSON"
    try:
        response = _http_get(url)
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return [
            int(c) for c in response.json().get("IdentifierList", {}).get("CID", [])
        ]
    except requests.RequestException as e:
        logger.error(f"PubChem CID-list request failed: {e}")
        raise PubChemAPIError(f"Failed to connect to PubChem: {e}") from e


def search_pubchem_candidates(query: str, max_results: int = 10) -> list:
    """Lightweight candidate descriptors for an ambiguous name query.

    Returns a list of ``{cid, title, formula, mw}`` dicts (best-match order,
    capped at ``max_results``), or ``[]`` if nothing matches. Uses one batch
    property request rather than fetching each full structure.
    """
    cids = search_cids_by_name(query)[:max_results]
    if not cids:
        return []
    cid_str = ",".join(str(c) for c in cids)
    url = (
        f"{PUBCHEM_BASE_URL}/compound/cid/{cid_str}"
        f"/property/MolecularFormula,MolecularWeight,Title/JSON"
    )
    try:
        response = _http_get(url)
        response.raise_for_status()
        props = response.json().get("PropertyTable", {}).get("Properties", [])
    except requests.RequestException as e:
        logger.error(f"PubChem property request failed: {e}")
        raise PubChemAPIError(f"Failed to connect to PubChem: {e}") from e

    # Preserve the CID search order (the property endpoint may reorder).
    by_cid = {int(p.get("CID")): p for p in props if p.get("CID") is not None}
    out = []
    for cid in cids:
        p = by_cid.get(cid)
        if p is None:
            continue
        try:
            mw = float(p.get("MolecularWeight", 0) or 0)
        except (TypeError, ValueError):
            mw = 0.0
        out.append(
            {
                "cid": cid,
                "title": p.get("Title") or f"CID {cid}",
                "formula": p.get("MolecularFormula", "?"),
                "mw": mw,
            }
        )
    return out


@lru_cache(maxsize=50)
def get_molecule_sdf(cid: int, conformer_3d: bool = True) -> str:
    """
    Retrieve molecule SDF from PubChem by CID.

    Args:
        cid: PubChem Compound ID
        conformer_3d: If True, fetch 3D conformer; if False, fetch 2D structure

    Returns:
        str: SDF file content

    Raises:
        PubChemAPIError: If API request fails
        MoleculeNotFoundError: If CID not found
    """
    record_type = "3d" if conformer_3d else "2d"
    url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/record/SDF"

    params = {}
    if conformer_3d:
        params["record_type"] = "3d"

    try:
        logger.debug(f"Fetching {record_type.upper()} SDF for CID {cid}")
        response = _http_get(url, params=params)

        if response.status_code == 404:
            if conformer_3d:
                # Try falling back to 2D if 3D not available
                logger.warning(f"No 3D structure for CID {cid}, trying 2D")
                return get_molecule_sdf(cid, conformer_3d=False)
            raise MoleculeNotFoundError(f"CID {cid} not found in PubChem")

        response.raise_for_status()
        sdf_content: str = str(response.text)

        logger.info(f"Retrieved {record_type.upper()} SDF for CID {cid}")
        return sdf_content

    except requests.RequestException as e:
        logger.error(f"PubChem SDF request failed: {e}")
        raise PubChemAPIError(f"Failed to retrieve molecule: {e}") from e


def _separate_fragments(mol: Any, min_gap: float = 3.0) -> None:
    """Push disconnected fragments (e.g. a salt's counterion) apart, in place.

    RDKit's ``EmbedMolecule`` places multiple fragments in one coordinate frame
    and frequently overlaps them — a counterion can land ~1.4 Å from the cation,
    which distance-based bond perception then reads as a (hyper)valent bond and
    the renderer rejects ("Valence of atom N is …, larger than allowed"). This
    is why salts like methylene blue (cation + Cl⁻) failed.

    After embedding, translate every non-largest fragment radially outward from
    the main fragment so the closest inter-fragment gap is at least ``min_gap``
    Å. Operates on the existing conformer; atom order is preserved. No-op for
    single-fragment molecules.
    """
    if not RDKIT_AVAILABLE or mol.GetNumConformers() == 0:
        return
    frags = Chem.GetMolFrags(mol)  # tuple of atom-index tuples, order preserved
    if len(frags) <= 1:
        return

    import numpy as np
    from rdkit.Geometry import Point3D

    conf = mol.GetConformer()
    pos = {i: np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())}
    main = max(frags, key=len)
    main_c = np.mean([pos[i] for i in main], axis=0)
    main_r = max((float(np.linalg.norm(pos[i] - main_c)) for i in main), default=0.0)
    for frag in frags:
        if frag is main:
            continue
        fc = np.mean([pos[i] for i in frag], axis=0)
        fr = max((float(np.linalg.norm(pos[i] - fc)) for i in frag), default=0.0)
        direction = fc - main_c
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        shift = (main_c + direction * (main_r + fr + min_gap)) - fc
        for i in frag:
            new = pos[i] + shift
            conf.SetAtomPosition(
                i, Point3D(float(new[0]), float(new[1]), float(new[2]))
            )
            pos[i] = new


def sdf_to_xyz(sdf_content: str) -> Tuple[str, Dict[str, Any]]:
    """
    Convert SDF content to XYZ format string.

    Args:
        sdf_content: SDF file content as string

    Returns:
        Tuple of (xyz_string, metadata_dict)
        xyz_string format: "n_atoms\\ncomment\\natom x y z\\n..."
        metadata includes: formula, molecular_weight, charge

    Raises:
        ValueError: If SDF parsing fails
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SDF to XYZ conversion")

    try:
        # Parse SDF with RDKit, keeping any explicit hydrogens (3D PubChem SDFs
        # already carry them with real coordinates).
        mol = Chem.MolFromMolBlock(sdf_content, removeHs=False)

        if mol is None:
            raise ValueError("Failed to parse SDF content")

        # Add any missing hydrogens *with* coordinates. Without addCoords the
        # new H default to the origin, which — combined with a 2D SDF — yields a
        # degenerate geometry (atoms piled at 0,0,0) that bond perception then
        # reads as absurd valences.
        mol = Chem.AddHs(mol, addCoords=True)

        # Re-embed in 3D whenever there is no conformer, or the conformer is 2D
        # (PubChem's 3D→2D fallback). A flat conformer must not be returned as a
        # "3D" structure.
        conf = mol.GetConformer() if mol.GetNumConformers() else None
        coords_embedded = conf is None or not conf.Is3D()
        if coords_embedded:
            if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                except Exception:
                    pass
            # Salts/counterions embed jammed together — separate them so bond
            # perception doesn't see a bonded counterion.
            _separate_fragments(mol)

        # Extract coordinates and build XYZ string
        conf = mol.GetConformer()
        xyz_lines = [str(mol.GetNumAtoms())]

        # Get molecular formula
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        xyz_lines.append(f"PubChem molecule: {formula}")

        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol:3s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

        xyz_string = "\n".join(xyz_lines)

        # Gather metadata
        metadata = {
            "formula": formula,
            "molecular_weight": Descriptors.MolWt(mol),
            "charge": Chem.GetFormalCharge(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "coords_embedded": coords_embedded,
        }

        logger.debug(f"Converted SDF to XYZ: {metadata['formula']}")
        return xyz_string, metadata

    except Exception as e:
        logger.error(f"SDF to XYZ conversion failed: {e}")
        raise ValueError(f"Failed to convert SDF to XYZ: {e}") from e


def fetch_molecule(
    name: str, conformer_3d: bool = True
) -> Tuple[str, Dict[str, Any], int]:
    """
    High-level function to fetch molecule from PubChem by name.

    Performs search, retrieves SDF, and converts to XYZ in one call.

    Args:
        name: Molecule name (common or IUPAC)
        conformer_3d: If True, fetch 3D structure; if False, 2D

    Returns:
        Tuple of (xyz_string, metadata_dict, cid)

    Raises:
        PubChemError: If any step fails
    """
    logger.info(f"Fetching molecule '{name}' from PubChem")

    # Search for CID
    cid = search_molecule_by_name(name)

    # Get SDF
    sdf_content = get_molecule_sdf(cid, conformer_3d=conformer_3d)

    # Convert to XYZ
    xyz_string, metadata = sdf_to_xyz(sdf_content)

    # Add CID to metadata
    metadata["pubchem_cid"] = cid
    metadata["pubchem_name"] = name

    logger.info(f"Successfully fetched '{name}' (CID: {cid})")
    return xyz_string, metadata, cid


def get_common_molecules() -> Dict[str, str]:
    """
    Get a curated list of common molecules for educational use.

    Returns:
        Dict mapping display names to PubChem search names
    """
    return {
        # Simple molecules
        "Water (H₂O)": "water",
        "Hydrogen (H₂)": "hydrogen",
        "Oxygen (O₂)": "oxygen",
        "Nitrogen (N₂)": "nitrogen",
        "Carbon Dioxide (CO₂)": "carbon dioxide",
        "Ammonia (NH₃)": "ammonia",
        "Methane (CH₄)": "methane",
        # Organic molecules
        "Ethanol (CH₃CH₂OH)": "ethanol",
        "Acetic Acid (CH₃COOH)": "acetic acid",
        "Acetone (CH₃COCH₃)": "acetone",
        "Benzene (C₆H₆)": "benzene",
        "Toluene (C₆H₅CH₃)": "toluene",
        "Phenol (C₆H₅OH)": "phenol",
        # Biochemical molecules
        "Glucose (C₆H₁₂O₆)": "glucose",
        "Glycine (NH₂CH₂COOH)": "glycine",
        "Alanine (CH₃CH(NH₂)COOH)": "alanine",
        "Caffeine": "caffeine",
        "Aspirin": "aspirin",
        "Vitamin C": "ascorbic acid",
        # Ions (may need special handling)
        "Hydronium (H₃O⁺)": "hydronium",
        "Hydroxide (OH⁻)": "hydroxide",
        "Ammonium (NH₄⁺)": "ammonium",
    }


def student_friendly_fetch(name: str) -> Tuple[Optional[str], str]:
    """
    Fetch molecule with student-friendly error messages.

    Args:
        name: Molecule name to search

    Returns:
        Tuple of (xyz_string, message)
        xyz_string is None if fetch failed
        message describes success or failure
    """
    try:
        xyz_string, metadata, cid = fetch_molecule(name, conformer_3d=True)

        message = (
            f"✓ Found '{name}' in PubChem!\n"
            f"  CID: {cid}\n"
            f"  Formula: {metadata['formula']}\n"
            f"  Atoms: {metadata['num_atoms']} "
            f"({metadata['num_heavy_atoms']} heavy atoms)\n"
            f"  Molecular Weight: {metadata['molecular_weight']:.2f} g/mol"
        )

        return xyz_string, message

    except MoleculeNotFoundError:
        message = (
            f"❌ Could not find '{name}' in PubChem.\n"
            f"   Try:\n"
            f"   • Check spelling (e.g., 'ethanol' not 'ethonal')\n"
            f"   • Use IUPAC name (e.g., 'ethanol' not 'alcohol')\n"
            f"   • Use common name (e.g., 'water' not 'dihydrogen monoxide')\n"
            f"   • Search manually at: https://pubchem.ncbi.nlm.nih.gov/"
        )
        return None, message

    except PubChemAPIError:
        message = (
            "❌ Connection to PubChem failed.\n"
            "   • Check your internet connection\n"
            "   • Try again in a moment\n"
            "   • Use preset molecules if problem persists"
        )
        return None, message

    except Exception as e:
        message = (
            f"❌ Error fetching molecule: {str(e)}\n"
            f"   Please try a different molecule or contact your instructor."
        )
        logger.error(f"Unexpected error in student_friendly_fetch: {e}", exc_info=True)
        return None, message


def check_pubchem_availability() -> bool:
    """
    Check if PubChem API is accessible.

    Returns:
        bool: True if PubChem is accessible, False otherwise
    """
    # M10 audit fix (2026-07-14): this used to call requests.get() directly,
    # bypassing the shared client-side rate limiter (_throttle()) that every
    # other function in this module goes through — a burst of concurrent
    # availability checks (e.g. several students in a classroom clicking
    # "check connection" around the same time) could exceed PubChem's
    # server-side throttle. It also hardcoded timeout=5 instead of the
    # config.PUBCHEM_AVAILABILITY_TIMEOUT_S constant defined specifically
    # for this probe, so changing that constant silently had no effect
    # here. Calls _throttle() directly (not the full _http_get retry loop —
    # this is meant to be a quick, no-retry reachability probe) and uses
    # the configured timeout.
    try:
        url = f"{PUBCHEM_BASE_URL}/compound/cid/962/property/MolecularFormula/JSON"
        _throttle()
        response = requests.get(url, timeout=config.PUBCHEM_AVAILABILITY_TIMEOUT_S)
        return bool(response.status_code == 200)
    except Exception:
        return False


# ============================================================================
# SMILES Input and 2D Structure Rendering
# ============================================================================


def smiles_to_xyz(smiles: str, optimize_3d: bool = True) -> Tuple[str, Dict[str, Any]]:
    """
    Convert SMILES string to XYZ coordinates with 3D structure generation.

    Args:
        smiles: SMILES string (e.g., "CCO" for ethanol)
        optimize_3d: If True, generate and optimize 3D coordinates with UFF

    Returns:
        Tuple of (xyz_string, metadata_dict)
        xyz_string format: "n_atoms\\ncomment\\natom x y z\\n..."
        metadata includes: formula, molecular_weight, charge, smiles

    Raises:
        ValueError: If SMILES parsing fails or 3D generation fails
        ImportError: If RDKit is not available
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES conversion")

    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        if optimize_3d:
            # Generate conformer
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result != 0:
                # Try with random coords if embedding fails
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)

            # Optimize with UFF force field
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except Exception:
                logger.warning("UFF optimization failed, using unoptimized coordinates")

            _separate_fragments(mol)  # keep salt counterions apart

        # Extract coordinates
        if mol.GetNumConformers() == 0:
            raise ValueError("Failed to generate 3D coordinates")

        conf = mol.GetConformer()
        xyz_lines = [str(mol.GetNumAtoms())]

        # Get molecular formula
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        xyz_lines.append(f"Generated from SMILES: {smiles} ({formula})")

        # Build XYZ string
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol:3s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

        xyz_string = "\n".join(xyz_lines)

        # Gather metadata
        metadata = {
            "formula": formula,
            "molecular_weight": Descriptors.MolWt(mol),
            "charge": Chem.GetFormalCharge(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol),
        }

        logger.info(f"Converted SMILES '{smiles}' to XYZ: {metadata['formula']}")
        return xyz_string, metadata

    except Exception as e:
        logger.error(f"SMILES to XYZ conversion failed: {e}")
        raise ValueError(f"Failed to convert SMILES to XYZ: {e}") from e


def inchi_to_xyz(inchi: str, optimize_3d: bool = True) -> Tuple[str, Dict[str, Any]]:
    """Convert an InChI string to XYZ coordinates via RDKit (no network).

    Mirrors :func:`smiles_to_xyz`: parse → add H → embed (ETKDG, seed 42) →
    UFF-optimize. Returns ``(xyz_string, metadata)``; metadata carries the
    canonical SMILES so the caller can label provenance consistently.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for InChI conversion")

    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            raise ValueError(f"Invalid InChI string: {inchi}")

        mol = Chem.AddHs(mol)
        if optimize_3d:
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result != 0:
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except Exception:
                logger.warning("UFF optimization failed, using unoptimized coordinates")

            _separate_fragments(mol)  # keep salt counterions apart

        if mol.GetNumConformers() == 0:
            raise ValueError("Failed to generate 3D coordinates")

        conf = mol.GetConformer()
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        xyz_lines = [str(mol.GetNumAtoms()), f"Generated from InChI ({formula})"]
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            xyz_lines.append(
                f"{atom.GetSymbol():3s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}"
            )

        metadata = {
            "formula": formula,
            "molecular_weight": Descriptors.MolWt(mol),
            "charge": Chem.GetFormalCharge(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "inchi": inchi,
            "canonical_smiles": Chem.MolToSmiles(mol),
        }
        logger.info(f"Converted InChI to XYZ: {formula}")
        return "\n".join(xyz_lines), metadata

    except Exception as e:
        logger.error(f"InChI to XYZ conversion failed: {e}")
        raise ValueError(f"Failed to convert InChI to XYZ: {e}") from e


def student_friendly_smiles_to_xyz(smiles: str) -> Tuple[Optional[str], str]:
    """
    Convert SMILES to XYZ with student-friendly error messages.

    Args:
        smiles: SMILES string

    Returns:
        Tuple of (xyz_string, message)
        xyz_string is None if conversion failed
        message describes success or failure
    """
    try:
        xyz_string, metadata = smiles_to_xyz(smiles, optimize_3d=True)

        message = (
            f"✓ Converted SMILES to 3D structure!\n"
            f"  SMILES: {smiles}\n"
            f"  Formula: {metadata['formula']}\n"
            f"  Atoms: {metadata['num_atoms']} "
            f"({metadata['num_heavy_atoms']} heavy atoms)\n"
            f"  Molecular Weight: {metadata['molecular_weight']:.2f} g/mol\n"
            f"  Canonical SMILES: {metadata['canonical_smiles']}"
        )

        return xyz_string, message

    except ValueError as e:
        message = (
            f"❌ Invalid SMILES string: {smiles}\n"
            f"   Error: {str(e)}\n\n"
            f"   SMILES Tips:\n"
            f"   • Check syntax (use RDKit/OpenBabel style)\n"
            f"   • Ethanol: CCO or C(C)O\n"
            f"   • Benzene: c1ccccc1 or C1=CC=CC=C1\n"
            f"   • Water: O (just the atom symbol)\n"
            f"   • Methane: C\n\n"
            f"   Resources:\n"
            f"   • SMILES Tutorial: https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html\n"
            f"   • Draw structure: https://pubchem.ncbi.nlm.nih.gov/edit3/index.html"
        )
        return None, message

    except ImportError:
        message = (
            "❌ RDKit is required for SMILES conversion.\n"
            "   Install with: conda install -c conda-forge rdkit"
        )
        return None, message

    except Exception as e:
        message = (
            f"❌ Error converting SMILES: {str(e)}\n"
            f"   Please try a different molecule or contact your instructor."
        )
        logger.error(
            f"Unexpected error in student_friendly_smiles_to_xyz: {e}", exc_info=True
        )
        return None, message


def generate_2d_structure_svg(
    smiles: Optional[str] = None,
    mol: Optional[object] = None,
    xyz_string: Optional[str] = None,
    width: int = 300,
    height: int = 300,
) -> Optional[str]:
    """
    Generate 2D structure diagram as SVG string.

    Can accept input as SMILES, RDKit Mol object, or XYZ string.

    Args:
        smiles: SMILES string (if provided)
        mol: RDKit Mol object (if provided)
        xyz_string: XYZ coordinate string (if provided)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        SVG string of 2D structure, or None if generation fails

    Raises:
        ImportError: If RDKit is not available
        ValueError: If no valid input provided
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for 2D structure rendering")

    try:

        from rdkit.Chem import Draw

        # Get RDKit molecule from input
        if mol is not None:
            rdkit_mol = mol
        elif smiles is not None:
            rdkit_mol = Chem.MolFromSmiles(smiles)
            if rdkit_mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
        elif xyz_string is not None:
            # Convert XYZ to SMILES (requires RDKit bond perception)
            # Parse XYZ
            lines = xyz_string.strip().split("\n")
            if len(lines) < 3:
                raise ValueError("Invalid XYZ format")

            # Build mol from XYZ
            from rdkit.Chem import rdDetermineBonds

            # Collect valid atom lines first so the conformer can be sized
            # up front and the atom index used for SetAtomPosition always
            # matches the just-added atom (skipped malformed lines used to
            # desync the two).
            atom_lines = []
            for line in lines[2:]:  # Skip first 2 lines (count + comment)
                parts = line.split()
                if len(parts) < 4:
                    continue
                atom_lines.append(parts)

            if not atom_lines:
                raise ValueError("No atom lines found in XYZ string")

            # Chem.Mol() is immutable — AddAtom only exists on RWMol.
            # Build on RWMol, then convert to an immutable Mol via
            # GetMol() before DetermineBonds (mirrors the working
            # Chem.MolFromXYZBlock() + DetermineBonds() pattern used
            # elsewhere in QuantUI, e.g. preopt.py).
            rw_mol = Chem.RWMol()
            conf = Chem.Conformer(len(atom_lines))

            for i, parts in enumerate(atom_lines):
                symbol = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

                atom = Chem.Atom(symbol)
                rw_mol.AddAtom(atom)
                conf.SetAtomPosition(i, (x, y, z))

            rw_mol.AddConformer(conf)
            rdkit_mol = rw_mol.GetMol()

            # Determine bonds
            rdDetermineBonds.DetermineBonds(rdkit_mol)
        else:
            raise ValueError("Must provide smiles, mol, or xyz_string")

        # Generate 2D coordinates for nice layout
        AllChem.Compute2DCoords(rdkit_mol)

        # Draw molecule to SVG
        drawer = Draw.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(rdkit_mol)
        drawer.FinishDrawing()
        svg: str = str(drawer.GetDrawingText())

        logger.debug("Generated 2D structure SVG")
        return svg

    except Exception as e:
        logger.error(f"2D structure generation failed: {e}")
        return None


def display_2d_structure(
    smiles: Optional[str] = None,
    mol: Optional[object] = None,
    xyz_string: Optional[str] = None,
    width: int = 400,
    height: int = 300,
):
    """
    Display 2D structure diagram in Jupyter notebook.

    Args:
        smiles: SMILES string (if provided)
        mol: RDKit Mol object (if provided)
        xyz_string: XYZ coordinate string (if provided)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        IPython display object or None if fails
    """
    try:
        from IPython.display import SVG
        from IPython.display import display as ipython_display

        svg = generate_2d_structure_svg(
            smiles=smiles, mol=mol, xyz_string=xyz_string, width=width, height=height
        )

        if svg:
            ipython_display(SVG(svg))
            return True
        else:
            print("⚠️  Could not generate 2D structure")
            return False

    except ImportError as e:
        logger.warning(f"Could not display 2D structure: {e}")
        print("⚠️  IPython display not available")
        return False
    except Exception as e:
        logger.error(f"2D structure display failed: {e}")
        print(f"⚠️  2D structure display failed: {e}")
        return False


def get_smiles_examples() -> Dict[str, str]:
    """
    Get example SMILES strings for educational use.

    Returns:
        Dict mapping molecule names to SMILES strings
    """
    return {
        # Simple molecules
        "Water": "O",
        "Ammonia": "N",
        "Methane": "C",
        "Ethane": "CC",
        "Propane": "CCC",
        # Functional groups
        "Methanol": "CO",
        "Ethanol": "CCO",
        "Acetic Acid": "CC(=O)O",
        "Acetone": "CC(=O)C",
        "Formaldehyde": "C=O",
        # Aromatics
        "Benzene": "c1ccccc1",
        "Toluene": "Cc1ccccc1",
        "Phenol": "Oc1ccccc1",
        "Aniline": "Nc1ccccc1",
        # Biochemical
        "Glycine": "NCC(=O)O",
        "Alanine": "CC(N)C(=O)O",
        "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
        # Common molecules
        "Carbon Dioxide": "O=C=O",
        "Hydrogen Peroxide": "OO",
        "Ethylene": "C=C",
        "Acetylene": "C#C",
    }


def validate_smiles(smiles: str) -> Tuple[bool, str]:
    """
    Validate a SMILES string.

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, message)
    """
    if not RDKIT_AVAILABLE:
        return False, "RDKit not available"

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES syntax"

        # Check if molecule is reasonable
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return False, "Molecule has no atoms"

        if num_atoms > 200:
            return (
                False,
                f"Molecule too large ({num_atoms} atoms). Consider smaller molecules for calculations.",
            )

        return True, f"Valid SMILES ({num_atoms} atoms)"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


# ============================================================================
# Smart input routing
# ============================================================================

# Standard InChIKey: 14 block chars - 10 block chars - 1 flag char.
_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
# A bare molecular formula, e.g. "H2O", "C6H6", "CO2", "Fe".
_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")
# Characters that only appear in SMILES, never in a common/IUPAC molecule name.
_SMILES_STRUCTURAL = set("=#()[]/\\@.%+")


def _looks_like_smiles(query: str) -> bool:
    """High-precision SMILES check: only ``True`` when RDKit parses it *and* it
    carries SMILES-only signals (structural punctuation or ring digits).

    Deliberately conservative — bare-letter tokens like ``CCO`` (which are valid
    SMILES *and* plausible names/formulas) are left for the provider chain
    to disambiguate, so we never misroute a plain name to a local parse.
    """
    if not RDKIT_AVAILABLE or " " in query:
        return False
    has_structural = any(c in _SMILES_STRUCTURAL for c in query)
    has_digit = any(c.isdigit() for c in query)
    if not (has_structural or has_digit):
        return False
    return Chem.MolFromSmiles(query) is not None


def classify_query(query: str) -> str:
    """Classify a user structure query so it can be routed to the right resolver.

    Returns one of ``"cid"``, ``"inchikey"``, ``"inchi"``, ``"smiles"``,
    ``"formula"``, or ``"name"``. ``smiles`` / ``inchi`` resolve locally via
    RDKit (no network); the rest go to PubChem. ``formula`` is currently routed
    like ``name`` (PubChem's async fastformula search is not used here).
    """
    q = query.strip()
    if not q:
        raise ValueError("Empty query")
    if q.startswith("InChI="):
        return "inchi"
    if _INCHIKEY_RE.match(q):
        return "inchikey"
    if re.fullmatch(r"(?:CID:?\s*)?\d+", q, flags=re.IGNORECASE):
        return "cid"
    if _looks_like_smiles(q):
        return "smiles"
    if _FORMULA_RE.match(q):
        return "formula"
    return "name"


def _coerce_cid(query: str) -> int:
    """Extract the integer CID from ``123`` / ``CID123`` / ``cid: 123`` forms."""
    return int(re.sub(r"[^\d]", "", query))


def fetch_structure(
    query: str, conformer_3d: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """Resolve any supported query type to ``(xyz_string, metadata)``.

    Routes by :func:`classify_query`: SMILES/InChI resolve locally via RDKit
    (no network); CID/InChIKey/name/formula go to PubChem. ``metadata`` always
    carries a ``source`` key (``"rdkit-smiles"`` / ``"rdkit-inchi"`` /
    ``"pubchem"``) and a ``conformer_origin`` key describing where the 3D
    coordinates came from, so the UI can be honest about provenance.

    Raises :class:`PubChemError` / :class:`ValueError` on failure (the
    student-friendly wrapper :func:`student_friendly_resolve` maps these to
    messages).
    """
    qtype = classify_query(query)
    q = query.strip()
    logger.info(f"Resolving structure query '{q}' classified as '{qtype}'")

    if qtype == "smiles":
        xyz, metadata = smiles_to_xyz(q, optimize_3d=conformer_3d)
        metadata["source"] = "rdkit-smiles"
        metadata["conformer_origin"] = "rdkit-embedded"
        return xyz, metadata

    if qtype == "inchi":
        xyz, metadata = inchi_to_xyz(q, optimize_3d=conformer_3d)
        metadata["source"] = "rdkit-inchi"
        metadata["conformer_origin"] = "rdkit-embedded"
        return xyz, metadata

    # Network branch: resolve to a CID, then fetch + convert the SDF.
    if qtype == "cid":
        cid = _coerce_cid(q)
    elif qtype == "inchikey":
        cid = search_cid_by_inchikey(q)
    else:  # "name" or "formula"
        cid = search_molecule_by_name(q)

    sdf_content = get_molecule_sdf(cid, conformer_3d=conformer_3d)
    xyz, metadata = sdf_to_xyz(sdf_content)
    metadata["source"] = "pubchem"
    metadata["pubchem_cid"] = cid
    metadata["query_type"] = qtype
    metadata["conformer_origin"] = (
        "rdkit-embedded" if metadata.get("coords_embedded") else "pubchem"
    )
    return xyz, metadata


def student_friendly_resolve(query: str) -> Tuple[Optional[str], str]:
    """Smart, type-aware structure fetch with student-friendly messages.

    Drop-in replacement for :func:`student_friendly_fetch` that also handles
    SMILES / InChI / CID / InChIKey input (the plain-name path is unchanged).
    Returns ``(xyz_string_or_None, message)``.
    """
    try:
        xyz_string, metadata = fetch_structure(query, conformer_3d=True)
    except (MoleculeNotFoundError, ValueError) as exc:
        return None, (
            f"❌ Could not resolve '{query}'.\n"
            f"   {exc}\n"
            f"   Try a different name, a SMILES (e.g. CCO), or check spelling.\n"
            f"   Search manually at: https://pubchem.ncbi.nlm.nih.gov/"
        )
    except PubChemAPIError:
        return None, (
            "❌ Connection to PubChem failed.\n"
            "   • Check your internet connection\n"
            "   • Try again in a moment\n"
            "   • Use a preset molecule if the problem persists"
        )
    except ImportError:
        return None, (
            "❌ RDKit is required for SMILES / InChI input.\n"
            "   Install with: conda install -c conda-forge rdkit"
        )
    except Exception as exc:  # pragma: no cover - unexpected
        logger.error(
            f"Unexpected error in student_friendly_resolve: {exc}", exc_info=True
        )
        return None, f"❌ Error resolving '{query}': {exc}"

    source = {
        "rdkit-smiles": "generated locally from SMILES",
        "rdkit-inchi": "generated locally from InChI",
        "pubchem": "PubChem",
    }.get(metadata.get("source", ""), metadata.get("source", "?"))
    origin = metadata.get("conformer_origin", "")
    origin_note = (
        " (2D structure embedded by RDKit)" if origin == "rdkit-embedded" else ""
    )
    message = (
        f"✓ Resolved '{query}' via {source}.\n"
        f"  Formula: {metadata.get('formula', '?')}\n"
        f"  Atoms: {metadata.get('num_atoms', '?')} "
        f"({metadata.get('num_heavy_atoms', '?')} heavy)\n"
        f"  Molecular weight: {metadata.get('molecular_weight', 0):.2f} g/mol{origin_note}"
    )
    return xyz_string, message
