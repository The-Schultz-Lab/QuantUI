"""NCI CACTUS Chemical Identifier Resolver (M-STRUCT STRUCT.3).

A chained fallback after PubChem. CACTUS resolves a wide range of identifiers
(common name, IUPAC name, CAS number, InChI, SMILES, formula) to a 3D SDF with
no API key. It often answers queries PubChem misses — CAS numbers in
particular.

Best-effort by design: every failure mode (miss, network error, malformed
response) raises one of the shared PubChem exception types so the provider
chain in :mod:`quantui.structure_providers` can treat all resolvers uniformly.
"""

import logging
from typing import Any, Dict, Tuple
from urllib.parse import quote

import requests

from . import config
from .pubchem import MoleculeNotFoundError, PubChemAPIError, sdf_to_xyz

logger = logging.getLogger(__name__)

# CACTUS resolver base. The ``/file?format=sdf&get3d=true`` form returns a 3D
# SDF; the bare ``/sdf`` form returns whatever (often 2D) CACTUS has.
CACTUS_BASE_URL = "https://cactus.nci.nih.gov/chemical/structure"


def _looks_like_sdf(text: str) -> bool:
    """CACTUS returns an HTML error page (HTTP 200) for unknown identifiers.

    A real SDF molfile always carries the ``V2000``/``V3000`` counts-line tag
    and the ``M  END`` terminator, so key off those rather than the status code.
    """
    return "M  END" in text and ("V2000" in text or "V3000" in text)


def resolve_to_sdf(identifier: str, conformer_3d: bool = True) -> str:
    """Resolve an identifier to SDF text via CACTUS.

    Tries the 3D endpoint first, then falls back to the plain ``/sdf`` form
    (whose coordinates RDKit will embed downstream if they are 2D).

    Raises:
        MoleculeNotFoundError: CACTUS has no structure for the identifier.
        PubChemAPIError: network/transport failure reaching CACTUS.
    """
    enc = quote(identifier, safe="")
    urls = []
    if conformer_3d:
        urls.append(f"{CACTUS_BASE_URL}/{enc}/file?format=sdf&get3d=true")
    urls.append(f"{CACTUS_BASE_URL}/{enc}/sdf")

    last_status = None
    try:
        for url in urls:
            logger.debug(f"CACTUS resolving: {url}")
            response = requests.get(url, timeout=config.CACTUS_TIMEOUT_S)
            last_status = response.status_code
            if response.status_code == 200 and _looks_like_sdf(response.text):
                return str(response.text)
    except requests.RequestException as e:
        logger.error(f"CACTUS request failed: {e}")
        raise PubChemAPIError(f"Failed to connect to CACTUS: {e}")

    raise MoleculeNotFoundError(
        f"CACTUS could not resolve '{identifier}' (last status: {last_status})"
    )


def fetch_from_cactus(
    identifier: str, conformer_3d: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """Resolve an identifier to ``(xyz_string, metadata)`` via CACTUS.

    Metadata carries ``source="cactus"`` and a ``conformer_origin`` describing
    whether RDKit had to embed the coordinates.
    """
    sdf_content = resolve_to_sdf(identifier, conformer_3d=conformer_3d)
    xyz, metadata = sdf_to_xyz(sdf_content)
    metadata["source"] = "cactus"
    metadata["conformer_origin"] = (
        "rdkit-embedded" if metadata.get("coords_embedded") else "cactus"
    )
    return xyz, metadata
