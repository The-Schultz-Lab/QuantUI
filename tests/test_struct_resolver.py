"""Tests for M-STRUCT STRUCT.1 (input-type routing) + STRUCT.2 (hardened client).

All tests are platform-independent: network is mocked, RDKit-only paths are
gated. No PySCF dependency.
"""

from unittest.mock import Mock, patch

import pytest
import requests

from quantui import config
from quantui.pubchem import (
    RDKIT_AVAILABLE,
    MoleculeNotFoundError,
    PubChemAPIError,
    classify_query,
    fetch_structure,
    get_molecule_sdf,
    search_cid_by_inchikey,
    search_molecule_by_name,
    student_friendly_resolve,
)

rdkit_only = pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit not installed")


# ============================================================================
# STRUCT.1 — classify_query routing
# ============================================================================


class TestClassifyQuery:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("962", "cid"),
            ("CID962", "cid"),
            ("cid: 2244", "cid"),
            ("InChI=1S/H2O/h1H2", "inchi"),
            ("XLYOFNOQVPJJNP-UHFFFAOYSA-N", "inchikey"),  # water InChIKey
            ("CC(=O)O", "smiles"),  # acetic acid — structural chars
            ("c1ccccc1", "smiles"),  # benzene — aromatic + ring digit
            ("[OH3+]", "smiles"),  # hydronium — brackets + charge
            ("H2O", "formula"),  # digit but not valid SMILES
            ("C6H6", "formula"),
            ("Fe", "formula"),
            ("water", "name"),
            ("carbon dioxide", "name"),  # space → never SMILES
            ("acetylsalicylic acid", "name"),
        ],
    )
    def test_classification(self, query, expected):
        # SMILES classification needs RDKit to validate; skip those when absent.
        if expected == "smiles" and not RDKIT_AVAILABLE:
            pytest.skip("rdkit not installed")
        assert classify_query(query) == expected

    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            classify_query("   ")

    @rdkit_only
    def test_bare_letter_smiles_routes_to_name_not_smiles(self):
        # "CCO" is valid SMILES *and* a plausible token; we deliberately leave
        # it to the (future) provider chain rather than misroute a name.
        assert classify_query("CCO") in ("formula", "name")


# ============================================================================
# STRUCT.2 — URL encoding
# ============================================================================


class TestUrlEncoding:
    @patch("quantui.pubchem.requests.get")
    def test_name_with_space_is_encoded(self, mock_get):
        mock_get.return_value = Mock(
            status_code=200, json=Mock(return_value={"IdentifierList": {"CID": [280]}})
        )
        search_molecule_by_name("carbon dioxide")
        url = mock_get.call_args[0][0]
        assert "carbon%20dioxide" in url
        assert " " not in url

    @patch("quantui.pubchem.requests.get")
    def test_inchikey_endpoint_encoded(self, mock_get):
        mock_get.return_value = Mock(
            status_code=200, json=Mock(return_value={"IdentifierList": {"CID": [962]}})
        )
        cid = search_cid_by_inchikey("XLYOFNOQVPJJNP-UHFFFAOYSA-N")
        assert cid == 962
        assert "compound/inchikey/" in mock_get.call_args[0][0]


# ============================================================================
# STRUCT.2 — throttle / 503 back-off
# ============================================================================


class TestThrottleBackoff:
    @patch("quantui.pubchem.time.sleep")  # don't actually sleep
    @patch("quantui.pubchem.requests.get")
    def test_503_then_success_retries(self, mock_get, mock_sleep):
        resp_503 = Mock(status_code=503)
        resp_ok = Mock(
            status_code=200, json=Mock(return_value={"IdentifierList": {"CID": [962]}})
        )
        mock_get.side_effect = [resp_503, resp_ok]

        cid = search_molecule_by_name("water")
        assert cid == 962
        assert mock_get.call_count == 2  # one retry after the 503
        assert mock_sleep.called  # backed off between attempts

    @patch("quantui.pubchem.time.sleep")
    @patch("quantui.pubchem.requests.get")
    def test_persistent_503_exhausts_retries_then_errors(self, mock_get, mock_sleep):
        resp_503 = Mock(status_code=503)
        resp_503.raise_for_status.side_effect = requests.HTTPError("503")
        mock_get.return_value = resp_503

        with pytest.raises(PubChemAPIError):
            search_molecule_by_name("water")
        # Tried the full retry budget, then gave up.
        assert mock_get.call_count == config.PUBCHEM_MAX_RETRIES
        # Backed off at least once between attempts (sleep is also used by the
        # rate-limiter, so assert "happened" rather than an exact count).
        assert mock_sleep.called

    @patch("quantui.pubchem.requests.get")
    def test_non_503_not_retried(self, mock_get):
        # A 500 must surface immediately, not trigger the 503 retry loop.
        resp = Mock(status_code=500)
        resp.raise_for_status.side_effect = requests.HTTPError("500")
        mock_get.return_value = resp
        with pytest.raises(PubChemAPIError):
            search_molecule_by_name("water")
        assert mock_get.call_count == 1


# ============================================================================
# STRUCT.1 — fetch_structure routing (local vs network)
# ============================================================================


class TestFetchStructureRouting:
    @rdkit_only
    @patch("quantui.pubchem.requests.get")
    def test_smiles_resolves_locally_without_network(self, mock_get):
        mock_get.side_effect = AssertionError("network must not be touched for SMILES")
        xyz, meta = fetch_structure("CC(=O)O")  # acetic acid
        assert meta["source"] == "rdkit-smiles"
        assert meta["conformer_origin"] == "rdkit-embedded"
        assert meta["formula"] == "C2H4O2"
        assert xyz.strip().splitlines()[0].strip() == str(meta["num_atoms"])
        mock_get.assert_not_called()

    @rdkit_only
    @patch("quantui.pubchem.requests.get")
    def test_inchi_resolves_locally_without_network(self, mock_get):
        mock_get.side_effect = AssertionError("network must not be touched for InChI")
        xyz, meta = fetch_structure("InChI=1S/H2O/h1H2")  # water
        assert meta["source"] == "rdkit-inchi"
        assert meta["num_heavy_atoms"] == 1
        mock_get.assert_not_called()

    @rdkit_only
    @patch("quantui.pubchem.search_cid_by_inchikey")
    @patch("quantui.pubchem.get_molecule_sdf")
    def test_inchikey_routes_through_network(
        self, mock_sdf, mock_inchikey, sample_sdf_water
    ):
        mock_inchikey.return_value = 962
        mock_sdf.return_value = sample_sdf_water
        xyz, meta = fetch_structure("XLYOFNOQVPJJNP-UHFFFAOYSA-N")
        mock_inchikey.assert_called_once()
        assert meta["source"] == "pubchem"
        assert meta["pubchem_cid"] == 962
        assert meta["conformer_origin"] in ("pubchem", "rdkit-embedded")

    @rdkit_only
    @patch("quantui.pubchem.get_molecule_sdf")
    @patch("quantui.pubchem.search_molecule_by_name")
    def test_name_routes_through_pubchem(self, mock_search, mock_sdf, sample_sdf_water):
        mock_search.return_value = 962
        mock_sdf.return_value = sample_sdf_water
        _, meta = fetch_structure("water")
        mock_search.assert_called_once_with("water")
        assert meta["source"] == "pubchem"
        assert meta["query_type"] == "name"


# ============================================================================
# STRUCT.1/.2 — student-friendly wrapper
# ============================================================================


class TestStudentFriendlyResolve:
    @rdkit_only
    def test_smiles_success_message(self):
        # Use an unambiguous SMILES (structural chars) so it resolves locally
        # with no network. Bare-letter tokens like "CCO" classify as
        # formula/name by design and are left to the provider chain.
        xyz, msg = student_friendly_resolve("CC(=O)O")
        assert xyz is not None
        assert "generated locally from SMILES" in msg
        assert "C2H4O2" in msg

    @patch("quantui.pubchem.fetch_structure")
    def test_not_found_message(self, mock_fetch):
        mock_fetch.side_effect = MoleculeNotFoundError("nope")
        xyz, msg = student_friendly_resolve("zzxqq")
        assert xyz is None
        assert "Could not resolve" in msg

    @patch("quantui.pubchem.fetch_structure")
    def test_api_error_message(self, mock_fetch):
        mock_fetch.side_effect = PubChemAPIError("down")
        xyz, msg = student_friendly_resolve("water")
        assert xyz is None
        assert "Connection to PubChem failed" in msg


# ============================================================================
# STRUCT.2 — sdf_to_xyz provenance flag (regression guard)
# ============================================================================


class TestCoordsEmbeddedFlag:
    @rdkit_only
    def test_sdf_with_3d_coords_not_flagged_embedded(self, sample_sdf_water):
        from quantui.pubchem import sdf_to_xyz

        get_molecule_sdf.cache_clear()
        _, meta = sdf_to_xyz(sample_sdf_water)
        assert "coords_embedded" in meta
        # The fixture SDF already carries 3D coords, so RDKit should not embed.
        assert meta["coords_embedded"] is False

    @rdkit_only
    def test_2d_sdf_is_reembedded_to_3d_no_atoms_piled(self):
        """STRUCT.12 regression: a 2D SDF (H added without coords in the old
        code piled them at the origin → 'valence 13') must come back as a real
        3D structure with every atom at a distinct position."""
        import math

        from rdkit import Chem
        from rdkit.Chem import AllChem

        from quantui.pubchem import sdf_to_xyz

        m = Chem.MolFromSmiles("CCO")  # ethanol, no explicit H
        AllChem.Compute2DCoords(m)  # 2D conformer, flagged 2D
        sdf_2d = Chem.MolToMolBlock(m)

        xyz, meta = sdf_to_xyz(sdf_2d)
        assert meta["coords_embedded"] is True  # we re-embedded the 2D input

        coords = [
            [float(c) for c in line.split()[1:4]]
            for line in xyz.strip().splitlines()[2:]
        ]
        assert len(coords) == meta["num_atoms"] == 9  # C2H6O
        # No two atoms coincide (the old bug piled all H at 0,0,0).
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = math.dist(coords[i], coords[j])
                assert d > 0.5, f"atoms {i},{j} only {d:.3f} Å apart"
        # Genuinely 3D, not flat.
        assert max(abs(c[2]) for c in coords) > 1e-3


class TestSaltFragmentSeparation:
    """STRUCT.14 regression: a salt's counterion must not embed jammed into the
    cation (RDKit otherwise places it ~1.4 Å away → bond perception reads it as
    a bonded, hypervalent atom and the renderer rejects it, e.g. methylene
    blue)."""

    @rdkit_only
    def test_counterion_is_separated(self):
        import math

        from quantui.pubchem import smiles_to_xyz

        # Ammonium chloride: [NH4+] cation + Cl- counterion (two fragments).
        xyz, meta = smiles_to_xyz("[NH4+].[Cl-]")
        atoms = [
            (line.split()[0], [float(c) for c in line.split()[1:4]])
            for line in xyz.strip().splitlines()[2:]
        ]
        cl = next(p for sym, p in atoms if sym == "Cl")
        nearest = min(math.dist(cl, p) for sym, p in atoms if sym != "Cl")
        # A real bond is ~1.3-2.0 Å; the separated counterion must be well beyond.
        assert nearest > 2.5, f"Cl- only {nearest:.2f} Å from the cation"
