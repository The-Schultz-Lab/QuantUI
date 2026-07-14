"""
Tests for PubChem Integration Module

Tests both API functionality (mocked) and real network requests (marked).
"""

from unittest.mock import Mock, patch

import pytest
import requests

from quantui.pubchem import (
    RDKIT_AVAILABLE,
    MoleculeNotFoundError,
    PubChemAPIError,
    check_pubchem_availability,
    fetch_molecule,
    get_common_molecules,
    get_molecule_sdf,
    sdf_to_xyz,
    search_molecule_by_name,
    student_friendly_fetch,
)

rdkit_only = pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit not installed")

# ============================================================================
# Mocked API Tests (No Network Required)
# ============================================================================


class TestSearchMoleculeByName:
    """Test molecule name search functionality."""

    @patch("quantui.pubchem.requests.get")
    def test_search_water_success(self, mock_get):
        """Test successful search for water."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"IdentifierList": {"CID": [962]}}
        mock_get.return_value = mock_response

        cid = search_molecule_by_name("water")
        assert cid == 962

    @patch("quantui.pubchem.requests.get")
    def test_search_not_found(self, mock_get):
        """Test search for non-existent molecule."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(MoleculeNotFoundError):
            search_molecule_by_name("xyznonexistent123")

    @patch("quantui.pubchem.requests.get")
    def test_search_api_error(self, mock_get):
        """Test API connection failure."""
        mock_get.side_effect = requests.RequestException("Connection failed")

        with pytest.raises(PubChemAPIError):
            search_molecule_by_name("water")

    @patch("quantui.pubchem.requests.get")
    def test_search_empty_result(self, mock_get):
        """Test search returning empty CID list."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"IdentifierList": {"CID": []}}
        mock_get.return_value = mock_response

        with pytest.raises(MoleculeNotFoundError):
            search_molecule_by_name("unknown")


class TestGetMoleculeSDF:
    """Test SDF retrieval functionality."""

    @patch("quantui.pubchem.requests.get")
    def test_get_sdf_3d_success(self, mock_get, sample_sdf_water):
        """Test successful 3D SDF retrieval."""
        get_molecule_sdf.cache_clear()  # avoid cross-test cache pollution
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sample_sdf_water
        mock_get.return_value = mock_response

        sdf = get_molecule_sdf(962, conformer_3d=True)
        assert sdf == sample_sdf_water

        # Verify 3D parameter was passed
        call_args = mock_get.call_args
        assert call_args[1]["params"] == {"record_type": "3d"}

    @patch("quantui.pubchem.requests.get")
    def test_get_sdf_2d_fallback(self, mock_get, sample_sdf_water):
        """Test fallback to 2D when 3D not available."""
        get_molecule_sdf.cache_clear()  # avoid cross-test cache pollution
        # First call (3D) returns 404, second call (2D) succeeds
        mock_response_404 = Mock()
        mock_response_404.status_code = 404

        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.text = sample_sdf_water

        mock_get.side_effect = [mock_response_404, mock_response_200]

        sdf = get_molecule_sdf(962, conformer_3d=True)
        assert sdf == sample_sdf_water
        assert mock_get.call_count == 2

    @patch("quantui.pubchem.requests.get")
    def test_get_sdf_not_found(self, mock_get):
        """Test CID not found."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(MoleculeNotFoundError):
            get_molecule_sdf(999999999, conformer_3d=False)


class TestSDFToXYZ:
    """Test SDF to XYZ conversion (requires RDKit)."""

    @rdkit_only
    def test_sdf_to_xyz_water(self, sample_sdf_water):
        """Test conversion of water SDF to XYZ."""
        xyz_string, metadata = sdf_to_xyz(sample_sdf_water)

        # Check XYZ format
        lines = xyz_string.strip().split("\n")
        assert len(lines) >= 3  # n_atoms, comment, at least 1 atom

        # Check metadata
        assert "formula" in metadata
        assert "molecular_weight" in metadata
        assert "charge" in metadata
        assert "num_atoms" in metadata

        # Water should have 3 atoms (O + 2H)
        assert metadata["num_atoms"] == 3

    @rdkit_only
    def test_sdf_to_xyz_invalid(self):
        """Test conversion of invalid SDF."""
        invalid_sdf = "Not a valid SDF file"

        with pytest.raises(ValueError):
            sdf_to_xyz(invalid_sdf)

    @rdkit_only
    def test_sdf_to_xyz_metadata_fields(self, sample_sdf_water):
        """Test all metadata fields are present."""
        xyz_string, metadata = sdf_to_xyz(sample_sdf_water)

        required_fields = [
            "formula",
            "molecular_weight",
            "charge",
            "num_atoms",
            "num_heavy_atoms",
        ]

        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"


class TestFetchMolecule:
    """Test high-level molecule fetching."""

    @rdkit_only
    @patch("quantui.pubchem.get_molecule_sdf")
    @patch("quantui.pubchem.search_molecule_by_name")
    def test_fetch_molecule_complete(self, mock_search, mock_get_sdf, sample_sdf_water):
        """Test complete molecule fetch workflow (requires RDKit for sdf_to_xyz)."""
        mock_search.return_value = 962
        mock_get_sdf.return_value = sample_sdf_water

        xyz_string, metadata, cid = fetch_molecule("water", conformer_3d=True)

        # Verify search was called
        mock_search.assert_called_once_with("water")

        # Verify SDF retrieval was called
        mock_get_sdf.assert_called_once_with(962, conformer_3d=True)

        # Verify results
        assert cid == 962
        assert "pubchem_cid" in metadata
        assert metadata["pubchem_cid"] == 962
        assert "pubchem_name" in metadata
        assert metadata["pubchem_name"] == "water"
        assert isinstance(xyz_string, str)

    @patch("quantui.pubchem.search_molecule_by_name")
    def test_fetch_molecule_not_found(self, mock_search):
        """Test fetching non-existent molecule."""
        mock_search.side_effect = MoleculeNotFoundError("Not found")

        with pytest.raises(MoleculeNotFoundError):
            fetch_molecule("nonexistent_molecule_xyz123")


class TestStudentFriendlyFetch:
    """Test student-friendly wrapper function."""

    @patch("quantui.pubchem.fetch_molecule")
    def test_student_friendly_success(self, mock_fetch):
        """Test successful fetch with friendly message."""
        mock_fetch.return_value = (
            "O 0 0 0\nH 1 0 0\nH 0 1 0",
            {
                "formula": "H2O",
                "molecular_weight": 18.015,
                "num_atoms": 3,
                "num_heavy_atoms": 1,
            },
            962,
        )

        xyz_string, message = student_friendly_fetch("water")

        assert xyz_string is not None
        assert "Found 'water' in PubChem" in message
        assert "CID: 962" in message
        assert "H2O" in message

    @patch("quantui.pubchem.fetch_molecule")
    def test_student_friendly_not_found(self, mock_fetch):
        """Test friendly error for molecule not found."""
        mock_fetch.side_effect = MoleculeNotFoundError("Not found")

        xyz_string, message = student_friendly_fetch("xyznotfound")

        assert xyz_string is None
        assert "Could not find" in message
        assert "Try:" in message

    @patch("quantui.pubchem.fetch_molecule")
    def test_student_friendly_api_error(self, mock_fetch):
        """Test friendly error for API failure."""
        mock_fetch.side_effect = PubChemAPIError("Connection failed")

        xyz_string, message = student_friendly_fetch("water")

        assert xyz_string is None
        assert "Connection to PubChem failed" in message
        assert "internet connection" in message


class TestCommonMolecules:
    """Test common molecules helper function."""

    def test_get_common_molecules_returns_dict(self):
        """Test that function returns a dictionary."""
        common = get_common_molecules()
        assert isinstance(common, dict)

    def test_get_common_molecules_has_water(self):
        """Test that water is in common molecules."""
        common = get_common_molecules()
        # Check if any key contains water
        water_found = any("water" in value.lower() for value in common.values())
        assert water_found

    def test_get_common_molecules_values_are_strings(self):
        """Test that all values are search strings."""
        common = get_common_molecules()
        for display_name, search_name in common.items():
            assert isinstance(display_name, str)
            assert isinstance(search_name, str)
            assert len(search_name) > 0


class TestCheckPubChemAvailability:
    """Test PubChem connectivity check."""

    @patch("quantui.pubchem.requests.get")
    def test_check_available(self, mock_get):
        """Test successful connectivity check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = check_pubchem_availability()
        assert result is True

    @patch("quantui.pubchem.requests.get")
    def test_check_unavailable(self, mock_get):
        """Test failed connectivity check."""
        mock_get.side_effect = requests.RequestException("Connection failed")

        result = check_pubchem_availability()
        assert result is False

    @patch("quantui.pubchem.requests.get")
    @patch("quantui.pubchem._throttle")
    def test_check_uses_shared_throttle(self, mock_throttle, mock_get):
        """M10 audit fix (2026-07-14): must go through the shared
        client-side rate limiter like every other PubChem call, not call
        requests.get() directly. A burst of concurrent availability
        checks (e.g. several students clicking "check connection" at
        once) could otherwise exceed PubChem's server-side throttle.
        """
        mock_get.return_value = Mock(status_code=200)
        check_pubchem_availability()
        assert mock_throttle.called

    @patch("quantui.pubchem.requests.get")
    def test_check_uses_configured_availability_timeout(self, mock_get):
        """M10 audit fix: must use config.PUBCHEM_AVAILABILITY_TIMEOUT_S,
        not a hardcoded value the config constant can't actually control.
        """
        from quantui import config

        mock_get.return_value = Mock(status_code=200)
        check_pubchem_availability()
        _, kwargs = mock_get.call_args
        assert kwargs.get("timeout") == config.PUBCHEM_AVAILABILITY_TIMEOUT_S


class TestHttpGetMaxRetriesGuard:
    """M10 audit fix (2026-07-14): _http_get must never return None.

    Regression: `for attempt in range(config.PUBCHEM_MAX_RETRIES):` iterated
    zero times when PUBCHEM_MAX_RETRIES was 0 (or negative), leaving the
    ``response`` initializer (None) as the return value — every caller
    then hit AttributeError on response.status_code / .raise_for_status().
    """

    @patch("quantui.pubchem.requests.get")
    def test_zero_max_retries_still_attempts_once(self, mock_get):
        from quantui import config
        from quantui.pubchem import _http_get

        mock_get.return_value = Mock(status_code=200)
        with patch.object(config, "PUBCHEM_MAX_RETRIES", 0):
            result = _http_get("http://example.com/probe")
        assert result is not None
        assert mock_get.call_count == 1

    @patch("quantui.pubchem.requests.get")
    def test_negative_max_retries_still_attempts_once(self, mock_get):
        from quantui import config
        from quantui.pubchem import _http_get

        mock_get.return_value = Mock(status_code=200)
        with patch.object(config, "PUBCHEM_MAX_RETRIES", -1):
            result = _http_get("http://example.com/probe")
        assert result is not None
        assert mock_get.call_count == 1


# ============================================================================
# Integration Tests (Require Network) - Marked and Skipped by Default
# ============================================================================


@pytest.mark.network
@pytest.mark.integration
@pytest.mark.slow
class TestPubChemIntegration:
    """
    Integration tests with real PubChem API.

    Run with: pytest -m network
    Skipped by default to avoid network dependencies.
    """

    def test_real_search_water(self):
        """Test real API search for water."""
        try:
            cid = search_molecule_by_name("water")
            assert cid == 962
        except PubChemAPIError:
            pytest.skip("PubChem not accessible")

    @rdkit_only
    def test_real_fetch_caffeine(self):
        """Test real fetch of caffeine molecule."""
        try:
            xyz_string, metadata, cid = fetch_molecule("caffeine")

            assert cid > 0
            assert xyz_string is not None
            assert len(xyz_string) > 0
            assert "formula" in metadata
            assert "C" in metadata["formula"]  # Caffeine contains carbon
        except PubChemAPIError:
            pytest.skip("PubChem not accessible")

    def test_real_student_friendly_glucose(self):
        """Test student-friendly fetch of glucose."""
        try:
            xyz_string, message = student_friendly_fetch("glucose")

            assert xyz_string is not None
            assert "Found 'glucose' in PubChem" in message
        except Exception:
            pytest.skip("PubChem not accessible")


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test proper error handling and exceptions."""

    @patch("quantui.pubchem.requests.get")
    def test_timeout_handling(self, mock_get):
        """Test timeout error handling."""
        mock_get.side_effect = requests.Timeout("Request timed out")

        with pytest.raises(PubChemAPIError):
            search_molecule_by_name("water")

    @patch("quantui.pubchem.requests.get")
    def test_http_error_handling(self, mock_get):
        """Test HTTP error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Error")
        mock_get.return_value = mock_response

        with pytest.raises(PubChemAPIError):
            search_molecule_by_name("water")

    def test_sdf_conversion_no_rdkit(self):
        """Test SDF conversion failure handling.

        Raises ValueError for invalid SDF when RDKit is available;
        raises ImportError when RDKit is not installed.
        """
        invalid_sdf = "INVALID SDF CONTENT"

        with pytest.raises((ValueError, ImportError)):
            sdf_to_xyz(invalid_sdf)


# ============================================================================
# Caching Tests
# ============================================================================


class TestCaching:
    """Test LRU caching of SDF retrieval."""

    @patch("quantui.pubchem.requests.get")
    def test_sdf_caching(self, mock_get, sample_sdf_water):
        """Test that repeated SDF requests are cached."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = sample_sdf_water
        mock_get.return_value = mock_response

        # Clear cache first
        get_molecule_sdf.cache_clear()

        # First call - should hit the API
        sdf1 = get_molecule_sdf(962, conformer_3d=True)
        assert mock_get.call_count == 1

        # Second call with same parameters - should use cache
        sdf2 = get_molecule_sdf(962, conformer_3d=True)
        assert mock_get.call_count == 1  # Still 1, not 2

        assert sdf1 == sdf2

        # Different parameters - should hit API again
        _ = get_molecule_sdf(962, conformer_3d=False)
        assert mock_get.call_count == 2


# ============================================================================
# generate_2d_structure_svg — XYZ input path (H4 audit fix, 2026-07-14)
# ============================================================================


class TestGenerate2DStructureSvgFromXyz:
    """The xyz_string= branch used to always return None.

    Regression: it called AddAtom on an immutable Chem.Mol() (only RWMol
    supports AddAtom), so every call raised AttributeError internally —
    silently swallowed by the function's broad except-and-return-None —
    and a stray Chem.Conformer() with no atom count plus an atom-index
    that didn't skip malformed lines the same way could desync positions
    from atoms even if construction had worked.
    """

    @rdkit_only
    def test_returns_svg_for_valid_xyz(self):
        from quantui.pubchem import generate_2d_structure_svg

        xyz = "3\nWater\nO 0.0 0.0 0.0\nH 0.757 0.587 0.0\nH -0.757 0.587 0.0\n"
        svg = generate_2d_structure_svg(xyz_string=xyz)
        assert svg is not None
        assert "<svg" in svg

    @rdkit_only
    def test_skips_malformed_lines_without_desyncing_positions(self):
        # A blank/short line interleaved with valid atom lines must not
        # shift which coordinate lands on which atom.
        from quantui.pubchem import generate_2d_structure_svg

        xyz = (
            "3\nWater\n"
            "O 0.0 0.0 0.0\n"
            "\n"  # malformed / skippable line
            "H 0.757 0.587 0.0\n"
            "H -0.757 0.587 0.0\n"
        )
        svg = generate_2d_structure_svg(xyz_string=xyz)
        assert svg is not None
        assert "<svg" in svg

    @rdkit_only
    def test_invalid_xyz_returns_none_not_raises(self):
        from quantui.pubchem import generate_2d_structure_svg

        svg = generate_2d_structure_svg(xyz_string="not\nxyz")
        assert svg is None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not network"])
