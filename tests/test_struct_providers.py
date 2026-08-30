"""Tests for M-STRUCT STRUCT.3 (NCI CACTUS) + STRUCT.4 (provider chain).

Platform-independent: network mocked, RDKit-only paths gated, no PySCF.
"""

from unittest.mock import Mock, patch

import pytest
import requests

from quantui import cactus
from quantui.pubchem import RDKIT_AVAILABLE, MoleculeNotFoundError, PubChemAPIError
from quantui.structure_providers import (
    ResolvedStructure,
    resolve_structure,
    student_friendly_resolve,
)

rdkit_only = pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit not installed")

_WATER_META = {
    "formula": "H2O",
    "num_atoms": 3,
    "num_heavy_atoms": 1,
    "charge": 0,
    "molecular_weight": 18.02,
    "conformer_origin": "pubchem",
    "source": "pubchem",
}
_WATER_XYZ = "3\nwater\nO 0 0 0\nH 1 0 0\nH 0 1 0"


# ============================================================================
# STRUCT.3 — CACTUS resolver
# ============================================================================


class TestCactus:
    def test_looks_like_sdf(self, sample_sdf_water):
        assert cactus._looks_like_sdf(sample_sdf_water) is True
        assert cactus._looks_like_sdf("<html>Page not found</html>") is False

    @patch("quantui.cactus.requests.get")
    def test_resolve_to_sdf_success(self, mock_get, sample_sdf_water):
        mock_get.return_value = Mock(status_code=200, text=sample_sdf_water)
        sdf = cactus.resolve_to_sdf("aspirin")
        assert "M  END" in sdf
        assert "compound" not in mock_get.call_args[0][0]  # uses CACTUS, not pubchem

    @patch("quantui.cactus.requests.get")
    def test_resolve_to_sdf_tries_3d_then_2d(self, mock_get, sample_sdf_water):
        # 3D endpoint returns an HTML error page; 2D endpoint returns real SDF.
        mock_get.side_effect = [
            Mock(status_code=200, text="<html>error</html>"),
            Mock(status_code=200, text=sample_sdf_water),
        ]
        sdf = cactus.resolve_to_sdf("weirdmol", conformer_3d=True)
        assert "M  END" in sdf
        assert mock_get.call_count == 2

    @patch("quantui.cactus.requests.get")
    def test_resolve_to_sdf_miss_raises_not_found(self, mock_get):
        mock_get.return_value = Mock(status_code=404, text="not found")
        with pytest.raises(MoleculeNotFoundError):
            cactus.resolve_to_sdf("zzznotreal")

    @patch("quantui.cactus.requests.get")
    def test_resolve_to_sdf_network_error_raises_api_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("down")
        with pytest.raises(PubChemAPIError) as exc_info:
            cactus.resolve_to_sdf("aspirin")
        # L audit fix (ruff B904): the original ConnectionError must be
        # chained via `raise ... from e`, not swallowed, so tracebacks show
        # the real root cause instead of just "during handling of ...".
        assert isinstance(exc_info.value.__cause__, requests.ConnectionError)

    @rdkit_only
    @patch("quantui.cactus.requests.get")
    def test_fetch_from_cactus_returns_xyz(self, mock_get, sample_sdf_water):
        mock_get.return_value = Mock(status_code=200, text=sample_sdf_water)
        xyz, meta = cactus.fetch_from_cactus("water")
        assert meta["source"] == "cactus"
        assert meta["num_atoms"] == 3

    @rdkit_only
    @patch("quantui.cactus.requests.get")
    def test_fetch_from_cactus_keeps_metal_complex_source_coords(
        self, mock_get, sample_sdf_metal_complex_2d
    ):
        """M-METAL MET.1: the CACTUS load path must not scatter a coordination complex.

        PubChem/CACTUS SDFs carry no metal-donor bond records, so RDKit's default
        re-embed + fragment separation would push ligands away from the metal.
        ``fetch_from_cactus`` → ``sdf_to_xyz`` must keep the source coordinates
        so downstream GFN-FF pre-opt can relax the flat layout into 3D.
        """
        mock_get.return_value = Mock(status_code=200, text=sample_sdf_metal_complex_2d)
        xyz, meta = cactus.fetch_from_cactus("cisplatin")
        assert meta["source"] == "cactus"
        assert meta["metal_detected"] is True
        assert meta["coords_embedded"] is False

        lines = xyz.strip().split("\n")[2:]
        coords = {}
        for line in lines:
            parts = line.split()
            symbol = parts[0]
            xyz_tuple = tuple(float(v) for v in parts[1:4])
            coords.setdefault(symbol, []).append(xyz_tuple)

        pt = coords["Pt"][0]
        for n in coords["N"]:
            dist = sum((a - b) ** 2 for a, b in zip(pt, n)) ** 0.5
            assert dist == pytest.approx(2.0, abs=1e-4)


# ============================================================================
# STRUCT.4 — provider chain ordering
# ============================================================================


class TestProviderChain:
    @rdkit_only
    @patch("quantui.pubchem.requests.get")
    def test_smiles_resolves_locally_no_network(self, mock_get):
        mock_get.side_effect = AssertionError("network must not be touched")
        result = resolve_structure("CC(=O)O")
        assert isinstance(result, ResolvedStructure)
        assert result.source == "rdkit-smiles"
        assert result.is_offline is True
        mock_get.assert_not_called()

    @patch("quantui.structure_providers.fetch_structure")
    def test_exact_library_hit_short_circuits_network(self, mock_fetch):
        mock_fetch.side_effect = AssertionError("library hit must not hit network")
        result = resolve_structure("H2O")  # exact MOLECULE_LIBRARY key
        assert result.source == "library"
        assert result.formula == "H2O"
        assert result.num_atoms == 3
        mock_fetch.assert_not_called()

    @patch("quantui.structure_providers.fetch_structure")
    def test_name_resolves_via_pubchem(self, mock_fetch):
        mock_fetch.return_value = (_WATER_XYZ, dict(_WATER_META))
        result = resolve_structure("dihydrogen monoxide")
        assert result.source == "pubchem"
        mock_fetch.assert_called_once()

    @patch("quantui.structure_providers.cactus.fetch_from_cactus")
    @patch("quantui.structure_providers.fetch_structure")
    def test_pubchem_miss_falls_through_to_cactus(self, mock_fetch, mock_cactus):
        mock_fetch.side_effect = MoleculeNotFoundError("pubchem miss")
        mock_cactus.return_value = (_WATER_XYZ, {**_WATER_META, "source": "cactus"})
        result = resolve_structure("64-19-7")  # a CAS number
        assert result.source == "cactus"
        mock_fetch.assert_called_once()
        mock_cactus.assert_called_once()

    @patch("quantui.structure_providers.cactus.fetch_from_cactus")
    @patch("quantui.structure_providers.fetch_structure")
    def test_network_down_uses_offline_fuzzy_fallback(self, mock_fetch, mock_cactus):
        mock_fetch.side_effect = PubChemAPIError("network down")
        mock_cactus.side_effect = PubChemAPIError("network down")
        # "water" isn't an exact key but appears in the H2O entry description.
        result = resolve_structure("water")
        assert result.source == "library-offline-fallback"
        assert result.formula == "H2O"

    @patch("quantui.structure_providers.cactus.fetch_from_cactus")
    @patch("quantui.structure_providers.fetch_structure")
    def test_total_miss_raises(self, mock_fetch, mock_cactus):
        mock_fetch.side_effect = MoleculeNotFoundError("miss")
        mock_cactus.side_effect = MoleculeNotFoundError("miss")
        with pytest.raises(MoleculeNotFoundError):
            resolve_structure("zzqxnotarealmolecule")

    @patch("quantui.structure_providers.fetch_structure")
    def test_allow_network_false_skips_resolvers(self, mock_fetch):
        mock_fetch.side_effect = AssertionError("network disabled")
        result = resolve_structure("water", allow_network=False)
        assert result.source == "library-offline-fallback"
        mock_fetch.assert_not_called()


# ============================================================================
# STRUCT.4 — friendly wrapper messaging
# ============================================================================


class TestFriendlyResolve:
    def test_library_message_mentions_offline(self):
        xyz, msg = student_friendly_resolve("H2O")
        assert xyz is not None
        assert "bundled library (offline)" in msg
        assert "H2O" in msg

    @patch("quantui.structure_providers.cactus.fetch_from_cactus")
    @patch("quantui.structure_providers.fetch_structure")
    def test_offline_fallback_message(self, mock_fetch, mock_cactus):
        mock_fetch.side_effect = PubChemAPIError("down")
        mock_cactus.side_effect = PubChemAPIError("down")
        xyz, msg = student_friendly_resolve("water")
        assert xyz is not None
        assert "offline fallback" in msg

    @patch("quantui.structure_providers.cactus.fetch_from_cactus")
    @patch("quantui.structure_providers.fetch_structure")
    def test_not_found_message(self, mock_fetch, mock_cactus):
        mock_fetch.side_effect = MoleculeNotFoundError("miss")
        mock_cactus.side_effect = MoleculeNotFoundError("miss")
        xyz, msg = student_friendly_resolve("zzqxnotreal")
        assert xyz is None
        assert "Could not resolve" in msg
