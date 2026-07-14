"""
Tests for QuantUI Utils Module

Tests utility functions for username detection, file operations,
validation, formatting, and error handling.

Removed vs. source repo (SLURM-specific helpers no longer present):
  - TestGenerateJobID (generate_job_id removed)
  - TestValidateSLURMCommand (validate_slurm_command removed)
  - TestFormatWalltime (format_walltime removed)
  - TestParseJobID (parse_slurm_job_id removed)
  - TestFormatMemory (format_memory removed)
  - TestParseJobTimestamp (parse_job_timestamp removed)
  - TestUtilsIntegration.test_create_user_directories (dirs removed from config)
  - TestUtilsIntegration.test_generate_and_parse_job_workflow (generate_job_id removed)
"""

import os
from unittest.mock import patch

import pytest

from quantui import utils

# ============================================================================
# Username and Path Tests
# ============================================================================


class TestGetUsername:
    """Test username detection."""

    @patch.dict(os.environ, {"JUPYTERHUB_USER": "student01"})
    def test_get_username_jupyterhub(self):
        """Test username from JUPYTERHUB_USER."""
        username = utils.get_username()
        assert username == "student01"

    @patch.dict(os.environ, {"USER": "johndoe", "JUPYTERHUB_USER": ""})
    def test_get_username_unix(self):
        """Test username from USER (Unix)."""
        username = utils.get_username()
        assert username == "johndoe"

    @patch.dict(os.environ, {"USERNAME": "janedoe", "USER": "", "JUPYTERHUB_USER": ""})
    def test_get_username_windows(self):
        """Test username from USERNAME (Windows)."""
        username = utils.get_username()
        assert username == "janedoe"

    @patch.dict(
        os.environ, {"USER": "", "USERNAME": "", "JUPYTERHUB_USER": ""}, clear=True
    )
    def test_get_username_none_found(self):
        """Test error when no username can be detected."""
        with pytest.raises(RuntimeError, match="Could not detect username"):
            utils.get_username()

    @patch.dict(os.environ, {"USER": "test user"})
    def test_get_username_sanitized(self):
        """Test that username is sanitized."""
        username = utils.get_username()
        # Space should be replaced with underscore
        assert username == "test_user"


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_sanitize_simple(self):
        """Test sanitizing simple filename."""
        result = utils.sanitize_filename("simple_file")
        assert result == "simple_file"

    def test_sanitize_spaces(self):
        """Test replacing spaces with underscores."""
        result = utils.sanitize_filename("my file name")
        assert result == "my_file_name"

    def test_sanitize_special_chars(self):
        """Test removing special characters."""
        result = utils.sanitize_filename("file@#$%name!.txt")
        assert result == "filename.txt"

    def test_sanitize_mixed(self):
        """Test sanitizing complex filename."""
        result = utils.sanitize_filename("Test File (2023)@v2.dat")
        # Spaces -> underscores, special chars removed except dot/dash/underscore
        assert "Test_File" in result
        assert "@" not in result
        assert "(" not in result


class TestEnsureDirectory:
    """Test directory creation."""

    def test_ensure_directory_creates(self, tmp_path):
        """Test that directory is created if it doesn't exist."""
        test_dir = tmp_path / "new_directory"
        assert not test_dir.exists()

        result = utils.ensure_directory(test_dir)

        assert test_dir.exists()
        assert test_dir.is_dir()
        assert result == test_dir

    def test_ensure_directory_existing(self, tmp_path):
        """Test that existing directory is not affected."""
        test_dir = tmp_path / "existing_directory"
        test_dir.mkdir()
        assert test_dir.exists()

        result = utils.ensure_directory(test_dir)

        assert test_dir.exists()
        assert result == test_dir

    def test_ensure_directory_nested(self, tmp_path):
        """Test creating nested directories."""
        test_dir = tmp_path / "level1" / "level2" / "level3"
        assert not test_dir.exists()

        result = utils.ensure_directory(test_dir)

        assert test_dir.exists()
        assert result == test_dir


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidateAtomSymbol:
    """Test atomic symbol validation."""

    def test_validate_common_atoms(self):
        """Test validation of common elements."""
        assert utils.validate_atom_symbol("H") is True
        assert utils.validate_atom_symbol("C") is True
        assert utils.validate_atom_symbol("N") is True
        assert utils.validate_atom_symbol("O") is True

    def test_validate_transition_metals(self):
        """Test validation of transition metals."""
        assert utils.validate_atom_symbol("Fe") is True
        assert utils.validate_atom_symbol("Cu") is True
        assert utils.validate_atom_symbol("Zn") is True

    def test_validate_invalid_symbols(self):
        """Test rejection of invalid symbols."""
        assert utils.validate_atom_symbol("X") is False
        assert utils.validate_atom_symbol("Xx") is False
        assert utils.validate_atom_symbol("123") is False
        assert utils.validate_atom_symbol("") is False

    def test_validate_with_whitespace(self):
        """Test validation strips whitespace."""
        assert utils.validate_atom_symbol(" H ") is True
        assert utils.validate_atom_symbol("\tC\n") is True


class TestValidateAtomSymbolExtendedElements:
    """M1 audit fix (2026-07-14): VALID_ATOMS covers the full periodic table.

    Regression: VALID_ATOMS previously stopped at Kr (Z=36), rejecting real
    structures resolved via PubChem/CACTUS/SMILES for any heavier element
    (iodine in thyroxine, tin/antimony in organometallics, gold/platinum
    complexes, ...) even though molecule.py's own error text listed iodine
    as a valid example.
    """

    def test_iodine_valid(self):
        assert utils.validate_atom_symbol("I") is True

    def test_heavy_elements_valid(self):
        for sym in ("Sn", "Sb", "Ag", "Au", "Pt", "Pb", "Bi", "Hg", "Xe"):
            assert utils.validate_atom_symbol(sym) is True, sym

    def test_full_periodic_table_length(self):
        from quantui import config

        assert len(config.VALID_ATOMS) == 118

    def test_valid_atoms_derived_from_atomic_numbers(self):
        """VALID_ATOMS and ATOMIC_NUMBERS must never be able to drift apart."""
        from quantui import config

        assert config.VALID_ATOMS == list(config.ATOMIC_NUMBERS.keys())


class TestValidateCoordinates:
    """Test coordinate validation."""

    def test_validate_valid_coordinates(self):
        """Test validation of valid coordinates."""
        assert utils.validate_coordinates([0.0, 0.0, 0.0]) is True
        assert utils.validate_coordinates([1.5, -2.3, 4.7]) is True

    def test_validate_integers(self):
        """Test that integers are accepted."""
        assert utils.validate_coordinates([0, 1, 2]) is True
        assert utils.validate_coordinates([1, -1, 0]) is True

    def test_validate_wrong_length(self):
        """Test rejection of wrong number of coordinates."""
        assert utils.validate_coordinates([0.0, 0.0]) is False  # Only 2
        assert utils.validate_coordinates([0.0, 0.0, 0.0, 0.0]) is False  # 4

    def test_validate_non_numeric(self):
        """Test rejection of non-numeric coordinates."""
        assert utils.validate_coordinates([0.0, "abc", 0.0]) is False
        assert utils.validate_coordinates(["x", "y", "z"]) is False

    def test_validate_wrong_type(self):
        """Test rejection of wrong data types."""
        assert utils.validate_coordinates("0.0 0.0 0.0") is False
        assert utils.validate_coordinates(123) is False
        assert utils.validate_coordinates(None) is False


class TestValidateCharge:
    """Test charge validation."""

    def test_validate_valid_charges(self):
        """Test validation of valid charges."""
        assert utils.validate_charge(0) is True
        assert utils.validate_charge(1) is True
        assert utils.validate_charge(-1) is True
        assert utils.validate_charge(5) is True

    def test_validate_extreme_charges(self):
        """Test rejection of unreasonable charges."""
        assert utils.validate_charge(100) is False
        assert utils.validate_charge(-100) is False

    def test_validate_boundary_charges(self):
        """Test boundary values."""
        assert utils.validate_charge(10) is True
        assert utils.validate_charge(-10) is True
        assert utils.validate_charge(11) is False
        assert utils.validate_charge(-11) is False

    def test_validate_non_integer(self):
        """Test rejection of non-integer charges."""
        assert utils.validate_charge(1.5) is False
        assert utils.validate_charge("1") is False


class TestValidateMultiplicity:
    """Test multiplicity validation."""

    def test_validate_valid_multiplicities(self):
        """Test validation of valid multiplicities."""
        assert utils.validate_multiplicity(1) is True  # Singlet
        assert utils.validate_multiplicity(2) is True  # Doublet
        assert utils.validate_multiplicity(3) is True  # Triplet

    def test_validate_zero_or_negative(self):
        """Test rejection of zero or negative multiplicities."""
        assert utils.validate_multiplicity(0) is False
        assert utils.validate_multiplicity(-1) is False

    def test_validate_too_large(self):
        """Test rejection of unreasonably large multiplicities."""
        assert utils.validate_multiplicity(11) is False
        assert utils.validate_multiplicity(100) is False

    def test_validate_boundary(self):
        """Test boundary value."""
        assert utils.validate_multiplicity(10) is True


# ============================================================================
# Formatting Tests
# ============================================================================


class TestFormatFileSize:
    """Test file size formatting."""

    def test_format_bytes(self):
        """Test formatting bytes."""
        assert utils.format_file_size(512) == "512.0 B"

    def test_format_kilobytes(self):
        """Test formatting kilobytes."""
        assert utils.format_file_size(1024) == "1.0 KB"
        assert utils.format_file_size(2048) == "2.0 KB"

    def test_format_megabytes(self):
        """Test formatting megabytes."""
        assert utils.format_file_size(1024 * 1024) == "1.0 MB"
        assert utils.format_file_size(5 * 1024 * 1024) == "5.0 MB"

    def test_format_gigabytes(self):
        """Test formatting gigabytes."""
        assert utils.format_file_size(1024 * 1024 * 1024) == "1.0 GB"


class TestTruncateString:
    """Test string truncation."""

    def test_truncate_short_string(self):
        """Test that short strings are not truncated."""
        text = "Short text"
        result = utils.truncate_string(text, max_length=100)
        assert result == text

    def test_truncate_long_string(self):
        """Test truncating long strings."""
        text = "A" * 200
        result = utils.truncate_string(text, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_truncate_custom_length(self):
        """Test truncating with custom length."""
        text = "A" * 100
        result = utils.truncate_string(text, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestStudentFriendlyError:
    """Test student-friendly error message generation."""

    def test_command_not_found(self):
        """Test error message for missing commands."""
        error = Exception("command not found: sbatch")
        message = utils.student_friendly_error(error, "submitting job")

        assert "System Error" in message
        assert "Required software not found" in message
        assert "submitting job" in message

    def test_permission_denied(self):
        """Test error message for permission errors."""
        error = Exception("Permission denied")
        message = utils.student_friendly_error(error, "writing file")

        assert "Permission Error" in message
        assert "don't have access" in message

    def test_connection_error(self):
        """Test error message for connection errors."""
        error = Exception("Connection timeout")
        message = utils.student_friendly_error(error, "connecting to cluster")

        assert "Connection Error" in message
        assert "network" in message.lower()

    def test_generic_error(self):
        """Test generic error message."""
        error = Exception("Something unexpected happened")
        message = utils.student_friendly_error(error, "processing data")

        assert "Error" in message
        assert "processing data" in message
        assert "Something unexpected happened" in message


# ============================================================================
# Timestamp Tests
# ============================================================================


class TestGetTimestamp:
    """Test timestamp generation."""

    def test_get_timestamp_format(self):
        """Test that timestamp is in ISO format."""
        timestamp = utils.get_timestamp()

        # Should be ISO format: YYYY-MM-DDTHH:MM:SS.ffffff
        assert "T" in timestamp
        assert "-" in timestamp
        assert ":" in timestamp

    def test_get_timestamp_unique(self):
        """Test that timestamps are unique (or very close)."""
        ts1 = utils.get_timestamp()
        import time

        time.sleep(0.01)  # Small delay
        ts2 = utils.get_timestamp()

        # Should be different (or at least not fail comparison)
        assert isinstance(ts1, str)
        assert isinstance(ts2, str)


# ============================================================================
# Session Resource Tests
# ============================================================================


class TestSessionCanHandle:
    """Tests for session_can_handle and get_session_resources."""

    def test_false_when_pyscf_unavailable(self):
        result = utils.session_can_handle(1, 1, pyscf_available=False)
        assert result is False

    def test_false_when_cores_exceed_available(self, monkeypatch):
        monkeypatch.setattr(utils, "get_session_resources", lambda: (2, 16))
        result = utils.session_can_handle(8, 1, pyscf_available=True)
        assert result is False

    def test_false_when_memory_exceeds_available(self, monkeypatch):
        monkeypatch.setattr(utils, "get_session_resources", lambda: (8, 4))
        result = utils.session_can_handle(1, 32, pyscf_available=True)
        assert result is False

    def test_true_when_within_limits(self, monkeypatch):
        monkeypatch.setattr(utils, "get_session_resources", lambda: (8, 16))
        result = utils.session_can_handle(2, 4, pyscf_available=True)
        assert result is True

    def test_true_when_memory_unknown(self, monkeypatch):
        """When psutil is missing, memory constraint is skipped."""
        monkeypatch.setattr(utils, "get_session_resources", lambda: (8, None))
        result = utils.session_can_handle(2, 999, pyscf_available=True)
        assert result is True

    def test_get_session_resources_returns_tuple(self):
        cores, mem = utils.get_session_resources()
        assert isinstance(cores, int)
        assert cores >= 1
        assert mem is None or (isinstance(mem, int) and mem >= 0)


# ============================================================================
# Integration Tests
# ============================================================================


class TestUtilsIntegration:
    """Test combined utility function workflows."""

    def test_validation_chain(self):
        """Test validating multiple components."""
        # Valid molecule components
        assert utils.validate_atom_symbol("H") is True
        assert utils.validate_coordinates([0.0, 0.0, 0.0]) is True
        assert utils.validate_charge(0) is True
        assert utils.validate_multiplicity(1) is True

        # Invalid components
        assert utils.validate_atom_symbol("X") is False
        assert utils.validate_coordinates([0.0, 0.0]) is False
        assert utils.validate_charge(100) is False
        assert utils.validate_multiplicity(0) is False
