"""
QuantUI Utilities Module

Helper functions for validation, session resource detection, and general
utilities used across the application. SLURM-specific helpers (job ID
parsing, walltime formatting, job directory management) have been removed.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from . import config

# M14 audit fix (2026-07-14): logging.basicConfig() configures the *root*
# logger process-wide the moment this module is imported (which
# quantui/__init__ always does) — a library must never do this, since it
# silently hijacks/duplicates whatever logging setup the host application
# or notebook already has. quantui/__init__.py already attaches a
# NullHandler to the package logger; host apps that want console output
# can call logging.basicConfig() themselves, optionally using
# config.LOG_LEVEL / config.LOG_FORMAT as defaults.
logger = logging.getLogger(__name__)


def get_username() -> str:
    """
    Detect the current username from environment variables.

    Tries multiple environment variables to handle different systems:
    - JUPYTERHUB_USER (JupyterHub)
    - USER (Linux/Mac)
    - USERNAME (Windows)

    Returns:
        str: The detected username

    Raises:
        RuntimeError: If username cannot be detected
    """
    username = (
        os.getenv("JUPYTERHUB_USER") or os.getenv("USER") or os.getenv("USERNAME")
    )

    if not username:
        raise RuntimeError(
            "Could not detect username. Please ensure you are running in a "
            "supported environment (JupyterHub, Linux, Mac, or Windows)."
        )

    username = sanitize_filename(username)
    logger.info(f"Detected username: {username}")
    return username


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string for safe use in filenames and paths.

    Args:
        filename: The string to sanitize

    Returns:
        str: Sanitized string safe for use in filenames
    """
    filename = filename.replace(" ", "_")
    filename = re.sub(r"[^\w\-.]", "", filename)
    return filename


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory

    Returns:
        Path: The path object (for chaining)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def validate_atom_symbol(symbol: str) -> bool:
    """
    Validate if a string is a valid atomic symbol.

    Args:
        symbol: Atomic symbol to validate (e.g., 'H', 'C', 'O')

    Returns:
        bool: True if valid, False otherwise
    """
    return symbol.strip() in config.VALID_ATOMS


def validate_coordinates(coords: List[float]) -> bool:
    """
    Validate that coordinates are a list of 3 numbers.

    Args:
        coords: List of coordinate values

    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(coords, (list, tuple)):
        return False
    if len(coords) != 3:
        return False
    try:
        [float(x) for x in coords]
        return True
    except (ValueError, TypeError):
        return False


def validate_charge(charge: int) -> bool:
    """
    Validate molecular charge.

    Args:
        charge: Charge value

    Returns:
        bool: True if valid (reasonable range), False otherwise
    """
    if not isinstance(charge, int):
        return False
    return -10 <= charge <= 10


def validate_multiplicity(multiplicity: int) -> bool:
    """
    Validate spin multiplicity.

    Args:
        multiplicity: Multiplicity value (2S+1)

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        multiplicity = int(multiplicity)
        return 1 <= multiplicity <= 10
    except (ValueError, TypeError):
        return False


def student_friendly_error(error: Exception, context: str = "") -> str:
    """
    Convert technical errors into student-friendly messages.

    Args:
        error: The exception that occurred
        context: Additional context about what was being attempted

    Returns:
        str: User-friendly error message
    """
    error_str = str(error).lower()

    if "command not found" in error_str or "no such file" in error_str:
        return (
            f"System Error: Required software not found. "
            f"Please contact your instructor.\n"
            f"Technical details: {context}"
        )

    if "permission denied" in error_str:
        return (
            f"Permission Error: You don't have access to perform this operation. "
            f"Please contact your instructor.\n"
            f"Technical details: {context}"
        )

    if "connection" in error_str or "timeout" in error_str:
        return (
            f"Connection Error: Cannot reach the requested service. "
            f"Please check your network connection or try again later.\n"
            f"Technical details: {context}"
        )

    return (
        f"Error: Something went wrong while {context}. "
        f"Please try again or contact your instructor if the problem persists.\n"
        f"Technical details: {str(error)}"
    )


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        str: Formatted size (e.g., "1.5 MB")
    """
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        str: ISO format timestamp
    """
    return datetime.now().isoformat()


def truncate_string(s: str, max_length: int = 100) -> str:
    """
    Truncate a string to maximum length with ellipsis.

    Args:
        s: String to truncate
        max_length: Maximum length

    Returns:
        str: Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - 3] + "..."


def get_session_resources() -> Tuple[int, Optional[int]]:
    """
    Detect available CPU cores and memory in the current process environment.

    Returns:
        Tuple of (available_cores, available_memory_gb).
        available_memory_gb is None when psutil is not installed.
    """
    try:
        # sched_getaffinity respects cgroup/container CPU limits (Linux only)
        available_cores = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError, NotImplementedError):
        # Windows / macOS fallback, or restricted container without sched_getaffinity
        available_cores = os.cpu_count() or 1
    try:
        import psutil

        mem_gb: Optional[int] = psutil.virtual_memory().available // (1024**3)
    except ImportError:
        mem_gb = None
    return available_cores, mem_gb


def session_can_handle(
    estimated_cores: int,
    estimated_memory_gb: int,
    pyscf_available: bool = True,
) -> bool:
    """
    Return True if the current session can likely run a calculation locally.

    Args:
        estimated_cores: Number of cores the job needs.
        estimated_memory_gb: Memory the job needs in GB.
        pyscf_available: Whether PySCF is importable in this environment.

    Returns:
        True if local execution looks feasible, False otherwise.
    """
    if not pyscf_available:
        return False
    cores, mem_gb = get_session_resources()
    if estimated_cores > cores:
        return False
    if mem_gb is not None and estimated_memory_gb > mem_gb:
        return False
    return True
