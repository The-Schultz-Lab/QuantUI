"""
Execution backends for QuantUI (local in-kernel and future SLURM batch).

Implements the v0.1 contract in ``QuantUI-development-tracking``:
``TODO/EXECUTION-BACKEND-CONTRACT.md``.
"""

from .base import (
    BackendCapabilities,
    CalculationRequest,
    CalculationResult,
    ProgressEvent,
)
from .local import LocalBackend
from .registry import JobRecord, JobRegistry
from .slurm import SlurmBackend
from .slurm_errors import (
    ErrorTranslation,
    format_error_for_student,
    format_error_html,
    translate_slurm_error,
)

__all__ = [
    "BackendCapabilities",
    "CalculationRequest",
    "CalculationResult",
    "ProgressEvent",
    "ErrorTranslation",
    "JobRecord",
    "JobRegistry",
    "LocalBackend",
    "SlurmBackend",
    "format_error_for_student",
    "format_error_html",
    "translate_slurm_error",
]
