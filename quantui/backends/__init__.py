"""
Execution backends for QuantUI: local in-kernel runs and SLURM batch jobs.

The shared request / progress / result types are in :mod:`quantui.backends.base`.
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
