"""
SLURM error translator for student-friendly error messages.

Salvaged from the legacy QuantUI archive (``quantui/slurm_errors.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ErrorTranslation:
    """A matched error with student-friendly explanation."""

    category: str
    summary: str
    advice: str
    raw_excerpt: str


_PATTERNS: List[Tuple[str, str, str, str]] = [
    (
        r"QOSMaxSubmitJobPerUserLimit",
        "Job Limit Reached",
        "You've reached the maximum number of jobs you can have in the queue.",
        "Wait for some of your current jobs to finish, or cancel jobs you no longer need.",
    ),
    (
        r"Invalid account|InvalidAccount",
        "Account Problem",
        "Your cluster account is not properly configured.",
        "Contact your instructor to make sure your account is activated on the cluster.",
    ),
    (
        r"ReqNodeNotAvail",
        "Resources Unavailable",
        "The compute nodes you requested are not currently available.",
        "Try again later, or reduce the number of cores or memory you requested.",
    ),
    (
        r"DependencyNeverSatisfied",
        "Job Dependency Failed",
        "This job was waiting for another job that failed or was cancelled.",
        "Check the status of the earlier job. You may need to resubmit both jobs.",
    ),
    (
        r"NodeDown|NodeDrained",
        "Cluster Issue",
        "A compute node is down for maintenance.",
        "This is a cluster-level issue — please report it to your instructor.",
    ),
    (
        r"PartitionNotAvail|Invalid partition",
        "Partition Problem",
        "The requested SLURM partition (queue) is not available.",
        "Check that the partition name is correct. Ask your instructor which partitions "
        "are available to you.",
    ),
    (
        r"Exceeded.*memory|oom-kill|Out of memory|OutOfMemory",
        "Out of Memory",
        "Your calculation ran out of memory and was killed by the system.",
        "Try increasing the memory allocation, or use a smaller basis set (e.g. STO-3G "
        "instead of cc-pVTZ) to reduce memory requirements.",
    ),
    (
        r"TIME LIMIT|TimeLimit|TIMEOUT|DUE TO TIME LIMIT",
        "Time Limit Exceeded",
        "Your job ran out of time before finishing.",
        "Request more walltime, or simplify the calculation (smaller basis set, "
        "fewer atoms) so it runs faster.",
    ),
    (
        r"CANCELLED",
        "Job Cancelled",
        "This job was cancelled.",
        "If you didn't cancel it yourself, it may have been cancelled by the system "
        "due to resource limits. Check with your instructor if this keeps happening.",
    ),
    (
        r"SCF not converged",
        "Calculation Did Not Converge",
        "The self-consistent field (SCF) calculation could not find a stable solution.",
        "This can happen with tricky molecules. Try:\n"
        "  1. Use a smaller basis set (STO-3G) first\n"
        "  2. Double-check the charge and multiplicity\n"
        "  3. Pre-optimize the geometry before submitting",
    ),
    (
        r"Electron number.*not.*consistent|electron count",
        "Electron Count Mismatch",
        "The charge and multiplicity you entered don't match the number of electrons.",
        "Go back to the molecule input and check that the charge and multiplicity are "
        "correct.",
    ),
    (
        r"basis.*not found|BasisNotFoundError",
        "Basis Set Not Found",
        "PySCF doesn't have the requested basis set for one of your elements.",
        "Try a different basis set. STO-3G and 6-31G work for most common elements.",
    ),
    (
        r"LinearDependency|linear dependency",
        "Linear Dependency Warning",
        "The basis set has near-linear-dependent functions. Results may be unreliable.",
        "Try a smaller basis set, or check that your molecular geometry is reasonable.",
    ),
    (
        r"No such file or directory",
        "File Not Found",
        "A required file could not be found.",
        "This usually means the calculation directory was deleted or moved. "
        "Try resubmitting the job.",
    ),
    (
        r"Permission denied",
        "Permission Denied",
        "You don't have permission to access a file or directory.",
        "Make sure you're running the calculation in your own directory. "
        "Contact your instructor if the problem persists.",
    ),
    (
        r"Connection refused|Connection timed out",
        "Connection Problem",
        "Could not connect to the cluster.",
        "Check your network connection. The cluster may be down for maintenance.",
    ),
]

_COMPILED_PATTERNS: List[Tuple[re.Pattern[str], str, str, str]] = [
    (re.compile(pat, re.IGNORECASE), cat, summ, adv)
    for pat, cat, summ, adv in _PATTERNS
]


def translate_slurm_error(stderr: str) -> Optional[ErrorTranslation]:
    if not stderr or not stderr.strip():
        return None

    for regex, category, summary, advice in _COMPILED_PATTERNS:
        match = regex.search(stderr)
        if match:
            start = max(0, match.start() - 40)
            end = min(len(stderr), match.end() + 40)
            excerpt = stderr[start:end].strip()
            return ErrorTranslation(
                category=category,
                summary=summary,
                advice=advice,
                raw_excerpt=excerpt,
            )

    return None


def format_error_for_student(stderr: str) -> str:
    if not stderr or not stderr.strip():
        return ""

    translation = translate_slurm_error(stderr)
    if translation is not None:
        lines = [
            f"❌ {translation.category}: {translation.summary}",
            "",
            f"💡 What to do: {translation.advice}",
            "",
            f"Technical details: {translation.raw_excerpt}",
        ]
        return "\n".join(lines)

    truncated = stderr.strip()[:500]
    return f"❌ Error encountered:\n\n{truncated}"


def format_error_html(stderr: str) -> str:
    if not stderr or not stderr.strip():
        return ""

    translation = translate_slurm_error(stderr)
    if translation is not None:
        return (
            '<div style="border:1px solid #d32f2f; border-radius:6px; '
            'padding:10px 14px; margin:6px 0; background:#fff5f5; max-width:620px;">'
            f'<div style="font-weight:bold; color:#d32f2f; font-size:14px;">'
            f"❌ {translation.category}: {translation.summary}</div>"
            f'<div style="margin-top:8px; font-size:13px;">'
            f"<b>💡 What to do:</b> {translation.advice}</div>"
            f'<details style="margin-top:8px; font-size:12px; color:#666;">'
            f"<summary>Technical details</summary>"
            f"<pre style='white-space:pre-wrap; margin:4px 0;'>{translation.raw_excerpt}</pre>"
            f"</details></div>"
        )

    truncated = stderr.strip()[:500]
    return (
        '<div style="border:1px solid #d32f2f; border-radius:6px; '
        'padding:10px 14px; margin:6px 0; background:#fff5f5; max-width:620px;">'
        '<div style="font-weight:bold; color:#d32f2f; font-size:14px;">'
        "❌ Error encountered</div>"
        f'<pre style="white-space:pre-wrap; margin:6px 0; font-size:12px;">'
        f"{truncated}</pre></div>"
    )
