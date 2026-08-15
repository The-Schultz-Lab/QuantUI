"""Density fitting (resolution of the identity, RI) opt-in for the SCF path.

Density fitting approximates the expensive four-centre two-electron integrals
with three-centre ones via an auxiliary ("fitting") basis. In PySCF it is a
single call: ``mf = mf.density_fit()``.

It is **not** a blanket win. Measured on aspirin (198 basis functions),
B3LYP/6-31G*: TD-DFT was ~1.6x faster, but a small SCF was slightly *slower*
(building the auxiliary integrals costs more than it saves at that size), while
the total-energy change was ~0.008 kcal/mol — far below chemical accuracy. So
DF is a *targeted* speedup, exposed as an opt-in that is **off by default**; the
per-calc-type default is only set once the size crossover has been measured
(roadmap M-DF, DF.2). This module is the plumbing that makes it selectable.

Mirrors :mod:`quantui.gpu_offload`: a single ``try_density_fit`` helper that
never raises, plus a settings gate. It must be applied to the freshly built
``mf`` (after the xc/D3 setup) **before** any PCM/solvent wrap and **before**
``gpu_offload.try_to_gpu`` — gpu4pyscf manages its own density fitting, and
``mf.to_gpu()`` can reject an already-fitted or PCM-wrapped object.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def _density_fit_enabled_in_settings() -> bool:
    """Return the user's density-fitting preference.

    Defaults to ``False`` (exact four-centre integrals, no approximation) on any
    read failure — the conservative choice for a teaching tool, and the opposite
    of the GPU gate, which fails *open* so a settings glitch never blocks
    hardware. Here a settings glitch must never silently switch on an
    approximation.
    """
    try:
        from .user_settings import UserSettings

        return bool(UserSettings.load().compute.density_fit)
    except Exception:  # pragma: no cover - defensive; never block a calc
        return False


def try_density_fit(
    mf: Any,
    *,
    enabled: Optional[bool] = None,
    auxbasis: Optional[str] = None,
) -> Tuple[Any, bool]:
    """Apply ``mf.density_fit()`` when density fitting is enabled.

    Parameters
    ----------
    mf:
        A freshly constructed PySCF mean-field object, already carrying its xc
        functional / D3 wrap but **not** yet PCM-wrapped or moved to GPU.
    enabled:
        Tri-state. ``None`` (default) reads the user's ``compute.density_fit``
        setting; pass an explicit bool to override (used by callers that already
        hold the preference, and by tests).
    auxbasis:
        Optional auxiliary basis name. ``None`` lets PySCF choose its default
        fitting set (DF.3 will revisit whether a matched auxbasis is better).

    Returns
    -------
    (mf, used_df):
        The possibly-fitted mean-field object and whether fitting was applied.
        Never raises: on any failure the original ``mf`` is returned unchanged
        with ``used_df=False`` — a checkpoint-style invariant that density
        fitting must never break a calculation.
    """
    if enabled is None:
        enabled = _density_fit_enabled_in_settings()
    if not enabled:
        return (mf, False)
    try:
        mf_df = mf.density_fit(auxbasis=auxbasis) if auxbasis else mf.density_fit()
        return (mf_df, True)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "mf.density_fit() failed, falling back to exact four-centre "
            "integrals: %s",
            exc,
        )
        return (mf, False)
