"""Shared SCF-convergence rescue helper (M-SCF-ROBUST SCFR.1/SCFR.2).

QuantUI used to build ``dft.RKS(mol)``/``dft.UKS(mol)`` (or ``scf.RHF``/
``scf.UHF``) and call bare ``mf.kernel()`` — no ``level_shift``, no
``init_guess`` override, nothing exposed anywhere — at nine independent call
sites across the package. Every calculation ran on PySCF's bare defaults,
with zero robustness net, on exactly the population most likely to need one:
open-shell transition-metal complexes.

This broke a real class deployment: CHEM-3200 Lab 2's Mn²⁺ hexaaquo
single-point (B3LYP/def2-SVP, sextet, high-spin d⁵) oscillated 40-60 Ha every
~9 cycles and never converged under plain defaults, even though the request
itself was correct. See ``QuantUI-development-tracking/TODO/GOTCHAS.md`` and
``TODO/roadmaps/52-m-scf-robust-open-shell-convergence-roadmap.md`` for the
full writeup and the two rescue strategies validated against that failure.

This module is the shared helper both roadmap docs call for: every SCF
``.kernel()`` call in the package should go through :func:`run_scf_with_rescue`
instead of calling ``mf.kernel()`` directly.

Two independent rescue stages, tried in order, only when the plain attempt
doesn't converge:

1. **Same-basis bootstrap** (primary). Converge UHF (RHF for closed-shell) at
   the *same* mol/basis first — cheap, usually easy — then hand its converged
   density to the target calculation as ``dm0``, replacing PySCF's default
   ``minao`` guess (or the caller's own ``dm0``, if one was passed and didn't
   help).
2. **Level-shift fallback.** ``level_shift=0.3, init_guess='atom',
   max_cycle=100`` when the bootstrap alone isn't enough.

Both rescues were validated against the real failing case (bare PySCF, no
QuantUI involved) and land on the identical correct energy that plain
defaults reach too, when they happen to converge — confirming this is a
genuinely marginal/unstable SCF case, not two different chemistries, and
that either rescue is safe to try automatically without changing results for
calculations that already converge.

A calculation whose first attempt converges is completely unaffected — the
rescue stages never run, so there is zero behavior change (same energy, same
iteration count) for the common case. This is the module's one hard
invariant; see SCFR.4's regression tests.
"""

from __future__ import annotations

import logging
from typing import IO, Any, Optional

logger = logging.getLogger(__name__)

# Provenance markers (SCFR.5) — stamped onto ``mf.scf_rescue_stage`` so a
# result that silently benefited from a rescue can be identified as such
# later, mirroring M-CHECKPOINT's ``[checkpoint]`` log-line convention. Set
# on *every* call, including the common converged-on-first-try case, so its
# absence can never be mistaken for "wasn't tried".
SCF_RESCUE_NONE = "none"  # first attempt converged (or rescue=False) — no-op
SCF_RESCUE_BOOTSTRAP = "bootstrap"  # same-basis HF/UHF density bootstrap rescued it
SCF_RESCUE_LEVEL_SHIFT = "level_shift"  # level-shift + init_guess='atom' fallback
SCF_RESCUE_FAILED = (
    "failed"  # both stages tried (or max_stage capped it), still not converged
)


def run_scf_with_rescue(
    mf: Any,
    dm0: Optional[Any] = None,
    *,
    rescue: bool = True,
    max_stage: int = 2,
    stream: Optional[IO[str]] = None,
    **kernel_kwargs: Any,
) -> float:
    """Run ``mf.kernel()``, automatically rescuing SCF non-convergence.

    Drop-in replacement for ``mf.kernel(dm0=dm0, **kernel_kwargs)`` (or
    ``mf.kernel(**kernel_kwargs)`` when ``dm0`` is ``None``) that retries a
    non-converged result through up to two validated rescue stages before
    giving up. ``mf.converged`` reflects whichever attempt this returns, and
    ``mf.scf_rescue_stage`` (one of the ``SCF_RESCUE_*`` constants above) is
    always set so the caller can log/report what happened.

    Args:
        mf: A PySCF SCF/KS mean-field object (RHF/UHF/RKS/UKS, or a
            solvent-wrapped one) that has not yet had ``.kernel()`` called.
            Mutated in place (rescue stages may set ``level_shift``,
            ``init_guess``, ``max_cycle`` on it) — same as calling
            ``mf.kernel()`` directly would let a caller do.
        dm0: Optional starting density matrix for the *first* attempt only
            (e.g. a checkpoint warm-start guess, M-CHECKPOINT CHK.1). If the
            first attempt is unstable, the bootstrap rescue below builds and
            uses its own density instead — a warm start that was itself part
            of the problem shouldn't be retried unchanged.
        rescue: When ``False``, behaves exactly like calling
            ``mf.kernel(...)`` directly — no rescue attempted, whatever
            ``mf.converged``/energy comes back is final. Lets a batch or
            reproducibility caller opt out (SCFR.3).
        max_stage: ``1`` = bootstrap only; ``2`` (default) = bootstrap, then
            the level-shift fallback if the bootstrap alone isn't enough.
        stream: Optional writable stream for status lines (the
            ``progress_stream`` convention already used across the calc
            modules). Rescue-stage messages go to ``logger`` always, and to
            *stream* too when given.
        **kernel_kwargs: Extra keyword arguments forwarded to every
            ``mf.kernel(...)`` call besides ``dm0``.

    Returns:
        The final SCF energy, exactly as ``mf.kernel(...)`` would have
        returned it.
    """

    def _status(msg: str) -> None:
        logger.info("[scf_rescue] %s", msg)
        if stream is not None:
            try:
                stream.write(f"[scf_rescue] {msg}\n")
            except Exception:  # noqa: BLE001 — best-effort status only
                pass

    if dm0 is not None:
        energy = float(mf.kernel(dm0=dm0, **kernel_kwargs))
    else:
        energy = float(mf.kernel(**kernel_kwargs))

    if mf.converged or not rescue:
        mf.scf_rescue_stage = SCF_RESCUE_NONE if mf.converged else SCF_RESCUE_FAILED
        return energy

    _status(
        f"SCF did not converge on first attempt (E={energy!r}) — retrying "
        "with a same-basis HF/UHF bootstrap density."
    )
    from pyscf import scf as _scf

    mol = mf.mol
    bootstrap = _scf.UHF(mol) if mol.spin != 0 else _scf.RHF(mol)
    bootstrap.verbose = 0
    bootstrap.kernel()
    energy = float(mf.kernel(dm0=bootstrap.make_rdm1(), **kernel_kwargs))
    if mf.converged:
        _status("Bootstrap rescue succeeded.")
        mf.scf_rescue_stage = SCF_RESCUE_BOOTSTRAP
        return energy

    if max_stage < 2:
        _status("Bootstrap alone did not converge; max_stage=1, not retrying further.")
        mf.scf_rescue_stage = SCF_RESCUE_FAILED
        return energy

    _status(
        "Bootstrap alone did not converge — retrying with level_shift=0.3, "
        "init_guess='atom', max_cycle=100."
    )
    mf.level_shift = 0.3
    mf.init_guess = "atom"
    mf.max_cycle = max(mf.max_cycle, 100)
    energy = float(mf.kernel(**kernel_kwargs))
    if mf.converged:
        _status("Level-shift rescue succeeded.")
        mf.scf_rescue_stage = SCF_RESCUE_LEVEL_SHIFT
    else:
        _status(
            "WARNING: still not converged after both rescue stages. "
            "Reporting the unconverged result as-is."
        )
        mf.scf_rescue_stage = SCF_RESCUE_FAILED
    return energy
