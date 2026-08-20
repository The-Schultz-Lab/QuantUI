"""Pre-run guards for inorganic / metal calculations (M-METAL MET.5).

Two mid-run PySCF failures are common the moment a student loads a
transition-metal complex, and both surface as cryptic tracebacks deep inside a
background thread:

* a basis set with **no parameters for an element** — e.g. the default
  ``6-31G`` on Pt raises ``BasisNotFoundError: Basis set not found for Pt``; and
* a **charge / multiplicity inconsistent with the electron count** — e.g. an odd
  electron count with the default multiplicity 1 raises
  ``Electron number N and spin S are not consistent``.

These functions catch both **before the run starts** and return a plain-language
message the app shows in place of launching the doomed calculation. Pure logic —
no widgets and no calculation; the only dependency is PySCF's own basis loader,
used as the source of truth so the check matches exactly what a run would hit.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

# def2 basis sets QuantUI ships that carry effective core potentials for heavy
# elements (so they cover the whole periodic table, unlike the Pople / cc sets).
_ECP_BASIS_SUGGESTION = "def2-SVP or def2-TZVP"


def basis_unsupported_elements(basis: str, elements: Iterable[str]) -> List[str]:
    """Return the unique elements ``basis`` has no parameters for (order-preserved).

    Uses ``pyscf.gto.basis.load`` — the same lookup the calculation performs — so
    the verdict matches what a run would actually hit. Never raises: any loader
    error is treated as "unsupported" for that element (the conservative choice,
    since the run would then fail too).
    """
    from pyscf import gto

    bad: List[str] = []
    seen = set()
    for el in elements:
        if el in seen:
            continue
        seen.add(el)
        try:
            gto.basis.load(basis, el)
        except Exception:
            bad.append(el)
    return bad


def ecp_for_basis(basis: str, elements: Iterable[str]) -> dict:
    """Return the ``mol.ecp`` mapping ``basis`` needs over ``elements``.

    Basis sets like **LANL2DZ** and the **def2** family bundle an effective core
    potential (ECP) for heavy elements, but PySCF only applies it when
    ``mol.ecp`` is set *as well as* ``mol.basis``. Set only the basis and the
    heavy atom is run all-electron against a valence-only basis: PySCF keeps the
    full electron count, warns ``ECP not specified``, and produces garbage
    energies and gradients — a geometry optimisation then walks off into
    nonsense (fmax in the thousands, energy sliding without converging).

    This returns ``{element: basis}`` for exactly the elements that carry an ECP
    under ``basis`` (via the same ``pyscf.gto.basis.load_ecp`` lookup a run
    performs), so a caller can write::

        mol.ecp = ecp_for_basis(basis, molecule.atoms)  # {} for all-electron sets

    Pople / cc / STO sets have no ECP table, so this returns ``{}`` and the
    caller leaves ``mol.ecp`` at its (empty) default. Never raises: a missing
    ECP table is treated as "no ECP for that element".
    """
    from pyscf import gto

    ecp: dict = {}
    seen = set()
    for el in elements:
        if el in seen:
            continue
        seen.add(el)
        try:
            if gto.basis.load_ecp(basis, el):
                ecp[el] = basis
        except Exception:  # noqa: BLE001 — no ECP table for this basis/element
            pass
    return ecp


def check_basis_coverage(elements: Iterable[str], basis: str) -> Optional[str]:
    """Message if ``basis`` lacks any element, else ``None``."""
    bad = basis_unsupported_elements(basis, elements)
    if not bad:
        return None
    els = ", ".join(bad)
    return (
        f"The basis set '{basis}' has no parameters for {els}. "
        f"Transition metals and other heavy elements need an ECP basis — switch "
        f"to {_ECP_BASIS_SUGGESTION} (these cover the whole periodic table via "
        f"effective core potentials) and run again."
    )


def check_charge_multiplicity(n_electrons: int, multiplicity: int) -> Optional[str]:
    """Message if ``multiplicity`` is impossible for ``n_electrons``, else ``None``.

    The number of unpaired electrons is ``multiplicity - 1``; it cannot exceed
    the electron count, and it must have the same parity as it (an odd electron
    count is only compatible with an even multiplicity, and vice versa).
    """
    if multiplicity < 1:
        return f"Multiplicity must be at least 1 (got {multiplicity})."
    n_unpaired = multiplicity - 1
    if n_unpaired > n_electrons:
        return (
            f"Multiplicity {multiplicity} needs {n_unpaired} unpaired "
            f"electrons, but the molecule has only {n_electrons}. Lower the "
            f"multiplicity."
        )
    if (n_electrons - n_unpaired) % 2 != 0:
        needs = "an even" if n_electrons % 2 else "an odd"
        suggestion = 2 if n_electrons % 2 else 1
        parity = "odd" if n_electrons % 2 else "even"
        return (
            f"{n_electrons} electrons with multiplicity {multiplicity} is "
            f"impossible: an {parity} electron count needs {needs} multiplicity "
            f"(e.g. {suggestion}). Re-check the charge and multiplicity — a metal "
            f"centre's oxidation state fixes its d-electron count and spin state."
        )
    return None


def preflight_messages(
    elements: Iterable[str],
    n_electrons: int,
    basis: str,
    multiplicity: int,
) -> List[str]:
    """Return the list of blocking pre-run problems (empty = OK to run)."""
    elements = list(elements)
    messages: List[str] = []
    basis_msg = check_basis_coverage(elements, basis)
    if basis_msg:
        messages.append(basis_msg)
    spin_msg = check_charge_multiplicity(n_electrons, multiplicity)
    if spin_msg:
        messages.append(spin_msg)
    return messages
