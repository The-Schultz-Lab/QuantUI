"""
QuantUI Calculator Module

Generates standalone PySCF Python scripts that students can download and
run independently. This is an "Export Script" feature — the primary
calculation path in QuantUI is session_calc.run_in_session(), not
batch script submission.
"""

import logging
from pathlib import Path

from . import config
from .molecule import Molecule

logger = logging.getLogger(__name__)


class PySCFCalculation:
    """
    Generates standalone PySCF scripts for a given molecule and method.

    The primary use in QuantUI is the "Export Script" button in the
    notebook, which lets students download a self-contained .py file they
    can study or run outside the notebook environment.
    """

    def __init__(
        self,
        molecule: Molecule,
        method: str = "RHF",
        basis: str = "6-31G",
    ):
        """
        Initialize a PySCF calculation.

        Args:
            molecule: Molecule object to calculate
            method: Calculation method (RHF, UHF)
            basis: Basis set name

        Raises:
            ValueError: If method is not supported
        """
        self.molecule = molecule
        # Match case-insensitively against the canonical (mixed-case)
        # SUPPORTED_METHODS list, but store the CANONICAL spelling — not
        # method.upper(). Several entries are mixed-case (e.g. "wB97X-D",
        # "CAM-B3LYP", "M06-L"); uppercasing turned "wB97X-D" into
        # "WB97X-D", which is in neither SUPPORTED_METHODS nor the xc-alias
        # tables the generated script relies on, breaking Export Script
        # and the educational-notes lookup for every mixed-case method.
        _method_upper = method.strip().upper()
        self.method = next(
            (m for m in config.SUPPORTED_METHODS if m.upper() == _method_upper),
            method,
        )
        self.basis = basis

        if self.method not in config.SUPPORTED_METHODS:
            raise ValueError(
                f"Method '{method}' not supported. "
                f"Supported methods: {', '.join(config.SUPPORTED_METHODS)}"
            )

        if self.basis not in config.SUPPORTED_BASIS_SETS:
            logger.warning(
                f"Basis set '{basis}' not in standard list. "
                f"Proceeding anyway, but it may not be available in PySCF."
            )

        logger.info(
            f"Initialized {self.method}/{self.basis} calculation for "
            f"{self.molecule.get_formula()}"
        )

    def generate_calculation_script(self, output_path: Path) -> str:
        """
        Generate a standalone Python script for the calculation.

        The script runs independently (no QuantUI required) and saves
        results to results.npz next to the script file.

        Args:
            output_path: Path where the script will be saved

        Returns:
            str: The generated script content
        """
        geometry = self.molecule.to_pyscf_format()
        spin = self.molecule.multiplicity - 1
        formula = self.molecule.get_formula()
        job_name = f"{formula}_{self.method}_{self.basis}"

        script_content = config.PYSCF_SCRIPT_TEMPLATE.format(
            job_name=job_name,
            method=self.method,
            basis=self.basis,
            geometry=geometry,
            charge=self.molecule.charge,
            spin=spin,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        logger.info(f"Generated calculation script: {output_path}")
        return script_content

    def get_description(self) -> str:
        """
        Get a human-readable description of the calculation.

        Returns:
            str: Description string
        """
        method_names = {
            "RHF": "Restricted Hartree-Fock",
            "UHF": "Unrestricted Hartree-Fock",
        }
        method_full = method_names.get(self.method, self.method)
        return (
            f"{method_full} calculation of {self.molecule.get_formula()} "
            f"using {self.basis} basis set"
        )

    def get_educational_notes(self) -> str:
        """
        Get educational notes about the calculation method and basis set.

        Returns:
            str: Markdown-formatted educational description
        """
        notes = []

        info = config.METHOD_INFO.get(self.method)
        if info:
            notes.append(f"**{info['label']}**: {info['description']}")
            notes.append(f"*Best used for:* {info['use_for']}")

        if self.basis == "STO-3G":
            notes.append(
                "**STO-3G basis**: Minimal basis set — 3 Gaussians per orbital. "
                "Very fast but low accuracy. Good for learning, not research."
            )
        elif "6-31G" in self.basis:
            note = (
                "**6-31G family**: Split-valence basis sets with a good balance of "
                "speed and accuracy. The * adds polarization functions for better "
                "description of molecular bonding and lone pairs."
            )
            # 6-31G* and 6-31G(d) are the same basis set in two notations; say so
            # rather than leave a reader thinking their textbook's spelling is a
            # different, unavailable option.
            alias = config.pople_notation_alias(self.basis)
            if alias:
                note += (
                    f" Equivalently written **{alias}** — the same basis set, "
                    "either spelling is accepted."
                )
            notes.append(note)
        elif "cc-pV" in self.basis:
            notes.append(
                "**Correlation-consistent basis sets (cc-pVXZ)**: High-quality basis "
                "sets designed for systematic convergence. More expensive but more "
                "accurate; best paired with correlated methods."
            )
        elif "def2" in self.basis:
            notes.append(
                "**def2 basis sets**: Karlsruhe basis sets optimised for DFT. "
                "def2-SVP is a good default for DFT calculations; def2-TZVP gives "
                "near-complete-basis accuracy for most properties."
            )
        elif self.basis.upper().startswith("LANL"):
            notes.append(
                "**LANL2DZ**: A double-zeta basis with a Los Alamos effective "
                "core potential (ECP) on heavy elements — it replaces the core "
                "electrons of a heavy metal with a potential, making transition "
                "and heavy-metal calculations tractable. A common alternative to "
                "the def2 sets for the heaviest centres."
            )

        if self.molecule.multiplicity > 1:
            notes.append(
                f"**Spin multiplicity {self.molecule.multiplicity}**: This molecule "
                f"has {self.molecule.multiplicity - 1} unpaired electron(s). "
                "UHF is appropriate for open-shell systems."
            )

        return "\n\n".join(notes)


def create_calculation(
    molecule: Molecule,
    method: str,
    basis: str,
) -> PySCFCalculation:
    """
    Factory function to create a PySCF calculation.

    Args:
        molecule: Molecule to calculate
        method: Calculation method
        basis: Basis set

    Returns:
        PySCFCalculation: Initialized calculation object
    """
    return PySCFCalculation(molecule=molecule, method=method, basis=basis)
