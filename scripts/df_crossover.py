#!/usr/bin/env python3
"""Measure the density-fitting (RI) speed / accuracy crossover — M-DF, DF.2.

QuantUI ships density fitting as an opt-in that is **off by default**
(``quantui.density_fitting``). The default is deliberately unset per calc type
until the size crossover is measured on real hardware, because DF is not a
blanket win: on aspirin/6-31G* it made TD-DFT ~1.6x faster but a small SCF
slightly *slower*. This script produces the curve that decides the defaults.

It is a **standalone measurement tool**, meant for a short NCShare / local
session — not part of the shipped package and not imported by it. It uses raw
PySCF so the numbers isolate the DF effect from any QuantUI overhead (the app's
``try_density_fit`` does the same ``mf.density_fit()`` call under the hood).

For each molecule x basis it times, back to back:
  * plain 4-centre SCF        vs  density-fitted SCF
  * plain TD-DFT solve        vs  density-fitted TD-DFT solve   (optional)
and reports wall-time speedups plus the total-energy difference in kcal/mol.
Optionally repeats the whole thing on a GPU via gpu4pyscf (``--gpu``), which
also exercises the DF.6 compose order ``mf.density_fit().to_gpu()``.

Examples
--------
    # Quick CPU sanity pass (small, ~1 min):
    python scripts/df_crossover.py --molecules water,benzene --bases 6-31G* \
        --no-tddft --warmup 1

    # The real DF.2 curve (run on an NCShare compute node; can take a while):
    OMP_NUM_THREADS=8 python scripts/df_crossover.py \
        --molecules benzene,naphthalene,aspirin,ibuprofen \
        --bases def2-SVP,def2-TZVP --states 10 \
        --csv df_crossover.csv --markdown df_crossover.md

    # H200 node (conda quantui[gpu-cudaXXx] env):
    python scripts/df_crossover.py --molecules aspirin,ibuprofen \
        --bases def2-SVP --states 10 --gpu

Paste the printed markdown table into
TODO/roadmaps/39-m-df-density-fitting-roadmap.md (DF.2) in the planning repo.
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ── A size ladder of neutral, closed-shell molecules (SMILES) ────────────────
# Chosen to sweep basis-function count, which is what DF's crossover tracks.
# Add your own with --molecules "name:SMILES,name:SMILES" or by editing here.
_DEFAULT_SMILES: dict[str, str] = {
    "water": "O",
    "benzene": "c1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    # ~44 atoms — a genuine "large" point for the tail of the curve.
    "cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
}


def _geometry_from_smiles(smiles: str) -> str:
    """Embed a 3-D geometry with RDKit (ETKDG + MMFF), return a PySCF atom block."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D  # deterministic: same geometry every run
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"RDKit could not embed {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    lines = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return "\n".join(lines)


@dataclass
class Cell:
    """One (molecule, basis, mode, device) measurement."""

    molecule: str
    basis: str
    device: str  # "cpu" | "gpu"
    n_atoms: int = 0
    n_basis: int = 0
    scf_plain_s: Optional[float] = None
    scf_df_s: Optional[float] = None
    tddft_plain_s: Optional[float] = None
    tddft_df_s: Optional[float] = None
    e_plain: Optional[float] = None
    e_df: Optional[float] = None
    exc_plain_ev: Optional[float] = None
    exc_df_ev: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    # ── derived ──
    @property
    def scf_speedup(self) -> Optional[float]:
        if self.scf_plain_s and self.scf_df_s:
            return self.scf_plain_s / self.scf_df_s
        return None

    @property
    def tddft_speedup(self) -> Optional[float]:
        if self.tddft_plain_s and self.tddft_df_s:
            return self.tddft_plain_s / self.tddft_df_s
        return None

    @property
    def total_speedup(self) -> Optional[float]:
        if self.scf_plain_s is None or self.scf_df_s is None:
            return None
        plain = self.scf_plain_s + (self.tddft_plain_s or 0.0)
        df = self.scf_df_s + (self.tddft_df_s or 0.0)
        return plain / df if df else None

    @property
    def de_kcal(self) -> Optional[float]:
        if self.e_plain is not None and self.e_df is not None:
            return abs(self.e_plain - self.e_df) * 627.5094740631
        return None

    @property
    def dexc_mev(self) -> Optional[float]:
        if self.exc_plain_ev is not None and self.exc_df_ev is not None:
            return abs(self.exc_plain_ev - self.exc_df_ev) * 1000.0
        return None


def _build_scf(mol, xc: str, device: str, density_fit: bool, auxbasis):
    """Build an RKS mean field, applying DF and/or GPU as requested.

    DF is applied to the base object *before* ``to_gpu()`` — the same order the
    QuantUI app uses (M-DF), and the order this script exists partly to
    validate on GPU (DF.6).
    """
    from pyscf import dft

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    if density_fit:
        mf = mf.density_fit(auxbasis=auxbasis) if auxbasis else mf.density_fit()
    if device == "gpu":
        mf = mf.to_gpu()
    return mf


def _run_one(
    mol,
    xc: str,
    device: str,
    density_fit: bool,
    auxbasis,
    n_states: int,
):
    """Return (scf_s, e_scf, tddft_s, first_exc_ev). TD-DFT parts are None when
    ``n_states == 0``."""
    t0 = time.perf_counter()
    mf = _build_scf(mol, xc, device, density_fit, auxbasis)
    e = float(mf.kernel())
    scf_s = time.perf_counter() - t0

    tddft_s = None
    first_exc = None
    if n_states > 0:
        from pyscf import tdscf

        t1 = time.perf_counter()
        td = tdscf.TDDFT(mf)
        td.nstates = n_states
        td.verbose = 0
        td.kernel()
        tddft_s = time.perf_counter() - t1
        # Excitation energies come back in Hartree; take the lowest.
        exc = [float(x) for x in td.e]
        if exc:
            first_exc = min(exc) * 27.211386245988  # Ha -> eV
    return scf_s, e, tddft_s, first_exc


def measure(
    name: str,
    atom_block: str,
    basis: str,
    xc: str,
    device: str,
    n_states: int,
    auxbasis,
) -> Cell:
    from pyscf import gto

    cell = Cell(molecule=name, basis=basis, device=device)
    try:
        mol = gto.M(atom=atom_block, basis=basis, verbose=0)
        cell.n_atoms = mol.natm
        cell.n_basis = int(mol.nao_nr())
    except Exception as exc:  # noqa: BLE001 — record and skip this cell
        cell.notes.append(f"build failed: {exc}")
        return cell

    for df_on in (False, True):
        try:
            scf_s, e, td_s, exc = _run_one(mol, xc, device, df_on, auxbasis, n_states)
        except (
            Exception
        ) as exc_:  # noqa: BLE001 — one mode failing shouldn't kill the row
            cell.notes.append(f"{'df' if df_on else 'plain'} failed: {exc_}")
            continue
        if df_on:
            cell.scf_df_s, cell.e_df, cell.tddft_df_s, cell.exc_df_ev = (
                scf_s,
                e,
                td_s,
                exc,
            )
        else:
            cell.scf_plain_s, cell.e_plain, cell.tddft_plain_s, cell.exc_plain_ev = (
                scf_s,
                e,
                td_s,
                exc,
            )
    return cell


def _fmt(x, spec="{:.2f}") -> str:
    return spec.format(x) if x is not None else "—"


def _render_table(cells: list[Cell], markdown: bool) -> str:
    cols = [
        ("molecule", 12),
        ("basis", 10),
        ("dev", 4),
        ("nbf", 5),
        ("scf+", 7),
        ("scf_df", 7),
        ("scf x", 6),
        ("td+", 7),
        ("td_df", 7),
        ("td x", 6),
        ("tot x", 6),
        ("dE kcal", 8),
        ("dExc meV", 8),
    ]

    def row(vals):
        if markdown:
            return "| " + " | ".join(str(v) for v in vals) + " |"
        return "  ".join(f"{str(v):<{w}}" for (_, w), v in zip(cols, vals))

    lines = [row([c for c, _ in cols])]
    if markdown:
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
    else:
        lines.append("-" * (sum(w for _, w in cols) + 2 * len(cols)))
    for c in cells:
        lines.append(
            row(
                [
                    c.molecule,
                    c.basis,
                    c.device,
                    c.n_basis,
                    _fmt(c.scf_plain_s),
                    _fmt(c.scf_df_s),
                    _fmt(c.scf_speedup, "{:.2f}x"),
                    _fmt(c.tddft_plain_s),
                    _fmt(c.tddft_df_s),
                    _fmt(c.tddft_speedup, "{:.2f}x"),
                    _fmt(c.total_speedup, "{:.2f}x"),
                    _fmt(c.de_kcal, "{:.4f}"),
                    _fmt(c.dexc_mev, "{:.2f}"),
                ]
            )
        )
        for note in c.notes:
            lines.append(f"    ! {c.molecule}/{c.basis}/{c.device}: {note}")
    return "\n".join(lines)


def _write_csv(cells: list[Cell], path: str) -> None:
    fields = [
        "molecule",
        "basis",
        "device",
        "n_atoms",
        "n_basis",
        "scf_plain_s",
        "scf_df_s",
        "scf_speedup",
        "tddft_plain_s",
        "tddft_df_s",
        "tddft_speedup",
        "total_speedup",
        "de_kcal_mol",
        "dexc_mev",
        "notes",
    ]
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for c in cells:
            w.writerow(
                {
                    "molecule": c.molecule,
                    "basis": c.basis,
                    "device": c.device,
                    "n_atoms": c.n_atoms,
                    "n_basis": c.n_basis,
                    "scf_plain_s": c.scf_plain_s,
                    "scf_df_s": c.scf_df_s,
                    "scf_speedup": c.scf_speedup,
                    "tddft_plain_s": c.tddft_plain_s,
                    "tddft_df_s": c.tddft_df_s,
                    "tddft_speedup": c.tddft_speedup,
                    "total_speedup": c.total_speedup,
                    "de_kcal_mol": c.de_kcal,
                    "dexc_mev": c.dexc_mev,
                    "notes": "; ".join(c.notes),
                }
            )


def _recommend(cells: list[Cell]) -> list[str]:
    """Derive a per-calc-type DF recommendation from the measured cells.

    Descriptive, not prescriptive: it reports where DF *actually* started
    winning in this run (the SCF crossover bracket) and the observed TD-DFT and
    accuracy behaviour, so the numbers set the policy rather than a guess. This
    is what DF.2 needs on top of the raw table.
    """
    cpu = [c for c in cells if c.device == "cpu" and c.scf_speedup is not None]
    cpu.sort(key=lambda c: c.n_basis)
    lines: list[str] = ["# Recommendation (from THIS run's data):"]
    if not cpu:
        lines.append("#   not enough completed cells to recommend.")
        return lines

    # SCF crossover: the bracket between the largest system where DF lost and
    # the smallest where it won.
    last_loss = max((c.n_basis for c in cpu if c.scf_speedup < 1.0), default=None)
    first_win = min((c.n_basis for c in cpu if c.scf_speedup >= 1.0), default=None)
    if first_win is None:
        lines.append(
            "#   SCF: DF never beat 4-centre in this run — keep SCF DF OFF, "
            "or extend the ladder to larger systems."
        )
    elif last_loss is None:
        lines.append(
            f"#   SCF: DF won at every size measured (down to {first_win} bf) — "
            "SCF DF ON is defensible; probe smaller if you want the floor."
        )
    else:
        lines.append(
            f"#   SCF crossover between ~{last_loss} bf (DF slower) and "
            f"~{first_win} bf (DF faster): default SCF DF ON above ~{first_win} bf."
        )

    # TD-DFT: usually a win well before SCF.
    td = [c for c in cpu if c.tddft_speedup is not None]
    if td:
        lo = min(c.tddft_speedup for c in td)
        hi = max(c.tddft_speedup for c in td)
        verdict = "ON" if lo >= 1.0 else "mixed — inspect per size"
        lines.append(
            f"#   TD-DFT: speedup {lo:.2f}x–{hi:.2f}x over {len(td)} cells → "
            f"default TD-DFT DF {verdict}."
        )

    # Accuracy ceiling actually observed.
    des = [c.de_kcal for c in cpu if c.de_kcal is not None]
    dxs = [c.dexc_mev for c in td if c.dexc_mev is not None]
    if des:
        lines.append(
            f"#   Accuracy cost: max {max(des):.4f} kcal/mol (energy)"
            + (f", {max(dxs):.2f} meV (first excitation)" if dxs else "")
            + " — compare against 1 kcal/mol chemical accuracy."
        )
    lines.append(
        "#   Then set quantui/density_fitting per-calc-type defaults from these "
        "numbers (still a small, data-backed follow-up — not guessed here)."
    )
    return lines


def _parse_molecules(spec: Optional[str]) -> dict[str, str]:
    if not spec:
        return {
            k: _DEFAULT_SMILES[k]
            for k in ("benzene", "naphthalene", "aspirin", "ibuprofen")
        }
    out: dict[str, str] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:  # name:SMILES
            name, smi = token.split(":", 1)
            out[name.strip()] = smi.strip()
        elif token in _DEFAULT_SMILES:
            out[token] = _DEFAULT_SMILES[token]
        else:
            raise SystemExit(
                f"Unknown molecule {token!r}. Known: "
                f"{', '.join(_DEFAULT_SMILES)} — or pass name:SMILES."
            )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure the density-fitting speed/accuracy crossover (M-DF DF.2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--molecules",
        help="Comma list of known names and/or name:SMILES pairs "
        f"(known: {', '.join(_DEFAULT_SMILES)}). "
        "Default: benzene,naphthalene,aspirin,ibuprofen.",
    )
    ap.add_argument(
        "--bases",
        default="def2-SVP,def2-TZVP",
        help="Comma list of orbital bases (default: def2-SVP,def2-TZVP).",
    )
    ap.add_argument("--xc", default="b3lyp", help="DFT functional (default: b3lyp).")
    ap.add_argument(
        "--states",
        type=int,
        default=10,
        help="TD-DFT states to solve (default 10; 0 or --no-tddft to skip).",
    )
    ap.add_argument("--no-tddft", action="store_true", help="Skip the TD-DFT solve.")
    ap.add_argument(
        "--auxbasis",
        default=None,
        help="Auxiliary (fitting) basis for DF. Default lets PySCF auto-derive a "
        "matched set (recommended for Pople bases). Try 'def2-universal-jfit' "
        "with def2 orbital bases to probe DF.3.",
    )
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="Also measure on GPU via gpu4pyscf (needs a CUDA device + install). "
        "Exercises the DF.6 compose order mf.density_fit().to_gpu().",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up runs to discard before timing (default 1; absorbs import/JIT).",
    )
    ap.add_argument(
        "--max-basis",
        type=int,
        default=1200,
        help="Skip any (molecule, basis) whose n_basis exceeds this (default 1200) "
        "so a stray huge case can't stall the session.",
    )
    ap.add_argument("--csv", help="Write per-cell results to this CSV path.")
    ap.add_argument("--markdown", help="Write a markdown table to this path.")
    args = ap.parse_args(argv)

    n_states = 0 if args.no_tddft else max(0, args.states)
    molecules = _parse_molecules(args.molecules)
    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    devices = ["cpu"] + (["gpu"] if args.gpu else [])

    # Report the compute context — CPU affinity / thread count matters a lot for
    # these timings on a shared node (see the M-GPU CPU-affinity note).
    n_threads = os.environ.get("OMP_NUM_THREADS", "(unset — PySCF default)")
    try:
        avail = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        avail = os.cpu_count()
    print(
        f"# df_crossover — xc={args.xc}  states={n_states}  "
        f"auxbasis={args.auxbasis or 'auto'}"
    )
    print(
        f"# OMP_NUM_THREADS={n_threads}  cores_visible={avail}  "
        f"warmup={args.warmup}"
    )
    if args.gpu:
        print("# GPU pass enabled (gpu4pyscf)")
    print()

    # Pre-embed geometries once (RDKit is the slow part for big molecules).
    geoms: dict[str, str] = {}
    for name, smi in molecules.items():
        try:
            geoms[name] = _geometry_from_smiles(smi)
        except Exception as exc:  # noqa: BLE001
            print(f"# skip {name}: geometry failed ({exc})", file=sys.stderr)

    # Optional warm-up: a tiny fixed calc so import/JIT cost isn't charged to
    # the first real cell.
    for _ in range(args.warmup):
        try:
            from pyscf import gto

            _wm = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
            _run_one(_wm, args.xc, "cpu", False, None, 0)
            _run_one(_wm, args.xc, "cpu", True, None, 0)
        except Exception:  # noqa: BLE001 — warm-up is best-effort
            break

    cells: list[Cell] = []
    for name in geoms:
        for basis in bases:
            for device in devices:
                # Cheap pre-check of size so we can skip before the slow build.
                try:
                    from pyscf import gto

                    probe = gto.M(atom=geoms[name], basis=basis, verbose=0)
                    nbf = int(probe.nao_nr())
                except Exception as exc:  # noqa: BLE001
                    c = Cell(molecule=name, basis=basis, device=device)
                    c.notes.append(f"build failed: {exc}")
                    cells.append(c)
                    continue
                if nbf > args.max_basis:
                    c = Cell(molecule=name, basis=basis, device=device, n_basis=nbf)
                    c.notes.append(
                        f"skipped: n_basis {nbf} > --max-basis {args.max_basis}"
                    )
                    cells.append(c)
                    print(
                        f"# skip {name}/{basis}/{device}: n_basis={nbf} "
                        f"> {args.max_basis}"
                    )
                    continue
                print(
                    f"# running {name}/{basis}/{device} (n_basis={nbf}) …", flush=True
                )
                cells.append(
                    measure(
                        name,
                        geoms[name],
                        basis,
                        args.xc,
                        device,
                        n_states,
                        args.auxbasis,
                    )
                )

    print()
    print(_render_table(cells, markdown=False))

    if args.csv:
        _write_csv(cells, args.csv)
        print(f"\n# wrote CSV: {args.csv}")
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(_render_table(cells, markdown=True) + "\n")
        print(f"# wrote markdown: {args.markdown}")

    # A one-line reading of the crossover so the takeaway isn't buried.
    print("\n# Reading the table:")
    print("#   scf x / td x / tot x > 1.0  ->  DF is faster for that cell.")
    print("#   Watch where 'scf x' crosses 1.0 as n_basis grows — that is the")
    print("#   SCF crossover DF.2 needs. TD-DFT usually wins well before SCF does.")
    print("#   dE kcal / dExc meV are the accuracy cost (expect << 1 kcal/mol).")
    print()
    for line in _recommend(cells):
        print(line)
    print(
        "\n# DF.3: to compare the auto-derived auxbasis against a matched one, "
        "run\n#   this twice — once as-is, once with "
        "--auxbasis def2-universal-jfit (def2 bases) — and diff dE/timing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
