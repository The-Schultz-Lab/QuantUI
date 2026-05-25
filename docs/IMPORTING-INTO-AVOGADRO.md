# Importing QuantUI results into Avogadro / IQmol / Jmol

QuantUI saves every calculation as a *result folder* under `~/.quantui/results/`.
Each folder ships with portable, standards-compliant files that the wider
quantum-chemistry ecosystem already knows how to read. No screen-scraping,
no lock-in, no waiting on QuantUI to add a feature you can already get from
the tool you already use.

This page is a quick cross-reference: **"I want to do X — which file do I open
in which tool?"**

## The big table

| What you want to do | QuantUI file (in result folder) | Recommended external tool(s) |
| --- | --- | --- |
| View molecular orbitals in 3D | `result.molden` | Avogadro · IQmol · Jmol |
| Animate vibrational normal modes | `result.molden` (from a Frequency calc) | Avogadro |
| Plot or replay a geometry-optimization or PES-scan trajectory | `trajectory.xyz` (any tool) or `trajectory.traj` (ASE) | VMD · Avogadro · ASE-GUI |
| Render an orbital isosurface from a saved cube | `HOMO.cube` / `LUMO.cube` / etc. | Avogadro · VMD · ChimeraX |
| Open spectrum data in Excel / a notebook | `*_data_*.csv` (per-panel: IR, UV-Vis, orbitals, PES) | LibreOffice Calc · Excel · `pandas.read_csv` |
| Share the whole result with a collaborator | `<result-folder>.zip` (use **Export bundle** in the Analysis tab) | Any unzip tool |
| Edit a structure and re-run elsewhere | `trajectory.traj` (last frame) | ASE-GUI |

## Where the files live

After a calculation finishes, open the **Files tab** in QuantUI and select
the result folder. You will see a tree like this:

```text
2026-05-25_14-32-11-394021_H2O_B3LYP_6-31Gs/
├── result.json           ← machine-readable result metadata
├── result.molden         ← MOs + (for freq) vibrations  ← EXPORT.1 / EXPORT.2
├── pyscf.log             ← raw PySCF output
├── orbitals.npz          ← MO coefficients (for QuantUI re-render)
├── thumbnail.png         ← preview card image
├── trajectory.xyz        ← geo-opt / PES frames (multi-frame XYZ)  ← EXPORT.3
├── trajectory.traj       ← geo-opt / PES frames (ASE binary)       ← EXPORT.7
├── ir_data_<ts>.csv      ← IR-spectrum (freq+intensity) data       ← EXPORT.4
├── uv_data_<ts>.csv      ← UV-Vis-spectrum data                    ← EXPORT.4
├── orb_data_<ts>.csv     ← orbital-diagram data                    ← EXPORT.4
├── pes_data_<ts>.csv     ← PES-scan data                           ← EXPORT.4
└── isosurfaces/
    ├── H2O_HOMO_<ts>.cube
    └── H2O_LUMO_<ts>.cube                                          ← EXPORT.5
```

Files marked `← EXPORT.X` were added in the M-EXPORT milestone (session 54,
QuantUI 0.2.0). Older result folders may not have them.

## Per-tool quick start

### Avogadro 2

Avogadro is the easiest cross-platform viewer for QuantUI outputs.

- **View MOs:** `File → Open → result.molden` → menu **Analysis → Orbitals**.
  Pick an orbital from the list, then **Extensions → Surfaces → Generate**.
- **Animate vibrations:** open the *same* `result.molden` from a Frequency
  calculation → menu **Extensions → Vibrational Modes** → pick a frequency
  → **Start Animation**. QuantUI writes `[FREQ]`, `[FR-COORD]`, and
  `[FR-NORM-COORD]` blocks per the Molden spec.
- **Replay a geometry optimization:** `File → Open → trajectory.xyz` and
  use the frame slider at the bottom of the viewport.
- **Render an isosurface from a cube file:** `File → Open → <orbital>.cube`
  → **Extensions → Surfaces → Generate** (the cube is already on a grid).

### IQmol

Excellent for MO visualization with smooth navigation between orbitals.

- **MOs:** `File → Open → result.molden`. The orbital tree appears in the
  side panel; double-click an orbital to render its isosurface.
- IQmol does not animate vibrations from Molden files. For vibrations,
  use Avogadro.

### Jmol

Useful when you want a script-driven viewer for batch screenshots or
publications.

- **MOs:** `load result.molden` → `mo HOMO` (or any orbital index).
- **Trajectories:** `load trajectory.xyz` autoloads all frames; `frame next`
  cycles them.
- **Cubes:** `isoSurface s1 cutoff 0.05 "HOMO.cube"`.

### VMD

The best tool for large trajectories (PES scans with hundreds of points,
long MD-style replays).

- **Trajectories:** `vmd -m trajectory.xyz`. VMD auto-detects multi-frame
  XYZ.
- **Cubes:** `mol new HOMO.cube` then **Graphics → Representations →
  Isosurface**.

### ASE-GUI (graphical) and `ase` (Python)

ASE round-trips the binary `.traj` file with per-frame energies preserved.

- **Graphical:** `ase gui trajectory.traj` opens an interactive viewer.
  Slice with `ase gui trajectory.traj@0:10:2`.
- **Edit + save as a new starting point:**
  `ase gui trajectory.traj` → manipulate atoms → **File → Save as…**.
  Re-import the saved geometry into QuantUI for a follow-up calculation.
- **Python post-processing:**

  ```python
  from ase.io import read
  frames = read("trajectory.traj", index=":")
  for f in frames:
      print(f.get_potential_energy())  # eV (ASE convention)
  ```

  The `.xyz` trajectory uses the *extended-XYZ* convention with
  `energy=<value> Hartree` per frame, so `ase.io.read("trajectory.xyz", ":")`
  also works.

### Plain Python (Excel, pandas)

Every spectrum / diagram panel exports its data as a per-trace CSV via
the **📋 Copy data** button. The file is also written to the result folder
as `<panel>_data_<timestamp>.csv`. The format is one section per trace:

```text
# trace 1
x,y
400,0.0
401,0.012
...
```

This parses cleanly with stdlib `csv.reader`, `pandas.read_csv`, Excel,
LibreOffice Calc, or anything else that knows how to read comma-separated
values with comment lines.

## Bundle export

The **Export bundle** button in the Analysis tab zips an entire result
folder. The archive lands as a sibling of the result directory:

```text
~/.quantui/results/2026-05-25_14-32-11-394021_H2O_B3LYP_6-31Gs.zip
```

Share that one file and your collaborator gets every artifact above —
no need to walk them through which file does what.

## Troubleshooting

- **Avogadro 1.2 doesn't show vibrations.** Upgrade to Avogadro 2; the v1
  branch is no longer maintained. Avogadro 2 reads QuantUI's Molden
  vibration blocks natively.
- **`result.molden` is missing for an older result.** Auto-export was
  added in session 54 (QuantUI 0.2.0). Older results don't have a
  `.molden`; re-running the calc regenerates one.
- **IQmol can't open the file.** IQmol's parser is stricter than
  Avogadro's. If you see a parse error, open the file in Avogadro first
  to confirm it's well-formed — usually a sign of a half-written file
  from an interrupted run.
- **Cube files render in Avogadro but the colors are inverted.** Toggle
  **Extensions → Surfaces → Color by Phase**. Cube sign conventions vary
  between codes; QuantUI uses PySCF's default (gpu4pyscf matches).

## Related reading

- [Molden file format spec](https://www.theochem.ru.nl/molden/molden_format.html)
- [Extended-XYZ specification](https://wiki.fysik.dtu.dk/ase/ase/io/formatoptions.html#extended-xyz)
- [ASE trajectory file format](https://wiki.fysik.dtu.dk/ase/ase/io/trajectory.html)
- [Cube file format (Gaussian convention)](https://gaussian.com/cubegen/)

## Roadmap link

This page closes work-package **EXPORT.6** in [M-EXPORT](https://github.com/NCCU-Schultz-Lab/QuantUI/blob/main/CHANGELOG.md).
The companion exports (Molden, multi-frame XYZ, ASE `.traj`, per-panel CSV,
cube + bundle) are tracked as EXPORT.1, EXPORT.2, EXPORT.3, EXPORT.4,
EXPORT.5, and EXPORT.7 in the same milestone.
