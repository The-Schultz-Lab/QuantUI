# QuantUI Repository Audit — Findings

**Date:** 2026-07-14
**Scope:** Full read of the core calculation, persistence, resolver, and infrastructure modules (`quantui/*.py`, ~29k LOC), targeted review of the large UI modules (`app*.py`, `benchmarks.py`, visualization modules), plus config/CI/packaging. Static analysis (ruff, bug-focused rule set) came back clean apart from `raise … from` hygiene, so everything below was found by manual review.

Findings are ordered by severity. File/line references are to the audited commit (`310a8cf`).

---

## High — functional bugs

### H1. Orbital isosurface fails for charged and all open-shell molecules
`orbital_visualization.generate_cube_from_arrays()` builds the PySCF molecule as:

```python
mol = gto.M(atom=mol_atom, basis=mol_basis, unit="Angstrom", spin=spin)   # orbital_visualization.py:673
```

No `charge` is passed, and the only call site passes neither `charge` nor `spin`:

```python
generate_cube_from_arrays(mol_atom, mol_basis, mo_coeff, orb_idx, cube_path)   # app_visualization.py:1207
```

PySCF raises at `mol.build()` whenever the electron count implied by `charge=0` is inconsistent with `spin=0` — i.e. for **every odd-electron system**: radicals (multiplicity 2), and charged species like H₃O⁺, NH₄⁺, OH⁻. The isosurface panel therefore errors out for these systems even though the MO coefficients are available and valid. `generate_cube_file()` (line 597) has the same omission.

**Fix:** thread `charge` and `spin` (`multiplicity − 1`) from the result object through to both cube generators.

### H2. Frequency results store `pyscf_mol_atom` in **Bohr**; every consumer assumes **Ångström**
`freq_calc.py:286` populates the field from PySCF internals:

```python
pyscf_mol_atom = [(str(s), list(map(float, c))) for s, c in mol._atom]   # mol._atom is in Bohr
```

whereas `session_calc.py:529` and `optimizer.py` populate the same field in Ångström (as documented at `session_calc.py:90`). Consumers all assume Å:

- `results_storage.save_molden()` (line 303) builds `mol.atom = …` with PySCF's default Å unit → the `[Atoms]`/`[GTO]` sections of Molden files exported from **frequency runs have the geometry inflated by ×1.889** (Bohr values read as Å), while the `[FR-COORD]` block (which the Molden spec defines in Bohr, written verbatim at line 368) is coincidentally correct — so the exported file is internally inconsistent and renders wrong in Avogadro/IQmol.
- `results_storage.save_orbitals()` persists the Bohr coordinates to `orbitals_meta.json`, so **history replay** of a frequency result feeds Bohr coordinates into `generate_cube_from_arrays(..., unit="Angstrom")`.
- `app_visualization.py:968` caches the same field for live isosurface rendering.

**Fix:** in `freq_calc`, build the field from `molecule.atoms`/`molecule.coordinates` (like `session_calc` does), or convert `mol._atom` with `param.BOHR`; and have `_append_molden_vibrations` do an explicit Å→Bohr conversion for `[FR-COORD]` so its input convention matches everything else.

### H3. "Export Script" and the method-notes panel are broken for `wB97X-D`
`calculator.PySCFCalculation.__init__` upper-cases the method then checks membership against the mixed-case list:

```python
self.method = method.upper()                      # "wB97X-D" -> "WB97X-D"
if self.method not in config.SUPPORTED_METHODS:   # list contains "wB97X-D"
    raise ValueError(...)                          # calculator.py:47-53
```

The UI passes display names straight from the dropdown (options = `SUPPORTED_METHODS`, `app_builders.py:579`), so:

- **Export Script** (`app_exports.on_export`, line 17) fails with "Method 'wB97X-D' not supported".
- **Educational notes** (`app_runflow.update_notes`, line 1068) silently disappear for wB97X-D — the exception is swallowed at line 1089.

Even if construction succeeded, the generated script's `_XC_ALIAS` / `_NEEDS_D3` tables (config.py:444-450) key on `'wB97X-D'` and would not match the upper-cased `'WB97X-D'`, producing a script that sets an invalid `mf.xc`. Same latent issue applies to any future mixed-case method name.

**Fix:** compare case-insensitively while preserving the canonical display name (mirror `session_calc.resolve_xc`'s approach).

### H4. `generate_2d_structure_svg()`'s XYZ input path can never work
`pubchem.py:808-825` builds the molecule with the immutable class:

```python
rdkit_mol = Chem.Mol()
...
rdkit_mol.AddAtom(atom)        # AddAtom exists only on Chem.RWMol → AttributeError
```

The AttributeError is swallowed by the broad `except` at line 841, so `xyz_string=` input always returns `None`. Two further latent bugs in the same block: the conformer index uses the raw line index `i` (lines with `len(parts) < 4` are skipped but `i` still advances, desynchronizing atoms and coordinates), and `Chem.Conformer()` is created without the atom count.

**Fix:** use `Chem.RWMol()`, track a separate atom counter, size the conformer, and call `.GetMol()` at the end.

### H5. XYZ parser silently drops the first atom for standard files with a blank or `#` title line
`molecule.parse_xyz_input()` strips blank lines and full-line comments **before** detecting the `count / title / atoms…` header layout (lines 364-405). For a completely standard file such as:

```
3
                      ← blank title line (very common)
O 0.0 0.0 0.0
H ...
H ...
```

the blank line is removed, then `start_idx = 2` skips the count line **and the `O` atom line**. The count mismatch is only a `logger.warning` (line 528), so the user gets a 2-atom water with no visible error. The same happens when the title line starts with `#` or `!`. A pasted `N\natom…` block without any title line also loses its first atom.

**Fix:** detect the header on the raw line list (or only strip comments after header detection), and consider promoting the count-mismatch warning to a hard error.

---

## Medium — incorrect behavior on specific paths / edge cases

### M1. Element support is H–Kr, but error text and the resolver chain suggest otherwise
`config.VALID_ATOMS` stops at Kr (Z=36), yet:

- The parse error help text explicitly lists **`I` (iodine) as a valid symbol** (`molecule.py:485`).
- PubChem / CACTUS / SMILES resolution happily returns structures containing I, Sn, Ag, … which then fail `Molecule` validation with a misleading message (e.g. searching *thyroxine* or *amiodarone* resolves, then can't be loaded).
- `Molecule.get_electron_count()` returns 0 for unknown elements (`atomic_numbers.get(atom, 0)`), which would silently corrupt electron counts if the validation gate were ever relaxed.

### M2. MP2 / CCSD / CCSD(T) silently use an RHF/ROHF reference for open-shell molecules
`session_calc.py:331-337` builds `scf.RHF(mol)` for these methods regardless of spin. With multiplicity > 1, PySCF's `scf.RHF` factory returns ROHF; the post-HF kernels may fail or produce results inconsistent with the closed-shell "RHF reference" the docstrings and result fields describe. No guard or warning exists.

### M3. UHF results silently lack dipole moment and Mulliken charges
`session_calc.py:490` skips both for `method_upper == "UHF"` only — while UKS (open-shell DFT) flows through the same extraction successfully. Both properties are well-defined for UHF; the skip appears stale rather than principled, and the result card just shows nothing.

### M4. NMR silently falls back to B3LYP/6-31G* TMS reference constants
`nmr_calc.py:319-322`: any method/basis combination absent from `NMR_REFERENCE_SHIELDINGS` (e.g. M06-2X, wB97X-D, def2-TZVP, cc-pVTZ) silently uses the B3LYP/6-31G* constants, and `NMRResult` does not record which reference was applied. Chemical shifts can be systematically offset by several ppm (¹³C) with no indication to the student. The lookup is also case-sensitive.

### M5. IR-intensity worker rebuilds RHF/UHF from spin alone; failure mode contradicts the docs
`freq_ir_workers.run_displaced_scf` (line 149) and the serial `_displaced_scf_dipole` (freq_calc.py:390) pick `RHF if spin == 0 else UHF`. A UHF-singlet parent run becomes RHF in the displaced SCFs, and the parent's `(2, nao, nao)` UHF `dm0` initial guess then has the wrong shape. The resulting exception aborts the whole IR-intensity step — and contrary to `freq_ir_workers`' module docstring ("the driver … falls back to the serial loop so the user's calc still completes"), `freq_calc.py:555` does **not** fall back to serial; it skips IR intensities entirely.

### M6. PES scan records the wrong geometry and NaN-poisoned statistics on failed points
`pes_scan.py:329-333`: a failed scan point appends the **original input molecule** (not the geometry the scan had actually reached) to `coordinates_list`, so trajectory animation shows a bogus frame; the parallel `float("nan")` energy then makes `PESScanResult.energy_hartree` (`min()` over the list) and `summary()`'s min/max/barrier order-dependently NaN, since Python's `min`/`max` propagate NaN only when it is the first element.

### M7. PES scan uses ASE's deprecated `FixInternals(angles=…, dihedrals=…)` radian kwargs
`pes_scan.py:288-298`. ASE ≥ 3.21 deprecated these in favor of `angles_deg` / `dihedrals_deg`; the legacy kwargs are scheduled for removal. With `ase>=3.22.0` unpinned above, this is a forward-compatibility break waiting to happen (and worth a compatibility test now).

### M8. Event log: full rewrite per event + read/rewrite race
`calc_log.log_event()` appends (one lock section) then calls `prune_events()`, which reads the whole file and rewrites it in **two separate lock sections** (calc_log.py:968-994). An append from another thread between the read and the rewrite is silently lost. It's also O(file-size) work on *every* event — O(N²) over a session.

### M9. `save_thumbnail` docstring promises "silently skips … any error" but only guards the import
`results_storage.py:700-711` — only `ImportError` is caught; a `savefig` failure (e.g. disk full, font issues) propagates into the caller's save path.

### M10. PubChem client edge cases
- `_http_get` returns `None` (annotated `Response`) if `PUBCHEM_MAX_RETRIES` is ever configured < 1 (`pubchem.py:70-89`) → `AttributeError` at every caller.
- `check_pubchem_availability()` bypasses the shared throttle and hardcodes `timeout=5` instead of `config.PUBCHEM_AVAILABILITY_TIMEOUT_S` (line 547) — the config constant is defined but unused.

### M11. Export filenames embed the basis set verbatim, including `*`
`app_exports.py` builds names like `H2O_RHF_6-31G*.py` / `.xyz` / `.mol` / `.pdb`. `*` is an invalid filename character on Windows (where the non-PySCF UI is supported and CI runs), so these exports fail there; on POSIX it produces glob-hostile filenames. `results_storage._safe_name()` already exists and solves exactly this — it just isn't used here.

### M12. Method-notes markdown mangling
`app_runflow.update_notes` (lines 1075-1078) converts `**bold**` to HTML by replacing only the **first** `**` pair; `get_educational_notes()` emits many bold spans, so literal `**` markers leak into the rendered panel.

### M13. The `quantui` CLI imports the entire GUI stack
`cli.py`'s header says it "deliberately avoids importing from the GUI side … so it stays fast on import", but `from quantui.calc_log import …` first executes `quantui/__init__.py`, which unconditionally runs `from .app import QuantUIApp` (line 175) → ipywidgets, IPython, and the full app module are imported for `quantui log tail`. This makes the CLI both slow and hard-dependent on notebook packages, defeating the stated design.

### M14. `logging.basicConfig()` at library import time
`utils.py:19` configures the **root** logger the moment `quantui.utils` is imported (which `quantui/__init__` always does). A library must not do this — it hijacks/duplicates the logging configuration of any host application or notebook. Use per-module loggers only (the package already installs a `NullHandler` correctly in `__init__.py`).

---

## Low — inefficiencies, data nits, docs, hygiene

1. **Estimator re-reads the entire perf log on every UI refresh** — `calc_log.estimate_time()` parses all of `perf_log.jsonl` (kept indefinitely by design) per call; `update_estimate` fires on widget changes. Estimates get progressively slower over the app's lifetime; consider caching with an mtime check.
2. **Basis-count table nit** — `_BASIS_FUNCTIONS["6-31G**"]["He"] = 2` (calc_log.py:138): 6-31G** places p-polarization on He as well → should be 5, inconsistent with `"H": 5` in the same table.
3. **Inconsistent Bohr constants** — `optimizer.py:59` uses `0.529177249`, `freq_calc.py:359` uses `0.52917721092`. Harmless numerically, but pick one (CODATA 2018: 0.529177210903) and share it like `HARTREE_TO_EV`.
4. **Duplicated data/constants** — the 36-entry atomic-number dict appears twice in `molecule.py` (lines 139, 553) and again in `benchmarks.py`; `HARTREE_TO_EV` is re-declared locally in `results_storage.py` (twice), `comparison.py`, and `optimizer.py:475`.
5. **Docstring drift** —
   - `session_calc.run_in_session` docstring says verbose "Default: 3"; the signature default is 4.
   - `pubchem.search_molecule_by_name` docstring says "None otherwise" but the function raises.
   - `pyproject.toml` comments claim `pyscf.prop.infrared` is in pyscf ≥ 2.13 core and "NMR is accessed via pyscf.nmr (core)"; `freq_calc.py:350` and `nmr_calc.py:162` say the opposite (and implement workarounds accordingly).
   - `freq_ir_workers` "serial fallback" claim (see M5).
6. **Python 3.9 support is claimed but untested** — `requires-python = ">=3.9"` and a 3.9 classifier, but CI runs only 3.10/3.11. Care was clearly taken (e.g. the `StrEnum` shim in `viz_backend_router.py`), but nothing verifies it.
7. **`raise … from` missing** in 12 exception-wrapping sites in `pubchem.py`, `cactus.py`, `molecule.py`, `app_visualization.py` (ruff B904) — masks root causes in tracebacks. (B904 is deliberately ignored in `pyproject.toml`; worth revisiting for the network client at least.)
8. **`StepProgress` renders labels/messages without HTML-escaping** (`progress.py:96-103`) — a failure message containing `<`/`>` (easy from a parse error echoing user input) breaks the widget rendering.
9. **`nmr_calc` monkey-patches PySCF process-wide on every call** — `pyscf.prop.nmr.rhf.gen_vind` and `.rks.get_vxc_giao` are replaced globally (nmr_calc.py:207, 294). Pragmatic, but it should be done once (idempotently) at import of the patched feature, with a version guard, so behavior doesn't depend on whether an NMR calc has run.
10. **`results_storage.list_results()` sort order** — collision-suffixed directories sort lexicographically (`…_1`, `…_10`, `…_2`); cosmetic, only affects same-microsecond collisions.
11. **`save_result` writes some fields un-coerced** — `energy_hartree` / `homo_lumo_gap_ev` go through `getattr` without `_opt_float`; a duck-typed result carrying numpy scalars would make `json.dumps` raise. All current internal producers coerce to `float`, so this is latent.
12. **`prune_events` keeps malformed entries forever** (calc_log.py:989) — deliberate, but it means a corrupt line never ages out of the 7-day log.
13. **Single atoms are rejected by `parse_xyz_input`** (molecule.py:517) — atomic calculations are legitimate QC targets (and PySCF handles them); the restriction plus its wording is a design limitation worth revisiting.
14. **`_cmd_analytics_build`** assigns `tool` from `_open_in_browser` and never uses it (cli.py:226).

---

## Verified-clean areas (for the record)

- `user_settings.py`, `vib_cache.py` — careful atomic writes, schema versioning, clamping; no issues found.
- `c_stderr.py` fd-2 capture — correct save/restore ordering, including the restore-before-read subtlety.
- `gpu_offload.py` — sound fallback design; the CuPy-array handling added across `session_calc` is consistent.
- `viz_backend_router.py` — routing logic exhaustive and correct, including the 3.9/3.10 `StrEnum` shim.
- `molecule_library.py` — read-only per-call SQLite connections, correct manifest fallback, `lru_cache` on the preset dict (so the `config.MOLECULE_LIBRARY` PEP 562 shim is *not* a per-access reload).
- `benchmarks.py` calibration worker loop — terminate/skip/stop/timeout handling and the exitcode/signal diagnostics are solid.
- Ruff (pyflakes/bugbear/pylint-error rules): no undefined names, no unused variables, no mutable default arguments anywhere in `quantui/`, `tests/`, `scripts/`.

## Suggested priorities

1. **H1 + H2** together — both corrupt the flagship analysis/export features (isosurface, Molden) for whole classes of molecules, and share the fix surface (thread charge/spin/units through result objects).
2. **H5** — silent data loss on a very common input format.
3. **H3, H4** — small, contained fixes.
4. **M1, M4** — correctness-of-science issues that are invisible to students, which is the worst kind in a teaching tool.
