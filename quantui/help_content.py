"""
Educational help content for QuantUI notebook widgets.

Provides collapsible HTML help panels that explain quantum chemistry
concepts to students.  Each help topic is a short, jargon-light
explanation with concrete examples and tips.

Usage in a notebook cell::

    from quantui.help_content import help_panel
    display(help_panel("charge"))
    display(help_panel("multiplicity"))
"""

from __future__ import annotations

from typing import Dict

import ipywidgets as widgets

from . import theme as _theme

# ---------------------------------------------------------------------------
# Help text bank — keys used by help_panel()
# ---------------------------------------------------------------------------

HELP_TOPICS: Dict[str, Dict[str, str]] = {
    "getting_started": {
        "title": "Getting Started",
        "body": (
            "<p><b>How to run a calculation:</b></p>"
            "<ol>"
            "<li>Select or enter a molecule in <b>Molecule Input</b> — browse the "
            "<b>Library</b>, paste <b>XYZ</b>, or use <b>Online Search</b> "
            "(name, SMILES, CID, or InChI). See the "
            "<i>Finding a molecule</i> help topic.</li>"
            "<li>Choose a calculation type, method, and basis set in "
            "<b>Calculation Setup</b></li>"
            "<li>Click <b>Run Calculation</b> — results appear in the "
            "<b>Results</b> tab immediately</li>"
            "<li>View energy-level diagrams, trajectories, and spectra in the "
            "<b>Analysis</b> tab</li>"
            "<li>Optionally compare results in <b>Compare</b>, or use "
            "<b>History</b> to reload a previous run</li>"
            "</ol>"
            "<p><b>Platform note:</b> PySCF calculations require Linux, macOS, "
            "or WSL. On Windows, run the pre-built container: "
            "<code>apptainer run quantui.sif</code></p>"
            "<p>Each dropdown in the Calculate tab has a <b>?</b> button for "
            "context-sensitive help on that specific option.</p>"
        ),
    },
    "calc_type": {
        "title": "Calculation types — what does each one do?",
        "body": (
            "<p>The <b>Calc. Type</b> dropdown chooses what QuantUI computes for "
            "your molecule at the selected method / basis.</p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            f"<tr style='border-bottom:1px solid {_theme.css.BORDER_LEGACY};'>"
            "  <th style='padding:3px 12px; text-align:left;'>Type</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Computes</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Cost</th></tr>"
            "<tr><td style='padding:3px 12px;'><b>Single Point</b></td>"
            "  <td style='padding:3px 12px;'>Energy + properties at the input "
            "geometry (no atoms move)</td>"
            "  <td style='padding:3px 12px;'>Fastest</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>Geometry Opt</b></td>"
            "  <td style='padding:3px 12px;'>Relaxes the structure to a minimum-"
            "energy geometry</td>"
            "  <td style='padding:3px 12px;'>Moderate</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>Frequency</b></td>"
            "  <td style='padding:3px 12px;'>Vibrational modes + IR spectrum; "
            "confirms a true minimum (no imaginary modes). On HPC clusters with "
            "many CPU cores, admins can set "
            "<code>QUANTUI_FREQ_PARALLEL=1</code> to parallelize the IR "
            "finite-difference step on CPU while the main SCF still uses the "
            "GPU when available.</td>"
            "  <td style='padding:3px 12px;'>Higher</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>UV-Vis (TD-DFT)</b></td>"
            "  <td style='padding:3px 12px;'>Electronic excitations / absorption "
            "spectrum (needs a DFT functional)</td>"
            "  <td style='padding:3px 12px;'>Higher</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>NMR Shielding</b></td>"
            "  <td style='padding:3px 12px;'>Isotropic shielding → predicted "
            "¹H / ¹³C chemical shifts</td>"
            "  <td style='padding:3px 12px;'>Higher</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>PES Scan</b></td>"
            "  <td style='padding:3px 12px;'>Energy profile along one bond / "
            "angle / dihedral coordinate</td>"
            "  <td style='padding:3px 12px;'>Many points</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>Reorganization Energy</b></td>"
            "  <td style='padding:3px 12px;'>Marcus 4-point λ for charge transfer "
            "(hole / electron channels)</td>"
            "  <td style='padding:3px 12px;'>Slowest (2–3 optimizations)</td></tr>"
            "</table>"
            "<p><b>Typical workflow:</b> optimize the geometry first "
            "(<b>Geometry Opt</b>), then run <b>Frequency</b>, <b>UV-Vis</b>, or "
            "<b>NMR</b> on the optimized structure for meaningful results — most "
            "spectra are only valid at a minimum-energy geometry.</p>"
        ),
    },
    "method": {
        "title": "RHF vs UHF — which method should I use?",
        "body": (
            "<p>Both methods approximate the electronic wavefunction using "
            "Hartree–Fock theory, but they treat electron spin differently.</p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            f"<tr style='border-bottom:1px solid {_theme.css.BORDER_LEGACY};'>"
            "  <th style='padding:3px 12px; text-align:left;'>Method</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Full name</th>"
            "  <th style='padding:3px 12px; text-align:left;'>When to use</th></tr>"
            "<tr><td style='padding:3px 12px;'><b>RHF</b></td>"
            "  <td style='padding:3px 12px;'>Restricted Hartree–Fock</td>"
            "  <td style='padding:3px 12px;'>All electrons are paired → multiplicity = 1</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>UHF</b></td>"
            "  <td style='padding:3px 12px;'>Unrestricted Hartree–Fock</td>"
            "  <td style='padding:3px 12px;'>Any unpaired electrons → multiplicity ≥ 2</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>B3LYP</b></td>"
            "  <td style='padding:3px 12px;'>DFT hybrid functional</td>"
            "  <td style='padding:3px 12px;'>General organic chemistry, good accuracy</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>PBE</b></td>"
            "  <td style='padding:3px 12px;'>DFT GGA functional</td>"
            "  <td style='padding:3px 12px;'>Large molecules, speed-critical</td></tr>"
            "</table>"
            "<p><b>Quick guide:</b></p>"
            "<ul>"
            "<li>Neutral H₂O, CH₄, NH₃ → <b>RHF</b> (singlet)</li>"
            "<li>O₂ (triplet) → <b>UHF</b></li>"
            "<li>OH radical → <b>UHF</b> (doublet)</li>"
            "<li>General organic molecules → <b>B3LYP/6-31G*</b> for research quality</li>"
            "</ul>"
            "<p>If you pick RHF for an open-shell molecule, the calculation will "
            "likely fail or give wrong energies. QuantUI will auto-switch to UHF "
            "when you set multiplicity &gt; 1.</p>"
        ),
    },
    "basis_set": {
        "title": "Choosing a basis set",
        "body": (
            "<p>A <b>basis set</b> is the mathematical toolkit used to describe "
            "electron orbitals. Larger basis sets are more accurate but slower.</p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            f"<tr style='border-bottom:1px solid {_theme.css.BORDER_LEGACY};'>"
            "  <th style='padding:3px 12px; text-align:left;'>Basis set</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Speed</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Accuracy</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Best for</th></tr>"
            "<tr><td style='padding:3px 12px;'>STO-3G</td>"
            "  <td style='padding:3px 12px;'>⚡ Very fast</td>"
            "  <td style='padding:3px 12px;'>Low</td>"
            "  <td style='padding:3px 12px;'>Learning, quick tests</td></tr>"
            "<tr><td style='padding:3px 12px;'>3-21G</td>"
            "  <td style='padding:3px 12px;'>⚡ Fast</td>"
            "  <td style='padding:3px 12px;'>Low–Medium</td>"
            "  <td style='padding:3px 12px;'>Quick estimates</td></tr>"
            "<tr><td style='padding:3px 12px;'>6-31G</td>"
            "  <td style='padding:3px 12px;'>Moderate</td>"
            "  <td style='padding:3px 12px;'>Medium</td>"
            "  <td style='padding:3px 12px;'>General purpose</td></tr>"
            "<tr><td style='padding:3px 12px;'>6-31G*</td>"
            "  <td style='padding:3px 12px;'>Moderate</td>"
            "  <td style='padding:3px 12px;'>Good</td>"
            "  <td style='padding:3px 12px;'>Research-quality (recommended)</td></tr>"
            "<tr><td style='padding:3px 12px;'>cc-pVDZ</td>"
            "  <td style='padding:3px 12px;'>Slower</td>"
            "  <td style='padding:3px 12px;'>High</td>"
            "  <td style='padding:3px 12px;'>Correlation-consistent studies</td></tr>"
            "<tr><td style='padding:3px 12px;'>cc-pVTZ</td>"
            "  <td style='padding:3px 12px;'>🐢 Slow</td>"
            "  <td style='padding:3px 12px;'>Very high</td>"
            "  <td style='padding:3px 12px;'>Benchmark / publication</td></tr>"
            "</table>"
            "<p><b>Recommendation:</b> Start with <b>STO-3G</b> for learning. "
            "Use <b>6-31G*</b> for serious work. Only use cc-pVTZ if you need "
            "high-accuracy results and have time to wait.</p>"
            "<p><b>Transition metals and heavy elements:</b> the Pople "
            "(<code>6-31G</code>…) and Dunning (<code>cc-pV*</code>) sets do "
            "<b>not</b> cover most metals, so a calculation on, say, a platinum "
            "or cobalt complex will stop with a message asking you to switch. "
            "Use <b>def2-SVP</b> or <b>def2-TZVP</b> — these carry effective "
            "core potentials that cover the whole periodic table (or "
            "<b>LANL2DZ</b>, an ECP basis for the heaviest centres). Remember to "
            "set the <b>charge and multiplicity</b> from the metal's oxidation "
            "state, and for a reliable starting geometry load one of the "
            "bundled inorganic examples (cisplatin, hexaamminecobalt(III), "
            "ferrocene) or paste your own coordinates in the <b>XYZ Input</b> "
            "tab rather than relying on an online name search, which often "
            "returns a disconnected salt form for coordination compounds.</p>"
            # UXP2.1: the two Pople notations are a recurring source of
            # confusion — a reader who only knows 6-31G(d) can conclude the
            # 6-31G* in the dropdown is a different set they can't select.
            "<h4 style='margin:14px 0 4px'>Reading the names: two notations, "
            "one basis set</h4>"
            "<p>Pople basis sets are written two equivalent ways. The starred "
            "and parenthesised forms are <b>the same basis set</b> — QuantUI's "
            "dropdown uses the starred spelling, and either is accepted:</p>"
            "<table style='border-collapse:collapse; margin:4px 0 8px 0;'>"
            "<tr style='border-bottom:1px solid #ddd;'>"
            "  <th style='padding:3px 12px; text-align:left;'>Starred</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Parenthesised</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Adds</th></tr>"
            "<tr><td style='padding:3px 12px;'><b>6-31G*</b></td>"
            "  <td style='padding:3px 12px;'><b>6-31G(d)</b></td>"
            "  <td style='padding:3px 12px;'>d functions on heavy atoms</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>6-31G**</b></td>"
            "  <td style='padding:3px 12px;'><b>6-31G(d,p)</b></td>"
            "  <td style='padding:3px 12px;'>…plus p functions on hydrogens</td></tr>"
            "</table>"
            "<p>Two more markers you will meet in the literature:</p>"
            "<ul>"
            "<li><b>+ and ++</b> add <i>diffuse</i> functions, which extend the "
            "basis further from the nucleus — needed for <b>anions</b>, lone "
            "pairs and weakly-bound species. <code>6-31+G*</code> puts them on "
            "heavy atoms; <code>6-31++G**</code> on hydrogens too.</li>"
            "<li><b>Dunning sets have no star notation at all.</b> "
            "<code>cc-pVDZ</code> / <code>cc-pVTZ</code> include polarisation "
            "by construction (that is the <i>p</i> in pV), so a missing "
            "<code>*</code> does not mean a missing feature. The diffuse "
            "counterpart is the <code>aug-</code> prefix "
            "(<code>aug-cc-pVDZ</code>), not a <code>+</code>.</li>"
            "</ul>"
        ),
    },
    "density_fitting": {
        "title": "Density fitting (RI) — a speed vs. accuracy trade-off",
        "body": (
            "<p><b>Density fitting</b> (also called the <b>resolution of the "
            "identity</b>, or RI) is a shortcut for the most expensive part of a "
            "DFT or Hartree–Fock calculation: the electron–electron repulsion "
            "integrals. Instead of computing them exactly, it approximates them "
            "using a smaller auxiliary basis. You are trading a tiny amount of "
            "accuracy for speed.</p>"
            "<p><b>It is off by default</b>, and it is a toggle on the Status "
            "tab. It is not a blanket win, so QuantUI does not turn it on for "
            "you — here is why.</p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            f"<tr style='border-bottom:1px solid {_theme.css.BORDER_LEGACY};'>"
            "  <th style='padding:3px 12px; text-align:left;'>Calculation</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Effect of DF</th></tr>"
            "<tr><td style='padding:3px 12px;'>TD-DFT (UV-Vis) and larger "
            "molecules</td>"
            "  <td style='padding:3px 12px;'>⚡ Noticeably faster — this is the "
            "case it is built for</td></tr>"
            "<tr><td style='padding:3px 12px;'>Small single points</td>"
            "  <td style='padding:3px 12px;'>Roughly neutral, sometimes slightly "
            "<i>slower</i> (building the auxiliary integrals costs more than it "
            "saves for a small molecule)</td></tr>"
            "</table>"
            "<p><b>How much accuracy do you lose?</b> On a real test case "
            "(aspirin, B3LYP/6-31G*), the total energy changed by about "
            "<b>0.008 kcal/mol</b> — hundreds of times smaller than "
            "&#8216;chemical accuracy&#8217; (1 kcal/mol), and far below the "
            "error of the functional itself. For most purposes it is "
            "invisible.</p>"
            "<p><b>When to switch it on:</b> longer TD-DFT runs or larger "
            "systems where you feel the wait. <b>When to leave it off:</b> when "
            "you want the number to match a textbook or a reference exactly, or "
            "for small quick calculations where it will not help. If you are "
            "unsure, leaving it off is the safe, exact choice.</p>"
        ),
    },
    "homo_lumo": {
        "title": "What is the HOMO-LUMO gap?",
        "body": (
            "<p>The <b>HOMO-LUMO gap</b> is the energy difference between the "
            "Highest Occupied Molecular Orbital (HOMO) and the Lowest Unoccupied "
            "Molecular Orbital (LUMO).</p>"
            "<ul>"
            "<li><b>Large gap</b> → chemically stable, electrically insulating, "
            "absorbs UV light (colorless)</li>"
            "<li><b>Small gap</b> → more reactive, semiconductor-like, may absorb "
            "visible light (colored)</li>"
            "</ul>"
            "<p><b>Typical values:</b></p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            f"<tr style='border-bottom:1px solid {_theme.css.BORDER_LEGACY};'>"
            "  <th style='padding:3px 12px; text-align:left;'>Molecule</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Gap (eV)</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Character</th></tr>"
            "<tr><td style='padding:3px 12px;'>H₂O</td>"
            "  <td style='padding:3px 12px;'>~12 eV</td>"
            "  <td style='padding:3px 12px;'>Insulator</td></tr>"
            "<tr><td style='padding:3px 12px;'>Benzene</td>"
            "  <td style='padding:3px 12px;'>~5 eV</td>"
            "  <td style='padding:3px 12px;'>Aromatic, UV absorber</td></tr>"
            "<tr><td style='padding:3px 12px;'>Beta-carotene</td>"
            "  <td style='padding:3px 12px;'>~2 eV</td>"
            "  <td style='padding:3px 12px;'>Orange pigment, visible absorber</td></tr>"
            "</table>"
            "<p><b>Note:</b> Hartree–Fock systematically overestimates the HOMO-LUMO "
            "gap. DFT methods (B3LYP, PBE) give more realistic values for comparison "
            "with experiment.</p>"
        ),
    },
    "reading_results": {
        "title": "Reading your results",
        "body": (
            "<p>After a calculation completes, QuantUI reports several key quantities:</p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            f"<tr style='border-bottom:1px solid {_theme.css.BORDER_LEGACY};'>"
            "  <th style='padding:3px 12px; text-align:left;'>Quantity</th>"
            "  <th style='padding:3px 12px; text-align:left;'>What it means</th></tr>"
            "<tr><td style='padding:3px 12px;'><b>Total energy</b></td>"
            "  <td style='padding:3px 12px;'>Electronic energy of the molecule. "
            "Absolute value has no chemical meaning — use <i>differences</i> between "
            "two calculations (e.g. reactant vs. product) to get reaction energies.</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>Ha (Hartree)</b></td>"
            "  <td style='padding:3px 12px;'>Atomic unit of energy. "
            "1 Ha = 27.211 eV = 627.5 kcal/mol.</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>eV (electron-volt)</b></td>"
            "  <td style='padding:3px 12px;'>Common unit for orbital energies. "
            "1 eV ≈ 23.06 kcal/mol.</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>HOMO-LUMO gap</b></td>"
            "  <td style='padding:3px 12px;'>See the HOMO-LUMO Gap help topic.</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>SCF converged</b></td>"
            "  <td style='padding:3px 12px;'><b>Yes</b> = the self-consistent field "
            "loop found a stable solution. <b>No</b> = the solution may be unreliable "
            "— try a different starting geometry or method.</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>SCF iterations</b></td>"
            "  <td style='padding:3px 12px;'>Number of cycles to reach convergence. "
            "Typical: 10–30 for simple molecules. More iterations = harder system.</td></tr>"
            "</table>"
            "<p><b>Tip:</b> The <i>Compare</i> tool (Advanced section) lets you "
            "run multiple calculations and view energy differences in one table.</p>"
        ),
    },
    "measure": {
        "title": "Click-to-measure (bond / angle / dihedral)",
        "body": (
            "<p>Click atoms directly in the Analysis tab's molecule viewer to "
            "read off geometry, the same way GaussView's picker works:</p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            f"<tr style='border-bottom:1px solid {_theme.css.BORDER_LEGACY};'>"
            "  <th style='padding:3px 12px; text-align:left;'>Clicks</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Shows</th></tr>"
            "<tr><td style='padding:3px 12px;'>1 atom</td>"
            "  <td style='padding:3px 12px;'>which atom is selected</td></tr>"
            "<tr><td style='padding:3px 12px;'>2 atoms</td>"
            "  <td style='padding:3px 12px;'>bond length, in Å</td></tr>"
            "<tr><td style='padding:3px 12px;'>3 atoms</td>"
            "  <td style='padding:3px 12px;'>+ the angle at the 2nd atom, in "
            "degrees</td></tr>"
            "<tr><td style='padding:3px 12px;'>4 atoms</td>"
            "  <td style='padding:3px 12px;'>+ the dihedral (torsion) angle, "
            "in degrees</td></tr>"
            "</table>"
            "<p>A 5th click starts a new chain from that atom. Clicking an "
            "atom already in the chain is ignored — repeating an atom makes "
            "the geometry undefined. Picked atoms are highlighted in the "
            "viewer; <b>Clear</b> resets the selection.</p>"
            "<p><b>Note:</b> this needs the <b>py3Dmol</b> viewer — if the "
            "panel shows a message instead of the picker, switch backends "
            "with the toggle above the viewer (or in Settings). Switching "
            "molecules or leaving the Analysis tab clears the current "
            "selection, so a measurement is never shown against the wrong "
            "structure.</p>"
            "<p>If four picked atoms include three that fall in a straight "
            "line, the dihedral has no defined plane — the panel reports "
            "this rather than showing a number.</p>"
        ),
    },
    "mulliken": {
        "title": "Mulliken populations (partial charges)",
        "body": (
            "<p><b>Mulliken population analysis</b> partitions the SCF electron "
            "density onto atoms, giving a partial charge on each atom (in units "
            "of the elementary charge <i>e</i>).</p>"
            "<ul>"
            "<li><b>Negative</b> charge → that atom has more electron density "
            "than its nuclear charge (electron-rich).</li>"
            "<li><b>Positive</b> charge → electron-deficient.</li>"
            "</ul>"
            "<p>The Analysis tab's <b>Mulliken Populations</b> accordion shows "
            "a per-atom table and a bar chart after a Single Point or Geometry "
            "Optimization. The sum of the charges should match the molecular "
            "charge you set.</p>"
            "<p>Two overlays on the 3D viewer (py3Dmol) are toggled from the "
            "same panel:</p>"
            "<ul>"
            "<li><b>Color atoms by charge</b> — red = electron-rich (−), "
            "blue = electron-deficient (+).</li>"
            "<li><b>Show dipole arrow</b> — green arrow along μ through the "
            "centre of mass. Needs the saved μ<sub>x</sub>, μ<sub>y</sub>, "
            "μ<sub>z</sub> components (re-run older History results to get "
            "them).</li>"
            "</ul>"
            "<p><b>Dipole moment</b> (Debye) comes from the full SCF density — "
            "not from summing Mulliken point charges. The two numbers are "
            "related but not identical.</p>"
            "<p><b>Caveat for students:</b> Mulliken charges are "
            "<i>basis-set dependent</i>. Comparing charges across very "
            "different basis sets (e.g. STO-3G vs. def2-TZVP) is unreliable; "
            "use them for qualitative polarity within one calculation.</p>"
        ),
    },
    "resuming_calculations": {
        "title": "Resuming an interrupted calculation",
        "body": (
            "<p>Long calculations save their progress as they go, so a run that "
            "is cancelled, fails, or is cut short by a closing laptop does not "
            "have to start over.</p>"
            "<p><b>If you are still in the same session</b> — the failure "
            "message itself tells you the run can be resumed. Leave every "
            "setting exactly as it was and return to the <i>Calculate</i> tab. "
            "A line appears just above the <b>Run</b> button:</p>"
            f"<blockquote style='border-left:3px solid {_theme.css.BORDER_LEGACY};padding:6px 12px;"
            f"margin:8px 0;font-size:13px;color:{_theme.css.TEXT_LABEL}'>"
            "&#9851; An interrupted run of this exact calculation was found — "
            "8 of 20 scan points already computed."
            "</blockquote>"
            "<p>A <b>Resume from checkpoint</b> checkbox appears with it, already "
            "ticked. Press <b>Run</b> and the finished work is reused. Untick it "
            "first if you would rather start clean.</p>"
            "<p><b>If you have closed and reopened QuantUI</b>, you no longer "
            "have to remember the settings. The <i>History</i> tab lists every "
            "unfinished calculation under <b>Unfinished calculations</b> — "
            "molecule, calculation type, level of theory, how much was "
            "completed and how long ago. Pick one and press <b>Load these "
            "settings</b>: the molecule and every setting are put back on the "
            "Calculate tab, where the resume offer then appears. <b>Discard</b> "
            "deletes a checkpoint you have finished with. The section is hidden "
            "entirely when nothing is unfinished.</p>"
            "<p><b>The settings must match.</b> The offer only appears when the "
            "calculation you have configured is <i>identical</i> to the "
            "interrupted one — same molecule, geometry, charge, multiplicity, "
            "method, basis and calculation type. Change any of them and the "
            "offer disappears, because resuming into a different calculation "
            "would silently mix two sets of results. To get the offer back, "
            "put the setting you changed back as it was.</p>"
            "<p><b>What gets saved, by calculation type:</b></p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            f"<tr style='border-bottom:1px solid {_theme.css.BORDER_LEGACY};'>"
            "  <th style='padding:3px 12px; text-align:left;'>Type</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Resuming picks up…</th></tr>"
            "<tr><td style='padding:3px 12px;'><b>Geometry Opt</b></td>"
            "  <td style='padding:3px 12px;'>From the last completed step. The "
            "optimizer also reloads what it had learned about the energy surface, "
            "so it does not lose momentum.</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>PES Scan</b></td>"
            "  <td style='padding:3px 12px;'>Every point already computed is "
            "reused; only the missing ones are run.</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>Single Point</b></td>"
            "  <td style='padding:3px 12px;'>Nothing to resume — but see warm "
            "starts below.</td></tr>"
            "<tr><td style='padding:3px 12px;'><b>Frequency</b></td>"
            "  <td style='padding:3px 12px;'>Not yet resumable.</td></tr>"
            "</table>"
            "<p><b>Warm starts happen automatically.</b> Separately from resuming, "
            "QuantUI reuses the converged electron density from an earlier run of "
            "the same molecule, charge, method and basis as the starting guess for "
            "a new one. The geometry does <i>not</i> have to match, so this speeds "
            "up a series of related calculations. You will see "
            "<code>&#9851; Warm start</code> in the Output tab when it happens. "
            "Nothing to switch on.</p>"
            "<p><b>Where it is stored.</b> In <code>~/.quantui/checkpoints</code>, "
            "separate from your saved results — a checkpoint exists for runs that "
            "never produced a result. They are cleaned up automatically after "
            "two weeks, and deleting the folder is always safe: the only cost is "
            "that an interrupted run can no longer be resumed.</p>"
            "<p><b>If a calculation keeps failing at the same point</b>, resuming "
            "will keep hitting the same failure. Untick <b>Resume from "
            "checkpoint</b> and change something — a smaller basis set, a better "
            "starting geometry — rather than retrying the same work.</p>"
        ),
    },
    "citing_pyscf": {
        "title": "How to cite PySCF",
        "body": (
            "<p>If you use QuantUI results in a report or publication, cite PySCF:</p>"
            f"<blockquote style='border-left:3px solid {_theme.css.BORDER_LEGACY};padding:6px 12px;"
            f"margin:8px 0;font-size:13px;color:{_theme.css.TEXT_LABEL}'>"
            "Q. Sun, X. Zhang, S. Banerjee, P. Bao, M. Barbry, N. S. Blunt, "
            "N. A. Bogdanov, G. H. Booth, J. Chen, Z.-H. Cui, J. J. Eriksen, "
            "Y. Gao, S. Guo, J. Hermann, M. R. Hermes, K. Koh, P. Koval, "
            "S. Lehtola, Z. Li, J. Liu, N. Mardirossian, J. D. McClain, M. Motta, "
            "B. Mussard, H. Q. Pham, A. Pulkin, W. Purwanto, P. J. Robinson, "
            "E. Ronca, E. R. Sayfutyarova, M. Scheurer, H. F. Schurkus, "
            "J. E. T. Smith, C. Sun, S.-N. Sun, S. Upadhyay, L. K. Wagner, "
            "X. Wang, A. White, J. D. Whitfield, M. J. Williamson, S. Wouters, "
            "J. Yang, J. M. Yu, T. Zhu, T. C. Berkelbach, S. Sharma, A. Y. Sokolov, "
            "and G. K.-L. Chan, "
            "<i>J. Chem. Phys.</i> <b>153</b>, 024109 (2020)."
            "</blockquote>"
            "<p>Also cite QuantUI (your instructor will provide the reference).</p>"
            "<p><b>BibTeX key:</b> <code>Sun2020</code> — search for "
            "'PySCF 2020' in Google Scholar or your reference manager.</p>"
        ),
    },
    "charge": {
        "title": "What is molecular charge?",
        "body": (
            "<p>The <b>charge</b> is the total electric charge of the molecule, "
            "measured in units of the elementary charge (<i>e</i>).</p>"
            "<ul>"
            "<li><b>0</b> — neutral molecule (most common)</li>"
            "<li><b>+1</b> — cation (one electron removed), e.g. NH₄⁺</li>"
            "<li><b>−1</b> — anion (one electron added), e.g. Cl⁻</li>"
            "</ul>"
            "<p><b>Tip:</b> If you are unsure, start with charge = 0. "
            "Most stable molecules are neutral.</p>"
        ),
    },
    "multiplicity": {
        "title": "What is spin multiplicity?",
        "body": (
            "<p><b>Multiplicity</b> = 2S + 1, where S is the total electron spin. "
            "It tells the computer how many unpaired electrons the molecule has.</p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            f"<tr style='border-bottom:1px solid {_theme.css.BORDER_LEGACY};'>"
            "  <th style='padding:3px 12px; text-align:left;'>Unpaired e⁻</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Multiplicity</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Name</th>"
            "  <th style='padding:3px 12px; text-align:left;'>Example</th></tr>"
            "<tr><td style='padding:3px 12px;'>0</td>"
            "  <td style='padding:3px 12px;'>1</td>"
            "  <td style='padding:3px 12px;'>Singlet</td>"
            "  <td style='padding:3px 12px;'>H₂O, CH₄</td></tr>"
            "<tr><td style='padding:3px 12px;'>1</td>"
            "  <td style='padding:3px 12px;'>2</td>"
            "  <td style='padding:3px 12px;'>Doublet</td>"
            "  <td style='padding:3px 12px;'>NO, OH radical</td></tr>"
            "<tr><td style='padding:3px 12px;'>2</td>"
            "  <td style='padding:3px 12px;'>3</td>"
            "  <td style='padding:3px 12px;'>Triplet</td>"
            "  <td style='padding:3px 12px;'>O₂</td></tr>"
            "</table>"
            "<p><b>Rule of thumb:</b> Use <b>1</b> (singlet) for most closed-shell "
            "molecules. Use <b>2</b> for radicals. Use <b>3</b> for O₂.</p>"
            "<p>The charge and multiplicity must be consistent with the electron count. "
            "QuantUI will warn you if the combination is impossible.</p>"
        ),
    },
    "external_tools": {
        "title": "Importing results into Avogadro / IQmol / Jmol",
        "body": (
            "<p>Every QuantUI result folder ships with portable, standards-"
            "compliant files. No screen-scraping — open the right file in "
            "the right tool.</p>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            f"<tr style='border-bottom:1px solid {_theme.css.BORDER_LEGACY};'>"
            "  <th style='padding:3px 12px; text-align:left;'>What you want to do</th>"
            "  <th style='padding:3px 12px; text-align:left;'>QuantUI file</th>"
            "  <th style='padding:3px 12px; text-align:left;'>External tool</th></tr>"
            "<tr><td style='padding:3px 12px;'>View MOs in 3D</td>"
            "  <td style='padding:3px 12px;'><code>result.molden</code></td>"
            "  <td style='padding:3px 12px;'>Avogadro, IQmol, Jmol</td></tr>"
            "<tr><td style='padding:3px 12px;'>Animate vibrations</td>"
            "  <td style='padding:3px 12px;'><code>result.molden</code> (freq)</td>"
            "  <td style='padding:3px 12px;'>Avogadro 2</td></tr>"
            "<tr><td style='padding:3px 12px;'>Replay a trajectory</td>"
            "  <td style='padding:3px 12px;'><code>trajectory.xyz</code> or <code>.traj</code></td>"
            "  <td style='padding:3px 12px;'>VMD, Avogadro, ASE-GUI</td></tr>"
            "<tr><td style='padding:3px 12px;'>Render an orbital isosurface</td>"
            "  <td style='padding:3px 12px;'><code>isosurfaces/&lt;orb&gt;.cube</code></td>"
            "  <td style='padding:3px 12px;'>Avogadro, VMD, ChimeraX</td></tr>"
            "<tr><td style='padding:3px 12px;'>Open spectrum data in Excel</td>"
            "  <td style='padding:3px 12px;'><code>*_data_*.csv</code></td>"
            "  <td style='padding:3px 12px;'>Excel, LibreOffice, pandas</td></tr>"
            "<tr><td style='padding:3px 12px;'>Share the whole result</td>"
            "  <td style='padding:3px 12px;'><code>&lt;result&gt;.zip</code> (Export bundle)</td>"
            "  <td style='padding:3px 12px;'>Any unzip tool</td></tr>"
            "<tr><td style='padding:3px 12px;'>Edit a structure and re-run</td>"
            "  <td style='padding:3px 12px;'><code>trajectory.traj</code></td>"
            "  <td style='padding:3px 12px;'>ASE-GUI</td></tr>"
            "</table>"
            "<p><b>Quick paths:</b></p>"
            "<ul>"
            "<li><b>Avogadro 2:</b> <code>File → Open → result.molden</code>; for "
            "vibrations use <b>Extensions → Vibrational Modes</b>.</li>"
            "<li><b>IQmol:</b> <code>File → Open → result.molden</code>; "
            "double-click an orbital in the side panel to render its isosurface.</li>"
            "<li><b>VMD:</b> <code>vmd -m trajectory.xyz</code> for large trajectories.</li>"
            "<li><b>ASE Python:</b> <code>frames = ase.io.read('trajectory.traj', ':')</code> "
            "— per-frame energies are preserved in eV.</li>"
            "</ul>"
            "<p><b>Find the files:</b> open the <b>Files tab</b>, browse to the "
            "result folder, and either preview each file there or open the folder "
            "in your OS file manager.</p>"
            "<p>Full guide with per-tool details, troubleshooting, and a sample "
            "result-folder layout: see <code>docs/IMPORTING-INTO-AVOGADRO.md</code> "
            "in the QuantUI repo.</p>"
        ),
    },
    "finding_structures": {
        "title": "Finding a molecule (library, search, SMILES)",
        "body": (
            "<p>The <b>Molecule Input</b> panel offers three ways to get a "
            "starting structure — all of which work fully offline except the "
            "Online Search.</p>"
            "<ul>"
            "<li><b>Library</b> — browse the bundled offline library: filter by "
            "<b>category</b> (amino acids, solvents, drugs, …) or type in the "
            "<b>search</b> box to match by name or formula. It includes a small "
            "curated set plus thousands of small molecules (QM9), so you can "
            "work with no internet connection.</li>"
            "<li><b>XYZ Input</b> — paste raw <code>element x y z</code> "
            "coordinates, one atom per line.</li>"
            "<li><b>Online Search</b> — type a <b>name</b> (aspirin), "
            "<b>SMILES</b> (<code>CC(=O)O</code>), <b>CID</b> (2244), or "
            "<b>InChI</b>. SMILES/InChI are built locally by RDKit (no network); "
            "names and CIDs are resolved by trying <b>PubChem</b>, then "
            "<b>NCI CACTUS</b> (which also understands CAS numbers), and finally "
            "the bundled library if you are offline. When a name matches several "
            "compounds (e.g. <i>xylene</i>), a <b>pick-list</b> appears so you "
            "choose the right one instead of guessing.</li>"
            "</ul>"
            "<p><b>Provenance:</b> the status line states where a structure came "
            "from — the library, a local SMILES build, PubChem, or CACTUS — so "
            "you know whether coordinates are experimental, DFT-optimized, or "
            "force-field-embedded. Either way, treat them as a <i>starting "
            "point</i> and run a geometry optimization for accurate results.</p>"
            "<p><b>Quick clean-up:</b> next to <i>Classical pre-optimize "
            "geometry</i>, click <b>Preview</b> to relax the structure with a "
            "fast force field (MMFF94/UFF). Use the controls under the viewer to "
            "play the relaxation, scrub to any step, or click <b>&#8644;</b> to "
            "flip between your input and the relaxed result for a direct "
            "comparison &mdash; then <b>Keep this geometry</b> to adopt it (it "
            "becomes the active structure your calculation runs on) or "
            "<b>Revert</b> to discard. Nothing is changed unless you Keep it.</p>"
        ),
    },
}

# All valid topic keys (for testing / discovery)
VALID_TOPICS = frozenset(HELP_TOPICS.keys())


# ---------------------------------------------------------------------------
# Widget builder
# ---------------------------------------------------------------------------

_PANEL_CSS = (
    "border: 1px solid #ddd; border-radius: 6px; padding: 8px 12px; "
    "margin: 4px 0 8px 0; background: #f8f9fa; max-width: 620px;"
)


def help_panel(topic: str) -> widgets.HTML:
    """
    Return a collapsible HTML help widget for a given topic.

    The widget uses a ``<details>/<summary>`` element so it starts
    collapsed and can be expanded by clicking.

    Args:
        topic: One of the keys in :data:`HELP_TOPICS` — ``'method'``,
               ``'basis_set'``, ``'homo_lumo'``, ``'reading_results'``,
               ``'citing_pyscf'``, ``'charge'``, or ``'multiplicity'``.

    Returns:
        ``ipywidgets.HTML`` widget ready for ``display()``.

    Raises:
        KeyError: If topic is not in :data:`HELP_TOPICS`.
    """
    if topic not in HELP_TOPICS:
        raise KeyError(
            f"Unknown help topic '{topic}'. "
            f"Valid topics: {', '.join(sorted(HELP_TOPICS))}"
        )

    entry = HELP_TOPICS[topic]
    html = (
        f'<details style="{_PANEL_CSS}">'
        f'<summary style="cursor:pointer; font-weight:bold; color:#0366d6;">'
        f'ℹ️ {entry["title"]}</summary>'
        f'<div style="margin-top:6px; font-size:13px;">{entry["body"]}</div>'
        f"</details>"
    )
    return widgets.HTML(value=html)
