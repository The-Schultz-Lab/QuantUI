# Features

From molecule input to spectra and history replay &mdash; everything runs in your
local Python kernel.

<div class="quantui-landing" markdown="0">
<div class="section" style="padding-top: 0;">
<div class="features-grid">

<div class="card">
  <div class="feature-card__icon">⚗️</div>
  <div class="feature-card__title">Molecule Input</div>
  <p class="feature-card__body">
    Paste XYZ, browse an indexed three-tier bundled library (presets +
    curated + ~1,900 QM9 structures, searchable by name/formula), or run
    a structure search by name, SMILES, InChI, CID, or CAS &mdash;
    PubChem &rarr; NCI CACTUS &rarr; an <strong>offline</strong>
    bundled-library fallback, so the search still works with no network.
  </p>
</div>

<div class="card">
  <div class="feature-card__icon">🔬</div>
  <div class="feature-card__title">3D Visualization</div>
  <p class="feature-card__body">
    py3Dmol-first interactive viewer with a capability-aware
    backend router. Molecules, optimization trajectories,
    vibrational modes, and orbital isosurfaces all render inline
    &mdash; and <strong>fully offline</strong> (3Dmol.js is vendored,
    never fetched from a CDN). Tunable playback FPS + on-disk cache
    for instant replay.
  </p>
</div>

<div class="card">
  <div class="feature-card__icon">⚡</div>
  <div class="feature-card__title">Calculations</div>
  <p class="feature-card__body">
    RHF, UHF, nine DFT functionals, MP2, CCSD, and CCSD(T) &mdash;
    with six calculation types: single point, geometry optimization,
    frequencies/thermochemistry, TD-DFT UV-Vis, NMR shielding,
    and 1D PES scans. PCM implicit solvation included.
  </p>
</div>

<div class="card">
  <div class="feature-card__icon">📊</div>
  <div class="feature-card__title">Spectra &amp; Analysis</div>
  <p class="feature-card__body">
    IR spectrum (stick + Lorentzian-broadened), UV-Vis plot,
    orbital energy-level diagram with HOMO/LUMO isosurfaces,
    and <sup>1</sup>H/<sup>13</sup>C NMR chemical shifts vs TMS.
    Side-by-side comparison table for multiple calculations.
  </p>
</div>

<div class="card">
  <div class="feature-card__icon">📂</div>
  <div class="feature-card__title">Exports &amp; History</div>
  <p class="feature-card__body">
    Every calc auto-saves to a timestamped directory and replays
    after a kernel restart. Export structures (XYZ, MOL/SDF, PDB),
    orbital data (Molden), trajectories (multi-frame XYZ, ASE
    <code class="inline-code">.traj</code>), cube files, spectra
    as HTML, full result bundles as <code class="inline-code">.zip</code>,
    or any run as a standalone <code class="inline-code">.py</code> script.
  </p>
</div>

<div class="card">
  <div class="feature-card__icon">🚀</div>
  <div class="feature-card__title">GPU Acceleration</div>
  <p class="feature-card__body">
    Optional NVIDIA GPU offload via
    <a class="hero__link" href="https://github.com/pyscf/gpu4pyscf" target="_blank" rel="noopener">gpu4pyscf</a>
    &mdash; RHF, UHF, RKS/UKS DFT, and TD-DFT auto-migrate to GPU
    when available. Numerical IR-intensity SCFs also offload. Set
    <code class="inline-code">QUANTUI_DISABLE_GPU=1</code> to force
    CPU; the result card always shows which device produced the numbers.
  </p>
</div>

<div class="card">
  <div class="feature-card__icon">📈</div>
  <div class="feature-card__title">Time Estimator &amp; Calibration</div>
  <p class="feature-card__body">
    Four-tier calibration suite anchors a per-machine time-prediction
    model with GPU-vs-CPU partitioning, IQR outlier rejection, and
    variance-aware confidence labels. Pre-run estimates show in the
    Calculate tab; predicted-vs-actual accuracy accrues automatically
    in the analytics dashboard.
  </p>
</div>

<div class="card">
  <div class="feature-card__icon">⌨️</div>
  <div class="feature-card__title">CLI &amp; Analytics</div>
  <p class="feature-card__body">
    The <code class="inline-code">quantui</code> CLI inspects the
    event log (<code class="inline-code">log tail</code>), probes
    GPU availability (<code class="inline-code">gpu check</code>),
    and builds a self-contained HTML analytics dashboard
    (<code class="inline-code">analytics build --open</code>) with
    GPU-vs-CPU speedup tables, method usage, and estimator-accuracy
    tracking. See the <a href="CLI.md">CLI reference</a>.
  </p>
</div>

<div class="card">
  <div class="feature-card__icon">🖥️</div>
  <div class="feature-card__title">Voil&agrave; App Mode</div>
  <p class="feature-card__body">
    Serve the notebook as a polished widget-only UI with
    <code class="inline-code">voila</code>. Light/Dark themes,
    inline log viewer, and an in-app bug-report form. Equally at
    home in a research group or a classroom.
  </p>
</div>

<div class="card">
  <div class="feature-card__icon">🧲</div>
  <div class="feature-card__title">Inorganic &amp; Coordination Complexes</div>
  <p class="feature-card__body">
    First-class transition-metal support: 14 bundled metal complexes
    (octahedral / tetrahedral / square-planar) with correct charge and
    spin, a pre-run guard that catches a metal on an incompatible basis
    (nudges to def2-SVP / LANL2DZ) and an impossible multiplicity, a
    <strong>spin-state helper</strong> that suggests high/low-spin
    multiplicities from oxidation state + geometry, optional
    <strong>GFN-FF (xtb)</strong> pre-optimization for metals, and a
    viewer that draws the coordination bonds.
  </p>
</div>

</div>
</div>
</div>
