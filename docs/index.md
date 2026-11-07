---
hide:
  - toc
  - title
---

<div class="quantui-landing" markdown="0">

<section class="hero quantui-bleed">
  <div class="container">
    <div class="hero__inner">
      <div class="hero__text">
        <div class="hero__eyebrow">
          <a class="pill" href="https://pypi.org/project/quantui/" target="_blank" rel="noopener">pip install quantui</a>
          <span class="pill pill--teal">No cluster required</span>
          <span class="pill pill--teal">Runs offline</span>
          <span class="pill pill--teal">GPU-ready</span>
        </div>
        <h1 class="hero__title">Free, open, and<br>interactive quantum chemistry</h1>
        <p class="hero__subtitle">
          An all-in-one frontend for open-source computational chemistry
          packages: select and visualize your system, run DFT, and analyze
          the outputs with interactive plots
        </p>
        <div class="hero__actions">
          <a class="btn btn--primary" href="https://github.com/The-Schultz-Lab/QuantUI">View on GitHub</a>
          <a class="btn btn--outline" href="installation/">Get started</a>
          <a class="btn btn--outline" href="viewer/app/voici/render/viewer.html">Open the Viewer</a>
        </div>
        <div class="hero__meta">
          <span class="hero__stat">Python 3.9&ndash;3.11</span>
          <span class="hero__sep">&middot;</span>
          <span class="hero__stat">4,100+ tests</span>
          <span class="hero__sep">&middot;</span>
          <span class="hero__stat">MIT License</span>
          <span class="hero__sep">&middot;</span>
          <span class="hero__stat">Linux &middot; macOS &middot; WSL &middot; Windows</span>
        </div>
      </div>
      <div class="hero__visual" aria-hidden="true">
        <svg class="atom-svg" viewBox="0 0 280 280" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="7" result="blur"/>
              <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <filter id="halo" x="-80%" y="-80%" width="260%" height="260%">
              <feGaussianBlur stdDeviation="22" result="blur"/>
              <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
          </defs>
          <circle cx="140" cy="140" r="48" fill="rgba(37,99,235,0.20)" filter="url(#halo)"/>
          <ellipse class="ring ring--1" cx="140" cy="140" rx="115" ry="33" stroke="#0891b2" stroke-width="1.4" fill="none" opacity="0.70"/>
          <ellipse class="ring ring--2" cx="140" cy="140" rx="115" ry="33" stroke="#0891b2" stroke-width="1.4" fill="none" opacity="0.55"/>
          <ellipse class="ring ring--3" cx="140" cy="140" rx="115" ry="33" stroke="#3b82f6" stroke-width="1.4" fill="none" opacity="0.42"/>
          <circle class="ring ring--1" cx="255" cy="140" r="5.5" fill="#67e8f9"/>
          <circle class="ring ring--2" cx="255" cy="140" r="4.5" fill="#93c5fd"/>
          <circle class="ring ring--3" cx="255" cy="140" r="4" fill="#60a5fa"/>
          <circle cx="140" cy="140" r="20" fill="rgba(37,99,235,0.25)" filter="url(#glow)"/>
          <circle cx="140" cy="140" r="14" fill="#2563eb" filter="url(#glow)"/>
          <circle cx="140" cy="140" r="8" fill="#60a5fa"/>
          <circle cx="137" cy="137" r="3" fill="rgba(255,255,255,0.45)"/>
        </svg>
      </div>
    </div>
  </div>
</section>

<section class="section quantui-bleed" id="examples">
  <div class="container">
    <h2 class="section__title">See it in action</h2>
    <p class="section__subtitle">
      Real QuantUI output. <em>Interactive</em> panels are live: drag to rotate, press play.
    </p>
    <div class="gallery-grid">
      <figure class="gallery-figure">
        <img src="images/cisplatin_lumo.png" alt="Cisplatin LUMO isosurface" loading="lazy">
        <figcaption>
          <strong>Cisplatin LUMO</strong>
          <span>B3LYP / LANL2DZ.</span>
        </figcaption>
      </figure>
      <figure class="gallery-figure">
        <img src="images/aspartame_orbital_diagram.png" alt="Aspartame orbital energy-level diagram" loading="lazy">
        <figcaption>
          <strong>Aspartame orbital energies</strong>
          <span>HOMO&ndash;LUMO gap 6.09 eV, B3LYP / 6-31G*.</span>
        </figcaption>
      </figure>
      <figure class="gallery-figure">
        <img src="images/benzene_ir_spectrum.png" alt="Benzene IR spectrum" loading="lazy">
        <figcaption>
          <strong>Benzene IR spectrum</strong>
          <span>&omega;B97X-D / 6-31G.</span>
        </figcaption>
      </figure>
      <figure class="gallery-figure">
        <img src="images/cisplatin_optimization.png" alt="Cisplatin geometry-optimization energy convergence" loading="lazy">
        <figcaption>
          <strong>Geometry optimization</strong>
          <span>Cisplatin, 21 steps, B3LYP / LANL2DZ.</span>
        </figcaption>
      </figure>
      <figure class="gallery-figure">
        <iframe src="examples/cisplatin_trajectory.html" title="Cisplatin optimization trajectory (interactive)" loading="lazy"></iframe>
        <figcaption>
          <span class="gallery-badge">Interactive</span>
          <strong>Optimization trajectory</strong>
          <span>Cisplatin, step by step.
            <a href="examples/cisplatin_trajectory.html" target="_blank" rel="noopener">Open full &nearr;</a></span>
        </figcaption>
      </figure>
      <figure class="gallery-figure">
        <iframe src="examples/benzene_vib_mode.html" title="Benzene vibrational mode (interactive)" loading="lazy"></iframe>
        <figcaption>
          <span class="gallery-badge">Interactive</span>
          <strong>Vibrational mode</strong>
          <span>A benzene normal mode.
            <a href="examples/benzene_vib_mode.html" target="_blank" rel="noopener">Open full &nearr;</a></span>
        </figcaption>
      </figure>
    </div>
    <p class="section__cta">
      <a class="btn btn--primary" href="gallery/">Full gallery &rarr;</a>
    </p>
  </div>
</section>

<section class="quickstart quantui-bleed">
  <div class="container--narrow">
    <h2 class="quickstart__title">Ready to run your first calculation?</h2>
    <p class="quickstart__text">
      Install with pip, launch the app, then try the five tutorials.
    </p>
    <a class="btn btn--primary" href="installation/">Installation guide &rarr;</a>
  </div>
</section>

</div>
