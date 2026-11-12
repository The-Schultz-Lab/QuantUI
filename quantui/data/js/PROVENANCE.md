# Vendored JavaScript assets

## 3Dmol-min.js

- **Version:** 3Dmol.js 2.5.4
- **Source:** https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js
- **License:** BSD-3-Clause — full text (with the bundled GLmol, Three.js and
  jQuery notices) in `3Dmol-LICENSE.txt`, copied from the npm `3dmol@2.5.4`
  package's `LICENSE`. `3Dmol-min.js.LICENSE.txt` is only the bundle's
  webpack banner.
- **Why vendored:** py3Dmol's generated HTML loads 3Dmol.js from the jsDelivr
  CDN at render time. That fetch fails silently with no network (offline
  classroom deployment) or under a restrictive CSP, leaving every 3D view
  (molecule, trajectory, vibration, orbital isosurface) blank. We ship the
  exact build py3Dmol targets and load it from a `data:` URI instead — see
  `quantui/viz_assets.py`. No new Python dependency.

To update: download the matching `3dmol@<version>/build/3Dmol-min.js`, bump
`THREEDMOL_VERSION` in `viz_assets.py`, and confirm the version still matches
py3Dmol's `view()` constructor default `js=` URL (so the generated viewer
calls stay API-compatible). Patch-level skew is tolerated (py3Dmol 2.5.5
targets 3Dmol.js 2.5.5 while 2.5.4 is vendored); `pyproject.toml` caps py3Dmol
below the next minor, and `tests/test_viz_assets_version.py` fails when the
installed py3Dmol targets a different major.minor than the vendored bundle —
re-vendor and raise that cap together.
