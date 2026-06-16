# Vendored JavaScript assets

## 3Dmol-min.js

- **Version:** 3Dmol.js 2.5.4
- **Source:** https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js
- **License:** BSD-3-Clause (see `3Dmol-min.js.LICENSE.txt`)
- **Why vendored:** py3Dmol's generated HTML loads 3Dmol.js from the jsDelivr
  CDN at render time. That fetch fails silently with no network (offline
  classroom deployment) or under a restrictive CSP, leaving every 3D view
  (molecule, trajectory, vibration, orbital isosurface) blank. We ship the
  exact build py3Dmol targets and load it from a `data:` URI instead — see
  `quantui/viz_assets.py`. No new Python dependency.

To update: download the matching `3dmol@<version>/build/3Dmol-min.js`, bump
`THREEDMOL_VERSION` in `viz_assets.py`, and confirm the version still matches
py3Dmol's `view()` constructor default `js=` URL (so the generated viewer
calls stay API-compatible).
