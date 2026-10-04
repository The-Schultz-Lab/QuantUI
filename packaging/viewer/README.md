# QuantUI Viewer — lightweight History + Analysis

A cut-down QuantUI for browsing results on any laptop: only the **History** and
**Analysis** tabs, pointed at a results folder. No quantum engine (PySCF) is
installed, so nothing here can run a calculation.

It is the normal app in viewer mode — `QuantUIApp(viewer=True)` — so every
panel is the same code as the full app. From a dev install:

```bash
quantui view path/to/results     # or just `quantui view` and pick the folder in the app
```

Two ways to ship it, both built by
[`.github/workflows/viewer-builds.yml`](../../.github/workflows/viewer-builds.yml).
Where students get them:

- **Browser build** — published with the docs by
  [`pages.yml`](../../.github/workflows/pages.yml) at
  `https://the-schultz-lab.github.io/QuantUI/viewer/app/voici/render/viewer.html`
  (CDN Pyodide), linked from the docs page
  [`docs/viewer.md`](../../docs/viewer.md) (`/viewer/`). Rebuilt on every
  release and on docs / `packaging/viewer/` changes to `main`.
- **Installers** — attached to each GitHub Release by `viewer-builds.yml`
  when the release is published.

| | Desktop installer (`installer/`) | Browser build (`browser/`) |
| --- | --- | --- |
| User gets | `.exe` / `.pkg` / `.sh` plus a "QuantUI Viewer" shortcut | A web page, nothing installed |
| Results folder | Any folder on disk (type the path, **Open**) | Upload a `.zip` of it (**Upload .zip**) |
| Offline | Yes | Only if Pyodide is self-hosted (see below) |
| Size | ~240 MB download, ~1 GB installed (Linux build) | ~20–60 MB first load from CDN; ~550 MB site if self-hosted |
| Isosurfaces | Not available (needs PySCF) | Not available (needs PySCF) |
| Signing | **Unsigned** — see below | None needed |

## Desktop installer

[conda constructor](https://github.com/conda/constructor) bundles Python,
Voilà and the viewer's conda-forge dependencies; `post_install.*` installs the
QuantUI wheel and creates the shortcut (Desktop + Start menu on Windows,
`~/Applications/QuantUI Viewer.app` on macOS, a `.desktop` entry on Linux). The
shortcut runs `quantui view`, which starts Voilà on a free `127.0.0.1` port and
opens the browser. **Exit** in the app stops it.

The installers are **not code-signed**. First run:

- **Windows** — SmartScreen shows "Windows protected your PC": click
  **More info → Run anyway**. Choose "Just Me" (no admin rights needed).
- **macOS 15+** — the `.pkg` is blocked the first time. Open **System Settings →
  Privacy & Security**, scroll to the message about the installer, click
  **Open Anyway**, and enter your password. Needs an admin account. The app the
  installer creates is made on your Mac, not downloaded, so it should open
  without a further prompt (not yet confirmed on a real Mac).
- **Linux** — `bash QuantUI-Viewer-*.sh` (needs a terminal once).

Free signing for the Windows installer is available to open-source projects
through the [SignPath Foundation](https://signpath.org/); macOS notarization has
no free route.

## Browser build

[Voici](https://github.com/voila-dashboards/voici) turns the viewer notebook
into a static site whose Python kernel is Pyodide (CPython compiled to
WebAssembly). `build.sh` bundles the QuantUI wheel and its pure-Python
dependencies into the site, so nothing is fetched from PyPI at runtime.

```bash
python packaging/viewer/browser/make_samples.py samples     # demo results (needs PySCF)
packaging/viewer/browser/build.sh dist/quantui-*.whl samples site [pyodide-X.tar.bz2]
python -m http.server -d site 8000    # then open http://localhost:8000/voici/render/viewer.html
```

Serve it over HTTP (GitHub Pages works); `file://` cannot run it. Without a
Pyodide tarball the site loads Pyodide from the jsDelivr CDN; pass the tarball
from the [Pyodide releases](https://github.com/pyodide/pyodide/releases) to
self-host it (offline-capable after the first visit, but the full distribution
is ~550 MB unpruned).

Browser differences, all handled in code:

- **No threads.** `Thread.start()` raises in Pyodide, so background renders
  (vibrational animation, isosurface, export) run inline instead
  (`app_visualization._start_daemon`).
- **3D viewer is py3Dmol only.** plotlymol needs RDKit, which Pyodide lacks.
- **No Exit button** (there is no server to stop) and no local disk: results
  arrive as an uploaded `.zip`, unpacked into the page's in-memory filesystem.
  They are gone when the tab closes, and never leave the computer.

Version pins in `build.sh` matter: `jupyterlite-pyodide-kernel` 0.7.2 does not
load under `voici` 0.10.0 (built against a newer JupyterLab); 0.7.0 does.

## Known gaps

- Orbital isosurfaces need PySCF to compute the cube file, so they are not
  available in either build yet. Fix: have the full app save HOMO/LUMO cubes at
  calculation time, or evaluate orbitals on a grid with numpy alone.
- Voilà's page template loads Font Awesome icons from a CDN; offline, the
  button icons are missing (labels still show).
