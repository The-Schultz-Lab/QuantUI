# QuantUI Viewer

The Viewer is a lightweight QuantUI for looking at results you already have.
It has only the **History** and **Analysis** tabs: energies, orbital diagrams,
IR / Raman / UV-Vis spectra, optimization trajectories, vibrational modes and
populations. It cannot run calculations.

There are two ways to use it. Neither needs git or a terminal.

## In your browser (nothing to install)

**[Open the QuantUI Viewer](https://the-schultz-lab.github.io/QuantUI/viewer/app/voici/render/viewer.html){ .md-button .md-button--primary }**

1. Wait for the page to load. The first visit downloads Python into the
   browser and can take a minute; later visits are faster.
2. It opens on a small set of demo results (water). To look at your own,
   zip your results folder (right-click it → *Compress* / *Send to →
   Compressed (zipped) folder*) and click **Upload .zip**.
3. Pick a calculation in the **History** tab, then **→ View Analysis**.

Your files never leave your computer: the zip is unpacked inside the browser
tab and is gone when you close it. Use a recent Chrome, Edge, Firefox or
Safari.

## Desktop app (works offline)

Download the installer for your computer from the
[QuantUI releases page](https://github.com/The-Schultz-Lab/QuantUI/releases)
(under *Assets* of the latest release):

| Computer | File |
| --- | --- |
| Windows | `QuantUI-Viewer-<version>-Windows-x86_64.exe` |
| Mac (Apple silicon) | `QuantUI-Viewer-<version>-MacOSX-arm64.pkg` |
| Linux | `QuantUI-Viewer-<version>-Linux-x86_64.sh` |

The installers are **not code-signed**, so your computer warns you the first
time:

- **Windows:** "Windows protected your PC" → click **More info**, then
  **Run anyway**. Choose *Just Me* when asked (no admin rights needed).
- **Mac:** the installer is blocked the first time you open it. Open
  **System Settings → Privacy & Security**, scroll down to the message about
  the QuantUI installer, click **Open Anyway** and enter your password.
- **Linux:** run `bash QuantUI-Viewer-*.sh` in a terminal.

Then start **QuantUI Viewer** from the Start menu / Desktop (Windows), the
Applications folder in your home folder (Mac), or your applications menu
(Linux). Your browser opens on the Viewer: type the path of your results
folder into **Results folder** and click **Open**. Click **Exit** when you are
done.

## Not available yet

- **Orbital isosurfaces, density and ESP surfaces** need PySCF to compute,
  which the Viewer does not include. Use the full QuantUI for those.
- The browser Viewer shows molecules with py3Dmol only.
