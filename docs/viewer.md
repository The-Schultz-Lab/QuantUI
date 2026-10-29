# QuantUI Viewer

A results-only QuantUI: the **History** and **Analysis** tabs, no calculations. Orbital surfaces are limited to the HOMO and LUMO saved with each result (QuantUI 0.11.0 or later). No git or terminal needed.

## In your browser

**[Open the QuantUI Viewer](https://the-schultz-lab.github.io/QuantUI/viewer/app/voici/render/viewer.html){ .md-button .md-button--primary }**

1. Wait for it to load (the first visit takes about a minute).
2. Zip your results folder and click **Upload .zip**. It opens on demo results until you do.
3. Pick a calculation in **History**, then **→ View Analysis**.

Files stay on your computer. Use a recent Chrome, Edge, Firefox or Safari.

## Desktop app (offline)

Download the installer from the latest [release](https://github.com/The-Schultz-Lab/QuantUI/releases) (*Assets*):

| Computer | File |
| --- | --- |
| Windows | `QuantUI-Viewer-<version>-Windows-x86_64.exe` |
| Mac (Apple silicon) | `QuantUI-Viewer-<version>-MacOSX-arm64.pkg` |
| Linux | `QuantUI-Viewer-<version>-Linux-x86_64.sh` |

The installers are not code-signed, so the first run warns you:

- **Windows:** **More info** → **Run anyway**; choose *Just Me* (no admin rights).
- **Mac:** **System Settings → Privacy & Security** → **Open Anyway**.
- **Linux:** `bash QuantUI-Viewer-*.sh`

Start **QuantUI Viewer** from your apps menu, enter your results folder under **Results folder**, click **Open**. Click **Exit** when done.

## From an existing QuantUI install

If QuantUI is already installed (with the `[app]` extra), open the same viewer on a results folder with:

```bash
quantui view /path/to/results
```

See [`quantui view`](CLI.md#quantui-view) for the options.

## Needs the full app

- **Orbital surfaces other than the HOMO and LUMO, density and ESP surfaces**
  need PySCF to compute, which the Viewer does not include.
- **Thermochemistry** is shown as saved with each Frequency result (at its
  temperature and pressure); other temperatures or pressures need the full
  QuantUI. Results from older QuantUI versions may have none saved.
- The browser Viewer shows molecules with py3Dmol only.
