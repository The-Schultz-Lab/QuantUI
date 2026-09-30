#!/bin/bash
# Install the bundled QuantUI wheel, then add a "QuantUI Viewer" launcher:
# a .desktop entry on Linux, an .app bundle in ~/Applications on macOS.
set -euo pipefail
"$PREFIX/bin/python" -m pip install --no-deps --no-index --no-cache-dir "$PREFIX"/quantui-*.whl
rm -f "$PREFIX"/quantui-*.whl

if [ "$(uname)" = "Darwin" ]; then
    APP="$HOME/Applications/QuantUI Viewer.app"
    mkdir -p "$APP/Contents/MacOS"
    cat > "$APP/Contents/MacOS/QuantUI Viewer" <<SH
#!/bin/bash
exec "$PREFIX/bin/quantui" view
SH
    chmod +x "$APP/Contents/MacOS/QuantUI Viewer"
    cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>QuantUI Viewer</string>
  <key>CFBundleIdentifier</key><string>org.schultzlab.quantui.viewer</string>
  <key>CFBundleExecutable</key><string>QuantUI Viewer</string>
  <key>CFBundlePackageType</key><string>APPL</string>
</dict></plist>
PLIST
else
    APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$APPS"
    cat > "$APPS/quantui-viewer.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=QuantUI Viewer
Comment=Browse QuantUI results (History + Analysis)
Exec="$PREFIX/bin/quantui" view
Terminal=true
Categories=Science;Education;
DESK
fi
