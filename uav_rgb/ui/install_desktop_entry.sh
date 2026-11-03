#!/usr/bin/env bash
# Install a per-user Applications-menu entry and, when available, a Desktop shortcut.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$HOME/.local/bin"
APP_DIR="$HOME/.local/share/applications"
mkdir -p "$BIN_DIR" "$APP_DIR"
WRAPPER="$BIN_DIR/uav-navigation-studio"
LAUNCHER="$(printf '%q' "$ROOT/ui/launch_studio.sh")"
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
exec /usr/bin/env bash $LAUNCHER "\$@"
EOF
chmod +x "$WRAPPER"
cat > "$APP_DIR/uav-navigation-studio.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=UAV Navigation Studio
Comment=UAV RGB training, Gazebo simulation and experiment workbench
Exec="$WRAPPER"
TryExec="$WRAPPER"
Icon=applications-science
Terminal=false
Categories=Science;Education;Development;
StartupNotify=true
EOF
chmod 644 "$APP_DIR/uav-navigation-studio.desktop"
if [[ -d "$HOME/Desktop" ]]; then
  cp "$APP_DIR/uav-navigation-studio.desktop" "$HOME/Desktop/UAV Navigation Studio.desktop"
  chmod +x "$HOME/Desktop/UAV Navigation Studio.desktop"
  if command -v gio >/dev/null 2>&1; then
    gio set "$HOME/Desktop/UAV Navigation Studio.desktop" metadata::trusted true 2>/dev/null || true
  fi
fi
echo "Installed: $APP_DIR/uav-navigation-studio.desktop"
[[ -d "$HOME/Desktop" ]] && echo "Desktop shortcut: $HOME/Desktop/UAV Navigation Studio.desktop"
echo "Launch with: $WRAPPER"
