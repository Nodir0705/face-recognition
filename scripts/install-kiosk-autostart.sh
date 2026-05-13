#!/usr/bin/env bash
# Install the Chromium kiosk autostart for the 7" touchscreen.
# Run this AFTER scripts/install.sh and AFTER the attendance service is running.
#
# This assumes Raspberry Pi OS with the desktop ("Bookworm with desktop").
# On Bookworm the desktop session is Wayland by default — Chromium kiosk
# works on both Wayland and X11 sessions.

set -euo pipefail

# Make sure we have what we need
sudo apt install -y chromium-browser unclutter curl

# Per-user autostart entry (LXDE/labwc both honor ~/.config/autostart)
AUTOSTART_DIR="$HOME/.config/autostart"
mkdir -p "$AUTOSTART_DIR"

cat > "$AUTOSTART_DIR/attendance-kiosk.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Attendance Kiosk
Exec=/home/pi/attendance_system/scripts/run-kiosk.sh
X-GNOME-Autostart-enabled=true
NoDisplay=false
EOF

chmod +x /home/pi/attendance_system/scripts/run-kiosk.sh

# Disable screen blanking system-wide for good measure
sudo raspi-config nonint do_blanking 1 || true

echo
echo "Kiosk autostart installed."
echo "Reboot the Pi:  sudo reboot"
echo
echo "After reboot, Chromium should open fullscreen on /kiosk automatically."
echo "To exit kiosk mode for maintenance: Ctrl+Alt+F1 (TTY), then sudo systemctl restart lightdm"
