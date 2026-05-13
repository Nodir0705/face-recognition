#!/usr/bin/env bash
# Install script for Raspberry Pi 4 (64-bit Bookworm)
# Run from project root: bash scripts/install.sh

set -euo pipefail

echo "==> Updating apt"
sudo apt update

echo "==> Installing system packages"
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    libatlas-base-dev libopenblas-dev libgomp1 \
    libjpeg-dev libpng-dev libavformat-dev libavcodec-dev libswscale-dev \
    libgl1 libglib2.0-0 \
    python3-picamera2 \
    sqlite3

echo "==> Creating Python venv (with system site-packages for picamera2)"
python3 -m venv .venv --system-site-packages
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip wheel

echo "==> Installing Python deps"
pip install \
    "numpy<2.0" \
    "opencv-python-headless==4.10.0.84" \
    "insightface==0.7.3" \
    "onnxruntime==1.18.1" \
    "google-api-python-client==2.140.0" \
    "google-auth==2.33.0" \
    "google-auth-httplib2==0.2.0" \
    "flask==3.0.3" \
    "PyYAML==6.0.2"

echo "==> Pre-downloading InsightFace model bundle (~50MB)"
python3 -c "from insightface.app import FaceAnalysis; \
            app = FaceAnalysis(name='buffalo_sc', \
                               providers=['CPUExecutionProvider']); \
            app.prepare(ctx_id=-1)"

echo "==> Setting up data directories"
mkdir -p data/faces data/logs

echo
echo "Install complete."
echo
echo "Next steps:"
echo "  1. Edit config/config.yaml  -- set google_sheets.spreadsheet_id"
echo "  2. Place service-account JSON at config/credentials.json"
echo "  3. Test:     source .venv/bin/activate"
echo "               PYTHONPATH=. python src/web/app.py"
echo "     Then visit http://<pi-ip>:5000/enroll  to enroll the first employee."
echo "     And     http://<pi-ip>:5000/kiosk   to see the recognition overlay."
echo "  4. Install services:"
echo "       sudo cp scripts/attendance.service scripts/attendance-sync.service \\"
echo "               /etc/systemd/system/"
echo "       sudo systemctl daemon-reload"
echo "       sudo systemctl enable --now attendance attendance-sync"
echo
echo "  5. CHANGE the admin password in /etc/systemd/system/attendance.service !"
echo
echo "  6. For the 7\" touchscreen kiosk (Chromium fullscreen):"
echo "       bash scripts/install-kiosk-autostart.sh"
echo "       sudo reboot"
echo
echo "  7. For remote admin access, see docs/remote_access.md"
