#!/usr/bin/env bash
# Launches Chromium fullscreen pointing at the kiosk URL.
# Called by the desktop autostart entry (see install-kiosk-autostart.sh).

set -euo pipefail

# Wait for the Flask app to be ready before opening the browser.
# We poll /kiosk until it returns 200, with a 60s timeout.
for i in $(seq 1 60); do
    if curl -fs -o /dev/null http://localhost:5000/kiosk; then
        break
    fi
    sleep 1
done

# Disable screen blanking + power management for the touchscreen
xset s off || true
xset -dpms || true
xset s noblank || true

# Hide the mouse cursor when idle (looks much cleaner on a kiosk)
unclutter -idle 0.5 -root &

# Clear any crash bubble from a previous unclean shutdown
PREF=~/.config/chromium/Default/Preferences
if [ -f "$PREF" ]; then
    sed -i 's/"exited_cleanly":false/"exited_cleanly":true/' "$PREF" || true
    sed -i 's/"exit_type":"[^"]*"/"exit_type":"Normal"/' "$PREF" || true
fi

# Launch Chromium in kiosk mode
# --noerrdialogs   suppresses error dialogs that could block the screen
# --disable-pinch  disables zoom on touch
# --kiosk          fullscreen, no chrome
# --incognito      no history / cookies persisted across sessions
exec chromium-browser \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --disable-translate \
    --disable-pinch \
    --overscroll-history-navigation=0 \
    --check-for-update-interval=31536000 \
    --autoplay-policy=no-user-gesture-required \
    --incognito \
    http://localhost:5000/kiosk
