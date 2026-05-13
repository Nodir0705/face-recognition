# Remote access from the admin laptop

Admins enroll new employees from their own laptop, while the Pi sits at the entrance running the camera. This requires the admin laptop to reach the Pi over the network.

## The simple case — same office LAN

If the laptop and the Pi are on the same WiFi/Ethernet:

1. Find the Pi's IP. On the Pi: `hostname -I` (returns e.g. `192.168.1.42`).
2. On the laptop, open `http://192.168.1.42:5000/enroll`.

That's it. The MJPEG stream from the Pi's camera plays in the laptop browser; the wizard works exactly as if you were standing at the Pi.

## Give the Pi a stable hostname

IPs change when a router reboots. Better to use a hostname:

```bash
# On the Pi:
sudo hostnamectl set-hostname attendance
sudo systemctl restart avahi-daemon
```

Now the laptop can reach the Pi at `http://attendance.local:5000/enroll` (mDNS works out of the box on macOS, modern Windows, and most Linux desktops).

## Multiple admins, multiple offices

If your company has more than one location or admins who work from home, you have two clean options:

**Option A — Tailscale (recommended for ease).** Install Tailscale on the Pi and on each admin's laptop. They get a private mesh network. Admins can reach `http://attendance:5000/enroll` from anywhere, no port forwarding, no VPN config.

```bash
# On the Pi:
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Each admin installs Tailscale on their laptop and joins the same tailnet. The Pi keeps no public exposure.

**Option B — Office VPN.** If you already have a corporate VPN (WireGuard, OpenVPN), put the Pi behind it. Admins on the VPN can reach the Pi by its internal IP.

## Do NOT port-forward to the public internet

The admin pages have basic auth, but exposing a Pi running a face-recognition kiosk to the public internet is asking for trouble. Use one of the LAN/VPN options above.

## Firewall on the Pi (recommended)

Only the admin pages (`/`, `/enroll`) need to be reachable from admin laptops; the kiosk view is served to `localhost` on the same Pi. With `ufw`:

```bash
sudo apt install -y ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from 192.168.0.0/16 to any port 5000  # adjust to your LAN
sudo ufw allow ssh                                    # if you SSH in
sudo ufw enable
```

This keeps the Pi reachable from the office LAN only.

## Testing the connection

From the admin laptop:

```bash
# Confirm the Pi is reachable
ping attendance.local

# Confirm the web service is up
curl -u admin:changeme http://attendance.local:5000/
```

If `ping` works but `curl` doesn't, the Flask service isn't running:

```bash
# On the Pi:
sudo systemctl status attendance
journalctl -u attendance -n 50
```

## Bandwidth note

The enrollment stream is MJPEG at 12 fps, ~75% JPEG quality, 1280×720. That's roughly 1–2 Mbit/s while the wizard is open. On a normal office WiFi, this is negligible. Just don't enroll over a 3G hotspot.
