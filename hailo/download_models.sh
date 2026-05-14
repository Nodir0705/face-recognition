#!/usr/bin/env bash
# Pull the two HEFs we need for the attendance pipeline from Hailo Model Zoo.
# Run this once on the Pi (the HEFs are arch-specific — they're "compiled" for
# the Hailo-8 chip). Output goes to ./models/ alongside this script.
#
# We try Hailo's S3 mirror first (no signup needed, fast). If that 404s the
# user can grab the same files from the Hailo Developer Zone (free signup)
# and drop them in models/ manually.

set -euo pipefail

cd "$(dirname "$0")/models"

# Hailo Model Zoo public S3 layout (verified 2026-05-14):
#   https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/<zoover>/<chip>/<model>.hef
# This URL pattern is what hailo-rpi5-examples/download_resources.sh uses
# internally — same source the AI Kit examples ship with.
#
# Override the version with HAILO_ZOO_VER env var if a newer release drops.

ZOOVER="${HAILO_ZOO_VER:-v2.14.0}"
BASE="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/${ZOOVER}/hailo8"

declare -A MODELS=(
    [scrfd_500m.hef]="$BASE/scrfd_500m.hef"
    [arcface_mobilefacenet.hef]="$BASE/arcface_mobilefacenet.hef"
)

echo "Downloading HEFs to $(pwd)"
for name in "${!MODELS[@]}"; do
    if [[ -s "$name" ]]; then
        echo "  [skip] $name (already present, $(du -h "$name" | cut -f1))"
        continue
    fi
    url="${MODELS[$name]}"
    echo "  [pull] $name <- $url"
    if ! curl -fL --retry 3 -o "$name.tmp" "$url"; then
        echo
        echo "  Mirror returned an error for $name."
        echo "  Fallback: download manually from https://hailo.ai/developer-zone/model-zoo/"
        echo "  (free signup) and drop the .hef into $(pwd)/"
        rm -f "$name.tmp"
        continue
    fi
    mv "$name.tmp" "$name"
done

echo
echo "Available HEFs:"
ls -lh *.hef 2>/dev/null || echo "  (none yet)"

echo
echo "Bundled fallbacks already on the Pi (in /usr/share/hailo-models/):"
echo "  scrfd_2.5g_h8l.hef    — heavier SCRFD compiled for Hailo-8L (runs on H8 too)"
echo "  resnet_v1_50_h10.hef  — generic backbone, not the right recognizer"
echo "If the S3 download failed you can point bench_hailo.py at the bundled"
echo "scrfd_2.5g_h8l.hef and skip --rec until you obtain arcface_mobilefacenet.hef."
