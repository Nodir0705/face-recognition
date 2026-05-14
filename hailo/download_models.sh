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

# Hailo Model Zoo public mirror layout (as of ZooModel 2.13 / HailoRT 4.20+):
#   https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/<chip>/<model>.hef
# Mirror them locally so the bench/daemon don't have to re-download.

BASE="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8"

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
