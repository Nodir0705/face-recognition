# Hailo-8 implementation

Third backend for the recognition pipeline, alongside `src/` (Python on CPU) and `cpp/` (C++ on CPU). Same models conceptually (SCRFD detector + ArcFace MobileFaceNet recognizer), executed on a Hailo-8 NPU via PCIe.

## What you have if you're reading this on `jarvis`

Already in place (set up 2026-05-14):

| Component | Version |
|---|---|
| Hailo-8 chip | `1e60:2864`, FW `4.23.0`, 400 MHz core |
| HailoRT runtime | `4.23.0` (apt: `hailort`) |
| Python binding | `hailo_platform 4.23.0` |
| Kernel driver | built from source against `6.18.29+rpt-rpi-2712`, auto-loads via `/etc/modules-load.d/hailo.conf` |
| PCIe link | Gen3 × 1 (8 GT/s) |
| Bundled HEFs | `/usr/share/hailo-models/` — includes `scrfd_2.5g_h8l.hef` (wrong chip variant — runs but underutilizes) |

**Not yet in place:** the right HEFs for our pipeline. See "Models" below.

## What this directory contains

```
hailo/
  README.md             # this file
  download_models.sh    # pulls scrfd_500m + arcface_mobilefacenet HEFs (Hailo-8 build)
  models/               # HEFs land here, gitignored
  bench_hailo.py        # latency benchmark — same CLI as scripts/bench_python.py
  recognize_hailo.py    # (planned) standalone daemon equivalent to src/recognize.py
```

## Models

We need two HEFs compiled for **Hailo-8** (not 8L) to match Python's `buffalo_sc`:

| HEF | What it replaces | Source |
|---|---|---|
| `scrfd_500m.hef` | `det_500m.onnx` | Hailo Model Zoo (public S3) |
| `arcface_mobilefacenet.hef` | `w600k_mbf.onnx` | Hailo Model Zoo (public S3) |

Pull both:

```bash
bash hailo/download_models.sh
```

If the S3 mirror 404s (Hailo occasionally moves things), grab them from <https://hailo.ai/developer-zone/model-zoo/> (free signup) and drop the `.hef` files into `hailo/models/`.

## Run the benchmark

From the project root, on the Pi:

```bash
# Python+Hailo benchmark — same surface as scripts/bench_python.py
python3 hailo/bench_hailo.py \
    --model hailo/models/scrfd_500m.hef \
    --rec   hailo/models/arcface_mobilefacenet.hef \
    --image samples/test.jpg \
    --iters 200 --warmup 30
```

Output ends with the same `SUMMARY  impl=hailo  det_p50=…` format the other two benches use, so a `diff` of three SUMMARY lines is your three-way comparison.

The Makefile target `make bench-hailo` does this automatically over SSH (rsyncs the directory to the Pi, runs the bench, returns the SUMMARY line).

## What's measured

`bench_hailo.py` times **preprocess + Hailo inference** per iteration. Notes:

- **Preprocessing** (resize, normalize, NHWC pack) runs on the Pi's CPU. With Gen3 PCIe and the Hailo-8 doing ~3-5 ms inference, preprocessing dominates the per-iteration time at this image size.
- **Postprocessing** (SCRFD anchor decode, NMS) is NOT in the timed loop — same cut as `cpp/bench.cpp`. The daemon would need it; the bench doesn't.
- Recognition pass on the largest face works the same as Python+CPU and C+++CPU benches: align via 5 landmarks, run the embedding model, time it.

## Honest expected numbers

Hailo's published benchmarks for `scrfd_500m` on Hailo-8 land around **3–5 ms per inference** at batch=1 input 640×640 (Hailo Model Zoo metrics). For `arcface_mobilefacenet` similar — around **1–2 ms per face**. Vs the ~280 ms / ~150 ms CPU numbers in the parent README, this is roughly a **30–50× speedup** on the model itself.

The catch: at this speed, the bottleneck shifts to **camera capture + preprocessing + postprocessing on the CPU**. Those don't change between backends. Practically:

- Per-frame end-to-end on Pi 5 + Hailo-8: roughly 15–25 ms
- Achievable throughput with our pipeline: **40–60 fps** (camera-limited)
- Real-time ✓

## Troubleshooting

- `RuntimeError: Failed to open device` — driver didn't load. `lsmod | grep hailo_pci`. If empty: `sudo modprobe hailo_pci`. If that fails the source needs to be rebuilt against the running kernel — see top of this file.
- `HEF version mismatch` — the HEF was compiled for a different HailoRT major version. `hailortcli fw-control identify` shows your firmware/runtime; HEFs from the same Model Zoo release should match.
- Slow inference (>10 ms for SCRFD-500m) — check `cat /sys/bus/pci/devices/0001:01:00.0/current_link_speed` reads `8.0 GT/s PCIe`. If it's `5.0` add `dtparam=pciex1_gen=3` to `/boot/firmware/config.txt` and reboot.
