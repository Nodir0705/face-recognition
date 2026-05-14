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
  README.md              # this file
  download_models.sh     # pulls scrfd_500m + arcface_mobilefacenet HEFs (Hailo-8 build)
  models/                # HEFs land here, gitignored
  bench_hailo.py         # latency benchmark — same CLI as scripts/bench_python.py
                          # also has --multi-face mode for N-face per-frame timing
  recognize_hailo.py     # standalone daemon equivalent to src/recognize.py
                          # writes to the same data/attendance.db the Flask app uses
  engine_adapter.py      # HailoFaceEngine class with the same API as
                          # src.face_engine.FaceEngine — drop-in for Flask
  embedding_compat.py    # validates whether existing Python-enrolled employees
                          # match against Hailo-computed embeddings (spoiler: no)
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

## Measured numbers

Real bench on jarvis (Pi 5 + Hailo-8 AI Kit, kernel 6.18.29, HailoRT 4.23.0, PCIe Gen3 x1, 2026-05-14), same input image, 200 iterations after 30 warmups:

| Stage | p50 | p95 | p99 | min–max |
|---|---|---|---|---|
| SCRFD-500m (640×640 in) | **4.12 ms** | 4.22 | 4.28 | 4.02 – 4.31 |
| ArcFace MobileFaceNet (112×112 in) | **1.46 ms** | 1.47 | 1.48 | 1.45 – 1.51 |
| **Total per face** | **5.59 ms** | 5.69 | 5.74 | 5.48 – 5.77 |

Head-to-head vs Python on the **same Pi 5 CPU**, same image, same iterations:

| Backend | det+rec p50 | p99 | Jitter | Throughput |
|---|---|---|---|---|
| Python + CPU (4 threads) | 48.08 ms | 65.33 ms | **±27 ms** | ~21 fps |
| **Hailo-8** | **5.59 ms** | **5.74 ms** | **±0.29 ms** | **~180 fps** |
| Speedup | **8.6×** | **11.4×** | **95× tighter** | **8.6×** |

The headline isn't only speed — **jitter is two orders of magnitude tighter**. CPU swings 42→70 ms on the same input, Hailo holds 5.48→5.77 ms. That's what makes Hailo "real-time" in a way the CPU isn't, even when averages are closer.

At this speed the recognition pipeline becomes camera-bound, not compute-bound. Practical throughput is whatever your camera + preprocessing can sustain — for a Pi Camera Module 3 at 720p that's ~30–60 fps depending on frame rate setting, and Hailo has plenty of headroom for multiple simultaneous detections per frame.

### Multi-face scaling (5 faces in one frame)

`bench_hailo.py --multi-face` runs 1 detect + N embed per iteration, simulating a queue of multiple people at the kiosk:

| Backend | per-frame p50 | p99 | Jitter | fps |
|---|---|---|---|---|
| Python+CPU (4 threads) | 124.21 ms | 135.70 ms | ±22 ms | 8.0 fps |
| **Hailo-8** | **20.37 ms** | **20.77 ms** | **±0.55 ms** | **49.1 fps** |
| Speedup | **6.1×** | **6.5×** | **40× tighter** | **6.1×** |

Hailo throughput at 5 faces ≈ **245 face recognitions/sec** on a single chip. Multi-face is where the advantage really shows up — Python drops to 8 fps (noticeable lag at the door), Hailo holds 49 fps (smooth).

## ⚠️ Embedding compatibility — re-enrollment required to switch backends

`hailo/embedding_compat.py` measures cosine similarity between the Python embedding (InsightFace `w600k_mbf.onnx`) and the Hailo embedding (`arcface_mobilefacenet.hef`) for the same face image with bit-identical alignment.

**Result: cos_sim ≈ -0.002.** The two HEF/ONNX are different trained checkpoints despite both being "MobileFaceNet ArcFace." Embeddings live in completely different vector spaces.

**Implication:** if you flip `recognition.backend` from `python` to `hailo` in `config.yaml`, all existing enrolled employees become unmatchable. The DB gallery must be cleared and everyone re-enrolled.

Two production paths:

1. **Re-enroll on switch (recommended).** Schedule an hour, run everyone through `/enroll` once with the new backend active. Done. From then on, Hailo is the source of truth.
2. **Compile InsightFace's `w600k_mbf.onnx` to a HEF yourself** with Hailo Dataflow Compiler (x86 + free SDK signup + a few hundred calibration images). Backwards-compatible embeddings, no re-enrollment. ~1 day of work.

Run the validator yourself before committing:
```bash
python3 hailo/embedding_compat.py \
    --det hailo/models/scrfd_500m.hef \
    --rec hailo/models/arcface_mobilefacenet.hef \
    --images samples/*.jpg
```

## Troubleshooting

- `RuntimeError: Failed to open device` — driver didn't load. `lsmod | grep hailo_pci`. If empty: `sudo modprobe hailo_pci`. If that fails the source needs to be rebuilt against the running kernel — see top of this file.
- `HEF version mismatch` — the HEF was compiled for a different HailoRT major version. `hailortcli fw-control identify` shows your firmware/runtime; HEFs from the same Model Zoo release should match.
- Slow inference (>10 ms for SCRFD-500m) — check `cat /sys/bus/pci/devices/0001:01:00.0/current_link_speed` reads `8.0 GT/s PCIe`. If it's `5.0` add `dtparam=pciex1_gen=3` to `/boot/firmware/config.txt` and reboot.
