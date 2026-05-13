# C++ implementation

A parallel C++ port of the recognition pipeline, kept alongside the Python so we can compare.

| Binary | What it is |
|---|---|
| `bench_cpp` | Latency benchmark — loads `det_500m.onnx` (and optionally `w600k_mbf.onnx`), runs N iterations on a fixed frame, reports p50/p95/p99 |
| `recognize_cpp` | Standalone recognition daemon — owns the camera, writes events to the same `data/attendance.db` the Python web app uses |

Both share `pipeline.hpp` (header-only) for SCRFD anchor decoding, NMS, ArcFace alignment, and embedding match math.

## Why a C++ version exists

The Python recognition path is dominated by ONNXRuntime's CPU kernels — those are already C++ under the hood, so the language boundary is most of what's left. Realistic expectation:

- **5–15 ms/frame saved** on the loop overhead (no GIL, no per-call dispatch)
- **No GIL contention** — the Flask MJPEG threads can serve previews without throttling recognition
- **Same model, same accuracy** — we load the exact same `buffalo_sc` ONNX files

If you want a bigger latency drop, change the model (e.g. a smaller SCRFD variant), not the language.

## Build

### 1. System packages

On Debian / Ubuntu / Raspberry Pi OS Bookworm:

```bash
sudo apt install -y \
    build-essential cmake pkg-config \
    libopencv-dev libsqlite3-dev
```

OpenCV 4.5+ is required for `cv::FaceDetectorYN` and the modern DNN module.
Bookworm ships 4.6 — fine. Older Buster ships 3.x — won't work.

### 2. ONNXRuntime C++

ONNXRuntime is **not** packaged in apt. Grab a release tarball:

**Linux x86_64 (dev laptop / desktop):**
```bash
cd ~
ORT_VER=1.18.1
wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz
tar -xzf onnxruntime-linux-x64-${ORT_VER}.tgz
# Now ~/onnxruntime-linux-x64-1.18.1/{include,lib} is the install root
```

**Linux aarch64 (Raspberry Pi 4 / 5 64-bit):**
```bash
cd ~
ORT_VER=1.18.1
wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-aarch64-${ORT_VER}.tgz
tar -xzf onnxruntime-linux-aarch64-${ORT_VER}.tgz
```

### 3. Configure + build

From the project root (so paths in the Makefile resolve):

```bash
cmake -S cpp -B cpp/build \
      -DONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-1.18.1 \
      -DNATIVE=ON
cmake --build cpp/build -j
```

Both binaries land in `cpp/build/`.

If you skip `-DONNXRUNTIME_ROOT`, CMake will look for ONNXRuntime under standard system paths — works only if you ran `make install` on a built ONNXRuntime tree.

`-DNATIVE=ON` adds `-march=native` (or `-mcpu=native` on the Pi). Drop it if you'll cross-compile and run on a different CPU.

## Models

Both binaries need the InsightFace `buffalo_sc` ONNX files. Easiest is to let the Python side download them once:

```bash
make install-dev          # if you haven't already
.venv/bin/python -c "from insightface.app import FaceAnalysis; \
    FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider']).prepare(ctx_id=-1)"
```

The bundle ends up at `~/.insightface/models/buffalo_sc/`:

```
det_500m.onnx          # SCRFD detector
w600k_mbf.onnx         # MobileFaceNet recognition (512-d ArcFace embeddings)
2d106det.onnx          # 106-pt landmarks (not used by the C++ daemon)
genderage.onnx         # not used anywhere
```

## Run

### Benchmark

```bash
# C++ benchmark — detector only, fixed test image
./cpp/build/bench_cpp \
    --model $HOME/.insightface/models/buffalo_sc/det_500m.onnx \
    --image samples/test.jpg \
    --iters 200 --warmup 30 --threads 2

# C++ benchmark — detector + recognizer (full per-face cost)
./cpp/build/bench_cpp \
    --model $HOME/.insightface/models/buffalo_sc/det_500m.onnx \
    --rec   $HOME/.insightface/models/buffalo_sc/w600k_mbf.onnx \
    --image samples/test.jpg \
    --iters 200 --warmup 30 --threads 2

# Python equivalent — same image, same iterations, same threads
python scripts/bench_python.py --image samples/test.jpg \
    --iters 200 --warmup 30 --threads 2 --rec
```

Both print a `SUMMARY  impl=...  det_p50=...` line — one diff and you have your number.

The Makefile shortcuts (`make bench-cpp`, `make bench-py`) wire the typical args together — see the Makefile.

### Daemon

⚠ Don't run the C++ daemon and the Python web app's recognition thread at the same time — they'd both try to open the camera. Stop the Python app first (or comment out the recognition thread for the comparison).

```bash
./cpp/build/recognize_cpp \
    --models $HOME/.insightface/models/buffalo_sc \
    --db     $PWD/data/attendance.db \
    --config $PWD/config/config.yaml \
    --camera 0
```

The daemon reads from the same SQLite database the Flask app writes to (WAL mode handles concurrent readers fine). Enroll employees from the Python web UI as usual; the C++ daemon picks them up at the next 60-second gallery reload.

## Limitations vs the Python implementation

These are intentional scope cuts to keep the C++ to two .cpp files plus one header:

- **No mask/sunglasses heuristic.** `src/occlusion.py` is not ported. The C++ daemon will match an occluded face — Python won't. Port if you need parity.
- **No blink/motion liveness gate.** `src/liveness.py` is not ported.
- **No MJPEG preview.** The Python web app keeps that role; C++ writes events headlessly.
- **No 106-pt landmark refinement.** Detection uses just the 5 SCRFD landmarks (which is also what FaceAnalysis uses for alignment).
- **YAML config is parsed by a tiny grep**, not yaml-cpp. Only the keys we actually need are read; comments and quotes are stripped, but nested lists / multi-line values are not understood.

## Honest performance notes

On a Raspberry Pi 4 (8 GB) with `buffalo_sc` and 320×320 detection, expect roughly:

|  | Python | C++ |
|---|---|---|
| Detect (p50) | ~280 ms | ~270 ms |
| Detect + recognize (p50) | ~430 ms | ~410 ms |
| Per-frame loop overhead | ~10 ms | ~1 ms |
| Throughput under MJPEG load | ~2 fps | ~3 fps |

The model itself dominates everything else. C++ wins clearly on **throughput under contention** (no GIL) and on the loop overhead, marginally on the model call. Numbers from a desktop x86 with 6 cores look much friendlier to both — most of the C++ vs Python gap on the Pi comes from the fact that ONNXRuntime is the same in both.

If you want **real** real-time, replace `det_500m.onnx` with a smaller model (e.g. SCRFD 2.5g + 160×160) — that's a 5–10× speedup, and it works equally well in either language.
