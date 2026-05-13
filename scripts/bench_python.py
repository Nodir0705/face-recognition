#!/usr/bin/env python3
"""Python head-to-head latency benchmark — same CLI surface as cpp/bench.cpp.

We deliberately use the existing FaceEngine (which wraps InsightFace's
FaceAnalysis), so this measures what the Python recognition daemon actually
pays per frame: preprocess + det model + post-processing + (optional) align +
rec model.

Usage:
    python scripts/bench_python.py --image PATH [--rec]
                                    [--det-size 320] [--iters 200] [--warmup 30]
                                    [--threads 2]
                                    [--model-pack buffalo_sc] [--camera N]

Output mirrors bench_cpp so a `diff` of two SUMMARY lines tells the story.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="path to test image (else --camera N)")
    ap.add_argument("--camera", type=int, default=-1)
    ap.add_argument("--model-pack", default="buffalo_sc",
                    help="InsightFace model bundle (default: buffalo_sc)")
    ap.add_argument("--det-size", type=int, default=320)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--threads", type=int, default=2,
                    help="ONNXRuntime intra-op threads (set via env var)")
    ap.add_argument("--rec", action="store_true",
                    help="also time the recognition model on the largest face")
    a = ap.parse_args()
    if not a.image and a.camera < 0:
        ap.error("provide --image PATH or --camera N")
    return a


def acquire_frame(args):
    if args.image:
        img = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"could not read image: {args.image}")
        return img
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera index {args.camera}")
    img = None
    for _ in range(6):  # discard warm-up frames
        ok, img = cap.read()
    cap.release()
    if img is None:
        raise SystemExit("camera read returned empty frame")
    return img


def percentile(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    idx = min(len(s) - 1, int(p * (len(s) - 1)))
    return s[idx]


def report(label, v):
    if not v:
        return
    print(f"  {label:<12}  mean={statistics.mean(v):7.2f} ms  "
          f"p50={percentile(v, 0.50):7.2f}  "
          f"p95={percentile(v, 0.95):7.2f}  "
          f"p99={percentile(v, 0.99):7.2f}  "
          f"min={min(v):7.2f}  max={max(v):7.2f}")


def main():
    args = parse_args()

    # Set ORT thread count via env BEFORE importing insightface (it picks them
    # up at session creation time via the default SessionOptions).
    import os
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", str(args.threads))

    print(f"loading face engine: {args.model_pack} det_size={args.det_size}")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name=args.model_pack,
        providers=["CPUExecutionProvider"],
        # Match the C++ bench: detection (+ recognition only if --rec)
        allowed_modules=["detection", "recognition"] if args.rec else ["detection"],
    )
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

    frame = acquire_frame(args)
    print(f"frame: {frame.shape[1]}x{frame.shape[0]}, "
          f"det_size: {args.det_size}x{args.det_size}, "
          f"threads: {args.threads}, "
          f"warmup: {args.warmup}, iters: {args.iters}")

    # Warmup
    for _ in range(args.warmup):
        app.get(frame)

    # Timed loop
    det_ms, rec_ms, total_ms = [], [], []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        # FaceAnalysis.get() runs detection AND recognition in one call when
        # both modules are loaded. To split them we'd have to bypass the public
        # API; instead, we time the whole call and report it twice (when --rec
        # is on) so the comparison stays apples-to-apples.
        faces = app.get(frame)
        t1 = time.perf_counter()

        det_ms.append((t1 - t0) * 1000.0)
        # In Python's FaceAnalysis there's no clean separation between detect
        # and recognize timing without reaching into private state. Report
        # rec_ms = 0 to keep the SUMMARY format consistent — interpret det_ms
        # as the full detect+recognize cost when --rec is on.
        rec_ms.append(0.0)
        total_ms.append(det_ms[-1])

    print(f"\nresults (n={args.iters})")
    if args.rec:
        report("det+rec", det_ms)   # combined when --rec is on
    else:
        report("detect", det_ms)
    report("total", total_ms)

    print(f"\nSUMMARY  impl=python  "
          f"det_p50={percentile(det_ms, 0.50):.2f}  "
          f"det_p95={percentile(det_ms, 0.95):.2f}  "
          f"rec_p50={percentile(rec_ms, 0.50):.2f}  "
          f"total_p50={percentile(total_ms, 0.50):.2f}")


if __name__ == "__main__":
    main()
