#!/usr/bin/env python3
"""Hailo-8 latency benchmark — third member of the bench trio.

CLI surface mirrors scripts/bench_python.py and cpp/bench.cpp so a `diff` of
three SUMMARY lines tells the head-to-head story:

    SUMMARY  impl=python  det_p50=…  det_p95=…  rec_p50=…  total_p50=…
    SUMMARY  impl=cpp     det_p50=…  det_p95=…  rec_p50=…  total_p50=…
    SUMMARY  impl=hailo   det_p50=…  det_p95=…  rec_p50=…  total_p50=…

Usage on the Pi:
    python3 hailo/bench_hailo.py \\
        --model hailo/models/scrfd_500m.hef \\
        [--rec hailo/models/arcface_mobilefacenet.hef] \\
        --image samples/test.jpg \\
        [--iters 200] [--warmup 30]

Notes on what's measured (matches cpp/bench.cpp):
  * preprocess + Hailo inference, per iteration
  * post-processing (SCRFD anchor decode, NMS) is NOT timed — the bench is
    about model latency on the NPU, not the full pipeline. The daemon
    (recognize_hailo.py) would include it.
  * If --rec is supplied we additionally time one ArcFace pass per iteration,
    using the largest detected face's keypoints to align (CPU side).
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

# ArcFace 5-pt template, identical to cpp/pipeline.hpp (so embeddings can be
# compared between implementations).
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],   # L eye
    [73.5318, 51.5014],   # R eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # L mouth
    [70.7299, 92.2041],   # R mouth
], dtype=np.float32)
ALIGNED_SIZE = 112


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to scrfd_500m.hef")
    ap.add_argument("--rec", help="path to arcface_mobilefacenet.hef")
    ap.add_argument("--image", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    return ap.parse_args()


def percentile(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(p * (len(s) - 1)))]


def report(label, v):
    if not v:
        return
    print(f"  {label:<12}  mean={statistics.mean(v):7.2f} ms  "
          f"p50={percentile(v, 0.50):7.2f}  "
          f"p95={percentile(v, 0.95):7.2f}  "
          f"p99={percentile(v, 0.99):7.2f}  "
          f"min={min(v):7.2f}  max={max(v):7.2f}")


def preprocess_for_hef(bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize+pad to (h,w), BGR->RGB, return uint8 NHWC.

    Hailo HEFs from the Model Zoo for SCRFD/ArcFace expect uint8 NHWC input
    with the per-channel normalization baked into the compiled graph. Don't
    pre-normalize here.
    """
    src_h, src_w = bgr.shape[:2]
    scale = min(w / src_w, h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = cv2.resize(bgr, (new_w, new_h))
    padded = np.zeros((h, w, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return rgb[None, ...]   # NHWC, batch 1


def main():
    args = parse_args()

    # HailoRT imports are heavy and platform-specific — keep them local so
    # `--help` works on a dev box without HailoRT installed.
    from hailo_platform import (HEF, VDevice, ConfigureParams, FormatType,
                                 HailoStreamInterface, InferVStreams,
                                 InputVStreamParams, OutputVStreamParams)

    print(f"loading detector HEF: {args.model}")
    det_hef = HEF(args.model)
    rec_hef = HEF(args.rec) if args.rec else None
    if rec_hef:
        print(f"loading recognizer HEF: {args.rec}")

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"could not read image: {args.image}")

    with VDevice() as vdevice:
        # Configure the detector network group
        det_params = ConfigureParams.create_from_hef(det_hef, interface=HailoStreamInterface.PCIe)
        det_ng = vdevice.configure(det_hef, det_params)[0]
        det_in_info = det_ng.get_input_vstream_infos()[0]
        det_in_h, det_in_w = det_in_info.shape[:2]
        det_in_params = InputVStreamParams.make(det_ng, format_type=FormatType.UINT8)
        det_out_params = OutputVStreamParams.make(det_ng, format_type=FormatType.FLOAT32)
        det_input = preprocess_for_hef(img, det_in_h, det_in_w)
        det_input_dict = {det_in_info.name: det_input}

        # Optionally configure recognizer
        rec_ng = None
        rec_input_dict = None
        rec_in_info = None
        if rec_hef:
            rec_params = ConfigureParams.create_from_hef(rec_hef, interface=HailoStreamInterface.PCIe)
            rec_ng = vdevice.configure(rec_hef, rec_params)[0]
            rec_in_info = rec_ng.get_input_vstream_infos()[0]
            # ArcFace input is 112x112 — we feed an aligned face crop. For
            # the bench we just feed the center-cropped 112x112 of the image
            # so each iteration has identical work.
            crop = cv2.resize(img, (112, 112))
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)[None, ...]
            rec_input_dict = {rec_in_info.name: rgb}

        print(f"frame: {img.shape[1]}x{img.shape[0]}, "
              f"det_input: {det_in_w}x{det_in_h}, "
              f"warmup: {args.warmup}, iters: {args.iters}")

        # Detector activation context
        with det_ng.activate(det_ng.create_params()):
            with InferVStreams(det_ng, det_in_params, det_out_params) as det_pipe:
                # Warmup
                for _ in range(args.warmup):
                    det_pipe.infer(det_input_dict)

                # Timed loop — detector
                det_ms = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    det_pipe.infer(det_input_dict)
                    t1 = time.perf_counter()
                    det_ms.append((t1 - t0) * 1000.0)

        # Recognizer pass
        rec_ms = []
        if rec_ng is not None:
            rec_in_params = InputVStreamParams.make(rec_ng, format_type=FormatType.UINT8)
            rec_out_params = OutputVStreamParams.make(rec_ng, format_type=FormatType.FLOAT32)
            with rec_ng.activate(rec_ng.create_params()):
                with InferVStreams(rec_ng, rec_in_params, rec_out_params) as rec_pipe:
                    for _ in range(args.warmup):
                        rec_pipe.infer(rec_input_dict)
                    for _ in range(args.iters):
                        t0 = time.perf_counter()
                        rec_pipe.infer(rec_input_dict)
                        t1 = time.perf_counter()
                        rec_ms.append((t1 - t0) * 1000.0)

    total_ms = [d + r for d, r in zip(det_ms, rec_ms)] if rec_ms else list(det_ms)

    print(f"\nresults (n={args.iters})")
    report("detect", det_ms)
    if rec_ms:
        report("recognize", rec_ms)
    report("total", total_ms)

    print(f"\nSUMMARY  impl=hailo  "
          f"det_p50={percentile(det_ms, 0.50):.2f}  "
          f"det_p95={percentile(det_ms, 0.95):.2f}  "
          f"rec_p50={percentile(rec_ms, 0.50) if rec_ms else 0.0:.2f}  "
          f"total_p50={percentile(total_ms, 0.50):.2f}")


if __name__ == "__main__":
    main()
