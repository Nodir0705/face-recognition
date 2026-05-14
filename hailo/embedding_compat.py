#!/usr/bin/env python3
"""hailo/embedding_compat.py — does Hailo's INT8 ArcFace match Python's FP32?

The production question this answers: if we flip the recognition backend from
Python (FP32 InsightFace) to Hailo (INT8-quantized ArcFace MobileFaceNet),
will the existing enrolled employees still match — or do we need to
re-enroll everyone?

What this script does:
  1. For each input image (one face per image), compute the embedding two ways:
       a) Python+CPU via InsightFace's FaceAnalysis (the production path)
       b) Hailo via SCRFD-detect-and-align then ArcFace HEF
  2. Report cosine similarity between the two embeddings per image
  3. Aggregate: mean, p50, min, max
  4. Recommend re-enrollment policy based on the lowest similarity vs the
     production match threshold (default 0.42)

Heuristics for the recommendation:
  * sim >= 0.85 across all faces → fully portable. Switch with no re-enroll.
  * 0.70 <= sim < 0.85 → likely safe. Spot-check a few employees in production
    before flipping the flag.
  * 0.50 <= sim < 0.70 → marginal. Re-enroll new hires on Hailo from day 1;
    legacy people may still match but expect occasional false negatives.
  * sim < 0.50 → re-enroll everyone before switching backends.

Usage:
    python3 hailo/embedding_compat.py \\
        --det hailo/models/scrfd_500m.hef \\
        --rec hailo/models/arcface_mobilefacenet.hef \\
        --images samples/face1.jpg samples/face2.jpg samples/face3.jpg

If --images is omitted we sweep `samples/*.jpg`.
"""

import argparse
import glob
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hailo"))

from recognize_hailo import HailoSCRFD, HailoArcFace, align_face


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="path to scrfd_500m.hef")
    ap.add_argument("--rec", required=True, help="path to arcface_mobilefacenet.hef")
    ap.add_argument("--images", nargs="*", default=[],
                    help="paths to face images (default: samples/*.jpg)")
    ap.add_argument("--match-threshold", type=float, default=0.42,
                    help="production cosine threshold (config.recognition.match_threshold)")
    return ap.parse_args()


def python_embedding(face_app, bgr):
    """Run InsightFace's FaceAnalysis to get the FP32 embedding for the largest face."""
    faces = face_app.get(bgr)
    if not faces:
        return None, None
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return f.normed_embedding.astype(np.float32), f.bbox


def hailo_embedding(det, rec, det_pipe, rec_pipe, bgr, score_threshold=0.5):
    """Detect → align → embed via Hailo. Returns (embedding, bbox)."""
    blob, scale = det.preprocess(bgr)
    outs = det_pipe.infer({det.input_name: blob})
    dets = det.decode(outs)
    if not dets:
        return None, None
    # Largest detected
    d = max(dets, key=lambda x: (x["bbox"][2] - x["bbox"][0]) *
                                  (x["bbox"][3] - x["bbox"][1]))
    kps = d["kps"] / scale
    bbox = tuple(v / scale for v in d["bbox"])
    aligned = align_face(bgr, kps)
    if aligned is None:
        return None, bbox
    emb = rec.embed(rec_pipe, aligned)
    return emb, bbox


def main():
    args = parse_args()

    images = args.images or sorted(glob.glob(str(PROJECT_ROOT / "samples" / "*.jpg")))
    if not images:
        sys.exit("no images — pass --images or drop face .jpg files in samples/")

    # ----- Python path -----
    print("loading Python InsightFace (buffalo_sc)…")
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(name="buffalo_sc",
                             providers=["CPUExecutionProvider"],
                             allowed_modules=["detection", "recognition"])
    face_app.prepare(ctx_id=-1, det_size=(320, 320))

    # ----- Hailo path -----
    print("opening Hailo device with scheduler…")
    from hailo_platform import VDevice, HailoSchedulingAlgorithm
    vparams = VDevice.create_params()
    vparams.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    sims = []
    bbox_iou = []
    skipped = []

    with VDevice(vparams) as vdevice:
        det = HailoSCRFD(vdevice, args.det)
        rec = HailoArcFace(vdevice, args.rec)
        with det.infer_pipeline() as dp, rec.infer_pipeline() as rp:
            print(f"\ncomparing {len(images)} image(s)…\n")
            print(f"  {'image':<40} {'cos_sim':>9}  {'py_box':<22}  {'hailo_box':<22}")
            print(f"  {'-'*40} {'-'*9}  {'-'*22}  {'-'*22}")
            for path in images:
                name = Path(path).name
                img = cv2.imread(path)
                if img is None:
                    skipped.append((name, "could not read"))
                    continue

                py_emb, py_box = python_embedding(face_app, img)
                hl_emb, hl_box = hailo_embedding(det, rec, dp, rp, img)

                if py_emb is None and hl_emb is None:
                    skipped.append((name, "no face from either path"))
                    continue
                if py_emb is None:
                    skipped.append((name, "Python found no face"))
                    continue
                if hl_emb is None:
                    skipped.append((name, "Hailo found no face"))
                    continue

                # Cosine sim — both vectors are L2-normalized
                sim = float(np.dot(py_emb, hl_emb))
                sims.append(sim)
                py_b = f"({py_box[0]:.0f},{py_box[1]:.0f},{py_box[2]:.0f},{py_box[3]:.0f})"
                hl_b = f"({hl_box[0]:.0f},{hl_box[1]:.0f},{hl_box[2]:.0f},{hl_box[3]:.0f})"
                print(f"  {name:<40} {sim:>9.4f}  {py_b:<22}  {hl_b:<22}")

    print()
    if skipped:
        print("skipped:")
        for n, why in skipped:
            print(f"  {n}: {why}")
        print()

    if not sims:
        sys.exit("no comparable image pairs — nothing to summarize")

    print(f"summary across {len(sims)} face(s):")
    print(f"  mean  cos_sim = {statistics.mean(sims):.4f}")
    print(f"  p50   cos_sim = {statistics.median(sims):.4f}")
    print(f"  min   cos_sim = {min(sims):.4f}")
    print(f"  max   cos_sim = {max(sims):.4f}")
    print()

    lo = min(sims)
    th = args.match_threshold
    print(f"production match threshold = {th:.2f}")
    if lo >= 0.85:
        verdict = ("FULLY PORTABLE", "Switch backends with no re-enrollment.")
    elif lo >= 0.70:
        verdict = ("LIKELY SAFE",
                   "Spot-check a few employees in production before flipping the flag. "
                   "Most will still match.")
    elif lo >= 0.50:
        verdict = ("MARGINAL",
                   "Re-enroll new hires on Hailo from day 1. Legacy people may still match "
                   "but expect occasional false negatives. Consider running both backends "
                   "in parallel during a transition window.")
    else:
        verdict = ("RE-ENROLL EVERYONE",
                   "Embeddings have drifted too far. Re-enroll all active employees on "
                   "Hailo before flipping the flag.")
    print(f"verdict: {verdict[0]}")
    print(f"         {verdict[1]}")

    # Headroom check: even if cos_sim is high, is it well above the match threshold?
    # If python_self_sim ~ 1.0 (same emb compared to itself) and python_vs_hailo = 0.85,
    # then a match against another enrolled person at 0.42 is still well-distinguished.
    print()
    print(f"headroom over threshold: {lo - th:+.2f} "
          f"({'comfortable' if lo - th > 0.30 else 'tight' if lo - th > 0.10 else 'risky'})")


if __name__ == "__main__":
    main()
