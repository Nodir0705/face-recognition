"""Enrollment script.

Usage:
    # Interactive: opens camera, captures 5 frames, shows preview
    python src/enroll.py --id E001 --name "Hong Gildong" --dept "Engineering"

    # From existing photos
    python src/enroll.py --id E002 --name "Kim Yuna" --photos /path/to/dir/

Best practices for enrollment photos (tell the employee):
  * Look straight at the camera, neutral expression
  * 3-5 photos total: head-on, slight left, slight right, with/without glasses
  * Same lighting conditions as the kiosk will use
  * No masks, no sunglasses
"""

import argparse
import sys
import time
from pathlib import Path
import shutil

import numpy as np
import cv2
import yaml

# Add src/ to path so we can run this as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import AttendanceDB
from face_engine import FaceEngine


def load_config():
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def capture_from_camera(engine: FaceEngine, num_shots: int = 5) -> list[np.ndarray]:
    """Open the Pi camera, capture `num_shots` good-quality frames."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("[!] picamera2 not installed — falling back to OpenCV/USB webcam")
        return _capture_opencv(engine, num_shots)

    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    picam.configure(config)
    picam.start()
    time.sleep(1.0)  # auto-exposure settle

    shots = []
    print(f"[*] Capturing {num_shots} shots. Press SPACE to capture, q to quit.")
    while len(shots) < num_shots:
        rgb = picam.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = engine.detect(bgr)

        preview = bgr.copy()
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(preview, f"Captured: {len(shots)}/{num_shots}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Enrollment", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            if len(faces) != 1:
                print(f"  ! need exactly 1 face, got {len(faces)} — skipped")
                continue
            shots.append(bgr.copy())
            print(f"  + captured ({len(shots)}/{num_shots})")
            time.sleep(0.3)

    picam.stop()
    cv2.destroyAllWindows()
    return shots


def _capture_opencv(engine: FaceEngine, num_shots: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(0)
    shots = []
    while len(shots) < num_shots:
        ok, frame = cap.read()
        if not ok:
            break
        faces = engine.detect(frame)
        preview = frame.copy()
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(preview, f"Captured: {len(shots)}/{num_shots}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Enrollment", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and len(faces) == 1:
            shots.append(frame.copy())
    cap.release()
    cv2.destroyAllWindows()
    return shots


def load_from_dir(path: Path) -> list[np.ndarray]:
    imgs = []
    for p in sorted(path.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            img = cv2.imread(str(p))
            if img is not None:
                imgs.append(img)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Employee ID (e.g. E001)")
    ap.add_argument("--name", required=True, help="Full name")
    ap.add_argument("--dept", default="", help="Department")
    ap.add_argument("--email", default="", help="Email")
    ap.add_argument("--photos", default=None,
                    help="Directory of pre-captured photos (skips camera)")
    ap.add_argument("--shots", type=int, default=5,
                    help="Number of camera shots if interactive")
    args = ap.parse_args()

    cfg = load_config()
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / cfg["paths"]["db"]
    faces_dir = project_root / cfg["paths"]["faces_dir"] / args.id
    faces_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading face engine (first run downloads ~50MB of models)…")
    engine = FaceEngine(
        model_pack=cfg["recognition"]["model_pack"],
        det_size=tuple(cfg["recognition"]["det_size"]),
    )

    if args.photos:
        images = load_from_dir(Path(args.photos))
        print(f"[*] Loaded {len(images)} images from disk")
    else:
        images = capture_from_camera(engine, args.shots)

    if len(images) < 2:
        print("[!] Need at least 2 good shots. Aborting.")
        sys.exit(1)

    embeddings = []
    for i, img in enumerate(images):
        faces = engine.detect(img)
        if len(faces) != 1:
            print(f"  ! image {i}: expected 1 face, got {len(faces)} — skipped")
            continue
        embeddings.append(faces[0].embedding)
        # Save the photo unless config says delete-after-enrollment
        if not cfg["privacy"]["delete_photos_after_enrollment"]:
            cv2.imwrite(str(faces_dir / f"{i:02d}.jpg"), img)

    if len(embeddings) < 2:
        print("[!] Got fewer than 2 usable embeddings. Aborting.")
        sys.exit(1)

    emb_arr = np.stack(embeddings).astype(np.float32)
    db = AttendanceDB(str(db_path))
    db.upsert_employee(
        emp_id=args.id, name=args.name,
        embeddings=emb_arr,
        department=args.dept, email=args.email,
    )

    print(f"[OK] Enrolled {args.name} ({args.id}) with {len(embeddings)} embeddings")

    if cfg["privacy"]["delete_photos_after_enrollment"]:
        shutil.rmtree(faces_dir, ignore_errors=True)
        print("[*] Source photos deleted (PIPA privacy mode)")


if __name__ == "__main__":
    main()
