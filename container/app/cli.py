"""
cli.py — fixed, standardised entrypoint for the leaderboard runner.

Two subcommands:

    info      print /app/STUDENT.json to stdout (valid JSON, exit 0)
    predict   run the ONNX model over /data/input/ and write
              /data/output/predictions.csv

CSV schema (header always written):
    image_path,xmin,ymin,xmax,ymax,confidence,class
One row per detected box. Images with no detections get a single row with
image_path filled and the other six fields empty.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

from detector import CatDetector

# Fixed paths the instructor mounts.
STUDENT_JSON = Path(os.environ.get("STUDENT_JSON", "/app/STUDENT.json"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/best.onnx"))
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/data/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
OUTPUT_CSV = OUTPUT_DIR / "predictions.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png"}
CSV_HEADER = ["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]

# Class names must match data.yaml. Single-class dataset here.
CLASS_NAMES = ("cat",)
# Low conf on purpose: mAP integrates the full PR curve, so keeping low-score
# boxes RAISES mAP. Do NOT bump this to 0.25 for the leaderboard.
CONF = float(os.environ.get("CONF", "0.001"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))


def cmd_info() -> int:
    with open(STUDENT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Re-dump so output is guaranteed valid, compact JSON on stdout.
    json.dump(data, sys.stdout)
    sys.stdout.write("\n")
    return 0


def _iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            yield path


def cmd_predict() -> int:
    if not INPUT_DIR.is_dir():
        print(f"error: input dir {INPUT_DIR} does not exist", file=sys.stderr)
        return 1

    # EXPLICITLY pass conf=0.001 to override detector.py's default 0.25 threshold
    # This preserves low-confidence boxes for full Precision-Recall curve integration!
    detector = CatDetector(str(MODEL_PATH), imgsz=640, conf=0.001, class_names=CLASS_NAMES)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_boxes = 0
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for img_path in _iter_images(INPUT_DIR):
            rel = img_path.relative_to(INPUT_DIR).as_posix()  # forward slashes
            n_images += 1
            try:
                dets = detector.predict(str(img_path))
            except Exception as exc:  # never let one bad image abort the run
                print(f"warning: failed on {rel}: {exc}", file=sys.stderr)
                dets = []

            if not dets:
                writer.writerow([rel, "", "", "", "", "", ""])
                continue

            for d in dets:
                n_boxes += 1
                writer.writerow(
                    [
                        rel,
                        f"{d['xmin']:.2f}",
                        f"{d['ymin']:.2f}",
                        f"{d['xmax']:.2f}",
                        f"{d['ymax']:.2f}",
                        f"{d['confidence']:.6f}",
                        d["class"],
                    ]
                )

    print(f"wrote {OUTPUT_CSV}: {n_images} images, {n_boxes} boxes", file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print("usage: cli.py {info|predict}", file=sys.stderr)
        return 2
    cmd = argv[0]
    if cmd == "info":
        return cmd_info()
    if cmd == "predict":
        return cmd_predict()
    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
