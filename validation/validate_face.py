"""Validate iris-driven eye gaze (両目) vs golden2.

Note: golden's 首/頭 keys (992 frames = every frame) come from the reference's POSE-based head
path; golden's 両目 keys (729 frames = face-detected frames only) come from the reference's
face-mesh iris path. So head is already nailed by the pose-only solve (4.2°), and
this test only validates the new face-mesh eye solve.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy._paths import FIXTURES  # noqa: E402
from mocapy.solve.eyes import solve_eyes, head_pitch_yaw_from_face_matrix  # noqa: E402


def qang(a, b):
    a = np.asarray(a) / np.linalg.norm(a)
    b = np.asarray(b) / np.linalg.norm(b)
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def main() -> int:
    gj = json.loads((FIXTURES / "golden2.bvh.golden.json").read_text("utf-8"))
    G_eyes = {round(k["time"] * 30): np.array(k["rot"])
              for k in gj["boneKeys"] if k["name"] == "両目"}

    fr_path = FIXTURES / "frames_face.json"
    if not fr_path.exists():
        print(f"First run:  python validation/detect_dump.py fixtures/test.mp4 fixtures/frames_face.json")
        return 2
    data = json.loads(fr_path.read_text("utf-8"))

    errs, mags_pred, mags_gold = [], [], []
    for f, fr in enumerate(data["frames"]):
        if fr["face"] is None or fr["face_matrix"] is None:
            continue
        gg = f + 1
        if gg not in G_eyes:
            continue
        pitch, yaw = head_pitch_yaw_from_face_matrix(np.asarray(fr["face_matrix"]))
        pred = solve_eyes(np.asarray(fr["face"]), head_pitch_rad=pitch, head_yaw_rad=yaw)
        errs.append(qang(pred, G_eyes[gg]))
        mags_pred.append(qang(pred, [0, 0, 0, 1]))
        mags_gold.append(qang(G_eyes[gg], [0, 0, 0, 1]))

    if not errs:
        print("FAIL — no overlapping frames"); return 1

    med = float(np.median(errs))
    p90 = float(np.percentile(errs, 90))
    print(f"両目 median err = {med:.2f}°   p90 = {p90:.2f}°   n = {len(errs)}")
    print(f"両目 median magnitude  pred: {np.median(mags_pred):.2f}°   golden: {np.median(mags_gold):.2f}°")

    ok = med < 8.0
    print("RESULT:", "PASS — eye gaze tracks golden (median < 8°)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
