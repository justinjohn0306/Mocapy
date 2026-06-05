"""Validate the finger solve vs golden2 (cj) using the reference's own captured hand annotations.

Isolates the SOLVER from hand detection: feeds the same hand `annotations` the reference's finger
solver consumed (golden2 handpose), reports per-joint rotation error vs the golden finger
bone keys. (The pipeline's raw hand_landmarker.task front-end differs from the reference's JS Hands,
which is a separate detection-quality gap — see fingers.py.)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy.solve.fingers import solve_finger, solve_thumb  # noqa: E402
from mocapy.vrm.skeleton import load_skeleton, compute_finger_axis_rot  # noqa: E402

from mocapy._paths import FIXTURES, ASSETS  # noqa: E402
FW = {0: "０", 1: "１", 2: "２", 3: "３"}
# (annotation key, m, MMD glyph)
FINGERS = [("index", 1, "人指"), ("middle", 2, "中指"), ("ring", 3, "薬指"), ("pinky", 4, "小指")]
# character side -> (handedness label, R)
SIDE = {"右": ("Left", -1.0), "左": ("Right", 1.0)}


def qang(a, b):
    a = np.asarray(a) / np.linalg.norm(a)
    b = np.asarray(b) / np.linalg.norm(b)
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def main() -> int:
    lm = json.loads((FIXTURES / "golden2.bvh.landmarks.json").read_text(encoding="utf-8"))
    gj = json.loads((FIXTURES / "golden2.bvh.golden.json").read_text(encoding="utf-8"))
    by_frame = {c["f"]: (json.loads(c["m"]) if isinstance(c["m"], str) else c["m"]) for c in lm}

    # Pull axis_rot tables from cj.vrm (the model golden2 was captured against).
    # This enables the reference's full THREEX finger-correction path (axis_rot per joint +
    # axis_rot_offset_inv premultiplication on the chain's boundary joint).
    skel = load_skeleton(ASSETS / "cj.vrm")
    axis_rot_by_mmd, axis_rot_offset_inv_by_mmd = compute_finger_axis_rot(skel)

    def rotmap(b):
        return {round(k["time"] * 30): np.array(k["rot"]) for k in gj["boneKeys"] if k["name"] == b}

    frames = [f for f in sorted(by_frame) if by_frame[f].get("handpose")]
    print(f"{'finger':10} {'j1':>6} {'j2':>6} {'j3':>6}   (median rot error vs golden, deg)")
    worst_mid = 0.0
    for side, (label, R) in SIDE.items():
        for key, m, glyph in FINGERS:
            g = [rotmap(side + glyph + FW[j]) for j in (1, 2, 3)]
            errs = [[], [], []]
            for f in frames:
                h = next((x for x in by_frame[f]["handpose"]
                          if x.get("label") == label and x.get("annotations")), None)
                if not h:
                    continue
                first_joint_offset_inv = axis_rot_offset_inv_by_mmd.get(side + glyph + "１")
                rr = solve_finger(h["annotations"], key, m, R, side,
                                  axis_rot_by_mmd=axis_rot_by_mmd,
                                  axis_rot_offset_inv_first_joint=first_joint_offset_inv,
                                  mmd_glyph=glyph)
                gg = f + 1
                for j in range(3):
                    if gg in g[j]:
                        errs[j].append(qang(rr[j], g[j][gg]))
            med = [np.median(e) if e else float("nan") for e in errs]
            worst_mid = max(worst_mid, med[1], med[2])  # middle/distal joints
            print(f"{side+glyph:10} {med[0]:5.1f}° {med[1]:5.1f}° {med[2]:5.1f}°")
        # thumb (親指０/１/２) — approximate (no model axis_rot offset port)
        gt = [rotmap(side + "親指" + FW[j]) for j in (0, 1, 2)]
        et = [[], [], []]
        for f in frames:
            h = next((x for x in by_frame[f]["handpose"]
                      if x.get("label") == label and x.get("annotations")
                      and "thumb" in x["annotations"]), None)
            if not h:
                continue
            thumb_offset_inv = axis_rot_offset_inv_by_mmd.get(side + "親指０")
            rr = solve_thumb(h["annotations"], R, side,
                             axis_rot_offset_inv_first_joint=thumb_offset_inv)
            gg = f + 1
            for j in range(3):
                if gg in gt[j]:
                    et[j].append(qang(rr[j], gt[j][gg]))
        medt = [np.median(e) if e else float("nan") for e in et]
        print(f"{side+'親指':10} {medt[0]:5.1f}° {medt[1]:5.1f}° {medt[2]:5.1f}°  (approx)")
    ok = worst_mid < 25.0
    print(f"\nworst middle/distal joint median: {worst_mid:.1f}°")
    print("RESULT:", "PASS — finger solve tracks golden (mid/distal < 25°)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
