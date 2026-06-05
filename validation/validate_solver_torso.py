"""Validate the torso solve (センター, 上半身, 上半身2) against golden boneKeys.

Feeds captured solver-input landmarks to mocapy.solve.torso, runs the shared
skin-blend, aligns capture frame f -> golden boneKey f+1, reports angular error.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy.solve.blend import BlendChannel  # noqa: E402
from mocapy.solve.torso import solve_center, solve_upper_body  # noqa: E402

from mocapy._paths import FIXTURES, ASSETS  # noqa: E402
def quat_angle_deg(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def main() -> int:
    lm = json.loads((FIXTURES / "golden2.bvh.landmarks.json").read_text(encoding="utf-8"))
    gj = json.loads((FIXTURES / "golden2.bvh.golden.json").read_text(encoding="utf-8"))
    by_frame = {c["f"]: (json.loads(c["m"]) if isinstance(c["m"], str) else c["m"]) for c in lm}

    def rotmap(b):
        return {round(k["time"] * 30): np.array(k["rot"]) for k in gj["boneKeys"] if k["name"] == b}

    golden = {"センター": rotmap("センター"), "上半身": rotmap("上半身"), "上半身2": rotmap("上半身2")}
    blend = {b: BlendChannel(mocap_data_smoothing=2) for b in golden}
    errs = {b: [] for b in golden}

    for f in sorted(by_frame):
        kp3 = by_frame[f].get("posenet", {}).get("keypoints3D")
        if not kp3:
            continue
        S = solve_center(kp3)
        h, m = solve_upper_body(kp3, S)
        recs = {
            "センター": blend["センター"].add_rot("センター", S, e_timing=0.5),
            "上半身": blend["上半身"].add_rot("上半身", h, e_timing=0.5),
            "上半身2": blend["上半身2"].add_rot("上半身2", m, e_timing=0.5),
        }
        g = f + 1
        if f < 40:
            continue
        for b in golden:
            if g in golden[b]:
                errs[b].append(quat_angle_deg(recs[b], golden[b][g]))

    ok = True
    for b in ("センター", "上半身", "上半身2"):
        e = np.array(errs[b])
        thr = 0.5
        passed = np.median(e) < thr
        ok = ok and passed
        print(f"{b:8} n={len(e):4} median={np.median(e):.4f}° mean={np.mean(e):.4f}° "
              f"p90={np.percentile(e, 90):.3f}°  {'PASS' if passed else 'FAIL'}")
    print("\nRESULT:", "PASS — torso reproduced" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
