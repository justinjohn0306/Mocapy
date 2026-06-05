"""Validate the first ported solver bone: センター rotation (hip-direction S).

S = setFromEuler( setFromVectorSpherical((1,0,0), hipDir) with z zeroed, "YZX" )
hipDir = (left_hip - right_hip).normalize(),  raw keypoints3D[23], [24].

Feeds the EXACT captured solver-input landmarks to the ported math and compares to
golden センター.rot. (st("hip") is passthrough in steady state, so the startup
transient ~first second may differ.)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy.solve.blend import BlendChannel  # noqa: E402
from mocapy.solve.torso import solve_center  # noqa: E402

from mocapy._paths import FIXTURES, ASSETS  # noqa: E402
L_HIP, R_HIP = 23, 24


def quat_angle_deg(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = abs(float(np.dot(a, b)))
    return math.degrees(2 * math.acos(min(1.0, d)))


def compute_S(kp3):
    return solve_center(kp3)


def main() -> int:
    lm = json.loads((FIXTURES / "golden2.bvh.landmarks.json").read_text(encoding="utf-8"))
    gj = json.loads((FIXTURES / "golden2.bvh.golden.json").read_text(encoding="utf-8"))

    # last capture per frame_count
    by_frame = {}
    for c in lm:
        m = json.loads(c["m"]) if isinstance(c["m"], str) else c["m"]
        by_frame[c["f"]] = m

    def golden_rot(bone):
        d = {}
        for k in gj["boneKeys"]:
            if k["name"] == bone:
                d[round(k["time"] * 30)] = np.array(k["rot"])
        return d

    center = golden_rot("センター")
    lower = golden_rot("下半身")
    print(f"golden センター keys: {len(center)}  下半身 keys: {len(lower)}")

    # Alignment: the recorder samples one frame after the solve's frame_count,
    # so capture frame f corresponds to golden boneKey f+1 (verified empirically).
    OFFSET = 1
    blend = BlendChannel(mocap_data_smoothing=2)  # timing regime; e=0.5 at ~33ms/frame
    raw_errs, blend_errs = [], []
    for f, m in sorted(by_frame.items()):
        kp3 = m.get("posenet", {}).get("keypoints3D")
        if not kp3:
            continue
        S = compute_S(kp3)
        if S is None:
            continue
        rec = blend.add_rot("センター", S, e_timing=0.5)  # the shared skin-blend layer
        g = f + OFFSET
        if f < 40 or g not in center:
            continue
        raw_errs.append(quat_angle_deg(S, center[g]))
        blend_errs.append(quat_angle_deg(rec, center[g]))

    raw = np.array(raw_errs)
    bl = np.array(blend_errs)
    print(f"\nセンター rotation vs golden (align +{OFFSET}, n={len(bl)}):")
    print(f"  raw S (no blend)   : median={np.median(raw):.3f}° mean={np.mean(raw):.3f}°")
    print(f"  + skin-blend (e=.5): median={np.median(bl):.4f}° mean={np.mean(bl):.4f}° "
          f"p90={np.percentile(bl, 90):.4f}°")

    ok = np.median(bl) < 0.1
    print("\nRESULT:", "PASS — センター reproduced (math + blend)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
