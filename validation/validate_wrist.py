"""Validate the forearm-twist (手捩) + wrist (手首) solve vs golden2 (cj).

Two checks:
  (1) GRANT self-consistency: rebuild 手捩 from l=手捩·手首 via the half-X-twist grant -> ~0deg.
  (2) END-TO-END from the pose: feed golden torso/arm + the pose wrist basis through
      solve_wrist; report 手捩 (the BVH-relevant twist) vs golden. ~11-15deg.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy.solve import three_math as tm  # noqa: E402
from mocapy.solve.wrist import solve_wrist, _M, _HE  # noqa: E402

from mocapy._paths import FIXTURES, ASSETS  # noqa: E402
FIXED = {"左": np.array([0.9977, -0.0093, 0.0666]), "右": np.array([-0.9977, -0.0093, 0.0666])}


def qang(a, b):
    a = np.asarray(a) / np.linalg.norm(a)
    b = np.asarray(b) / np.linalg.norm(b)
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def main() -> int:
    gj = json.loads((FIXTURES / "golden2.bvh.golden.json").read_text(encoding="utf-8"))
    lm = json.loads((FIXTURES / "golden2.bvh.landmarks.json").read_text(encoding="utf-8"))
    by_frame = {c["f"]: (json.loads(c["m"]) if isinstance(c["m"], str) else c["m"]) for c in lm}

    def mp(b):
        return {round(k["time"] * 30): np.array(k["rot"]) for k in gj["boneKeys"] if k["name"] == b}

    G = {b: mp(b) for b in ["センター", "上半身", "上半身2",
                            "左腕", "左ひじ", "右腕", "右ひじ",
                            "左手捩", "左手首", "右手捩", "右手首"]}

    # (1) grant self-consistency
    print("grant self-consistency (rebuild 手捩 from l=手捩·手首):")
    worst_grant = 0.0
    for side in ("左", "右"):
        M = _M[side]
        fa = np.array([M, 0.0, 0.0])
        errs = []
        for f, q in G[side + "手捩"].items():
            if f not in G[side + "手首"]:
                continue
            l = tm.quat_mul(q, G[side + "手首"][f])
            ex = tm.euler_from_matrix(tm.matrix_from_quat(l), "YZX")[0]
            d = -(ex * -M) * 0.5
            errs.append(qang(tm.quat_set_from_axis_angle(fa, d), q))
        med = float(np.median(errs))
        worst_grant = max(worst_grant, med)
        print(f"  {side}手捩: {med:.2f}°  (pure-X axis)")

    # (2) end-to-end from the pose (golden torso/arm feed)
    print("\nend-to-end 手捩 from pose (golden torso/arm):")
    worst_tw = 0.0
    for side in ("左", "右"):
        errs = []
        for f, m in by_frame.items():
            gg = f + 1
            if gg not in G[side + "手捩"]:
                continue
            kp = m.get("posenet", {}).get("keypoints3D")
            if not kp:
                continue
            S = G["センター"].get(gg); h = G["上半身"].get(gg); mm = G["上半身2"].get(gg)
            ude = G[side + "腕"].get(gg); hiji = G[side + "ひじ"].get(gg)
            if any(x is None for x in (S, h, mm, ude, hiji)):
                continue
            twist, _wrist = solve_wrist(kp, S, h, mm, ude, hiji, side)
            errs.append(qang(twist, G[side + "手捩"][gg]))
        med = float(np.median(errs))
        worst_tw = max(worst_tw, med)
        sig = float(np.median([qang(v, [0, 0, 0, 1]) for v in G[side + "手捩"].values()]))
        print(f"  {side}手捩: {med:.1f}°  (signal {sig:.1f}°)")

    # pure-X axis costs ~2° vs the exact tilted fixedAxis (which is 0.00°) but generalizes
    ok = worst_grant < 2.5 and worst_tw < 20.0
    print(f"\nRESULT:", "PASS — grant ~exact (pure-X), pose twist < 20°" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
