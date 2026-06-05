"""Within-run arm validation (armv run: boneKeys + armcap + landmarks together).

Sourcing (test B) is exact (~0.1°): solve_arm with flip_x=False reproduces the reference's b
from raw landmarks to within 0.4° p90.

Smoothing: the production pipeline uses blend-only e=0.5 (no je), which matches
the reference's dominant branch (`else` of `if(!p || d.score <= 0)` — when d.score>0 and p
truthy, je is skipped). The armv fixture has d.score=1 on every landmark so the reference
also skipped je every frame. Test A_v2 shows the production V2 chain (~6.5°/7.1°
median). Legacy A_v1 shows je+blend (~9.6°/9.8°) — kept as a regression baseline.
The ~6° residual is the irreducible floor; see arms.py docstring.

Tests:
  A_v2. output side : the reference's captured b -> blend (production)  vs golden 腕
  A_v1. output side : the reference's captured b -> je -> blend (legacy) vs golden 腕
  B.    sourcing    : my b from raw landmarks vs the reference's captured b
  C_v2. end-to-end  : raw landmarks -> my b -> blend (production) vs golden 腕
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy.solve.blend import BlendChannel  # noqa: E402
from mocapy.solve.torso import solve_center, solve_upper_body  # noqa: E402
from mocapy.solve.arms import solve_arm  # noqa: E402
from mocapy.detect.filters import OneEuroFilter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DT = 1000.0 / 30.0


def qang(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return math.degrees(2 * math.acos(min(1, abs(float(np.dot(a, b))))))


def main() -> int:
    base_dir = ROOT / "fixtures" if (ROOT / "fixtures" / "armv.bvh.golden.json").exists() else ROOT
    for suf in (".golden.json", ".armcap.json", ".landmarks.json"):
        if not (base_dir / ("armv.bvh" + suf)).exists():
            print("missing", "armv.bvh" + suf); return 2
    gj = json.loads((base_dir / "armv.bvh.golden.json").read_text(encoding="utf-8"))
    arm = json.loads((base_dir / "armv.bvh.armcap.json").read_text(encoding="utf-8"))
    lm = json.loads((base_dir / "armv.bvh.landmarks.json").read_text(encoding="utf-8"))
    by_frame = {c["f"]: (json.loads(c["m"]) if isinstance(c["m"], str) else c["m"]) for c in lm}

    def rotmap(b):
        return {round(k["time"] * 30): np.array(k["rot"]) for k in gj["boneKeys"] if k["name"] == b}
    golden = {"左腕": rotmap("左腕"), "右腕": rotmap("右腕")}

    last = {(a["fr"], a["c"]): a for a in arm}
    keys = sorted(last)

    def my_b(a, c):
        m = by_frame.get(a["fr"])
        if not m or not m.get("posenet", {}).get("keypoints3D"):
            return None
        kp3 = m["posenet"]["keypoints3D"]
        S = solve_center(kp3); h, M = solve_upper_body(kp3, S)
        ude, _ = solve_arm(kp3, S, h, M, c, flip_x=False)
        return ude

    def run(get_b, *, use_je):
        je = {"左": OneEuroFilter(30, 2, 2, 1, 4), "右": OneEuroFilter(30, 2, 2, 1, 4)}
        bl = {"左": BlendChannel(mocap_data_smoothing=2), "右": BlendChannel(mocap_data_smoothing=2)}
        errs = {"左": [], "右": []}
        for fr, c in keys:
            b = get_b(last[(fr, c)], c)
            if b is None:
                continue
            bone = c + "腕"
            if use_je:
                b = np.array(je[c].filter(list(b), fr * DT))
            rec = bl[c].add_rot(bone, b, e_timing=0.5)
            g = fr + 1
            if fr >= 40 and g in golden[bone]:
                errs[c].append(qang(rec, golden[bone][g]))
        return errs

    def report(label, errs):
        print(label)
        for c in ("左", "右"):
            e = np.array(errs[c])
            print(f"   {c}腕: median={np.median(e):.3f}° p90={np.percentile(e,90):.3f}°")

    xrb = lambda a, c: np.array(a["b"])  # noqa: E731

    report("A_v2. output side (the reference b -> blend, production) vs golden:",
           run(xrb, use_je=False))
    report("A_v1. output side (the reference b -> je -> blend, legacy) vs golden:",
           run(xrb, use_je=True))

    # B. sourcing
    bd = {"左": [], "右": []}
    for fr, c in keys:
        mb = my_b(last[(fr, c)], c)
        if mb is not None:
            bd[c].append(qang(mb, np.array(last[(fr, c)]["b"])))
    print("B. sourcing (my b from raw landmarks vs the reference b):")
    for c in ("左", "右"):
        e = np.array(bd[c])
        print(f"   {c}: median={np.median(e):.3f}° p90={np.percentile(e,90):.3f}°")

    report("C_v2. end-to-end (raw landmarks -> my b -> blend, production) vs golden:",
           run(my_b, use_je=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
