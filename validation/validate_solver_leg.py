"""Validate the leg FK solve (足 thigh, ひざ knee) against golden boneKeys.

Legs use FK here (use_legIK OFF). ひざ is ~exact; 足 has a ~7° residual from the
腰キャンセル/下半身 grant (documented). Feeds captured landmarks -> solve -> blend.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy.solve.blend import BlendChannel  # noqa: E402
from mocapy.solve.torso import solve_center, solve_lower_body  # noqa: E402
from mocapy.solve.legs import solve_leg  # noqa: E402

from mocapy._paths import FIXTURES, ASSETS  # noqa: E402
def qang(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return math.degrees(2 * math.acos(min(1, abs(float(np.dot(a, b))))))


def main() -> int:
    lm = json.loads((FIXTURES / "golden2.bvh.landmarks.json").read_text(encoding="utf-8"))
    gj = json.loads((FIXTURES / "golden2.bvh.golden.json").read_text(encoding="utf-8"))
    by_frame = {c["f"]: (json.loads(c["m"]) if isinstance(c["m"], str) else c["m"]) for c in lm}

    def rotmap(b):
        return {round(k["time"] * 30): np.array(k["rot"]) for k in gj["boneKeys"] if k["name"] == b}

    gD = rotmap("下半身")
    bD = BlendChannel(mocap_data_smoothing=2)
    ok = True

    # 下半身 (shared)
    eD = []
    lower_cache = {}
    for f in sorted(by_frame):
        kp3 = by_frame[f].get("posenet", {}).get("keypoints3D")
        if not kp3:
            continue
        S = solve_center(kp3)
        D = solve_lower_body(kp3, S)
        rd = bD.add_rot("下半身", D, e_timing=0.5)
        lower_cache[f] = (S, D)
        if f >= 40 and (f + 1) in gD:
            eD.append(qang(rd, gD[f + 1]))
    eD = np.array(eD)
    d_ok = np.median(eD) < 1.0
    ok = ok and d_ok
    print(f"下半身 : median={np.median(eD):.3f}° p90={np.percentile(eD,90):.3f}°  "
          f"{'PASS' if d_ok else 'FAIL'}")

    for s in ("左", "右"):
        gA, gK, gF = rotmap(s + "足"), rotmap(s + "ひざ"), rotmap(s + "足首")
        bA, bK, bF = (BlendChannel(mocap_data_smoothing=2) for _ in range(3))
        eA, eK, eF = [], [], []
        for f in sorted(by_frame):
            if f not in lower_cache:
                continue
            kp3 = by_frame[f]["posenet"]["keypoints3D"]
            S, D = lower_cache[f]
            thigh, knee, foot = solve_leg(kp3, S, D, s)
            ra = bA.add_rot(s + "足", thigh, e_timing=0.5)
            rk = bK.add_rot(s + "ひざ", knee, e_timing=0.5)
            rf = bF.add_rot(s + "足首", foot, e_timing=0.5)
            g = f + 1
            if f < 40:
                continue
            if g in gA:
                eA.append(qang(ra, gA[g]))
            if g in gK:
                eK.append(qang(rk, gK[g]))
            if g in gF:
                eF.append(qang(rf, gF[g]))
        eA, eK, eF = np.array(eA), np.array(eK), np.array(eF)
        a_ok, k_ok, f_ok = np.median(eA) < 1.5, np.median(eK) < 1.0, np.median(eF) < 2.5
        ok = ok and a_ok and k_ok and f_ok
        print(f"{s}足  : median={np.median(eA):.3f}° p90={np.percentile(eA,90):.3f}°  {'PASS' if a_ok else 'FAIL'}")
        print(f"{s}ひざ: median={np.median(eK):.3f}° p90={np.percentile(eK,90):.3f}°  {'PASS' if k_ok else 'FAIL'}")
        print(f"{s}足首: median={np.median(eF):.3f}° p90={np.percentile(eF,90):.3f}°  {'PASS' if f_ok else 'FAIL'}")
    print("\nRESULT:", "PASS — lower body + feet (下半身/足/ひざ/足首) reproduced" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
