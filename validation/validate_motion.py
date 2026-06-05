"""Validate the full Python BVH writer against the reference engine golden data.

Inputs (produced by the temporary dump in animate.js):
  golden_out.bvh             — reference BVH from the reference engine
  golden_out.bvh.golden.json — { boneKeys, hierarchy_list, scales, ... }

Runs mocapy.export.bvh.write_bvh on the dumped boneKeys and diffs the result
against golden_out.bvh, both byte-wise and numerically per channel.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy.export.bvh import build_hierarchy, write_bvh  # noqa: E402
from mocapy.vrm.skeleton import load_skeleton  # noqa: E402

from mocapy._paths import FIXTURES, ASSETS  # noqa: E402
def motion_matrix(bvh_text: str):
    lines = bvh_text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip() == "MOTION")
    frames = int(lines[i + 1].split(":")[1])
    rows = lines[i + 3:i + 3 + frames]
    return np.array([[float(x) for x in r.split()] for r in rows]), frames


def main() -> int:
    bvh_path = FIXTURES / "golden_out.bvh"
    json_path = FIXTURES / "golden_out.bvh.golden.json"
    if not bvh_path.exists() or not json_path.exists():
        print("Golden files not found yet:", bvh_path.name, json_path.name)
        return 2

    golden = json.loads(json_path.read_text(encoding="utf-8"))
    skel = load_skeleton(ASSETS / "cj.vrm")

    # VRM->MMD map joined on BVH display name via our hierarchy order.
    _, order = build_hierarchy(skel)
    name_mmd_by_bvh = {item["name"]: item.get("name_MMD") for item in golden["hierarchy_list"]}
    vrm_to_mmd = {vrm: name_mmd_by_bvh.get(bvh) for bvh, vrm in order}

    leg_scale = golden["left_leg_length"] / golden["model_para_left_leg_length"]
    pos_scale = golden["vrm_scale"] / golden["VRM_vrm_scale"]

    # The the reference golden BVH was generated without the leading rest frame, so
    # disable that option to keep frame counts aligned for byte comparison.
    mine = write_bvh(skel, golden["boneKeys"], vrm_to_mmd,
                     leg_scale=leg_scale, pos_scale=pos_scale,
                     prepend_rest_frame=False)
    ref = bvh_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    if mine == ref:
        print("EXACT MATCH: full BVH is byte-identical to golden_out.bvh")
        return 0

    print("Not byte-identical — numeric comparison of MOTION channels:\n")
    m_mat, m_frames = motion_matrix(mine)
    r_mat, r_frames = motion_matrix(ref)
    print(f"frames: mine={m_frames} ref={r_frames}")
    if m_mat.shape != r_mat.shape:
        print(f"!! shape mismatch mine={m_mat.shape} ref={r_mat.shape}")
        return 1
    diff = np.abs(m_mat - r_mat)
    print(f"channels/frame: {m_mat.shape[1]}")
    print(f"max abs channel error : {diff.max():.6e}")
    print(f"mean abs channel error: {diff.mean():.6e}")
    # Which column is worst?
    col = int(diff.max(axis=0).argmax())
    fr = int(diff[:, col].argmax())
    print(f"worst at frame {fr}, channel {col}: mine={m_mat[fr, col]} ref={r_mat[fr, col]}")

    # First literal line difference for insight.
    ml, rl = mine.splitlines(), ref.splitlines()
    for i in range(min(len(ml), len(rl))):
        if ml[i] != rl[i]:
            print(f"\nfirst line diff at {i}:\n  mine: {ml[i][:160]}\n  ref : {rl[i][:160]}")
            break

    # Pass if errors are float-noise level. The only cells that approach 1e-3 are
    # joints exactly at YXZ gimbal lock, where asin(~±1) amplifies quaternion ULPs.
    return 0 if diff.max() < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
