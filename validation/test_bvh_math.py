"""Unit tests for the BVH writer's number formatting and rotation math."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy.export.bvh import (  # noqa: E402
    js_num,
    quat_to_euler_yxz,
    quat_mul,
    write_bvh,
)
from mocapy.vrm.skeleton import load_skeleton  # noqa: E402

from mocapy._paths import FIXTURES, ASSETS  # noqa: E402
def test_js_num():
    cases = {
        0.0: "0",
        -0.0: "0",
        1 / 30: "0.03333333333333333",
        9.073646: "9.073646",
        0.125542283: "0.125542283",
        100.0: "100",
        -2.7508990047486215: "-2.7508990047486215",
    }
    for value, expected in cases.items():
        got = js_num(value)
        assert got == expected, f"js_num({value!r}) = {got!r}, expected {expected!r}"

    # Values taken verbatim from motion.bvh (exact JS formatting incl. exponent style).
    assert js_num(-5.960464477539062e-7) == "-5.960464477539062e-7"
    assert js_num(-0.12534432395355233) == "-0.12534432395355233"
    print("js_num: OK")


def test_euler_yxz():
    # Identity quaternion -> zero euler.
    ex, ey, ez = quat_to_euler_yxz((0, 0, 0, 1))
    assert abs(ex) < 1e-12 and abs(ey) < 1e-12 and abs(ez) < 1e-12

    # 90 deg about Y.
    s = math.sin(math.radians(45))
    ex, ey, ez = quat_to_euler_yxz((0, s, 0, s))
    assert abs(math.degrees(ey) - 90) < 1e-9
    assert abs(ex) < 1e-9 and abs(ez) < 1e-9

    # 90 deg about X.
    ex, ey, ez = quat_to_euler_yxz((s, 0, 0, s))
    assert abs(math.degrees(ex) - 90) < 1e-9
    print("euler_yxz: OK")


def test_quat_mul_identity():
    q = (0.1, 0.2, 0.3, math.sqrt(1 - 0.14))
    out = quat_mul(q, (0, 0, 0, 1))
    assert all(abs(a - b) < 1e-12 for a, b in zip(out, q))
    print("quat_mul: OK")


def test_write_bvh_smoke():
    skel = load_skeleton(ASSETS / "cj.vrm")
    # Two-frame synthetic motion on a couple of MMD-named tracks.
    bone_keys = [
        {"name": "下半身", "time": 0.0, "pos": [0, 0, 0], "rot": [0, 0, 0, 1]},
        {"name": "下半身", "time": 1.0, "pos": [0, 0, 0], "rot": [0, 0, 0, 1]},
        {"name": "センター", "time": 0.0, "pos": [0, 0, 0], "rot": [0, 0, 0, 1]},
        {"name": "センター", "time": 1.0, "pos": [1, 2, 3], "rot": [0, 0, 0, 1]},
    ]
    vrm_to_mmd = {"hips": "センター", "spine": "上半身"}
    text = write_bvh(skel, bone_keys, vrm_to_mmd)
    assert text.startswith("HIERARCHY")
    assert "MOTION" in text
    # the reference-style: rest frame at index 0 + the 31 boneKey timestamps (time 0..1s
    # is 31 distinct positions on the 30fps grid, shifted by +1/30 and bracketed
    # by a synthetic trailing key) = 32 BVH frames.
    assert "Frames: 32" in text
    print("write_bvh smoke: OK")


if __name__ == "__main__":
    test_js_num()
    test_euler_yxz()
    test_quat_mul_identity()
    test_write_bvh_smoke()
    print("\nAll BVH math tests passed.")
