"""Tests for the THREE.js math primitive ports."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy.solve import three_math as tm  # noqa: E402

ORDERS = ["XYZ", "YXZ", "ZXY", "ZYX", "YZX", "XZY"]


def test_euler_quat_roundtrip():
    # euler -> quat -> matrix -> euler should round-trip away from gimbal.
    e = np.array([0.3, -0.4, 0.15])
    for order in ORDERS:
        q = tm.quat_set_from_euler(e, order)
        assert abs(np.linalg.norm(q) - 1) < 1e-12
        e2 = tm.euler_from_quat(q, order)
        q2 = tm.quat_set_from_euler(e2, order)
        # quaternions equal up to sign
        assert min(np.abs(q - q2).max(), np.abs(q + q2).max()) < 1e-9, order
    print("euler<->quat roundtrip: OK")


def test_axis_angle_matches_euler():
    # 90deg about Y via axis-angle vs euler YXZ.
    qa = tm.quat_set_from_axis_angle(np.array([0, 1, 0]), math.pi / 2)
    qe = tm.quat_set_from_euler(np.array([0, math.pi / 2, 0]), "YXZ")
    assert min(np.abs(qa - qe).max(), np.abs(qa + qe).max()) < 1e-12
    print("axis-angle vs euler: OK")


def test_vector_spherical_maps_axis_to_target():
    # quat_set_from_vector_spherical(ref, t) should rotate ref onto direction t.
    for ref in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0])):
        for t in (np.array([0.2, 0.5, -0.84]), np.array([-0.6, 0.1, 0.79])):
            t = t / np.linalg.norm(t)
            q = tm.quat_set_from_vector_spherical(ref, t)
            out = tm.apply_quat_to_vec(ref, q)
            # The mapping aligns the axis with t (allow small spherical convention error).
            cos = float(np.dot(out / np.linalg.norm(out), t))
            assert cos > 0.999, f"ref={ref} t={t} cos={cos}"
    print("vector-spherical maps axis->target: OK")


def test_apply_quat_identity():
    v = np.array([0.3, -0.7, 0.2])
    out = tm.apply_quat_to_vec(v, np.array([0, 0, 0, 1.0]))
    assert np.abs(out - v).max() < 1e-12
    print("apply_quat identity: OK")


def test_set_from_unit_vectors():
    # The quaternion must rotate `from` onto `to`.
    for frm, to in [
        (np.array([1.0, 0, 0]), np.array([0.0, 1, 0])),
        (np.array([1.0, 0, 0]), np.array([0.3, 0.5, -0.81])),
        (np.array([0.0, -1, 0]), np.array([0.1, 0.2, 0.97])),
    ]:
        frm = frm / np.linalg.norm(frm)
        to = to / np.linalg.norm(to)
        q = tm.quat_set_from_unit_vectors(frm, to)
        out = tm.apply_quat_to_vec(frm, q)
        assert np.abs(out - to).max() < 1e-9, f"{frm}->{to} got {out}"
    # Opposite vectors (180 deg) must stay finite and ~aligned.
    q = tm.quat_set_from_unit_vectors(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]))
    out = tm.apply_quat_to_vec(np.array([1.0, 0, 0]), q)
    assert np.abs(out - np.array([-1.0, 0, 0])).max() < 1e-9
    print("set_from_unit_vectors: OK")


def test_matrix_from_quat_orthonormal():
    q = tm.quat_set_from_euler(np.array([0.5, 0.2, -0.3]), "XYZ")
    M = tm.matrix_from_quat(q)
    assert np.abs(M @ M.T - np.eye(3)).max() < 1e-12
    assert abs(np.linalg.det(M) - 1) < 1e-12
    print("matrix_from_quat orthonormal: OK")


if __name__ == "__main__":
    test_euler_quat_roundtrip()
    test_axis_angle_matches_euler()
    test_vector_spherical_maps_axis_to_target()
    test_set_from_unit_vectors()
    test_apply_quat_identity()
    test_matrix_from_quat_orthonormal()
    print("\nAll three_math tests passed.")
