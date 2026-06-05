"""Head / neck solve (首, 頭).

TWO PATHS:

(1) FACE-MESH (preferred when MediaPipe face landmarker is available):
    head_total_from_face_matrix() uses the facial_transformation_matrix MediaPipe
    computes directly from the 478-point face mesh. Tight (<2°), low-noise — recommended.

(2) POSE-ONLY (fallback when no face data):
    The original head_orientation() builds a face basis from nose + eye-outer corners
    out of the pose landmarks. Validated ~4° vs golden — works, but jitterier.

The total head rotation (relative to the spine 上半身2 frame) is split ~50/50 between
首 (neck) and 頭 (head); both golden bones have equal magnitude.

Pipeline applies a head_rot OneEuro([30,1,2,1,4]) (the reference's Dt filter) to the total before
splitting — quaternion-space filtering, otherwise the pose face z can spin the head 170°.

Inputs (pose path): face landmarks in PIXEL scale with z ([x*W, y*H, z*W]).
Inputs (face-mesh path): MediaPipe's 4x4 facial_transformation_matrix.
"""

from __future__ import annotations

import math

import numpy as np

from . import three_math as tm

NOSE, L_EYE_OUTER, R_EYE_OUTER = 0, 3, 6
_PITCH = math.pi / 3.5  # face->head pitch offset


def _scale_quat(q, k):
    q = q / np.linalg.norm(q)
    w = max(-1.0, min(1.0, q[3]))
    ang = 2 * math.acos(w)
    s = math.sqrt(max(1 - w * w, 0.0))
    if s < 1e-9:
        return np.array([0.0, 0, 0, 1.0])
    return tm.quat_set_from_axis_angle(q[:3] / s, ang * k)


def head_orientation(face_px):
    """World-ish head orientation `s` from nose + eye-outer corners (px+z)."""
    e = np.asarray(face_px[NOSE], dtype=float)
    t = np.asarray(face_px[L_EYE_OUTER], dtype=float)
    o = np.asarray(face_px[R_EYE_OUTER], dtype=float)
    u = t - o
    h = float(np.dot(e, u) - np.dot(o, u)) / float(np.dot(u, u))
    K = o + u * h
    i = e - K; i /= np.linalg.norm(i)
    n = t - o; n /= np.linalg.norm(n)
    a = np.cross(n, i); a /= np.linalg.norm(a)
    n2 = np.cross(i, a)
    Ve = tm.quat_set_from_basis([n2, i, a])
    return tm.quat_mul(tm.quat_conjugate(Ve), np.array([math.sin(_PITCH / 2), 0, 0, math.cos(_PITCH / 2)]))


def head_total(face_px, S, h_upper, m_upper2):
    """Total head rotation relative to the 上半身2 frame (before the 首/頭 split).
    Apply a head_rot OneEuro([30,1,2,1,4]) to this (the reference's Dt filter) before splitting —
    MediaPipe's pose face z is noisy and will otherwise flip the basis / spin the head."""
    s = head_orientation(face_px)
    return tm.quat_mul(tm.quat_conjugate(tm.quat_mul(tm.quat_mul(S, h_upper), m_upper2)), s)


def split_head(total, *, split=0.5):
    """Split the (filtered) total head rotation 50/50 into (首, 頭)."""
    return _scale_quat(total, 1 - split), _scale_quat(total, split)


def solve_head(face_px, S, h_upper, m_upper2, *, split=0.5):
    """Unfiltered (首, 頭) — for tests. The pipeline filters head_total first."""
    return split_head(head_total(face_px, S, h_upper, m_upper2), split=split)


# ─── Face-mesh path (preferred) ───────────────────────────────────────────────

def head_orientation_from_face_matrix(face_matrix):
    """World-ish head orientation from MediaPipe's 4x4 facial_transformation_matrix.

    MediaPipe outputs the head in a camera/world frame where +X = subject's right,
    +Y = up, +Z = into-camera. To match our MMD/VRM frame (avatar's +X = avatar's
    LEFT, +Y = up, +Z = forward-from-avatar) we negate X and Z of the rotation
    columns — equivalent to a Y-axis 180° flip pre-multiplied, the same convention
    used for VRM 0.x rest poses (see vrm/skeleton.py).
    """
    R = np.asarray(face_matrix, dtype=float)[:3, :3].copy()
    # 180° Y-flip in pre-multiply: negate the X and Z rows of R.
    R[0, :] *= -1.0
    R[2, :] *= -1.0
    # And in post-multiply (for the basis axes): negate cols 0 and 2.
    R[:, 0] *= -1.0
    R[:, 2] *= -1.0
    # Convert rotation matrix -> quaternion via matrix_from_quat's inverse.
    # Build a 4x4 transform so we can use tm.matrix_from_quat conventions.
    return _matrix_to_quat(R)


def _matrix_to_quat(R):
    """3x3 rotation -> quaternion [x,y,z,w] (numerically stable)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    return q / np.linalg.norm(q)


def head_total_from_face_matrix(face_matrix, S, h_upper, m_upper2):
    """Total head rotation relative to 上半身2, using MediaPipe's face matrix."""
    s = head_orientation_from_face_matrix(face_matrix)
    return tm.quat_mul(tm.quat_conjugate(tm.quat_mul(tm.quat_mul(S, h_upper), m_upper2)), s)
