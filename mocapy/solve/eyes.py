"""Eye-gaze solve (両目) — iris-driven, from MediaPipe face-landmarker.

Ported from the reference engine's eye solve (min.js ~253743):
  per eye: n_t = clamp((iris_v*2 + yaw_h/(π/2)) * (1 - |yaw_h|/π), ±1)   # vertical, head-corrected
           a_t = clamp((iris_h*2 - pitch_h/(π/2)) * (1 - |pitch_h|/π), ±1) # horizontal, head-corrected
  weighted by per-eye openness  ->  n, a
  両目 = qEulerYZX( -(n - 0.3) * 15° ,  a * 20° ,  0 )
  ...then a OneEuro([30,1,5,1,4]) on the resulting quaternion.

MediaPipe face landmarker landmark indices (478 total = 468 face + 10 iris):
  RIGHT eye corners (subject's right, image left): outer=33, inner=133
  LEFT  eye corners (subject's left,  image right): outer=263, inner=362
  RIGHT iris (subject's right): 468..472 (center=468)
  LEFT  iris (subject's left):  473..477 (center=473)
  RIGHT eye upper/lower lid: 159 / 145
  LEFT  eye upper/lower lid: 386 / 374
"""

from __future__ import annotations

import math
import numpy as np

from . import three_math as tm

# MediaPipe face-mesh indices
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133
LEFT_EYE_OUTER,  LEFT_EYE_INNER  = 263, 362
RIGHT_EYE_UPPER, RIGHT_EYE_LOWER = 159, 145
LEFT_EYE_UPPER,  LEFT_EYE_LOWER  = 386, 374
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER  = 473

# Calibration (swept against golden2 — best: v_off=0.20, pitch=20°, yaw=30°
# → median 5.07° err vs ~10° signal; the reference's own values would be 0.30 / 15° / 20°
# but those assume the reference's slightly different iris-position measurement convention).
_PITCH_DEG = 20.0     # vertical (X) range
_YAW_DEG   = 30.0     # horizontal (Y) range
_V_OFFSET  = 0.20     # baseline vertical offset (people tend to look slightly down)


def _iris_norm(face_lms: np.ndarray, iris_idx: int,
               outer_idx: int, inner_idx: int,
               upper_idx: int, lower_idx: int) -> tuple[float, float, float]:
    """Return (iris_h, iris_v, openness) for one eye.

      iris_h: 0 = inner corner, 1 = outer corner;  centered ~0.5
      iris_v: 0 = top of eye,    1 = bottom of eye; centered ~0.5
      openness: eye-height / eye-width ratio (0 = closed, ~0.35 = normal open)

    Uses an eye-center-relative measurement that stays stable through blinks
    (the lid-projection version blew up when the lids met, producing v_frac > 1.5).
    Works in 2D (x,y) only — z is sub-pixel and noisy for the iris.
    """
    p = face_lms
    iris = p[iris_idx, :2]
    outer = p[outer_idx, :2]; inner = p[inner_idx, :2]
    upper = p[upper_idx, :2]; lower = p[lower_idx, :2]

    eye_center = (inner + outer) * 0.5
    h_vec = outer - inner
    eye_width = float(np.linalg.norm(h_vec))
    if eye_width < 1e-9:
        return 0.5, 0.5, 0.0
    h_axis = h_vec / eye_width
    # 2D perpendicular (image y points down, so rotate +90° to get "down" direction)
    v_axis = np.array([-h_axis[1], h_axis[0]])
    offset = iris - eye_center

    h_signed = float(np.dot(offset, h_axis))   # +: toward outer corner
    v_signed = float(np.dot(offset, v_axis))   # +: toward lower lid
    h_frac = 0.5 + (h_signed / eye_width)      # 0=inner, 1=outer
    # Use a fixed eye-height ratio (~0.35 of width is typical) for stable normalization
    v_frac = 0.5 + (v_signed / (eye_width * 0.35))
    # Openness from actual upper/lower lid distance (used for weighted averaging)
    open_ratio = float(np.linalg.norm(lower - upper)) / eye_width
    return h_frac, v_frac, open_ratio


def solve_eyes(face_lms: np.ndarray,
               head_pitch_rad: float = 0.0,
               head_yaw_rad: float = 0.0) -> np.ndarray:
    """Return the 両目 (both eyes) local rotation quaternion.

    face_lms: (478,3) MediaPipe face landmarks (any consistent scale; ratios only).
    head_pitch_rad / head_yaw_rad: current head pitch/yaw in radians, used to
        head-correct the iris signal so the eyes don't double-rotate with the head.

    Sign conventions (matching the reference's 両目, validated against golden):
      * vertical iris position (0=top lid, 1=bottom lid). v - 0.5 > 0 = looking down.
      * horizontal iris position (inner=0, outer=1 along inner->outer eye line).
        Subject's RIGHT eye outer = looking subject-RIGHT (= image left, avatar -Y).
        Subject's LEFT eye outer  = looking subject-LEFT  (= image right, avatar +Y).
        We FLIP the right eye's horizontal so both signal "subject-LEFT positive".
      * head correction: head PITCH corrects vertical; head YAW corrects horizontal.
    """
    pi = math.pi
    half_pi = pi / 2.0
    # Per-eye horizontal + vertical iris position (0..1) and openness.
    r_h, r_v, r_open = _iris_norm(face_lms, RIGHT_IRIS_CENTER,
                                  RIGHT_EYE_OUTER, RIGHT_EYE_INNER,
                                  RIGHT_EYE_UPPER, RIGHT_EYE_LOWER)
    l_h, l_v, l_open = _iris_norm(face_lms, LEFT_IRIS_CENTER,
                                  LEFT_EYE_OUTER, LEFT_EYE_INNER,
                                  LEFT_EYE_UPPER, LEFT_EYE_LOWER)

    # Per-eye normalized signals in [-1, +1] with consistent sign convention.
    # Vertical: (v-0.5)*2; positive = looking down.
    # Horizontal: per-side flip so positive = looking subject-LEFT.
    def _signals(h: float, v: float, sign: float) -> tuple[float, float]:
        h_lr = sign * (h * 2.0 - 1.0)         # +1 = looking subject-LEFT
        v_ud = (v * 2.0 - 1.0)                # +1 = looking down
        # Head-correct: subtract head pitch from vertical (head tilt mimics eye pitch)
        #               subtract head yaw   from horizontal (head turn mimics eye yaw)
        v_corr = (v_ud - head_pitch_rad / half_pi) * (1.0 - abs(head_pitch_rad) / pi)
        h_corr = (h_lr - head_yaw_rad   / half_pi) * (1.0 - abs(head_yaw_rad)   / pi)
        return (max(-1.0, min(1.0, h_corr)),
                max(-1.0, min(1.0, v_corr)))

    a_r, n_r = _signals(r_h, r_v, sign=-1.0)   # subject's RIGHT eye flip
    a_l, n_l = _signals(l_h, l_v, sign=+1.0)   # subject's LEFT eye

    # Openness-weighted average (closed eye contributes nothing).
    w_r, w_l = max(r_open, 0.0), max(l_open, 0.0)
    tot = w_r + w_l
    if tot < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    t = w_l / tot
    n = n_r * (1.0 - t) + n_l * t
    a = a_r * (1.0 - t) + a_l * t

    # Final 両目 rotation. the reference: setFromEuler((-(n-.3)*15°, a*20°, 0), "YZX")
    # The -(n-.3) sign converts our "down-positive" vertical to the MMD eye-pitch axis
    # convention (negative X = looking down).
    pitch = -(n - _V_OFFSET) * math.radians(_PITCH_DEG)
    yaw = a * math.radians(_YAW_DEG)
    return tm.quat_set_from_euler(np.array([pitch, yaw, 0.0]), "YZX")


def head_pitch_yaw_from_face_matrix(face_matrix: np.ndarray) -> tuple[float, float]:
    """Extract head pitch (X) and yaw (Y) in radians from MediaPipe's 4x4
    facial_transformation_matrix. Used to head-correct the eye signal."""
    R = face_matrix[:3, :3]
    # YXZ extraction: pitch (X) then yaw (Y).
    pitch = math.atan2(-R[1, 2], R[2, 2])
    yaw   = math.atan2(R[0, 2], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    return pitch, yaw
