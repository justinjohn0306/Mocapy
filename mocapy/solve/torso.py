"""Torso solve: センター (root), 下半身 (lower body), 上半身/上半身2 (spine).

Ports the body-pose section of the solver (the reference solver worker-message
handler, chars ~100800-104300 + bone application ~121900). Inputs are world pose
landmarks ``keypoints3D`` in raw BlazePose order (33 pts).

Status:
  * solve_center (センター rotation): VALIDATED exact — median 0.043 deg vs golden
    (after the shared skin-blend, validation/validate_solver_center.py).
  * solve_upper_body (上半身 / 上半身2): VALIDATED — median 0.16 deg / 0.30 deg vs
    golden (with skin-blend). The yaw split is qY(p/2) for 上半身 and qY(p/2 - d) for
    上半身2, where d = shoulder-twist s.y and p = js_wrap(d) (JS remainder keeps the
    sign — the earlier Python-modulo version flipped negatives, the one real bug).
    `ue` (head-tilt offset) only applies when tilt_adjustment.enabled (default false),
    so it is omitted here.
"""

from __future__ import annotations

import math

import numpy as np

from . import three_math as tm

# raw BlazePose indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

_X = np.array([1.0, 0.0, 0.0])
_NEG_Y = np.array([0.0, -1.0, 0.0])


def _qY(theta: float) -> np.ndarray:
    return np.array([0.0, math.sin(theta / 2), 0.0, math.cos(theta / 2)])


def _js_mod(a: float, n: float) -> float:
    """JavaScript % (remainder, keeps the sign of the dividend)."""
    return math.fmod(a, n)


def _wrap(d: float) -> float:
    """JS: p=(0+d)%(2pi); if(p>pi)p=2pi-p; else if(p<0&&p<-pi)p+=2pi."""
    p = _js_mod(d, 2 * math.pi)
    if p > math.pi:
        p = 2 * math.pi - p
    elif p < 0 and p < -math.pi:
        p += 2 * math.pi
    return p


def hip_direction(kp3) -> np.ndarray:
    hL = np.array([kp3[L_HIP]["x"], kp3[L_HIP]["y"], kp3[L_HIP]["z"]])
    hR = np.array([kp3[R_HIP]["x"], kp3[R_HIP]["y"], kp3[R_HIP]["z"]])
    t = hL - hR
    return t / np.linalg.norm(t)


def solve_center(kp3) -> np.ndarray:
    """センター / lower-body root rotation S (VALIDATED exact).

    S = setFromEuler( setFromVectorSpherical((1,0,0), hipDir) with z zeroed, "YZX" ).
    (st("hip") smoothing is passthrough in steady state and is omitted.)
    """
    t = hip_direction(kp3)
    e = tm.euler_from_vector_spherical(_X, t)
    e[2] = 0.0  # g = e.z is kept by the JS for 下半身; here we zero it for センター
    return tm.quat_set_from_euler(e, "YZX")


def _qX(a: float) -> np.ndarray:
    return np.array([math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)])


def spine_pitch(kp3, S) -> float:
    """A = the 上半身 forward pitch (w.x), used by 下半身 and the leg-pitch frame."""
    n, u, f = shoulder_center(kp3)
    nn = n / np.linalg.norm(n)
    n_local = tm.apply_quat_to_vec(nn, tm.quat_conjugate(S))
    return float(tm.euler_from_vector_spherical(_NEG_Y, n_local)[0])


def solve_lower_body(kp3, S):
    """下半身 = setFromEuler((v, 0, g), "YZX")  — VALIDATED ~0.19° vs golden.

    g = hip lean-z (the e.z zeroed in solve_center); v = 0.25*(sum of leg thigh
    pitches in the S*qX(A) frame) + A (spine pitch). This is the bone 足 is local to:
    足 = conjugate(下半身) * thigh_aim.
    """
    # g: hip-direction euler z (before zeroing)
    hL = np.array([kp3[L_HIP]["x"], kp3[L_HIP]["y"], kp3[L_HIP]["z"]])
    hR = np.array([kp3[R_HIP]["x"], kp3[R_HIP]["y"], kp3[R_HIP]["z"]])
    t = hL - hR
    g = float(tm.euler_from_vector_spherical(_X, t / np.linalg.norm(t))[2])

    A = spine_pitch(kp3, S)
    frame = tm.quat_conjugate(tm.quat_mul(S, _qX(A)))
    # leg thigh pitches (side-swapped hip/knee like legs.py): 左<-24/26, 右<-23/25
    v_sum = 0.0
    for hip_i, knee_i in ((24, 26), (23, 25)):
        d = np.array([kp3[hip_i]["x"], kp3[hip_i]["y"], kp3[hip_i]["z"]]) - \
            np.array([kp3[knee_i]["x"], kp3[knee_i]["y"], kp3[knee_i]["z"]])
        d = tm.apply_quat_to_vec(d / np.linalg.norm(d), frame)
        v_sum += float(tm.euler_from_vector_spherical(_NEG_Y, d)[0])
    v = 0.25 * v_sum + A
    return tm.quat_set_from_euler(np.array([v, 0.0, g]), "YZX")


def shoulder_center(kp3) -> np.ndarray:
    u = np.array([kp3[L_SHOULDER]["x"], kp3[L_SHOULDER]["y"], kp3[L_SHOULDER]["z"]])
    f = np.array([kp3[R_SHOULDER]["x"], kp3[R_SHOULDER]["y"], kp3[R_SHOULDER]["z"]])
    return (u + f) * 0.5, u, f


def solve_upper_body(kp3, S, *, shoulder_tracking=True):
    """上半身 (spine bend + yaw) and 上半身2 (shoulder twist). Returns (h_upper, m_upper2).

    Validated to ~0.16°/0.07° vs golden after the skin-blend.
    """
    n, u, f = shoulder_center(kp3)
    nn = n / np.linalg.norm(n)
    n_local = tm.apply_quat_to_vec(nn, tm.quat_conjugate(S))
    w = tm.euler_from_vector_spherical(_NEG_Y, n_local)
    h = tm.quat_set_from_euler(w, "XZY")

    a = u - f
    a /= np.linalg.norm(a)
    a_local = tm.apply_quat_to_vec(a, tm.quat_conjugate(tm.quat_mul(S, h)))
    s = tm.euler_from_vector_spherical(_X, a_local)
    d = s[1]
    p = _wrap(d)

    h_upper = tm.quat_mul(h, _qY(p / 2))          # +p/2 yaw to upper body

    # r = shoulder_tracking ? 0.5 : 0 (shoulders are tracked here); s.z *= 1 - r.
    # The r*s.z portion drives the 肩 (shoulder) bones; the rest stays on 上半身2.
    r = 0.5 if shoulder_tracking else 0.0
    s = s.copy()
    s[2] *= (1.0 - r)
    m = tm.quat_set_from_euler(s, "YZX")
    m_upper2 = tm.quat_mul(m, _qY(p / 2 - d))     # remainder to 上半身2
    return h_upper, m_upper2
