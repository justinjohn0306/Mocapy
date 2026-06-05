"""Leg solve: 足 (thigh), ひざ (knee), 足首 (ankle), via 足ＩＫ 2-bone IK.

Joints (SIDE-SWAPPED like the arms — character 左 uses image-RIGHT landmarks):
  左: hip=raw23->24? NO — empirically 左足 uses raw 24/26/28 (right side), 右 uses 23/25/27.
Parent frame is S (センター rotation), via conjugate(S) — NOT the arm's torso-chain `le`.

足 (thigh) aim (pre-IK), VALIDATED to ~7° (golden mag ~14°):
  d  = (hip - knee).normalize().applyQuaternion(conjugate(S))
  c  = setFromVectorSpherical((0,-1,0), d)            # euler
  # ankle coupling overrides the yaw c.y:
  o  = (knee - ankle).normalize().applyQuaternion(conjugate(S))
  i  = o.applyQuaternion(conjugate(setFromEuler(c,"XZY")))
  i.z = max(-i.z, 0); i.x *= -1; i = i.normalize()
  e  = toSphericalCoords(i);  t = sqrt(min((pi - e[2])/(pi/2), 1))
  c.y = e[1] * t
  足 = setFromEuler(c, "XZY")

STATUS (legs use FK here — use_legIK is OFF in this config, so NO CCD IK needed):
  * ひざ (knee): VALIDATED ~0.67° vs golden. ひざ = setFromVectorSpherical((0,-1,0),
    i_fk) where i_fk = (knee-ankle).norm . conj(S) . conj(足_adjusted)  [NO z/x flip].
  * 足 (thigh): ~7° vs golden. ひざ (relative to 足) is exact, so the 足 residual is the
    腰キャンセル parent (waist-cancel grant of 下半身) — 足's recorded LOCAL is relative
    to 腰キャンセル左 (parent), which grant-cancels 下半身, while solve_thigh aims in the
    センター(S) frame. Closing 足 needs the 腰キャンセル/下半身 grant FK (and 下半身's
    leg-coupled v). Ce/He input OneEuro filters are minor.
  * 足首 (ankle): non-identity axis_rot, foot direction + grounding; unported.
"""

from __future__ import annotations

import math

import numpy as np

from . import three_math as tm

# side-swap: character 左 <- image-right landmarks (raw 24/26/28); 右 <- 23/25/27
HIP = {"左": 24, "右": 23}
KNEE = {"左": 26, "右": 25}
ANKLE = {"左": 28, "右": 27}
HEEL = {"左": 30, "右": 29}
TOE = {"左": 32, "右": 31}
_NEG_Y = np.array([0.0, -1.0, 0.0])
_FOOT_PITCH = -math.pi / 8  # model_quality != "Best" (else -pi/10)


def _qX(a):
    return np.array([math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)])


def _pt(kp3, i):
    k = kp3[i]
    return np.array([k["x"], k["y"], k["z"]])


def solve_thigh(kp3, S, side):
    """足 (thigh) pre-IK aim with ankle-yaw coupling. ~7° vs golden; IK refines further."""
    Sc = tm.quat_conjugate(S)
    hip = _pt(kp3, HIP[side]); knee = _pt(kp3, KNEE[side]); ankle = _pt(kp3, ANKLE[side])

    d = hip - knee
    d = tm.apply_quat_to_vec(d / np.linalg.norm(d), Sc)
    c = tm.euler_from_vector_spherical(_NEG_Y, d)
    l0 = tm.quat_set_from_euler(c, "XZY")

    o = knee - ankle
    o = tm.apply_quat_to_vec(o / np.linalg.norm(o), Sc)
    i = tm.apply_quat_to_vec(o, tm.quat_conjugate(l0))
    i = i.copy()
    i[2] = max(-i[2], 0.0)
    i[0] *= -1.0
    i = i / np.linalg.norm(i)
    e = tm.to_spherical_coords(i)
    t = math.sqrt(min((math.pi - e[2]) / (math.pi / 2), 1.0))
    c = c.copy()
    c[1] = e[1] * t
    return tm.quat_set_from_euler(c, "XZY")


def solve_ankle(kp3, S, thigh_aim, knee, side):
    """足首 (foot) — VALIDATED ~1.4° vs golden. parent_based on ひざ.

    Foot orientation from a heel/ankle/toe basis, conjugated + a -pi/8 pitch, expressed
    relative to the knee world frame (= S . thigh_aim . knee, since 下半身 cancels the
    腰キャンセル grant). Grounding/ratio/ue refinements omitted (minor).
    """
    heel = _pt(kp3, HEEL[side]); ankle = _pt(kp3, ANKLE[side]); toe = _pt(kp3, TOE[side])
    e = heel - ankle; e /= np.linalg.norm(e)
    t = heel - toe; t /= np.linalg.norm(t)
    o = np.cross(e, t); o /= np.linalg.norm(o)
    e2 = np.cross(t, o)
    Ve = tm.quat_set_from_basis([o, e2, t])
    i = tm.quat_mul(tm.quat_conjugate(Ve), _qX(_FOOT_PITCH))
    knee_world = tm.quat_mul(tm.quat_mul(S, thigh_aim), knee)
    return tm.quat_mul(tm.quat_conjugate(knee_world), i)


def solve_leg(kp3, S, lower_body, side):
    """Return (足, ひざ, 足首) local rotations. VALIDATED <1.5° vs golden.

    足 = conjugate(下半身) * thigh_aim  (足's parent 腰キャンセル grant-cancels to 下半身).
    ひざ = knee FK (relative to 足). 足首 = foot basis relative to the knee world frame.
    """
    thigh_aim = solve_thigh(kp3, S, side)
    knee = solve_knee(kp3, S, side)
    foot = solve_ankle(kp3, S, thigh_aim, knee, side)
    thigh = tm.quat_mul(tm.quat_conjugate(lower_body), thigh_aim)
    return thigh, knee, foot


def solve_knee(kp3, S, side):
    """ひざ (knee) FK — VALIDATED ~0.67° vs golden. Relative to the (adjusted) 足.

    Returns the knee local rotation; pair with solve_thigh's 足 for the leg.
    """
    Sc = tm.quat_conjugate(S)
    knee = _pt(kp3, KNEE[side]); ankle = _pt(kp3, ANKLE[side])
    thigh = solve_thigh(kp3, S, side)  # adjusted 足
    o = knee - ankle
    o = tm.apply_quat_to_vec(o / np.linalg.norm(o), Sc)
    i_fk = tm.apply_quat_to_vec(o, tm.quat_conjugate(thigh))
    return tm.quat_set_from_vector_spherical(_NEG_Y, i_fk)
