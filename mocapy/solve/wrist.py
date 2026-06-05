"""Wrist (手首) + forearm-twist (手捩) — pose-based, with the MMD twist-grant.

Chain (rig): 腕 -> 腕捩 -> ひじ -> 手捩 -> 手首. The BVH writer folds 手捩 into the lower
arm (leftLowerArm = ひじ·手捩) and maps 手首 -> leftHand.

SOLVE (ported from the `It` onProcessRotation callback, min.js ~155226, + the wrist pose
basis ~124701):
  1. Wrist pose orientation `c` from the pose wrist/pinky/index landmarks (world kp3):
       i=wrist, n=pinky, a=index ; side-vec s = (e==1 ? a-n : n-a)   # ORDER flips per side
       r = (i-(n+a)/2).norm ; r.x*=pe ; s.y*=pe ; s.z*=pe
       l = cross(s,r).norm ; s = cross(r,l) ; Ve = setFromBasis([s,r,l]) ; c = conj(Ve)
     Sourcing (mirrored video, pe=+1 global): 左 -> e=1, idx 16/18/20 ; 右 -> e=0, idx 15/17/19.
  2. Wrist bone-local `l` = the wrist pose brought into the forearm's frame:
       l = conj(ひじ)·conj(腕)·le·c·He[M]   (le=conj(S·上半身·上半身2); 肩/腕捩 ~= identity)
     He[M]: w[1]=eulerYZX(0,-pi/2,pi/2), w[-1]=eulerYZX(0,pi/2,-pi/2) (腕 axis_rot is identity).
  3. The 手捩 GRANT (回転付与) takes HALF the wrist X-twist about the forearm fixed axis:
       t = eulerYZX(l).x * -M ;  d = -t*0.5 ;  手捩 = axisAngle(fixedAxis, d)
       手首 = 手捩^-1 · l
     fixedAxis ~= (M,0,0) (forearm dir in MMD T-pose local; exact cj tilt is (±0.998,-0.009,0.066),
     pure-X is within 0.2deg and generalizes to any VRM). M = +1 (左) / -1 (右).

VALIDATED vs golden2: the GRANT formula is exact (0.00deg self-consistency). End-to-end from
the pose (golden torso/arm feed): 手捩 ~11deg (右) / 15deg (左) vs golden twist mag 21/29deg —
a real twist that pulls the BVH lower-arm off identity. 手首 (hand bone) ~30deg: the pose
gives the forearm-relative wrist, but golden refines the hand SWING with the hand landmarker
(the handpose branch) which pose-only `c` doesn't capture — so 手首 is approximate, 手捩 solid.
"""

from __future__ import annotations

import math

import numpy as np

from . import three_math as tm

# wrist/pinky/index pose-landmark base index is 15/17/19; +e picks the side.
# pe=+1 (mirrored video): 左 -> e=1 (raw 16/18/20), 右 -> e=0 (raw 15/17/19).
_E = {"左": 1, "右": 0}
_M = {"左": 1.0, "右": -1.0}
# He[M] = w[M] (腕 axis_rot is identity for standard rigs)
_HE = {
    1.0: tm.quat_set_from_euler(np.array([0.0, -math.pi / 2, math.pi / 2]), "YZX"),
    -1.0: tm.quat_set_from_euler(np.array([0.0, math.pi / 2, -math.pi / 2]), "YZX"),
}


def _pt(kp3, i):
    k = kp3[i]
    return np.array([k["x"], k["y"], k["z"]], dtype=float)


def wrist_basis(kp3, side, pe=1.0):
    """Wrist world orientation `c` from pose wrist/pinky/index landmarks."""
    e = _E[side]
    i = _pt(kp3, 15 + e)
    n = _pt(kp3, 17 + e)
    a = _pt(kp3, 19 + e)
    s = (a - n) if e == 1 else (n - a)          # side vector: order flips per side
    r = i - (n + a) / 2.0
    r /= np.linalg.norm(r)
    r[0] *= pe
    s = s / np.linalg.norm(s)
    s[2] *= pe
    s[1] *= pe
    l = np.cross(s, r)
    l /= np.linalg.norm(l)
    s = np.cross(r, l)
    return tm.quat_conjugate(tm.quat_set_from_basis([s, r, l]))


def solve_wrist(kp3, S, h_upper, m_upper2, ude, hiji, side, pe=1.0):
    """Return (手捩, 手首) local rotations for one side.

    手捩 (forearm twist) is solid; 手首 (hand bone) is an approximate pose-only swing.
    """
    M = _M[side]
    c = wrist_basis(kp3, side, pe)
    le = tm.quat_conjugate(tm.quat_mul(tm.quat_mul(S, h_upper), m_upper2))
    # l = conj(ひじ)·conj(腕)·le·c·He[M]  (wrist pose in the forearm's frame)
    l = tm.quat_mul(le, c)
    l = tm.quat_mul(l, _HE[M])
    l = tm.quat_mul(tm.quat_conjugate(ude), l)
    l = tm.quat_mul(tm.quat_conjugate(hiji), l)
    # twist grant: 手捩 takes half the wrist X-twist about the forearm axis
    ex = tm.euler_from_matrix(tm.matrix_from_quat(l), "YZX")[0]
    d = -(ex * -M) * 0.5
    twist = tm.quat_set_from_axis_angle(np.array([M, 0.0, 0.0]), d)
    wrist = tm.quat_mul(tm.quat_conjugate(twist), l)
    return twist, wrist
