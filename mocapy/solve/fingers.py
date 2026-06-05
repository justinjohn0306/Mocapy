"""Finger solve (人指/中指/薬指/小指) — from MediaPipe hand-landmarker.

Ported from the min.js finger loop (~126372-130600). Each finger's 3 joints are solved
down the chain: express the hand points in a palm-aligned basis, then for each joint take
the segment direction in the *previous joint's* frame, convert to an euler via
setFromVectorSpherical, apply the bend clamps, and record the joint quaternion.

Pipeline (per finger, 4 hand points base->tip):
  Ve = basis(palm0, middle0, index0, pinky0)         # hand orientation
  P  = Ve with x,z negated                            # into hand-local
  v[e] = P.apply((pt[e]-palm0) * (1,-1,1))            # finger points, hand-local
  for d in 0,1,2:
     re-root remaining points at v[d], carry them through M (prev joint's conj)
     dir = (v[d+1]-v[d]).norm
     s = eulerFromVectorSpherical((0,1,0), dir);  a = quat(s,"XZY");  M = conj(a)
     ...bend clamps on s (degenerate reset, s.x reduction, s.z taper by joint)...
     bone = quat(s,"XZY") with its rotation-axis z negated

Sourcing (mirrored video): character 左 <- image 'Right' hand, 右 <- 'Left'. R = +1 (左) / -1 (右).

The hand basis side-vector flips per side (左 index-pinky / 右 pinky-index) — this brought
the left hand's distal joints from ~13-23deg down to ~6-10deg.

VALIDATED vs golden2 (cj): middle/distal joints ~6-13deg (signal 4-40), proximal ~16-32deg
(the base joint carries the most clamp logic). Thumb (親指, the m==0 branch) IS ported but
approximate (~28-40deg proximal / ~26-33deg distal) — the model-specific 親指 axis_rot
offset corrections aren't ported, so the thumb curls plausibly but isn't golden-exact.
"""

from __future__ import annotations

import math

import numpy as np

from . import three_math as tm

_NY = np.array([0.0, 1.0, 0.0])
_FW = {0: "０", 1: "１", 2: "２", 3: "３"}

# annotation key -> (MMD finger glyph, m index, mediapipe landmark indices base->tip)
# m: 1=index 2=middle 3=ring 4=pinky (drives the s.z taper / ring-pinky base sign).
_FINGERS = [
    ("index", "人指", 1, (5, 6, 7, 8)),
    ("middle", "中指", 2, (9, 10, 11, 12)),
    ("ring", "薬指", 3, (13, 14, 15, 16)),
    ("pinky", "小指", 4, (17, 18, 19, 20)),
]
# character side -> (mediapipe handedness label, R sign)
_SIDE = {"左": ("Right", 1.0), "右": ("Left", -1.0)}


def annotations_from_landmarks(lms21, W, H):
    """MediaPipe 21 hand landmarks (normalized) -> the reference annotation dict in px+z scale.
    palm=[lm0], index=[5..8], etc. Scale matches pose_raw: [x*W, y*H, z*W]."""
    pts = np.asarray(lms21, dtype=float) * np.array([W, H, W])
    ann = {"palm": [pts[0]]}
    for key, _g, _m, idx in _FINGERS:
        ann[key] = [pts[i] for i in idx]
    ann["thumb"] = [pts[i] for i in (1, 2, 3, 4)]
    return ann


def hand_basis(ann, side, pe=1.0):
    """Hand orientation quaternion from palm/middle/index/pinky base points.

    The side-vector order flips per side (min.js: e==1?[index,pinky]:[pinky,index]) —
    左 -> index-pinky, 右 -> pinky-index. (Improves the left hand 18->12deg.)"""
    palm = np.asarray(ann["palm"][0], float)
    mid = np.asarray(ann["middle"][0], float)
    idx = np.asarray(ann["index"][0], float)
    pky = np.asarray(ann["pinky"][0], float)
    n = palm - mid
    n /= np.linalg.norm(n)
    n[0] *= pe
    a = (idx - pky) if side == "左" else (pky - idx)
    a /= np.linalg.norm(a)
    a[2] *= pe
    a[1] *= pe
    s = np.cross(a, n)
    s /= np.linalg.norm(s)
    a2 = np.cross(n, s)
    return tm.quat_set_from_basis([a2, n, s])


def _ang_between(u, v):
    return math.acos(max(-1.0, min(1.0, float(np.dot(u / np.linalg.norm(u), v / np.linalg.norm(v))))))


def _axis_z_flip(a):
    a = a / np.linalg.norm(a)
    w = max(-1.0, min(1.0, a[3]))
    ang = 2 * math.acos(w)
    s = math.sqrt(max(1 - w * w, 0.0))
    if s < 1e-9:
        return a
    ax = a[:3] / s
    ax[2] *= -1
    return tm.quat_set_from_axis_angle(ax, ang)


def _axis_correct(quat: np.ndarray, R: float, is_thumb: bool,
                     axis_rot: np.ndarray | None,
                     axis_rot_offset_inv: np.ndarray | None,
                     apply_offset_inv: bool) -> np.ndarray:
    """Full THREEX-mode the reference finger correction — min.js v0.34.0 ~130600 inlined.

    In the JS, `m` is the FINGER index (thumb=0, index=1, middle=2, ring=3,
    pinky=4); our `is_thumb` flag captures `m == 0`. `d` is the joint index
    within the finger.

    Steps:
      1. (axis, angle) = quat.toAxisAngle()
      2. axis.z *= -1
      3. axis.applyEuler((π/2, -π/2·R, 0), "YXZ")   — THREEX-mode rotation
      4. SKIP axis_rot for ALL thumb joints (`m == 0 && THREEX.enabled`);
         non-thumb fingers apply axis_rot.
      5. a = quat.setFromAxisAngle(axis, angle)
      6. If apply_offset_inv: a.premultiply(slerp(I, axis_rot_offset_inv, k))
         where k = 0.5 for thumb (m==0); k = 1.0 for non-thumb.
    """
    a = quat / np.linalg.norm(quat)
    w = max(-1.0, min(1.0, a[3]))
    ang = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-9:
        out = a
    else:
        axis = a[:3] / s
        axis[2] *= -1
        q_euler = tm.quat_set_from_euler(
            np.array([math.pi / 2.0, -math.pi / 2.0 * R, 0.0]), "YXZ")
        axis = tm.apply_quat_to_vec(axis, q_euler)
        if (not is_thumb) and axis_rot is not None:
            axis = tm.apply_quat_to_vec(axis, axis_rot)
        out = tm.quat_set_from_axis_angle(axis, ang)

    if apply_offset_inv and axis_rot_offset_inv is not None:
        k = 0.5 if is_thumb else 1.0
        ident = np.array([0.0, 0.0, 0.0, 1.0])
        q_off = np.array(tm.quat_slerp_arr(ident, axis_rot_offset_inv, k))
        out = tm.quat_premultiply(out, q_off)
    return out


def solve_finger(ann, finger_key, m, R, side, pe=1.0,
                 axis_rot_by_mmd: dict | None = None,
                 axis_rot_offset_inv_first_joint: np.ndarray | None = None,
                 mmd_glyph: str = ""):
    """Return [joint1, joint2, joint3] local quaternions for one finger (non-thumb).

    axis_rot_by_mmd: optional dict {MMD-bone-name → axis_rot quat}. When provided,
        applies the reference's full THREEX-mode finger correction (axis_rot multiplied into
        the per-joint axis).
    axis_rot_offset_inv_first_joint: axis_rot_offset_inv for the FIRST joint of
        this finger (the reference's `Ne[y].axis_rot_offset_inv`). Premultiplied at d == 0.
    mmd_glyph: the MMD finger name (人指/中指/薬指/小指) for axis_rot lookup.
    """
    Ve = hand_basis(ann, side, pe)
    P = Ve.copy()
    P[0] *= -1
    P[2] *= -1
    palm = np.asarray(ann["palm"][0], float)
    v = [tm.apply_quat_to_vec((np.asarray(pt, float) - palm) * np.array([1.0, -1.0, 1.0]), P)
         for pt in ann[finger_key]]
    f = 0
    M = None
    eulers = []
    rots = []
    suffixes = ("１", "２", "３")
    for d in range(3):
        pts = [v[e].copy() for e in range(4)]
        w = pts[d].copy()
        for e in range(d, 4):
            pts[e] = pts[e] - w
            if M is not None and e > d:
                pts[e] = tm.apply_quat_to_vec(pts[e], M)
        seg = pts[d + 1] - pts[d]
        nn = seg / np.linalg.norm(seg)
        s = tm.euler_from_vector_spherical(_NY, nn)
        a = tm.quat_set_from_euler(s, "XZY")
        M = tm.quat_conjugate(a)
        if abs(s[2]) > math.pi / 2.5 or s[0] > math.pi / 3:
            s = np.array([-abs(_ang_between(_NY, nn)), 0.0, 0.0])
        r = None
        if s[0] > 0:
            if d > f:
                v0x = eulers[0][0]
                s[0] *= (-min(abs(v0x) / (math.pi / 3), 1.0) if v0x < 0 else 0.0)
            else:
                s[0] *= 0.75
        if s[0] < 0:
            r = -math.pi / 1.8
        emin = min([e[0] for e in eulers] + [s[0]])
        if d == 0:
            if m > 2:
                s[2] = R * abs(s[2])
        else:
            t = 1 - (min(abs(emin + math.pi / 6) / (math.pi / 2 - math.pi / 6), 1.0)
                     if emin < -math.pi / 6 else 0.0)
            s[2] *= t * (0.75 if d == 1 else 0.5)
            if t < 1:
                ew = 2 * math.acos(max(-1.0, min(1.0, a[3])))
                s[0] = s[0] * t + np.sign(s[0]) * abs(ew) * (1 - t)
        if r is not None:
            s[0] = max(s[0], r)
        eulers.append(s.copy())

        quat_raw = tm.quat_set_from_euler(s, "XZY")
        if axis_rot_by_mmd:
            bone_name = side + mmd_glyph + suffixes[d]
            axis_rot = axis_rot_by_mmd.get(bone_name)
            offset_inv = axis_rot_offset_inv_first_joint if d == 0 else None
            rots.append(_axis_correct(quat_raw, R, is_thumb=False,
                                         axis_rot=axis_rot,
                                         axis_rot_offset_inv=offset_inv,
                                         apply_offset_inv=(d == 0)))
        else:
            rots.append(_axis_z_flip(quat_raw))
    return rots


def _finger_axis_correction(a, R):
    """VRM-mode finger axis fix: axis.z*=-1 then applyEuler((π/2, -π/2·R, 0), "YXZ").

    Ports the THREEX branch of min.js ~130700, applied to EVERY finger joint (not
    just the thumb). After extracting (axis, angle) from the bone quaternion, the
    natural curl axis on a VRM differs from the standard rig axis by a fixed
    offset (π/2 about X, ±π/2 about Y per side, in YXZ order). The ±π/2 term is
    side-dependent, so omitting this rotation creates LEFT/RIGHT asymmetry —
    which is exactly what the user observed in Blender (left-hand fingers looked
    different from the right).

    For non-thumb fingers there's also an `axis.applyQuaternion(Ne[_].axis_rot)`
    step in the JS that requires per-rig data (axis_rot from the model's bone
    rest pose). We don't currently thread rig data into the finger solver, so we
    skip that piece — this is approximate; see notes for why
    static rig.json values didn't help there either.
    """
    a = a / np.linalg.norm(a)
    w = max(-1.0, min(1.0, a[3]))
    ang = 2 * math.acos(w)
    s = math.sqrt(max(1 - w * w, 0.0))
    if s < 1e-9:
        return a
    axis = a[:3] / s
    axis[2] *= -1
    q_off = tm.quat_set_from_euler(np.array([math.pi / 2.0, -math.pi / 2.0 * R, 0.0]), "YXZ")
    axis = tm.apply_quat_to_vec(axis, q_off)
    return tm.quat_set_from_axis_angle(axis, ang)


# Back-compat alias — old code (solve_thumb) used this name.
_thumb_axis_correction = _finger_axis_correction


def solve_thumb(ann, R, side, pe=1.0,
                axis_rot_offset_inv_first_joint: np.ndarray | None = None):
    """Return [親指０, 親指１, 親指２] local quaternions (the m==0 thumb branch).

    Differs from the fingers: joints 0/1/2 (base_index 0); d=0,1 use setFromUnitVectors
    (not spherical); looser degenerate threshold (pi/1.1); and a thumb-specific clamp
    (z-offsets by ±pi/4,±pi/8; the distal joint zeroes x and maps |z| to a y/z fan).
    the reference's `m == 0` THREEX-mode skip of axis_rot applies to ALL thumb joints (the
    `m` in the JS is the finger index, not the joint index), so we pass
    axis_rot=None for every thumb joint.
    axis_rot_offset_inv_first_joint: when given, applies the reference's premultiplication
        at d==1 with the half-strength slerp (`m==0 ? 0.5 : 1`)."""
    Ve = hand_basis(ann, side, pe)
    P = Ve.copy()
    P[0] *= -1
    P[2] *= -1
    palm = np.asarray(ann["palm"][0], float)
    v = [tm.apply_quat_to_vec((np.asarray(pt, float) - palm) * np.array([1.0, -1.0, 1.0]), P)
         for pt in ann["thumb"]]
    sgn = R  # 左:+1 / 右:-1
    M = None
    rots = []
    for d in range(3):
        pts = [v[e].copy() for e in range(4)]
        w = pts[d].copy()
        for e in range(d, 4):
            pts[e] = pts[e] - w
            if M is not None and e > d:
                pts[e] = tm.apply_quat_to_vec(pts[e], M)
        seg = pts[d + 1] - pts[d]
        nn = seg / np.linalg.norm(seg)
        if d == 2:
            s = tm.euler_from_vector_spherical(_NY, nn)
            a = tm.quat_set_from_euler(s, "XZY")
        else:
            a = tm.quat_set_from_unit_vectors(_NY, nn)
            s = tm.euler_from_matrix(tm.matrix_from_quat(a), "XZY")
        M = tm.quat_conjugate(a)
        if abs(s[2]) > math.pi / 1.1 or s[0] > math.pi / 1.1:
            s = np.array([-abs(_ang_between(_NY, nn)), 0.0, 0.0])
        r = -math.pi / 1.25 if s[0] < 0 else None
        if d == 0:
            s[2] = s[2] * 1.25 + (math.pi / 4) * sgn
        elif d == 1:
            s[2] -= (math.pi / 8) * sgn
        else:  # d == 2 distal: zero the X bend, fan |z| into y/z
            s[0] = 0.0
            A = min(abs(s[2]) / (math.pi / 2), 1.0)
            if s[2] > 0:
                s[1] = -math.pi / 4 * min(A * 2, 1.0)
                s[2] = math.pi / 2 * A
            else:
                s[2] = -math.pi / 2 * A * 1.5
        if r is not None:
            s[0] = max(s[0], r)
        quat_raw = tm.quat_set_from_euler(s, "XZY")
        offset_inv = axis_rot_offset_inv_first_joint if d == 1 else None
        rots.append(_axis_correct(quat_raw, sgn, is_thumb=True,
                                     axis_rot=None,
                                     axis_rot_offset_inv=offset_inv,
                                     apply_offset_inv=(d == 1)))
    return rots


def solve_hands(hands, W, H, *,
                axis_rot_by_mmd: dict | None = None,
                axis_rot_offset_inv_by_mmd: dict | None = None):
    """{'Left'|'Right': (21,3)} -> {MMD bone: local quat} for both characters' fingers.

    axis_rot_by_mmd / axis_rot_offset_inv_by_mmd: dicts {MMD-bone-name → quat}
    from `mocapy.vrm.skeleton.compute_finger_axis_rot(skeleton)`. When given,
    enables the reference's full THREEX-mode finger correction (axis_rot multiplication +
    axis_rot_offset_inv premultiplication at the first joint). Without these
    the solver falls back to the rig-independent z-flip-only path.
    """
    out = {}
    for side, (label, R) in _SIDE.items():
        lms = hands.get(label)
        if lms is None:
            continue
        ann = annotations_from_landmarks(lms, W, H)
        for key, glyph, m, _idx in _FINGERS:
            first_joint = side + glyph + "１"
            offset_inv = (axis_rot_offset_inv_by_mmd.get(first_joint)
                          if axis_rot_offset_inv_by_mmd else None)
            for j, rot in enumerate(solve_finger(
                ann, key, m, R, side,
                axis_rot_by_mmd=axis_rot_by_mmd,
                axis_rot_offset_inv_first_joint=offset_inv,
                mmd_glyph=glyph,
            ), start=1):
                out[side + glyph + _FW[j]] = rot
        # 親指 — uses first-joint offset_inv (親指０) at d==1 per the reference
        thumb_first = side + "親指０"
        thumb_offset_inv = (axis_rot_offset_inv_by_mmd.get(thumb_first)
                            if axis_rot_offset_inv_by_mmd else None)
        for j, rot in enumerate(solve_thumb(
            ann, R, side, axis_rot_offset_inv_first_joint=thumb_offset_inv,
        )):
            out[side + "親指" + _FW[j]] = rot
    return out
