"""Arm solve: 腕 (upper arm) and ひじ (elbow). Rotation math VALIDATED.

The rotation formula was confirmed to reproduce the reference engine's captured intermediates
to 0.0000° (tests against arm*.bvh.armcap.json):

    le = conjugate(S · 上半身 · 上半身2)              # parent (torso-chain) frame, k=identity
    g  = (shoulder - elbow).normalize().applyQuaternion(le)
    腕  = setFromUnitVectors((F,0,0), g)               # F = +1 左, -1 右
    # elbow forearm direction e = (elbow - wrist) in (S·腕)-local frame:
    e  = (elbow - wrist).normalize().applyQuaternion(le).applyQuaternion(conjugate(腕))
    ひじ = setFromUnitVectors((F,0,0), e)
  (axis_rot is identity for 腕/ひじ on this rig, so the Ae correction is a no-op.)

SOURCING SOLVED: the character's 左 uses the image-RIGHT MediaPipe landmarks
(raw idx 12/14/16), 右 uses left (11/13/15) — a side-swap (mirror), NO coord flip.
F = +1 (左) / -1 (右). Verified within-run: with the reference's own `le`, this reproduces the reference's
captured arm rotation `b` to 0.000 deg.

REMAINING — the ~6° BVH residual is the IRREDUCIBLE floor (investigated 2026-05-31):
  golden(腕) . b^-1 averages to IDENTITY (0.3 deg) => b is the exact target, no missing
  transform. Pipeline runs blend-only (e=0.5) and that already matches the reference's dominant
  branch: the JS is `if(!p || d.score <= 0) { ...je-filter-or-reset... } else { /* skip je */ }`.
  The armv fixture has d.score=1 on every landmark, so the reference also took the else (skip) branch
  on every frame. There is NO second-stage bonefilter — `process_bones` in the JS is a
  lifecycle event name (`addEventListener("..._process_bones_after_IK", ...)`), not a filter.
  axis_rot is identity for 腕/ひじ on this rig so the Ae correction is also a no-op.
  Param sweep (je min_co ∈ {0.5..8} × β ∈ {0..8}, before/after blend; lookahead +1;
  adaptive e; double-EMA) — nothing beats blend-only's 6.5°/7.1° median against the reference's
  recorded b. The 6° is the gap between offline-blend(b) and online-blend(b) given
  identical input, and closing it would require the magnet/hand-IK system that drives
  the per-frame dynamic je gating.
"""

from __future__ import annotations

import numpy as np

from . import three_math as tm

# Mirror: the character's 左 (left) maps to the image-RIGHT landmarks (MediaPipe
# "right_*"), and vice versa. Confirmed within-run to 0.0 distance vs the reference's captured u.
SH = {"左": 12, "右": 11}
EL = {"左": 14, "右": 13}
WR = {"左": 16, "右": 15}
F_SIGN = {"左": 1.0, "右": -1.0}


def _pt(kp3, i, flip_x):
    k = kp3[i]
    v = np.array([k["x"], k["y"], k["z"]])
    if flip_x:
        v = v * np.array([-1.0, 1.0, 1.0])
    return v


def solve_arm(kp3, S, h_upper, m_upper2, side, *, flip_x=False):
    """Return (腕, ひじ) rotations for one side ('左'/'右'). Rotation core is exact;
    see module docstring for the still-open sourcing details (x-flip, je filter)."""
    le = tm.quat_conjugate(tm.quat_mul(tm.quat_mul(S, h_upper), m_upper2))
    F = F_SIGN[side]
    ref = np.array([F, 0.0, 0.0])

    sh = _pt(kp3, SH[side], flip_x)
    el = _pt(kp3, EL[side], flip_x)
    wr = _pt(kp3, WR[side], flip_x)

    g = sh - el
    g = tm.apply_quat_to_vec(g / np.linalg.norm(g), le)
    ude = tm.quat_set_from_unit_vectors(ref, g)

    e = el - wr
    e = tm.apply_quat_to_vec(e / np.linalg.norm(e), le)
    e = tm.apply_quat_to_vec(e, tm.quat_conjugate(ude))
    hiji = tm.quat_set_from_unit_vectors(ref, e)

    return ude, hiji
