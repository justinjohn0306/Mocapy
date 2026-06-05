"""MMD CCD IK — port of the jThree MMD plugin solver (v2.1.2_jThree.MMD.js ~4002-4098).

Reproduces the leg (足ＩＫ) / arm (腕ＩＫ) inverse kinematics: rotate the chain links so
the effector bone reaches a target position, with a per-step angle clamp (ik.control)
and an X-axis hinge constraint on limited links (the knee).

Minimal FK over a bone chain in rest pose (origins) + per-bone local quaternions.
Bone world transform (MMD convention):
  world_rot[b]  = world_rot[parent] * local_q[b]
  world_pos[b]  = world_pos[parent] + world_rot[parent] . (origin[b] - origin[parent])
"""

from __future__ import annotations

import math

import numpy as np

from . import three_math as tm


class Chain:
    """A bone chain in rest pose. names[0] is the root; each bone's parent is the
    previous entry unless `parents` is given. origins are world rest positions."""

    def __init__(self, names, origins, parents=None):
        self.names = list(names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.origin = {n: np.asarray(origins[n], dtype=float) for n in self.names}
        if parents is None:
            self.parent = {names[i]: (names[i - 1] if i > 0 else None) for i in range(len(names))}
        else:
            self.parent = dict(parents)
        self.local_q = {n: np.array([0.0, 0, 0, 1.0]) for n in self.names}

    def set_local(self, name, q):
        self.local_q[name] = np.asarray(q, dtype=float)

    def fk(self):
        """Compute world rotation (quat) and world position for every bone."""
        wr, wp = {}, {}
        for n in self.names:  # assumes parent precedes child
            p = self.parent[n]
            if p is None or p not in wr:
                wr[n] = self.local_q[n].copy()
                wp[n] = self.origin[n].copy()
            else:
                wr[n] = tm.quat_mul(wr[p], self.local_q[n])
                wp[n] = wp[p] + tm.apply_quat_to_vec(self.origin[n] - self.origin[p], wr[p])
        return wr, wp


def solve_ccd(chain: Chain, *, effector: str, target: np.ndarray,
              links, iteration: int, control: float):
    """CCD over `links` (each: {"bone":name, "limit_sign":None|+1|-1}); links[0] is
    nearest the effector. Mutates chain.local_q for the link bones. Mirrors the jThree
    loop incl. the X-axis hinge for limited links."""
    target = np.asarray(target, dtype=float)
    for _ in range(iteration):
        rotated = False
        for lk in links:
            wr, wp = chain.fk()
            link = lk["bone"]
            link_pos = wp[link]
            inv_link = tm.quat_conjugate(wr[link])  # global rotation inverse
            eff = tm.apply_quat_to_vec(wp[effector] - link_pos, inv_link)
            tgt = tm.apply_quat_to_vec(target - link_pos, inv_link)
            ne = np.linalg.norm(eff); nt = np.linalg.norm(tgt)
            if ne < 1e-9 or nt < 1e-9:
                continue
            eff /= ne; tgt /= nt
            dot = max(-1.0, min(1.0, float(np.dot(tgt, eff))))
            angle = math.acos(dot)
            if angle < 1e-5:
                continue
            if angle > control:
                angle = control
            axis = np.cross(eff, tgt)
            na = np.linalg.norm(axis)
            if na < 1e-9:
                continue
            axis /= na
            q = tm.quat_set_from_axis_angle(axis, angle)
            newq = tm.quat_mul(chain.local_q[link], q)
            sign = lk.get("limit_sign")
            if sign is not None:  # X-axis hinge (knee)
                w = max(-1.0, min(1.0, newq[3]))
                newq = np.array([sign * math.sqrt(1 - w * w), 0.0, 0.0, w])
            chain.local_q[link] = newq
            rotated = True
        if not rotated:
            break
    return chain
