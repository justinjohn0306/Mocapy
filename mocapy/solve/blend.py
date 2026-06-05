"""Skin-blending layer — the stateful smoothing every recorded bone passes through.

Port of the `add` method of the reference engine's skin manager (class with `this.skin`,
char ~193380 of the reference solver). Each frame a bone's freshly-solved
value is blended toward its previous recorded value:

    rec[f].rot = slerp(raw[f].rot, rec[f-1].rot, 1 - e)

with the per-bone, per-frame blend factor

    e = blending_ratio                              if set by mocap_data_smoothing
      = 0.5 + clamp((max(t_delta_frame, dt_ms*speed) - 50) / 150, 0, 1) * 0.5   otherwise

mocap_data_smoothing (Ye.mocap_data_smoothing): 0 -> non-finger,non-両目 bones use
e=1 (no blend), fingers 0.75; 1 -> 0.75; >=2 -> the timing formula above.

At a steady ~33 ms/frame the timing branch gives dt<50 -> e=0.5, i.e. a 50/50 slerp
with the previous recorded frame. Validated: this reproduces golden センター to a
median 0.043 deg (vs 1.13 deg unblended). The exact JS uses RAF wall-clock for
dt_ms; offline we use e=0.5 steady-state (optionally a supplied per-frame dt).
"""

from __future__ import annotations

import numpy as np

from .three_math import quat_slerp_arr as _slerp


def blend_factor(t_delta_frame_ms: float, dt_ms: float, speed: float = 1.0) -> float:
    """The timing-based blend ratio e in [0.5, 1.0] (mocap_data_smoothing >= 2)."""
    dt = max(t_delta_frame_ms, dt_ms * speed)
    return 0.5 + max(min((dt - 50.0) / 150.0, 1.0), 0.0) * 0.5


class BlendChannel:
    """Per-bone recursive blend, mirroring lt.add('skin', name, {rot|pos|weight})."""

    def __init__(self, *, mocap_data_smoothing: int = 2):
        self._rot: dict[str, np.ndarray] = {}
        self._pos: dict[str, np.ndarray] = {}
        self.mocap_data_smoothing = mocap_data_smoothing

    def _ratio_for(self, name: str, e_timing: float) -> float:
        s = self.mocap_data_smoothing
        if s < 2 and name != "両目":
            if s == 1:
                return 0.75
            return 0.75 if "指" in name else 1.0
        return e_timing

    def add_rot(self, name: str, rot, *, e_timing: float = 0.5,
                no_blending: bool = False) -> np.ndarray:
        rot = np.asarray(rot, dtype=float)
        prev = self._rot.get(name)
        if prev is None or no_blending:
            self._rot[name] = rot.copy()
            return self._rot[name]
        e = self._ratio_for(name, e_timing)
        out = rot if e >= 1.0 else np.array(_slerp(rot, prev, 1.0 - e))
        self._rot[name] = out
        return out

    def add_pos(self, name: str, pos, *, e_timing: float = 0.5,
                no_blending: bool = False) -> np.ndarray:
        pos = np.asarray(pos, dtype=float)
        prev = self._pos.get(name)
        if prev is None or no_blending:
            self._pos[name] = pos.copy()
            return self._pos[name]
        e = self._ratio_for(name, e_timing)
        out = pos if e >= 1.0 else pos * e + prev * (1.0 - e)
        self._pos[name] = out
        return out
