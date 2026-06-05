"""1:1 port of the reference engine's ``js/the reference filter source`` (quaternion-capable variant).

Types: 0 = scalar, 3 = vector (list/np), 4 = quaternion [x,y,z,w].
Pose landmarks use ``OneEuroFilter(freq=30, minCutOff=1, beta=1, dCutOff=2, type=3)``.

Note on determinism: the JS filter recomputes ``freq`` from wall-clock timestamps
between *live* processed frames, so a live capture is not bit-reproducible. For an
offline port we pass video presentation timestamps (deterministic, equivalent method).
"""

from __future__ import annotations

import math

import numpy as np


def _quat_conj(q):
    return [-q[0], -q[1], -q[2], q[3]]


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        ax * bw + aw * bx + ay * bz - az * by,
        ay * bw + aw * by + az * bx - ax * bz,
        az * bw + aw * bz + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_slerp(a, b, t):
    a = list(a)
    b = list(b)
    cos_half = sum(a[i] * b[i] for i in range(4))
    if cos_half < 0:
        b = [-v for v in b]
        cos_half = -cos_half
    if cos_half >= 1.0:
        return a
    sqr_sin = 1.0 - cos_half * cos_half
    if sqr_sin <= np.finfo(float).eps:
        s = 1 - t
        out = [s * a[i] + t * b[i] for i in range(4)]
        n = math.sqrt(sum(v * v for v in out))
        return [v / n for v in out]
    sin_half = math.sqrt(sqr_sin)
    half = math.atan2(sin_half, cos_half)
    ra = math.sin((1 - t) * half) / sin_half
    rb = math.sin(t * half) / sin_half
    return [a[i] * ra + b[i] * rb for i in range(4)]


class LowPassFilter:
    def __init__(self, alpha, type=0):
        self.set_alpha(alpha)
        self.y = None
        self.s = None
        self.type = type

    def set_alpha(self, alpha):
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha

    def filter(self, value, alpha=None):
        if alpha:
            self.set_alpha(alpha)
        if self.y is None:
            s = value
        elif isinstance(value, (list, tuple, np.ndarray)):
            if self.type == 4:
                s = _quat_slerp(self.s, value, self.alpha)
            else:
                s = [self.alpha * v + (1.0 - self.alpha) * self.s[i] for i, v in enumerate(value)]
        else:
            s = self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        self.s = s
        return s

    def last_value(self):
        return self.y


class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, type=0):
        if freq <= 0 or min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("freq, min_cutoff, d_cutoff must be > 0")
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.type = type
        self.x = LowPassFilter(self._alpha(min_cutoff), type)
        self.dx = LowPassFilter(self._alpha(d_cutoff), type)
        self.lasttime = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def _derivative(self, x):
        prev = self.x.last_value()
        if prev is None:
            if self.type == 3:
                return [0, 0, 0]
            if self.type == 4:
                return [0, 0, 0, 1]
            return 0
        if self.type == 3:
            dt = 1 / self.freq
            return [(x[i] - prev[i]) / dt for i in range(len(x))]
        if self.type == 4:
            dt = 1 / self.freq
            rate = 1.0 / dt
            dq = _quat_mul(x, _quat_conj(prev))
            return _quat_slerp([0, 0, 0, 1], dq, rate)
        return (x - prev) * self.freq

    def _derivative_magnitude(self, dx):
        if self.type == 3:
            return math.sqrt(sum(v * v for v in dx))
        if self.type == 4:
            return 2.0 * math.acos(max(-1.0, min(1.0, dx[3])))
        return abs(dx)

    def filter(self, x, timestamp=None, time_scale=1):
        """timestamp in milliseconds (matches the JS API)."""
        if self.lasttime is not None and timestamp is not None:
            self.freq = 1.0 / (max((timestamp - self.lasttime) / 1000.0, 1 / 60) * time_scale)
        self.lasttime = timestamp

        dx = self._derivative(x)
        edx = self._derivative_magnitude(self.dx.filter(dx, self._alpha(self.d_cutoff)))
        cutoff = self.min_cutoff + self.beta * edx
        return self.x.filter(x, self._alpha(cutoff))
