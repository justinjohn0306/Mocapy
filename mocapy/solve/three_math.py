"""Exact ports of the jThree (old THREE.js) math primitives the solver uses.

Conventions:
  * quaternions are np arrays [x, y, z, w]
  * eulers are np arrays [x, y, z] paired with an order string
  * rotation matrices are 3x3 np arrays with M[row][col] == THREE's m(row+1)(col+1)

Sources (bundled jThree): three.core.min.js — setFromEuler, setEulerFromRotationMatrix,
makeRotationFromQuaternion, toSphericalCoords, setFromVectorSpherical, setFromBasis.
The gimbal threshold in setEulerFromRotationMatrix is 0.99999 (jThree), not 0.9999999.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------
# Quaternion basics (THREE conventions)
# --------------------------------------------------------------------------

def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        ax * bw + aw * bx + ay * bz - az * by,
        ay * bw + aw * by + az * bx - ax * bz,
        az * bw + aw * bz + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_premultiply(a, q):
    """THREE: a.premultiply(q) == q * a."""
    return quat_mul(q, a)


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_slerp_arr(a, b, t):
    """THREE.Quaternion.slerp(a -> b, t). Returns np[xyzw]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if t == 0:
        return a.copy()
    if t == 1:
        return b.copy()
    cos_half = float(np.dot(a, b))
    if cos_half < 0:
        b = -b
        cos_half = -cos_half
    if cos_half >= 1.0:
        return a.copy()
    sqr_sin = 1.0 - cos_half * cos_half
    if sqr_sin <= np.finfo(float).eps:
        out = (1 - t) * a + t * b
        return out / np.linalg.norm(out)
    sin_half = math.sqrt(sqr_sin)
    half = math.atan2(sin_half, cos_half)
    ra = math.sin((1 - t) * half) / sin_half
    rb = math.sin(t * half) / sin_half
    return a * ra + b * rb


def quat_set_from_axis_angle(axis, angle):
    h = angle / 2
    s = math.sin(h)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h)])


def quat_set_from_unit_vectors(e, t):
    """THREE.Quaternion.setFromUnitVectors(from=e, to=t). e, t are unit vectors."""
    e = np.asarray(e, dtype=float)
    t = np.asarray(t, dtype=float)
    i = float(np.dot(e, t)) + 1.0
    if i < np.finfo(float).eps:
        i = 0.0
        if abs(e[0]) > abs(e[2]):
            q = np.array([-e[1], e[0], 0.0, i])
        else:
            q = np.array([0.0, -e[2], e[1], i])
    else:
        q = np.array([
            e[1] * t[2] - e[2] * t[1],
            e[2] * t[0] - e[0] * t[2],
            e[0] * t[1] - e[1] * t[0],
            i,
        ])
    return q / np.linalg.norm(q)


def quat_set_from_euler(e, order="XYZ"):
    """THREE.Quaternion.setFromEuler — all six orders."""
    i = math.cos(e[0] / 2); r = math.cos(e[1] / 2); n = math.cos(e[2] / 2)
    a = math.sin(e[0] / 2); o = math.sin(e[1] / 2); s = math.sin(e[2] / 2)
    if order == "XYZ":
        return np.array([a*r*n + i*o*s, i*o*n - a*r*s, i*r*s + a*o*n, i*r*n - a*o*s])
    if order == "YXZ":
        return np.array([a*r*n + i*o*s, i*o*n - a*r*s, i*r*s - a*o*n, i*r*n + a*o*s])
    if order == "ZXY":
        return np.array([a*r*n - i*o*s, i*o*n + a*r*s, i*r*s + a*o*n, i*r*n - a*o*s])
    if order == "ZYX":
        return np.array([a*r*n - i*o*s, i*o*n + a*r*s, i*r*s - a*o*n, i*r*n + a*o*s])
    if order == "YZX":
        return np.array([a*r*n + i*o*s, i*o*n + a*r*s, i*r*s - a*o*n, i*r*n - a*o*s])
    if order == "XZY":
        return np.array([a*r*n - i*o*s, i*o*n - a*r*s, i*r*s + a*o*n, i*r*n + a*o*s])
    raise ValueError(f"bad order {order}")


def matrix_from_quat(q):
    """THREE.Matrix4.makeRotationFromQuaternion -> 3x3 with M[row][col]."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _clamp(v, lo=-1.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def euler_from_matrix(M, order="XYZ"):
    """THREE.Vector3.setEulerFromRotationMatrix — jThree threshold 0.99999."""
    m11, m12, m13 = M[0]
    m21, m22, m23 = M[1]
    m31, m32, m33 = M[2]
    T = 0.99999
    if order == "XYZ":
        y = math.asin(_clamp(m13))
        if abs(m13) < T:
            x = math.atan2(-m23, m33); z = math.atan2(-m12, m11)
        else:
            x = math.atan2(m32, m22); z = 0.0
    elif order == "YXZ":
        x = math.asin(-_clamp(m23))
        if abs(m23) < T:
            y = math.atan2(m13, m33); z = math.atan2(m21, m22)
        else:
            y = math.atan2(-m31, m11); z = 0.0
    elif order == "ZXY":
        x = math.asin(_clamp(m32))
        if abs(m32) < T:
            y = math.atan2(-m31, m33); z = math.atan2(-m12, m22)
        else:
            y = 0.0; z = math.atan2(m21, m11)
    elif order == "ZYX":
        y = math.asin(-_clamp(m31))
        if abs(m31) < T:
            x = math.atan2(m32, m33); z = math.atan2(m21, m11)
        else:
            x = 0.0; z = math.atan2(-m12, m22)
    elif order == "YZX":
        z = math.asin(_clamp(m21))
        if abs(m21) < T:
            x = math.atan2(-m23, m22); y = math.atan2(-m31, m11)
        else:
            x = 0.0; y = math.atan2(m13, m33)
    elif order == "XZY":
        z = math.asin(-_clamp(m12))
        if abs(m12) < T:
            x = math.atan2(m32, m22); y = math.atan2(m13, m11)
        else:
            x = math.atan2(-m23, m33); y = 0.0
    else:
        raise ValueError(f"bad order {order}")
    return np.array([x, y, z])


def euler_from_quat(q, order="XYZ"):
    return euler_from_matrix(matrix_from_quat(q), order)


def quat_set_from_basis(rows):
    """THREE.Quaternion.setFromBasis(matrix). `rows` = 3x3 with M[row][col]
    (the solver builds it as nt.set(a..,n..,s..) i.e. rows = [a, n, s])."""
    e = euler_from_matrix(np.asarray(rows, dtype=float), "XYZ")
    return quat_set_from_euler(e, "XYZ")


# --------------------------------------------------------------------------
# Spherical helpers
# --------------------------------------------------------------------------

def to_spherical_coords(v, radius=None):
    """THREE.Vector3.toSphericalCoords -> [r, theta, phi]."""
    t, i, r = v[0], v[1], v[2]
    o = radius if radius is not None else math.sqrt(t*t + i*i + r*r)
    if o == 0:
        return [0.0, 0.0, 0.0]
    return [o, math.atan2(t, r), math.acos(_clamp(i / o))]


def euler_from_vector_spherical(e_ref, t):
    """THREE.Vector3.setFromVectorSpherical(e_ref, t) -> euler vec3.

    e_ref selects the branch: e_ref.y (axis ~Y) or e_ref.x (axis ~X)."""
    if e_ref[1]:
        r = np.array([-t[2], t[0], -t[1]], dtype=float)
        if e_ref[1] > 0:
            r = -r
        i = to_spherical_coords(r, 1)
        return np.array([i[1], 0.0, math.pi / 2 - i[2]])
    if e_ref[0]:
        r = np.array([-t[2], t[1], t[0]], dtype=float)
        if e_ref[0] < 0:
            r = -r
        i = to_spherical_coords(r, 1)
        return np.array([0.0, i[1], math.pi / 2 - i[2]])
    return np.zeros(3)


def quat_set_from_vector_spherical(e_ref, t):
    """THREE.Quaternion.setFromVectorSpherical(e_ref, t)."""
    order = "XZY" if e_ref[1] else "YZX"
    return quat_set_from_euler(euler_from_vector_spherical(e_ref, t), order)


def apply_quat_to_vec(v, q):
    """THREE.Vector3.applyQuaternion."""
    x, y, z = v
    qx, qy, qz, qw = q
    ix = qw * x + qy * z - qz * y
    iy = qw * y + qz * x - qx * z
    iz = qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    return np.array([
        ix * qw + iw * -qx + iy * -qz - iz * -qy,
        iy * qw + iw * -qy + iz * -qx - ix * -qz,
        iz * qw + iw * -qz + ix * -qy - iy * -qx,
    ])
