"""BVH exporter — a 1:1 port of the reference engine's ``js/the reference BVH writer``.

Two halves:
  * ``build_hierarchy`` — the HIERARCHY block, derived purely from the VRM rest
    pose (see ``mocapy.vrm.skeleton``). Validatable today against a reference .bvh.
  * ``write_bvh`` — the MOTION block, which consumes recorded ``boneKeys`` (the
    output of the recorder/solver stages, ported later). Implemented here per the
    spec so the writer is complete, but full numeric validation waits on golden
    boneKeys data.

Numbers are formatted with :func:`js_num`, which reproduces V8's
``Number.prototype.toString`` so output can be byte-compared with the JS writer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from mocapy.vrm.skeleton import Skeleton

VRM_SCALE = 10.0
TAB = "  "

# Region-based axis locking applied to each joint OFFSET (the reference BVH writer:38-48).
_RE_ZERO_Z = re.compile(r"spine|upperleg|shoulder", re.I)
_RE_ARM = re.compile(r"arm|hand|zzzIntermediate|zzzDistal", re.I)
_RE_ZERO_XZ = re.compile(r"leg|chest|neck|head", re.I)
_RE_FOOT = re.compile(r"foot|toes", re.I)


# ---------------------------------------------------------------------------
# V8-compatible number formatting (ECMAScript Number::toString, radix 10)
# ---------------------------------------------------------------------------

def js_num(x: float) -> str:
    """Format a float the way JavaScript's ``String(Number)`` does."""
    x = float(x)  # coerce numpy scalars (numpy 2.x repr is 'np.float64(...)')
    if x == 0:
        # JS prints both +0 and -0 as "0".
        return "0"
    if x != x:
        return "NaN"
    if x == float("inf"):
        return "Infinity"
    if x == float("-inf"):
        return "-Infinity"

    neg = x < 0
    # Python's repr gives the shortest round-tripping decimal, same family as V8.
    r = repr(abs(x))

    if "e" in r or "E" in r:
        mant, exp = re.split(r"[eE]", r)
        exp = int(exp)
    else:
        mant, exp = r, 0

    if "." in mant:
        int_part, frac_part = mant.split(".")
    else:
        int_part, frac_part = mant, ""

    digits = (int_part + frac_part).lstrip("0") or "0"
    # n = position of decimal point relative to the start of `digits`.
    leading_zeros = len(int_part + frac_part) - len((int_part + frac_part).lstrip("0"))
    n = len(int_part) + exp - leading_zeros
    digits = digits.rstrip("0") or "0"
    k = len(digits)

    if digits == "0":
        s = "0"
    elif k <= n <= 21:
        s = digits + "0" * (n - k)
    elif 0 < n <= 21:
        s = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        s = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        body = digits[0] if k == 1 else digits[0] + "." + digits[1:]
        s = f"{body}e{'+' if e >= 0 else '-'}{abs(e)}"

    return "-" + s if neg else s


def _join(vec) -> str:
    return " ".join(js_num(float(v)) for v in vec)


# ---------------------------------------------------------------------------
# HIERARCHY
# ---------------------------------------------------------------------------

def _bvh_name(vrm_name: str) -> str:
    # the reference BVH writer:31 — thumb renaming for BVH conventions.
    return vrm_name.replace("ThumbProximal", "ThumbIntermediate").replace(
        "ThumbMetacarpal", "ThumbProximal"
    )


def _offset(pos0_child: np.ndarray, pos0_parent: np.ndarray | None, name: str) -> np.ndarray:
    base = pos0_parent if pos0_parent is not None else np.zeros(3)
    pos = (pos0_child - base) * VRM_SCALE
    if _RE_ZERO_Z.search(name):
        pos[2] = 0.0
    elif _RE_ARM.search(name):
        length = float(np.linalg.norm(pos))
        pos[0] = np.sign(pos[0]) * length
        pos[1] = pos[2] = 0.0
    elif _RE_ZERO_XZ.search(name):
        length = float(np.linalg.norm(pos))
        pos[1] = np.sign(pos[1]) * length
        pos[0] = pos[2] = 0.0
    return pos


@dataclass
class HierarchyNode:
    name: str            # BVH (display) name
    vrm_name: str        # original VRM humanoid name
    offset: np.ndarray
    children: list["HierarchyNode"]


def build_hierarchy(skel: Skeleton):
    """Return (root HierarchyNode, ordered list of (bvh_name, vrm_name))."""
    order: list[tuple[str, str]] = []

    def build(vrm_name: str, parent_vrm: str | None) -> HierarchyNode:
        bone = skel.bones[vrm_name]
        pos0_parent = skel.bones[parent_vrm].pos0 if parent_vrm is not None else None
        offset = _offset(bone.pos0, pos0_parent, _bvh_name(vrm_name))
        node = HierarchyNode(_bvh_name(vrm_name), vrm_name, offset, [])
        order.append((node.name, vrm_name))
        for c in bone.children:
            node.children.append(build(c, vrm_name))
        return node

    root = build(skel.root, None)
    return root, order


def _end_site_offset(node: HierarchyNode, skel: Skeleton) -> np.ndarray:
    pos = node.offset.copy()
    if _RE_FOOT.search(node.name):
        # the reference BVH writer:108-111 — drop the end site to ground level.
        pos[1] = -skel.bones[node.vrm_name].pos0[1] * VRM_SCALE
    return pos


def _emit_hierarchy(node: HierarchyNode, skel: Skeleton, depth: int, is_root: bool) -> list[str]:
    pad = TAB * depth
    lines: list[str] = []
    if is_root:
        lines.append("HIERARCHY")
        lines.append("ROOT hips")
        channels = "6 Xposition Yposition Zposition Yrotation Xrotation Zrotation"
    else:
        lines.append(pad + "JOINT " + node.name)
        channels = "3 Yrotation Xrotation Zrotation"

    lines.append(pad + "{")
    lines.append(pad + TAB + "OFFSET " + _join(node.offset))
    lines.append(pad + TAB + "CHANNELS " + channels)

    if node.children:
        for c in node.children:
            lines += _emit_hierarchy(c, skel, depth + 1, False)
    else:
        lines.append(pad + TAB + "End Site")
        lines.append(pad + TAB + "{")
        lines.append(pad + TAB + TAB + "OFFSET " + _join(_end_site_offset(node, skel)))
        lines.append(pad + TAB + "}")

    lines.append(pad + "}")
    return lines


def hierarchy_text(skel: Skeleton) -> str:
    root, _ = build_hierarchy(skel)
    return "\n".join(_emit_hierarchy(root, skel, 0, True))


# ---------------------------------------------------------------------------
# MOTION
# ---------------------------------------------------------------------------
# Quaternion helpers transcribed from THREE.Quaternion so rounding matches the
# JS writer. Quaternions are [x, y, z, w].

def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        ax * bw + aw * bx + ay * bz - az * by,
        ay * bw + aw * by + az * bx - ax * bz,
        az * bw + aw * bz + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_slerp(a, b, t):
    """THREE.Quaternion.slerpQuaternions semantics."""
    if t == 0:
        return tuple(a)
    if t == 1:
        return tuple(b)
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    cos_half = aw * bw + ax * bx + ay * by + az * bz
    if cos_half < 0:
        bx, by, bz, bw = -bx, -by, -bz, -bw
        cos_half = -cos_half
    if cos_half >= 1.0:
        return (ax, ay, az, aw)
    sqr_sin = 1.0 - cos_half * cos_half
    if sqr_sin <= np.finfo(float).eps:
        s = 1 - t
        x, y, z, w = s * ax + t * bx, s * ay + t * by, s * az + t * bz, s * aw + t * bw
        n = (x * x + y * y + z * z + w * w) ** 0.5
        return (x / n, y / n, z / n, w / n)
    sin_half = sqr_sin ** 0.5
    half = np.arctan2(sin_half, cos_half)
    ratio_a = np.sin((1 - t) * half) / sin_half
    ratio_b = np.sin(t * half) / sin_half
    return (
        ax * ratio_a + bx * ratio_b,
        ay * ratio_a + by * ratio_b,
        az * ratio_a + bz * ratio_b,
        aw * ratio_a + bw * ratio_b,
    )


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def quat_to_euler_yxz(q):
    """Replicates THREE.Euler.setFromQuaternion(q, 'YXZ'); returns (ex, ey, ez) rad."""
    x, y, z, w = q
    # Rotation matrix elements (THREE.Matrix4.makeRotationFromQuaternion, column-major).
    m11 = 1 - 2 * (y * y + z * z)
    m21 = 2 * (x * y + w * z)
    m31 = 2 * (x * z - w * y)
    m22 = 1 - 2 * (x * x + z * z)
    m13 = 2 * (x * z + w * y)
    m23 = 2 * (y * z - w * x)
    m33 = 1 - 2 * (x * x + y * y)

    ex = np.arcsin(-_clamp(m23, -1.0, 1.0))
    # NOTE: the bundled three.js (jThree) uses 0.99999, not the modern 0.9999999.
    if abs(m23) < 0.99999:
        ey = np.arctan2(m13, m33)
        ez = np.arctan2(m21, m22)
    else:
        ey = np.arctan2(-m31, m11)
        ez = 0.0
    return ex, ey, ez


def _expand_to_30fps(keys):
    """Port of the reference BVH writer:142-172 — expand one bone's keys to dense 30fps.

    The outer gap-fill condition compares the key's frame number to the *count of
    original keys processed so far* (``_f > f`` with ``f`` a per-key counter == idx),
    not to the previous key's frame. Keys are NOT sorted (insertion order preserved),
    exactly as the JS does.
    """
    full = []
    for idx, k in enumerate(keys):
        f = round(k["time"] * 30)
        if f > idx and idx > 0:
            k_last = keys[idx - 1]
            f_last = round(k_last["time"] * 30)
            f_diff = f - f_last
            for i in range(1, f_diff):
                t = i / f_diff
                pos = [k_last["pos"][j] + (k["pos"][j] - k_last["pos"][j]) * t for j in range(3)]
                rot = quat_slerp(k_last["rot"], k["rot"], t)
                full.append({"time": (f_last + i) / 30, "pos": pos, "rot": list(rot)})
        full.append(k)
    return full


def write_bvh(skel: Skeleton, bone_keys, vrm_to_mmd, *, leg_scale=1.0, pos_scale=1.0,
              prepend_rest_frame: bool = True):
    """Build full BVH text from recorded boneKeys.

    bone_keys: flat list of {name(MMD), time, pos[3], rot[xyzw]}.
    vrm_to_mmd: dict mapping VRM humanoid bone name -> MMD bone name.

    NOTE: depends on the VRM->MMD map and recorded boneKeys produced by the
    recorder/solver stages (later phases). The math here mirrors the reference BVH writer;
    end-to-end numeric validation is pending golden boneKeys data.
    """
    root, order = build_hierarchy(skel)
    header = "\n".join(_emit_hierarchy(root, skel, 0, True))

    # the reference's CLI BVH output has a REST FRAME at index 0 (hips at exact rest, all
    # rotations zero), then mocap data from frame 1 onward (see
    # `the reference output` — frame 0 has 18 trailing zeros, frames 1+ have real
    # rotations). Reproduce this by:
    #   1. time-shifting every incoming boneKey by +1/30s
    #   2. prepending one identity-rotation rest key at t=0 per bone
    #   3. appending one synthetic trailing key (copy of the last real key) per
    #      bone at t = time_max + 2/30 — this is what the reference's recorder emits when
    #      the capture stops, and it bumps f_max by one so the LAST real mocap
    #      frame survives the writer's `dense[name][f_max-1]` truncation.
    # Net effect for an N-frame detection: BVH has N+1 frames (1 rest + N mocap).
    if prepend_rest_frame and bone_keys:
        bones_seen: list[str] = []
        bones_set: set[str] = set()
        last_per_bone: dict[str, dict] = {}
        orig_time_max = 0.0
        shifted_keys = []
        for k in bone_keys:
            if k["name"] not in bones_set:
                bones_set.add(k["name"])
                bones_seen.append(k["name"])
            orig_time_max = max(orig_time_max, k["time"])
            shifted = {**k, "time": k["time"] + 1.0 / 30.0}
            shifted_keys.append(shifted)
            last_per_bone[k["name"]] = shifted
        rest_keys = [{"name": n, "time": 0.0, "pos": [0.0, 0.0, 0.0],
                      "rot": [0.0, 0.0, 0.0, 1.0]} for n in bones_seen]
        synth_t = orig_time_max + 2.0 / 30.0
        trailing_keys = [{**last_per_bone[n], "time": synth_t} for n in bones_seen]
        bone_keys = rest_keys + shifted_keys + trailing_keys

    by_name: dict[str, list] = {}
    time_max = 0.0
    for k in bone_keys:
        by_name.setdefault(k["name"], []).append(k)
        time_max = max(time_max, k["time"])
    f_max = round(time_max * 30)

    dense = {}
    for name, keys in by_name.items():
        # NOTE: insertion order preserved (the JS writer does not sort).
        full = _expand_to_30fps(keys)
        if len(full) < f_max:
            last = full[-1]
            for i in range(len(full), f_max):
                full.append({"time": i / 30, "pos": last["pos"], "rot": last["rot"]})
        dense[name] = full

    hips_pos0 = skel.bones["hips"].pos0

    rows = []
    for f in range(f_max):
        data = []
        for bvh_name, vrm_name in order:
            name_mmd = vrm_to_mmd.get(vrm_name)
            if name_mmd and name_mmd not in dense and "足首" in (name_mmd or ""):
                name_mmd = name_mmd[0] + "足ＩＫ"
            if not name_mmd or name_mmd not in dense:
                data.append("0 0 0")
                continue

            k = dense[name_mmd][f]
            q = tuple(k["rot"])
            cell = ""
            if bvh_name == "hips":
                pos = np.array(k["pos"], dtype=float)
                if "全ての親" in dense:
                    pos = pos + np.array(dense["全ての親"][f]["pos"], dtype=float)
                pos = pos * pos_scale * leg_scale + hips_pos0 * VRM_SCALE
                cell += _join(pos) + " "
                if "下半身" in dense:
                    q = quat_mul(q, tuple(dense["下半身"][f]["rot"]))
            elif bvh_name == "spine":
                if "下半身" in dense:
                    q = quat_mul(quat_conjugate(tuple(dense["下半身"][f]["rot"])), q)
            elif re.search(r"(left|right)LowerArm", bvh_name):
                d = "左" if bvh_name.startswith("left") else "右"
                twist = d + "手捩"
                if twist in dense:
                    q = quat_mul(q, tuple(dense[twist][f]["rot"]))

            ex, ey, ez = quat_to_euler_yxz(q)
            deg = 180.0 / np.pi
            cell += " ".join(js_num(v * deg) for v in (ey, ex, ez))
            data.append(cell)
        rows.append(" ".join(data))

    motion = "\n".join([
        "MOTION",
        f"Frames: {f_max}",
        "Frame Time: " + js_num(1 / 30),
        "\n".join(rows),
    ])
    return header + "\n" + motion
