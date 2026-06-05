"""Extract a humanoid rest-pose skeleton from a VRM (glTF 2.0 binary) file.

This is the data the reference engine's the reference BVH writer reads as ``modelX.para.pos0``
(per-bone world rest position) and the VRM humanoid bone hierarchy. We parse the
glTF node tree directly instead of going through THREE.js, so the "bone names" we
work with are the VRM humanoid bone names (``hips``, ``leftUpperLeg``, ...).

Supports VRM 0.x (``extensions.VRM``) and VRM 1.0 (``extensions.VRMC_vrm``).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Bone:
    name: str                      # VRM humanoid bone name
    node: int                      # glTF node index
    parent: str | None             # nearest humanoid-ancestor bone name
    children: list[str] = field(default_factory=list)  # humanoid children, glTF order
    pos0: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world rest pos (m)


@dataclass
class Skeleton:
    root: str                      # always the humanoid root bone, "hips"
    bones: dict[str, Bone]
    vrm_version: int               # 0 or 1

    def ordered(self) -> list[str]:
        """Bone names in BVH traversal order (depth-first, glTF child order)."""
        out: list[str] = []

        def walk(name: str) -> None:
            out.append(name)
            for c in self.bones[name].children:
                walk(c)

        walk(self.root)
        return out


# ---------------------------------------------------------------------------
# glTF parsing
# ---------------------------------------------------------------------------

def _read_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    magic, version, total = struct.unpack("<4sII", data[:12])
    if magic != b"glTF":
        # Maybe a plain .gltf JSON file.
        return json.loads(data.decode("utf-8"))
    jlen, jtype = struct.unpack("<I4s", data[12:20])
    if jtype != b"JSON":
        raise ValueError(f"Unexpected first glb chunk type: {jtype!r}")
    return json.loads(data[20:20 + jlen].decode("utf-8"))


def _local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        # glTF stores column-major; reshape accordingly then transpose to row-major.
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T

    t = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    r = np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)  # x,y,z,w
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)

    x, y, z, w = r
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)

    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot * s  # scale columns
    m[:3, 3] = t
    return m


def _humanoid_bone_to_node(gltf: dict) -> tuple[dict[str, int], int]:
    ext = gltf.get("extensions", {})
    if "VRM" in ext:  # VRM 0.x
        hb = ext["VRM"]["humanoid"]["humanBones"]
        return {b["bone"]: b["node"] for b in hb}, 0
    if "VRMC_vrm" in ext:  # VRM 1.0
        hb = ext["VRMC_vrm"]["humanoid"]["humanBones"]
        return {name: spec["node"] for name, spec in hb.items()}, 1
    raise ValueError("No VRM humanoid extension found (expected 'VRM' or 'VRMC_vrm').")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_skeleton(vrm_path: str | Path) -> Skeleton:
    path = Path(vrm_path)
    gltf = _read_glb_json(path)
    nodes = gltf["nodes"]
    bone_to_node, vrm_version = _humanoid_bone_to_node(gltf)
    node_to_bone = {n: b for b, n in bone_to_node.items()}

    # Parent map for the full node tree.
    parent_of = [-1] * len(nodes)
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent_of[c] = i
    roots = [i for i in range(len(nodes)) if parent_of[i] == -1]

    # World matrices (rest pose) via DFS from scene roots.
    world = [None] * len(nodes)

    def compute(i: int, parent_world: np.ndarray) -> None:
        world[i] = parent_world @ _local_matrix(nodes[i])
        for c in nodes[i].get("children", []):
            compute(c, world[i])

    for r in roots:
        compute(r, np.eye(4, dtype=np.float64))

    if "hips" not in bone_to_node:
        raise ValueError("VRM humanoid has no 'hips' bone.")

    bones: dict[str, Bone] = {}

    # Build the humanoid tree: DFS from the hips node, carrying the nearest
    # humanoid-bone ancestor. Children are recorded in glTF child order, which is
    # exactly the order the reference engine's bvh_parse walks them.
    def walk(node_i: int, humanoid_parent: str | None) -> None:
        bone_name = node_to_bone.get(node_i)
        if bone_name is not None:
            pos = world[node_i][:3, 3].copy()
            bones[bone_name] = Bone(
                name=bone_name, node=node_i, parent=humanoid_parent, pos0=pos
            )
            if humanoid_parent is not None:
                bones[humanoid_parent].children.append(bone_name)
            next_parent = bone_name
        else:
            next_parent = humanoid_parent
        for c in nodes[node_i].get("children", []):
            walk(c, next_parent)

    walk(bone_to_node["hips"], None)

    # VRM 0.x models are authored facing +Z; three-vrm (which the reference engine uses to
    # produce `pos0`) rotates them 180 deg about Y to face the viewer, negating world
    # X and Z. Reproduce that so positions match the reference engine's frame.
    if vrm_version == 0:
        for b in bones.values():
            b.pos0[0] = -b.pos0[0]
            b.pos0[2] = -b.pos0[2]

    return Skeleton(root="hips", bones=bones, vrm_version=vrm_version)


# ---------------------------------------------------------------------------
# Default skeleton — shipped inside the package so users can produce BVH output
# without supplying a VRM. Extracted from cj.vrm (the bundled reference avatar)
# and frozen as a tiny JSON file in mocapy/_data/.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-bone axis_rot computation — port of the reference MMD notes.get_bone_axis_rotation() for
# the arm/finger branch (the only one we need). Used by the finger solver to
# apply the reference's full THREEX-mode correction:
#   axis.applyQuaternion(Ne[bone].axis_rot)
#   a.premultiply(slerp(identity, Ne[first_joint].axis_rot_offset_inv, …))
# where axis_rot is the bone's natural-frame quaternion and axis_rot_offset_inv
# = parent.axis_rot · conj(self.axis_rot).
# ---------------------------------------------------------------------------

# VRM bone name → MMD name, for the bones we need axis_rot of (arm chain + fingers).
_VRM_TO_MMD_FOR_AXIS: dict[str, str] = {}
for _lr, _j in (("left", "左"), ("right", "右")):
    _VRM_TO_MMD_FOR_AXIS[_lr + "Shoulder"] = _j + "肩"
    _VRM_TO_MMD_FOR_AXIS[_lr + "UpperArm"] = _j + "腕"
    _VRM_TO_MMD_FOR_AXIS[_lr + "LowerArm"] = _j + "ひじ"
    _VRM_TO_MMD_FOR_AXIS[_lr + "Hand"] = _j + "手首"
    for _f, (_mf, _sfx) in (("Thumb", ("親指", "０１２")),
                            ("Index", ("人指", "１２３")),
                            ("Middle", ("中指", "１２３")),
                            ("Ring", ("薬指", "１２３")),
                            ("Little", ("小指", "１２３"))):
        for _seg, _s in zip(("Proximal", "Intermediate", "Distal"), _sfx):
            _VRM_TO_MMD_FOR_AXIS[_lr + _f + _seg] = _j + _mf + _s


def _axis_rot_arm_branch(direction: np.ndarray, side_sign: int) -> np.ndarray:
    """get_bone_axis_rotation's RE_arm branch — for shoulder/arm/elbow/wrist/finger bones.

    direction: unit-vector pointing from THIS bone toward its next bone in the chain.
    side_sign: +1 for 左 (left), -1 for 右 (right).
    """
    from mocapy.solve.three_math import (
        quat_set_from_unit_vectors, apply_quat_to_vec, quat_set_from_basis,
    )
    x_axis = direction.copy()
    if abs(np.linalg.norm(x_axis)) < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    if side_sign == -1:
        x_axis = x_axis.copy()
        x_axis[0] *= -1
    q = quat_set_from_unit_vectors(np.array([1.0, 0.0, 0.0]), x_axis)
    z_axis = apply_quat_to_vec(np.array([0.0, 0.0, 1.0]), q)
    y_axis = -np.cross(x_axis, z_axis)
    ny = float(np.linalg.norm(y_axis))
    if ny > 1e-9:
        y_axis = y_axis / ny
    rot = quat_set_from_basis([x_axis, y_axis, z_axis])
    if side_sign == 1:
        rot = rot.copy()
        rot[1] *= -1; rot[2] *= -1
    return rot


def compute_finger_axis_rot(skel: "Skeleton") -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute axis_rot + axis_rot_offset_inv for the arm/finger bones of `skel`.

    Returns (axis_rot_by_mmd, axis_rot_offset_inv_by_mmd). Each is a dict
    `{MMD-bone-name → quaternion}`. Bones the rig doesn't expose are skipped.
    """
    from mocapy.solve.three_math import quat_mul, quat_conjugate

    # 1. Resolve each MMD-named bone to its VRM bone (and the VRM bone that's
    #    its "next in chain" — child for proximal/intermediate, end-extension
    #    fallback for distal).
    # Arm chain children: Shoulder → UpperArm → LowerArm → Hand.
    chain_child: dict[str, str] = {}
    for lr in ("left", "right"):
        chain_child[lr + "Shoulder"] = lr + "UpperArm"
        chain_child[lr + "UpperArm"] = lr + "LowerArm"
        chain_child[lr + "LowerArm"] = lr + "Hand"
        chain_child[lr + "Hand"] = None  # wrist — no in-chain child
        for f in ("Thumb", "Index", "Middle", "Ring", "Little"):
            chain_child[lr + f + "Proximal"] = lr + f + "Intermediate"
            chain_child[lr + f + "Intermediate"] = lr + f + "Distal"
            chain_child[lr + f + "Distal"] = None  # fingertip — no in-chain child

    axis_rot: dict[str, np.ndarray] = {}
    for vrm_name, mmd_name in _VRM_TO_MMD_FOR_AXIS.items():
        if vrm_name not in skel.bones:
            continue
        side_sign = 1 if mmd_name.startswith("左") else -1
        this_pos = skel.bones[vrm_name].pos0
        child_vrm = chain_child.get(vrm_name)
        if child_vrm and child_vrm in skel.bones:
            direction = skel.bones[child_vrm].pos0 - this_pos
        else:
            # Fallback: direction = this − parent (extension beyond the chain end).
            parent_vrm = skel.bones[vrm_name].parent
            if parent_vrm and parent_vrm in skel.bones:
                direction = this_pos - skel.bones[parent_vrm].pos0
            else:
                continue
        axis_rot[mmd_name] = _axis_rot_arm_branch(direction, side_sign)

    # 2. axis_rot_offset_inv = parent.axis_rot · conj(self.axis_rot). For finger
    #    proximals (first joint of each finger), the "parent" in the reference's chain is
    #    the ELBOW (ひじ), not the wrist. For non-first joints, parent is the
    #    previous joint in the same finger.
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    parent_in_chain: dict[str, str] = {}
    for side in ("左", "右"):
        for f_mmd in ("親指", "人指", "中指", "薬指", "小指"):
            suffixes = ("０", "１", "２") if f_mmd == "親指" else ("１", "２", "３")
            for i, s in enumerate(suffixes):
                bone = side + f_mmd + s
                if i == 0:
                    parent_in_chain[bone] = side + "ひじ"
                else:
                    parent_in_chain[bone] = side + f_mmd + suffixes[i - 1]
        # arms themselves: 腕's parent = identity, ひじ's parent = 腕
        parent_in_chain[side + "腕"] = None  # treat as identity
        parent_in_chain[side + "ひじ"] = side + "腕"

    offset_inv: dict[str, np.ndarray] = {}
    for mmd, q_self in axis_rot.items():
        parent_mmd = parent_in_chain.get(mmd)
        if parent_mmd is None or parent_mmd not in axis_rot:
            parent_axis = ident
        else:
            parent_axis = axis_rot[parent_mmd]
        offset_inv[mmd] = quat_mul(parent_axis, quat_conjugate(q_self))
    return axis_rot, offset_inv


def model_spine_length(skel: "Skeleton", vrm_scale: float = 10.0) -> float:
    """Compute the reference's `model.para.spine_length` for a Skeleton.

    Definition (MMD.js/the reference MMD notes:9143):
        `para.spine_length = (para.pos0['neck'][1] - para.pos0['leftUpperLeg'][1]) · vrm_scale`

    the reference's depth pipeline uses this as `spine_length_ref` — the multiplier on the
    3D anatomy length in `data3D.length = spine_length_ref · 2 · |3D-spine|`.
    For cj.vrm this evaluates to ≈ 5.42, vs the 2.5 placeholder we previously
    hardcoded. Using the right value brings our depth magnitudes in line with
    the reference's (mine was ~half the magnitude → XY scaling was also ~half).
    """
    if "neck" in skel.bones and "leftUpperLeg" in skel.bones:
        return float(skel.bones["neck"].pos0[1] - skel.bones["leftUpperLeg"].pos0[1]) * vrm_scale
    # Sensible fallback for rigs missing neck/upper-leg.
    return 5.42


def load_default_skeleton() -> Skeleton:
    """Return the bundled default Skeleton (cj.vrm's rest pose, ~36 bones).

    Lets `landmarks_to_bvh(..., vrm_path=None)` and `mocapy-solve` work without
    the caller supplying a VRM file. Bone naming is standard VRM-humanoid so the
    existing STANDARD_VRM_TO_MMD map applies unchanged.
    """
    import json
    from importlib.resources import files

    data_text = files("mocapy._data").joinpath("default_skeleton.json").read_text(encoding="utf-8")
    data = json.loads(data_text)
    bones: dict[str, Bone] = {}
    for nm, b in data["bones"].items():
        bones[nm] = Bone(
            name=nm,
            node=-1,
            parent=b["parent"],
            children=list(b["children"]),
            pos0=np.array(b["pos0"], dtype=np.float64),
        )
    return Skeleton(root=data["root"], bones=bones, vrm_version=int(data["vrm_version"]))
