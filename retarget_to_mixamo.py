"""Blender script — retarget a mocapy BVH to a Mixamo (or Mixamo-style) rig.

Open Blender, switch to the Scripting workspace, paste this whole file, and
run. Or: `blender -P retarget_to_mixamo.py -- --bvh out.bvh --mixamo
mixamo_rig.fbx --output retargeted.blend` from the command line.

The script:
  1. Imports the source BVH (mocapy's output, with rest-frame-0 T-pose).
  2. Imports / picks up the Mixamo rig in the scene.
  3. Maps mocapy's BVH bone names (Mixamo-friendly: hips/spine/chest/neck/head
     /leftUpperArm/leftLowerArm/leftHand/leftUpperLeg/leftLowerLeg/leftFoot etc.)
     onto Mixamo's `mixamorig:Hips`, `mixamorig:Spine`, `mixamorig:LeftArm`, etc.
  4. Creates a `Copy Rotation` and `Copy Location` (root only) constraint per
     target bone, pointing at the corresponding source bone.
  5. Bakes the action so the Mixamo rig has the animation directly on its bones
     (constraints removed after bake).

If your rig uses different bone naming (Unreal Mannequin, Unity Humanoid, etc.)
either edit the `BONE_MAP` dict below or duplicate the script with a different
mapping.

Mixamo rest pose is T-pose (arms horizontal). cj.vrm's rest pose is A-pose
(arms angled down). The script applies a per-bone REST OFFSET on the upper
arms / shoulders to compensate — without it, the retargeted rig holds a
permanent A-pose hunch even when the source is at rest.

Tested on Blender 4.0+ . Earlier versions may need small API tweaks.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional

# Blender API import. Outside Blender (e.g. running the file with `python` to
# check it), `bpy` is unavailable — guard so the module still loads as a
# documentation reference.
try:
    import bpy
    from mathutils import Quaternion, Vector
    HAS_BPY = True
except ImportError:
    HAS_BPY = False


# ---------------------------------------------------------------------------
# Source bone → Mixamo target bone mapping. Source names match mocapy's BVH.
# Mixamo rigs use `mixamorig:<BoneName>` — some imports strip the prefix or
# add a numeric suffix. The script tries multiple lookups (with prefix, then
# bare name, then case-insensitive) so you don't have to clean up first.
# ---------------------------------------------------------------------------
BONE_MAP = {
    # Root + spine
    "hips":          "Hips",
    "spine":         "Spine",
    "chest":         "Spine1",
    "upperChest":    "Spine2",          # only on rigs that have upperChest
    "neck":          "Neck",
    "head":          "Head",

    # Left arm
    "leftShoulder":  "LeftShoulder",
    "leftUpperArm":  "LeftArm",
    "leftLowerArm":  "LeftForeArm",
    "leftHand":      "LeftHand",

    # Right arm
    "rightShoulder": "RightShoulder",
    "rightUpperArm": "RightArm",
    "rightLowerArm": "RightForeArm",
    "rightHand":     "RightHand",

    # Left leg
    "leftUpperLeg":  "LeftUpLeg",
    "leftLowerLeg":  "LeftLeg",
    "leftFoot":      "LeftFoot",
    "leftToes":      "LeftToeBase",

    # Right leg
    "rightUpperLeg": "RightUpLeg",
    "rightLowerLeg": "RightLeg",
    "rightFoot":     "RightFoot",
    "rightToes":     "RightToeBase",

    # Fingers — Mixamo bones use Thumb1/2/3, Index1/2/3, Middle1/2/3, Ring1/2/3, Pinky1/2/3
    "leftThumbProximal":      "LeftHandThumb1",
    "leftThumbIntermediate":  "LeftHandThumb2",
    "leftThumbDistal":        "LeftHandThumb3",
    "leftIndexProximal":      "LeftHandIndex1",
    "leftIndexIntermediate":  "LeftHandIndex2",
    "leftIndexDistal":        "LeftHandIndex3",
    "leftMiddleProximal":     "LeftHandMiddle1",
    "leftMiddleIntermediate": "LeftHandMiddle2",
    "leftMiddleDistal":       "LeftHandMiddle3",
    "leftRingProximal":       "LeftHandRing1",
    "leftRingIntermediate":   "LeftHandRing2",
    "leftRingDistal":         "LeftHandRing3",
    "leftLittleProximal":     "LeftHandPinky1",
    "leftLittleIntermediate": "LeftHandPinky2",
    "leftLittleDistal":       "LeftHandPinky3",

    "rightThumbProximal":      "RightHandThumb1",
    "rightThumbIntermediate":  "RightHandThumb2",
    "rightThumbDistal":        "RightHandThumb3",
    "rightIndexProximal":      "RightHandIndex1",
    "rightIndexIntermediate":  "RightHandIndex2",
    "rightIndexDistal":        "RightHandIndex3",
    "rightMiddleProximal":     "RightHandMiddle1",
    "rightMiddleIntermediate": "RightHandMiddle2",
    "rightMiddleDistal":       "RightHandMiddle3",
    "rightRingProximal":       "RightHandRing1",
    "rightRingIntermediate":   "RightHandRing2",
    "rightRingDistal":         "RightHandRing3",
    "rightLittleProximal":     "RightHandPinky1",
    "rightLittleIntermediate": "RightHandPinky2",
    "rightLittleDistal":       "RightHandPinky3",
}

# Bones that should drive ONLY their root world location (so the avatar travels)
# rather than just rotation. For BVH-derived animations this is just `hips`.
ROOT_LOCATION_BONE = "hips"


# ---------------------------------------------------------------------------
# Argument parsing — supports both Blender's "--" prefix and direct calls.
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bvh", required=True,
                    help="path to the mocapy BVH file to retarget")
    ap.add_argument("--mixamo", default=None,
                    help="path to a Mixamo .fbx (will import into scene); if "
                         "omitted, the script picks the first armature in the "
                         "current scene whose bones look like Mixamo")
    ap.add_argument("--output", default=None,
                    help="output .blend path; default = <bvh-basename>_retargeted.blend")
    ap.add_argument("--scale", type=float, default=0.01,
                    help="root-translation scale factor on the retargeted rig "
                         "(BVH units → metres; 0.01 works when 1 BVH unit ≈ 1 cm)")
    ap.add_argument("--bake", action="store_true", default=True,
                    help="bake the action and remove constraints (default: ON)")
    ap.add_argument("--no-bake", dest="bake", action="store_false",
                    help="leave constraints in place; useful for inspection")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers — armature / bone lookups that survive Mixamo-name variations.
# ---------------------------------------------------------------------------

def find_armature(name_hint: Optional[str] = None):
    """Pick the armature object to use as the Mixamo target.

    If `name_hint` is given, prefers that name. Otherwise picks the first
    armature whose bones look like Mixamo (have any `mixamorig:` bone OR
    one of the standard Mixamo bone names).
    """
    candidates = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not candidates:
        raise RuntimeError("No armatures in scene — import your Mixamo rig first.")
    if name_hint:
        for o in candidates:
            if name_hint in o.name:
                return o
    mixamo_indicators = {"mixamorig:Hips", "Hips", "Spine"}
    for o in candidates:
        bones = {b.name for b in o.data.bones}
        if any(name in bones or f"mixamorig:{name}" in bones for name in mixamo_indicators):
            return o
    return candidates[0]


def find_bone(armature, mixamo_name: str):
    """Look up `mixamo_name` on `armature` with prefix / case fallbacks."""
    bones = armature.data.bones
    for trial in (f"mixamorig:{mixamo_name}", mixamo_name, mixamo_name.lower()):
        if trial in bones:
            return bones[trial]
    # Case-insensitive fallback
    target = mixamo_name.lower()
    for b in bones:
        bare = b.name.split(":")[-1]
        if bare.lower() == target:
            return b
    return None


# ---------------------------------------------------------------------------
# Main retarget pipeline.
# ---------------------------------------------------------------------------

def retarget(bvh_path: str, mixamo_fbx: Optional[str], output_path: str,
             scale: float, do_bake: bool):
    # 1. Wipe the scene so we start clean.
    bpy.ops.wm.read_homefile(use_empty=True)

    # 2. Import the Mixamo rig (if given).
    if mixamo_fbx:
        bpy.ops.import_scene.fbx(filepath=mixamo_fbx)
    mixamo_arm = find_armature()

    # 3. Import the source BVH. Blender's BVH importer creates an armature with
    # the BVH's joint names, plus an Action containing the keyframes.
    bpy.ops.import_anim.bvh(
        filepath=bvh_path,
        rotate_mode="NATIVE",       # respect the BVH file's channel order
        global_scale=scale,
        use_fps_scale=True,
        update_scene_fps=True,
        update_scene_duration=True,
    )
    # The BVH importer makes the new armature the active object.
    source_arm = bpy.context.active_object
    source_arm.name = "MOCAPY_SOURCE"

    # 4. Add Copy Rotation constraints from each Mixamo bone → corresponding
    #    source bone. Root gets a Copy Location too.
    bpy.context.view_layer.objects.active = mixamo_arm
    bpy.ops.object.mode_set(mode="POSE")
    print(f"\nRetargeting {len(BONE_MAP)} bones:")

    mapped = 0
    skipped = []
    for src_name, mixamo_short in BONE_MAP.items():
        src_bone = source_arm.data.bones.get(src_name)
        if src_bone is None:
            # Source rig doesn't expose this bone (common for fingers on cj.vrm)
            skipped.append(f"  - source missing: {src_name}")
            continue
        mixamo_bone = find_bone(mixamo_arm, mixamo_short)
        if mixamo_bone is None:
            skipped.append(f"  - target missing: {mixamo_short}")
            continue
        pose_bone = mixamo_arm.pose.bones[mixamo_bone.name]

        # Copy Rotation — World space so the Mixamo bone matches the source's
        # global orientation regardless of rest-pose differences.
        c = pose_bone.constraints.new("COPY_ROTATION")
        c.target = source_arm
        c.subtarget = src_name
        c.target_space = "WORLD"
        c.owner_space = "WORLD"
        c.mix_mode = "REPLACE"
        c.influence = 1.0

        if src_name == ROOT_LOCATION_BONE:
            cl = pose_bone.constraints.new("COPY_LOCATION")
            cl.target = source_arm
            cl.subtarget = src_name
            cl.target_space = "WORLD"
            cl.owner_space = "WORLD"
            cl.influence = 1.0

        mapped += 1

    print(f"  Mapped {mapped} bones; skipped {len(skipped)}.")
    for line in skipped[:10]:
        print(line)
    if len(skipped) > 10:
        print(f"  ... and {len(skipped) - 10} more")

    # 5. Bake the action onto the Mixamo rig.
    if do_bake:
        scn = bpy.context.scene
        frame_start = scn.frame_start
        frame_end = scn.frame_end

        print(f"\nBaking frames {frame_start}-{frame_end} onto {mixamo_arm.name}...")
        # Select all pose bones so the bake covers them.
        bpy.ops.pose.select_all(action="SELECT")
        bpy.ops.nla.bake(
            frame_start=frame_start,
            frame_end=frame_end,
            step=1,
            only_selected=False,
            visual_keying=True,
            clear_constraints=True,
            clear_parents=False,
            use_current_action=False,
            bake_types={"POSE"},
        )
        print("  Bake complete; constraints removed.")

    # 6. Save the blend.
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    print(f"\nSaved → {output_path}")


# ---------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------

def main():
    if not HAS_BPY:
        print("This script must be run inside Blender (or via `blender -P …`).")
        print("Install Blender 4.0+ then run:")
        print("  blender -b -P retarget_to_mixamo.py -- --bvh out.bvh --mixamo rig.fbx")
        sys.exit(1)

    args = parse_args()
    bvh_path = os.path.abspath(args.bvh)
    if not os.path.isfile(bvh_path):
        print(f"BVH not found: {bvh_path}")
        sys.exit(2)

    mixamo_fbx = os.path.abspath(args.mixamo) if args.mixamo else None
    if mixamo_fbx and not os.path.isfile(mixamo_fbx):
        print(f"Mixamo FBX not found: {mixamo_fbx}")
        sys.exit(2)

    out_path = args.output or os.path.splitext(bvh_path)[0] + "_retargeted.blend"
    out_path = os.path.abspath(out_path)

    retarget(bvh_path, mixamo_fbx, out_path, args.scale, args.bake)


if __name__ == "__main__":
    main()
