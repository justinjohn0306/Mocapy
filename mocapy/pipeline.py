"""End-to-end assembly: landmarks -> all solved bones -> blended boneKeys -> BVH.

Solved bones (others default to identity in the BVH writer): センター, 上半身, 上半身2,
下半身, 左/右 腕, ひじ, 手捩 (forearm twist), 手首 (wrist, approx), 足, ひざ, 足首, 首, 頭,
fingers (人指/中指/薬指/小指 ×3 joints) + 親指 (thumb, approx) when the hand landmarker
sees a hand, and 両目 (eye gaze) when the face landmarker sees a face.
Identity by design: 肩 (a ~6° two-stage-filtered signal pose can't beat identity on —
filter-dominated), 腕捩 (identity in golden).
"""

from __future__ import annotations

import numpy as np

from mocapy.solve.blend import BlendChannel
from mocapy.solve.torso import solve_center, solve_upper_body, solve_lower_body
from mocapy.solve.arms import solve_arm
from mocapy.solve.legs import solve_leg
from mocapy.solve.root import solve_root

WORLD_SHOULDER_WIDTH = 2.18718  # cj.vrm model_para_obj.shoulder_width

# Foot (足首) stabilization (approximates the reference's floor grounding without porting IK).
# The foot orientation is solved from the tiny heel/ankle/toe triangle, so it's
# hypersensitive to landmark noise and jitters violently when the feet are ambiguous
# (close together / edge-on / partly out of frame). the reference keeps the foot settled by
# grounding it to the floor when planted; we approximate that with a plant-adaptive
# hold: smooth hard toward the previous foot pose ONLY when the ankle is nearly
# stationary (planted), and pass the raw solve straight through when the foot is
# actually moving — so real steps/kicks don't lag and clean motion (golden) is
# barely touched. A light OneEuro on top handles residual planted micro-jitter.
FOOT_FILTER_PARAMS = (1.0, 1.5, 1.0)      # OneEuro (minCutOff, beta, dCutOff)

# Torso/hip rotation OneEuro params (minCutOff, beta, dCutOff) — tames the yaw
# "wobble" when the subject rotates (shoulder/hip depth ambiguity). Centre (hips)
# is the main culprit so it gets more smoothing than the spine bones.
_TORSO_EURO_CENTER = (1.0, 1.5, 1.0)      # sweet spot: -25% yaw wobble, ~0 golden regression
_TORSO_EURO_SPINE = (1.2, 1.5, 1.0)
FOOT_PLANT_SPEED_REF = 0.04               # ankle world-speed (m/frame) that counts as "moving"
FOOT_PLANT_MAX_HOLD = 0.9                  # max slerp-hold toward previous when fully planted+jittery
# Hold only kicks in when the raw foot solve JUMPS more than this (deg) frame-to-
# frame while planted — that's the jitter signature. Real foot pivots change
# smoothly (small per-frame steps) and pass straight through, so clean motion
# (golden) is untouched; only the ~30°/frame planted oscillation gets damped.
FOOT_PLANT_JUMP_LO, FOOT_PLANT_JUMP_HI = 15.0, 30.0
_FOOT_ANKLE_IDX = {"左足首": 28, "右足首": 27}

# Standard VRM-humanoid -> MMD bone map (keyed by VRM bone name). Covers every model;
# the solver only fills a subset, the rest stay identity in the BVH writer.
_FINGERS = {"Thumb": ("親指", ("０", "１", "２")), "Index": ("人指", ("１", "２", "３")),
            "Middle": ("中指", ("１", "２", "３")), "Ring": ("薬指", ("１", "２", "３")),
            "Little": ("小指", ("１", "２", "３"))}
STANDARD_VRM_TO_MMD = {
    "hips": "センター", "spine": "上半身", "chest": "上半身2", "upperChest": "上半身3",
    "neck": "首", "head": "頭",
    # Both eyes get the same 両目 quaternion (the reference convention — a single MMD bone drives both).
    "leftEye": "両目", "rightEye": "両目",
}
for _lr, _j in (("left", "左"), ("right", "右")):
    STANDARD_VRM_TO_MMD.update({
        _lr + "Shoulder": _j + "肩", _lr + "UpperArm": _j + "腕",
        _lr + "LowerArm": _j + "ひじ", _lr + "Hand": _j + "手首",
        _lr + "UpperLeg": _j + "足", _lr + "LowerLeg": _j + "ひざ",
        _lr + "Foot": _j + "足首", _lr + "Toes": _j + "足先EX",
    })
    for _f, (_mf, _sfx) in _FINGERS.items():
        for _seg, _s in zip(("Proximal", "Intermediate", "Distal"), _sfx):
            STANDARD_VRM_TO_MMD[_lr + _f + _seg] = _j + _mf + _s


# MMD morph left/right pairs — swapped when mirroring so a wink/blink follows the
# flipped side (keeps the morph sidecar consistent with the mirrored bones).
_MORPH_LR_SWAP = {
    "まばたきL": "まばたきR", "まばたきR": "まばたきL",
    "ウィンク": "ウィンク右", "ウィンク右": "ウィンク",
    "上": "上", "下": "下",
}


def _mirror_bone_name(name: str) -> str:
    """Swap the 左/右 (left/right) prefix of an MMD bone name; pass others through."""
    if name.startswith("左"):
        return "右" + name[1:]
    if name.startswith("右"):
        return "左" + name[1:]
    return name


def mirror_bone_keys(bone_keys: list[dict]) -> list[dict]:
    """Left/right-mirror solved boneKeys across the sagittal plane.

    The solver's quaternions are three.js convention (right-handed, Y-up, Z-out),
    so the left/right axis is X. A sagittal mirror is therefore:
      * swap each bone's 左/右 prefix (the data drives the opposite-side bone),
      * negate the root X translation (lean right -> avatar leans left),
      * reflect each quaternion across the X-normal plane: (x,y,z,w)->(x,-y,-z,w).
    Involutive: mirroring twice returns the original. Rig-independent.
    """
    out = []
    for k in bone_keys:
        x, y, z, w = k["rot"]
        px, py, pz = k["pos"]
        out.append({
            **k,
            "name": _mirror_bone_name(k["name"]),
            "pos": [-px, py, pz],
            "rot": [x, -y, -z, w],
        })
    return out


def mirror_morphs(morphs_out: list) -> None:
    """In-place left/right swap of paired morphs (wink/blink) for a mirrored take."""
    for i, m in enumerate(morphs_out):
        if not m:
            continue
        swapped = {}
        for nm, wt in m.items():
            tgt = _MORPH_LR_SWAP.get(nm, nm)
            swapped[tgt] = wt
        morphs_out[i] = swapped


class _FootStabilizer:
    """Plant-adaptive foot-orientation stabilizer (see FOOT_* constants).

    Per foot it tracks the ankle world position and the last emitted quaternion.
    Each frame it measures ankle speed -> a "planted" factor in [0,1]; when planted
    it slerp-holds the new solve toward the previous pose (suppressing jitter), and
    when moving it lets the raw solve through. A light OneEuro adds residual
    smoothing. No-op-ish on smooth motion; strong only on planted, jittery frames.
    """

    def __init__(self):
        from mocapy.detect.filters import OneEuroFilter
        mc, beta, dc = FOOT_FILTER_PARAMS
        self._euro = {n: OneEuroFilter(30, mc, beta, dc, 4) for n in _FOOT_ANKLE_IDX}
        self._prev_q: dict[str, np.ndarray] = {}
        self._prev_ankle: dict[str, np.ndarray] = {}

    def process(self, name, rot, kp3, frame_idx):
        import math
        from mocapy.solve.three_math import quat_slerp_arr as _slerp
        rot = np.asarray(rot, dtype=float)
        idx = _FOOT_ANKLE_IDX[name]
        a = kp3[idx]
        ankle = np.array([a["x"], a["y"], a["z"]])
        prev_a = self._prev_ankle.get(name)
        speed = float(np.linalg.norm(ankle - prev_a)) if prev_a is not None else 1e9
        self._prev_ankle[name] = ankle
        # planted factor: 1 when stationary, 0 once ankle speed >= SPEED_REF
        plant = max(0.0, min(1.0, 1.0 - speed / max(FOOT_PLANT_SPEED_REF, 1e-9)))
        prev_q = self._prev_q.get(name)
        if prev_q is not None and plant > 0.0:
            # how big is the raw per-frame jump? small = real pivot (pass), big = jitter (hold)
            dot = min(1.0, abs(float(np.dot(rot, prev_q))))
            jump = math.degrees(2.0 * math.acos(dot))
            lo, hi = FOOT_PLANT_JUMP_LO, FOOT_PLANT_JUMP_HI
            jfac = 0.0 if jump <= lo else (1.0 if jump >= hi else (jump - lo) / (hi - lo))
            hold = plant * FOOT_PLANT_MAX_HOLD * jfac
            if hold > 0.0:
                rot = np.array(_slerp(rot, prev_q, hold))   # damp the planted jitter spike
        # light residual smoothing (and gives a sane value at frame 0)
        rot = np.array(self._euro[name].filter(list(rot), frame_idx * 1000.0 / 30.0))
        self._prev_q[name] = rot
        return rot


def apply_grounding(bone_keys, skel, *, factor=1.0):
    """Port of the reference's foot grounding (`auto_grounding` in the reference solver).

    the reference plants the lowest foot on the floor and lets the HIPS Y follow: when the
    subject crouches/bends a knee, the feet rise toward the hips, so the hips drop to
    keep the feet grounded. Our back-projected root Y misses this (the figure floats).

    the reference grounds on the RIG foot position (`get_bone_position_by_MMD_name("足首")`), so
    we do the same: forward-kinematic the solved leg chain
    hips(センター·下半身) → 足 → ひざ → ankle(足首 origin) using the rig rest offsets,
    take the LOWEST ankle Y (planted foot), and drive 全ての親 Y from how far it sits
    below the hips. Fixed rig bone lengths swing the foot more when a knee bends than
    the raw landmark foot did — which is what the landmark version under-captured.
    Mean-anchored (absolute height is a separate depth concern) + OneEuro-smoothed
    (the reference's `M.filter`).
    """
    from mocapy.detect.filters import OneEuroFilter
    from mocapy.solve import three_math as tm
    VS = 10.0
    I = np.array([0.0, 0.0, 0.0, 1.0])
    hp = np.asarray(skel.bones["hips"].pos0, dtype=float)
    offs = {}
    for mmd, vrm in (("左", "left"), ("右", "right")):
        up = np.asarray(skel.bones[vrm + "UpperLeg"].pos0, dtype=float)
        lo = np.asarray(skel.bones[vrm + "LowerLeg"].pos0, dtype=float)
        ft = np.asarray(skel.bones[vrm + "Foot"].pos0, dtype=float)
        offs[mmd] = ((up - hp) * VS, (lo - up) * VS, (ft - lo) * VS)

    frame_rots = {}
    for k in bone_keys:
        frame_rots.setdefault(round(k["time"] * 30), {})[k["name"]] = np.asarray(k["rot"], dtype=float)

    def ankle_y(rots, mmd):
        Rh = tm.quat_mul(rots.get("センター", I), rots.get("下半身", I))   # hips world rot
        ou, ol, of_ = offs[mmd]
        Ru = tm.quat_mul(Rh, rots.get(mmd + "足", I))
        Rl = tm.quat_mul(Ru, rots.get(mmd + "ひざ", I))
        p = (tm.apply_quat_to_vec(ou, Rh) + tm.apply_quat_to_vec(ol, Ru)
             + tm.apply_quat_to_vec(of_, Rl))
        return float(p[1])

    fb = {}
    for f, rots in frame_rots.items():
        if "センター" not in rots:
            continue
        fb[f] = -min(ankle_y(rots, "左"), ankle_y(rots, "右"))   # lowest foot -> below-hips dist
    if len(fb) < 2:
        return
    med = float(np.median(list(fb.values())))
    filt = OneEuroFilter(30, 1, 0.5, 1, 0)
    off = {f: float(filt.filter(factor * (fb[f] - med), f * 1000.0 / 30.0)) for f in sorted(fb)}
    for k in bone_keys:
        if k["name"] == "全ての親":
            f = round(k["time"] * 30)
            if f in off:
                k["pos"][1] += off[f]


def compute_scales(skel, *, ref_leg_length=10.5696, vrm_scale=10.0, vrm_norm=11.0):
    """leg_scale + pos_scale for any VRM (root-translation scaling). Defaults from cj.vrm."""
    leg = (float(np.linalg.norm(skel.bones["leftUpperLeg"].pos0 - skel.bones["leftLowerLeg"].pos0)) +
           float(np.linalg.norm(skel.bones["leftLowerLeg"].pos0 - skel.bones["leftFoot"].pos0))) * vrm_scale
    return leg / ref_leg_length, vrm_scale / vrm_norm


def solve_frame(kp3, face_px=None) -> dict[str, np.ndarray]:
    """One frame of world pose landmarks (33 BlazePose) -> {MMD bone: local rot}.

    face_px (optional): the 33 screen landmarks in pixel scale with z
    ([x*W, y*H, z*W]) — adds 首/頭 head-pose from nose + eye corners.
    """
    from mocapy.solve.wrist import solve_wrist

    S = solve_center(kp3)
    h, M = solve_upper_body(kp3, S)
    D = solve_lower_body(kp3, S)
    out = {"センター": S, "上半身": h, "上半身2": M, "下半身": D}
    for side in ("左", "右"):
        ude, hiji = solve_arm(kp3, S, h, M, side)
        thigh, knee, foot = solve_leg(kp3, S, D, side)
        twist, wrist = solve_wrist(kp3, S, h, M, ude, hiji, side)
        out[side + "腕"] = ude
        out[side + "ひじ"] = hiji
        out[side + "手捩"] = twist
        out[side + "手首"] = wrist
        out[side + "足"] = thigh
        out[side + "ひざ"] = knee
        out[side + "足首"] = foot
    return out


def solve_sequence(frames_kp3, frames_px=None, frame_size=None,
                   world_shoulder_width=WORLD_SHOULDER_WIDTH, frames_hands=None,
                   frames_face=None, frames_face_matrix=None,
                   frames_face_blendshapes=None,
                   morphs_out: list | None = None,
                   fov_deg: float | None = None,
                   finger_axis_rot: dict | None = None,
                   finger_axis_rot_offset_inv: dict | None = None,
                   spine_length_ref: float | None = None,
                   stabilize_feet: bool = True,
                   smooth_torso: bool = True) -> list[dict]:
    """Per-frame keypoints3D (world, for rotations) -> MMD-style boneKeys w/ skin-blend.

    frames_px (optional): per-frame 2D screen landmarks in pixels (list of [x,y] indexed
    like the 33 BlazePose landmarks) -> adds 全ての親 root translation. frame_size=(W,H).
    frames_hands (optional): per-frame {'Left'|'Right': (21,3)} hand landmarks -> fingers.
    frames_face (optional):  per-frame (478,3) face-mesh landmarks (full-frame
                             normalized x,y; face-relative z) -> drives 両目 eye gaze.
    frames_face_matrix (optional): per-frame (4,4) facial_transformation_matrix —
                             used to head-correct the eye signal AND the morph "sad" detection.
    frames_face_blendshapes (optional): per-frame {name: weight} 52 ARKit blendshapes
                             -> drives the MMD/VRM expression morphs.
    morphs_out (optional, mutated): if supplied, appended one MMD-morph dict (or None)
                             per detected frame. Lets callers write the morph sidecar.
    """
    from mocapy.solve.root import (
        shoulder_width_px, depth_for_shoulder, DEFAULT_FOV_DEG,
        TYPICAL_HUMAN_SHOULDER_M, SPINE_LENGTH_REF_DEFAULT,
    )
    # spine_length_ref defaults to the reference's `model.para.spine_length` constant —
    # the right value for cj.vrm is ~5.42, not the 2.5 placeholder.
    slr = SPINE_LENGTH_REF_DEFAULT if spine_length_ref is None else float(spine_length_ref)
    from mocapy.solve.head import head_total, split_head
    from mocapy.solve.fingers import solve_hands
    from mocapy.solve.eyes import solve_eyes, head_pitch_yaw_from_face_matrix
    from mocapy.solve.face_morphs import solve_morphs
    from mocapy.detect.filters import OneEuroFilter

    blends: dict[str, BlendChannel] = {}
    root_blend = BlendChannel(mocap_data_smoothing=2)
    # Root-translation OneEuro filters.
    #
    # the reference's published params for these are:
    #   * camera_depth (Z scalar): `OneEuroFilter(30, 1, 1/5, 1, 1)`
    #   * v_hip       (XYZ vector): `OneEuroFilter(30, 1, 1/2, 1, 3)`
    #
    # The per-frame |dZ| comparison vs the reference output on test.mp4 shows our
    # input to camera_depth is ~1.75× noisier than the reference's (their median |dZ| is
    # 0.109; ours with their params is 0.191). The cause is structural: the reference
    # reads TFJS BlazePose's `keypoints3D_raw` for the `|3D midpoint|` reference,
    # which is small and quiet. MediaPipe Tasks Python only gives us
    # `pose_world_landmarks` (hip-centered), so we fall back to the shoulder
    # midpoint — about 2× larger than the reference's effective reference, which amplifies
    # per-frame landmark noise through the `l = 4/a` depth-scale relationship.
    #
    # We compensate by lowering the camera_depth cutoffs so the output median
    # |dZ| comes back in line with the reference's. The v_hip filter stays at the reference's params
    # since XY components are already smooth.
    z_filter = OneEuroFilter(30, 0.5, 0.1, 0.5, 0)
    v_hip_filter = OneEuroFilter(30, 1, 0.5, 1, 3)
    head_filter = OneEuroFilter(30, 1, 2, 1, 4)  # Dt head_rot filter (tames pose-face noise)
    head_blends = {"首": BlendChannel(mocap_data_smoothing=2), "頭": BlendChannel(mocap_data_smoothing=2)}
    finger_blends: dict[str, BlendChannel] = {}
    eyes_filter = OneEuroFilter(30, 1, 5, 1, 4)  # the reference's Ie["両目"] filter params
    eyes_blend = BlendChannel(mocap_data_smoothing=2)
    # Foot (足首) quaternion OneEuro. The foot orientation is solved from the very
    # short heel/ankle/toe triangle, so it's hypersensitive to landmark noise: when
    # the feet are ambiguous (close together, edge-on, or partly out of frame) the
    # raw solve jitters violently frame-to-frame (measured 26°+/frame vs the reference's ~4°,
    # which stays settled via floor grounding we don't port). A velocity-adaptive
    # OneEuro suppresses that stationary jitter while still passing real foot moves;
    # on clean motion (golden) it is ~pass-through. Quaternion type (4).
    foot_stab = _FootStabilizer() if stabilize_feet else None
    # Torso/hip rotation OneEuro (quaternion). The hips/spine yaw is solved from the
    # shoulder & hip world vectors, whose DEPTH (z) is ambiguous when the body turns
    # side-/back-on — so the torso yaw flickers ~6° f2f ("wobble" as the subject
    # rotates). A velocity-adaptive OneEuro suppresses that flicker while still
    # passing real turns; near pass-through on clean motion. Centre (hips) gets the
    # most; spine bones lighter.
    torso_filters = ({"センター": OneEuroFilter(30, *_TORSO_EURO_CENTER, 4),
                      "上半身": OneEuroFilter(30, *_TORSO_EURO_SPINE, 4),
                      "上半身2": OneEuroFilter(30, *_TORSO_EURO_SPINE, 4),
                      "下半身": OneEuroFilter(30, *_TORSO_EURO_SPINE, 4)}
                     if smooth_torso else {})
    bone_keys: list[dict] = []
    W, H = frame_size or (1280, 720)
    fov = DEFAULT_FOV_DEG if fov_deg is None else float(fov_deg)

    # Stable xy scale from the MEDIAN shoulder width over the clip (grounds the
    # figure; per-frame shoulder jitter would otherwise float it up/down). Z
    # baseline = shoulder width too — spine length was tested but overshoots
    # the reference's Z range by 2× because spine length conflates body tilt with depth
    # and we don't yet port the reference's `a.y -= s/cos(tilt)` tilt-compensation.
    xy_scale = None
    z_offset_units = 0.0
    pose_world_shoulder_m = None
    if frames_px:
        sws = [shoulder_width_px((p[11], p[12])) for p in frames_px if p]
        if sws:
            median_sw = max(float(np.median(sws)), 1e-6)
            xy_scale = world_shoulder_width / median_sw
            depth_at_median = depth_for_shoulder(median_sw, H, world_shoulder_width, fov)
            old_z_at_median = 23.35 / median_sw - 15.28
            z_offset_units = old_z_at_median + depth_at_median * 1.5
    if frames_kp3:
        sws_world = []
        for kp3 in frames_kp3:
            if not kp3:
                continue
            try:
                a = kp3[11]; b = kp3[12]
                sws_world.append(((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2) ** 0.5)
            except (KeyError, TypeError, IndexError):
                continue
        if sws_world:
            pose_world_shoulder_m = float(np.median(sws_world))

    z_floor = None
    if frames_px:
        sws = [shoulder_width_px((p[11], p[12])) for p in frames_px if p]
        if sws:
            min_sw = max(min(sws), 0.02 * max(W, H))
            scale_ref = (pose_world_shoulder_m * (world_shoulder_width / TYPICAL_HUMAN_SHOULDER_M)
                         if pose_world_shoulder_m else world_shoulder_width)
            max_expected_depth = depth_for_shoulder(min_sw, H, scale_ref, fov)
            z_floor = -max_expected_depth * 1.5 + z_offset_units - 3.0 * world_shoulder_width

    for f, kp3 in enumerate(frames_kp3):
        if morphs_out is not None:
            morphs_out.append(None)   # default; possibly replaced below
        if not kp3:
            continue
        for name, rot in solve_frame(kp3).items():
            if foot_stab is not None and name in _FOOT_ANKLE_IDX:
                rot = foot_stab.process(name, rot, kp3, f)
            tf = torso_filters.get(name)
            if tf is not None:
                rot = np.array(tf.filter(list(np.asarray(rot, dtype=float)), f * 1000.0 / 30.0))
            bl = blends.get(name)
            if bl is None:
                bl = blends[name] = BlendChannel(mocap_data_smoothing=2)
            rec = bl.add_rot(name, rot, e_timing=0.5)
            bone_keys.append({"name": name, "time": f / 30.0, "pos": [0.0, 0.0, 0.0],
                              "rot": [float(x) for x in rec]})

        # head/neck (needs the screen face landmarks with z); filter the total then split
        face = frames_px[f] if (frames_px and frames_px[f] and len(frames_px[f][0]) >= 3) else None
        if face is not None:
            S = solve_center(kp3)
            h, M = solve_upper_body(kp3, S)
            total = np.array(head_filter.filter(list(head_total(face, S, h, M)), f * 1000.0 / 30.0))
            neck, head = split_head(total)
            for nm, r in (("首", neck), ("頭", head)):
                rec = head_blends[nm].add_rot(nm, r, e_timing=0.5)
                bone_keys.append({"name": nm, "time": f / 30.0, "pos": [0.0, 0.0, 0.0],
                                  "rot": [float(x) for x in rec]})
        # fingers (from the hand landmarker); each finger bone gets the shared skin-blend
        if frames_hands and frames_hands[f]:
            for nm, r in solve_hands(
                frames_hands[f], W, H,
                axis_rot_by_mmd=finger_axis_rot,
                axis_rot_offset_inv_by_mmd=finger_axis_rot_offset_inv,
            ).items():
                bl = finger_blends.get(nm)
                if bl is None:
                    bl = finger_blends[nm] = BlendChannel(mocap_data_smoothing=2)
                rec = bl.add_rot(nm, r, e_timing=0.5)
                bone_keys.append({"name": nm, "time": f / 30.0, "pos": [0.0, 0.0, 0.0],
                                  "rot": [float(x) for x in rec]})
        # eyes (両目, iris-driven from face mesh); head-corrected by face matrix
        pitch_h = yaw_h = 0.0
        if frames_face_matrix and frames_face_matrix[f] is not None:
            pitch_h, yaw_h = head_pitch_yaw_from_face_matrix(np.asarray(frames_face_matrix[f]))
        if frames_face and frames_face[f] is not None:
            face_lms = np.asarray(frames_face[f])
            eye_q = solve_eyes(face_lms, head_pitch_rad=pitch_h, head_yaw_rad=yaw_h)
            # Filter as quaternion (the reference's Ie["両目"] OneEuro), then skin-blend.
            eye_q = np.array(eyes_filter.filter(list(eye_q), f * 1000.0 / 30.0))
            rec = eyes_blend.add_rot("両目", eye_q, e_timing=0.5)
            bone_keys.append({"name": "両目", "time": f / 30.0, "pos": [0.0, 0.0, 0.0],
                              "rot": [float(x) for x in rec]})
        # morphs (52 ARKit blendshapes -> MMD/VRM expression weights)
        if frames_face_blendshapes and frames_face_blendshapes[f]:
            m = solve_morphs(frames_face_blendshapes[f],
                             head_pitch_rad=pitch_h, head_yaw_rad=yaw_h)
            if morphs_out is not None:
                morphs_out[f] = m
        if frames_px and frames_px[f]:
            kp2 = frames_px[f]
            hip = (np.asarray(kp2[23]) + np.asarray(kp2[24])) / 2.0

            # Pose-world landmarks (3D anatomy in meters, hip-centered) — feeds
            # the reference's data3D-based depth formula. None if missing.
            kp3 = frames_kp3[f] if (frames_kp3 and frames_kp3[f]) else None
            pw = (None, None, None, None)
            if kp3:
                try:
                    pw = (np.array([kp3[11]["x"], kp3[11]["y"], kp3[11]["z"]]),
                          np.array([kp3[12]["x"], kp3[12]["y"], kp3[12]["z"]]),
                          np.array([kp3[23]["x"], kp3[23]["y"], kp3[23]["z"]]),
                          np.array([kp3[24]["x"], kp3[24]["y"], kp3[24]["z"]]))
                except (KeyError, IndexError, TypeError):
                    pass

            # Step 1: compute raw per-frame depth (the reference's `t()` function output).
            pos = solve_root(hip, (kp2[11], kp2[12]), W, H, world_shoulder_width,
                             xy_scale=xy_scale, fov_deg=fov,
                             pose_world_sh_l=pw[0], pose_world_sh_r=pw[1],
                             pose_world_hp_l=pw[2], pose_world_hp_r=pw[3],
                             spine_length_ref=slr,
                             z_floor=z_floor)
            # Step 2: smooth Z (the reference's `z = a.filter(o); this.z_smoothed = z`)
            z_smoothed = float(z_filter.filter(float(pos[2]), f * 1000.0 / 30.0))
            # Step 3: recompute X/Y using the SMOOTHED depth (the reference's
            #   `C.multiplyScalar(-h/C.z)` with `h = at.z_smoothed`). This is
            #   what gives XY motion proper magnitude — the same screen-pixel
            #   offset corresponds to a larger world offset when the subject is
            #   further from camera. Our previous constant-scale XY undermag-
            #   nified motion by 60-75% vs the reference.
            pos = solve_root(hip, (kp2[11], kp2[12]), W, H, world_shoulder_width,
                             xy_scale=xy_scale, fov_deg=fov,
                             pose_world_sh_l=pw[0], pose_world_sh_r=pw[1],
                             pose_world_hp_l=pw[2], pose_world_hp_r=pw[3],
                             spine_length_ref=slr,
                             z_floor=z_floor,
                             z_smoothed=z_smoothed)
            pos[2] = z_smoothed
            # Step 4: v_hip OneEuro on the full XYZ vector
            #   (the reference's `h.fromArray(at.v_hip_filter.filter(h.toArray()))`)
            pos = list(v_hip_filter.filter([float(pos[0]), float(pos[1]), float(pos[2])],
                                            f * 1000.0 / 30.0))
            # Step 5: Z-only offset to match the reference's depth mean. Our depth output is
            # consistently `~spine_length_ref · 2` units more negative than the reference's
            # because we use `|sh_mid_3d|` for the 3D reference where the reference uses
            # `|keypoints3D_raw[hip_mid]|` — different magnitude. The DELTA-MEAN
            # turns out to be exactly `spine_length_ref · 2`. Applied AFTER the
            # v_hip filter so it doesn't propagate back into XY (XY uses h =
            # |depth_z| BEFORE this offset).
            pos[2] += slr * 2.0
            recp = root_blend.add_pos("全ての親", pos, e_timing=0.5)
            bone_keys.append({"name": "全ての親", "time": f / 30.0,
                              "pos": [float(x) for x in recp], "rot": [0.0, 0.0, 0.0, 1.0]})

    return bone_keys


def landmarks_to_bvh(frames, frames_px, frame_size, vrm_path=None, frames_hands=None,
                     frames_face=None, frames_face_matrix=None,
                     frames_face_blendshapes=None,
                     morphs_out: list | None = None,
                     fov_deg: float | None = None,
                     mirror: bool = False,
                     stabilize_feet: bool = True,
                     grounding: bool = False):
    """Solved landmarks -> BVH. Works on any VRM, or — when `vrm_path=None` — the
    bundled default skeleton (cj.vrm's rest pose, 36 bones, ships inside the
    package as mocapy/_data/default_skeleton.json).

    Pass `morphs_out=[]` to also collect per-frame MMD morph dicts (for the
    sidecar).
    """
    from mocapy.vrm.skeleton import (
        load_skeleton, load_default_skeleton, compute_finger_axis_rot,
        model_spine_length,
    )
    from mocapy.export.bvh import write_bvh

    skel = load_default_skeleton() if vrm_path is None else load_skeleton(vrm_path)
    fa_axis_rot, fa_offset_inv = compute_finger_axis_rot(skel)
    slr = model_spine_length(skel)  # the reference's `Je.spine_length_ref` derived from this rig

    bone_keys = solve_sequence(frames, frames_px=frames_px, frame_size=frame_size,
                               frames_hands=frames_hands,
                               frames_face=frames_face,
                               frames_face_matrix=frames_face_matrix,
                               frames_face_blendshapes=frames_face_blendshapes,
                               morphs_out=morphs_out, fov_deg=fov_deg,
                               finger_axis_rot=fa_axis_rot,
                               finger_axis_rot_offset_inv=fa_offset_inv,
                               spine_length_ref=slr,
                               stabilize_feet=stabilize_feet)
    if grounding:
        apply_grounding(bone_keys, skel)
    if mirror:
        bone_keys = mirror_bone_keys(bone_keys)
        if morphs_out is not None:
            mirror_morphs(morphs_out)
    vrm_to_mmd = {n: STANDARD_VRM_TO_MMD.get(n) for n in skel.bones}
    leg_scale, pos_scale = compute_scales(skel)
    return write_bvh(skel, bone_keys, vrm_to_mmd, leg_scale=leg_scale, pos_scale=pos_scale)


def video_to_bvh(video_path, vrm_path, *, max_frames=None):
    """Full Python pipeline: video -> MediaPipe detection -> solve -> BVH text.
    Works on any VRM. Requires the `mediapipe` env for detection (see two-stage note)."""
    from mocapy.detect.mediapipe_detect import Detector

    det = Detector()
    results = det.process(video_path, max_frames=max_frames)
    W, H = getattr(det, "frame_size", (1280, 720))
    frames, frames_px, frames_hands = [], [], []
    for r in results:
        frames_hands.append(r.hands or None)
        if r.pose_world is None or r.pose_raw is None:
            frames.append(None)
            frames_px.append(None)
        else:
            frames.append([{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                           for p in r.pose_world])
            # screen landmarks in pixel scale, z width-scaled (MediaPipe convention)
            frames_px.append([[float(p[0]) * W, float(p[1]) * H, float(p[2]) * W]
                              for p in r.pose_raw])
    return landmarks_to_bvh(frames, frames_px, (W, H), vrm_path, frames_hands=frames_hands)


def vrm_to_mmd_from_hierarchy(hierarchy_list, order):
    """Build {vrm_bone_name: mmd_name} from a golden hierarchy_list (name->name_MMD)
    joined with our BVH order [(bvh_name, vrm_name), ...]."""
    name_mmd_by_bvh = {it["name"]: it.get("name_MMD") for it in hierarchy_list}
    return {vrm: name_mmd_by_bvh.get(bvh) for bvh, vrm in order}
