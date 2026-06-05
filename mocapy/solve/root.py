"""Root translation (全ての親 position) — direct port of the reference's depth pipeline.

the reference engine computes the avatar's world position via a multi-step pipeline
(`the reference solver:74833-78700`, the `at` IIFE). Decoded:

  1. Build a 2D segment from the subject's silhouette in the image:
       point2D[0] = shoulder-mid (with body-tilt compensation: `y -= s/cos(tilt)`)
       point2D[1] = hip-mid
  2. Pair it with a known 3D length:
       data3D.length = spine_length_ref · 2 · |3D-spine-midpoint|
  3. Back-project each 2D endpoint through a fixed virtual camera (FOV ≈ 50°,
     `the reference MMD notes.THREEX.camera.obj.fov`) — gives two unit rays t, o into the scene.
  4. Solve for the depth scale `l` that makes the rays' apparent separation
     match the known 3D length:
         a = sqrt((tx - ox·tz/oz)² + (ty - oy·tz/oz)²)   # 2D dist at unit z
         s = (tx - ox·tz/oz) + (ty - oy·tz/oz)           # sum of components
         r = s / a                                       # direction factor
         l = (sqrt(i² + 1) - 1) / (a / r)                # depth scale
       t.multiplyScalar(l)
  5. Final z = `-t.z · 1.5 · d` where `d` is a FOV correction factor
     (`d = 1 + (2·tan(fov/2)/0.9326 - 1)·0.5`) modulated by user's
     `hip_depth_scale_percent` (default 100 → no change).
  6. Subtract leg-length offset: `o -= hip_z_position_offset% · left_leg_length·2`
  7. Floor clamp against the rig's head/center bone positions:
         o = max(o, 5·n + (head.z - center.z))
         where n = spine_length / 4.97462
  8. Smooth with `camera_depth` OneEuro: `(30, 1, 1/5, 1, scalar)`

XY (the world X/Y offset) comes from the 2D hip-midpoint scaled by a stable
median shoulder width (the reference also derives X/Y from the same unprojected rays, but
with the same camera back-projection — equivalent up to coordinate convention).

The 0.9326153163099972 constant in the FOV correction = 2·tan(25°), i.e. it
normalizes against a 50° reference FOV. At fov=50° the correction is 1.0
(no scaling); tighter/wider FOVs scale the depth accordingly.
"""

from __future__ import annotations

import math

import numpy as np

# Default virtual-camera FOV — matches the reference's THREE.js scene camera.
DEFAULT_FOV_DEG = 50.0
# the reference's hardcoded depth magnitude multiplier (`-t.z * 1.5 * d`).
DEPTH_MAGNITUDE_MULT = 1.5
# Normalizer in the reference's FOV-correction formula. 0.9326... = 2·tan(25°), i.e. the
# value of `2·tan(fov/2)` at the reference 50° FOV. Picking the same value
# means `c = 1` at the reference FOV and we just inherit the reference's behaviour.
FOV_CORRECTION_REF = 0.9326153163099972
# `Je.spine_length_ref` in the reference — a model-dependent scale constant. For typical
# MMD humanoid models this lives around 2.5; using cj.vrm's measurements
# directly gives the same numeric range as out1.bvh.
SPINE_LENGTH_REF_DEFAULT = 2.5
# Typical human shoulder width in meters, for pose_world → BVH-unit fallback.
TYPICAL_HUMAN_SHOULDER_M = 0.40


def shoulder_width_px(shoulder_px):
    return float(np.linalg.norm(np.asarray(shoulder_px[0]) - np.asarray(shoulder_px[1])))


def _unproject_ray(px: float, py: float, frame_w: float, frame_h: float,
                   fov_deg: float) -> np.ndarray:
    """Unit ray from a perspective camera at origin (looking -Z) through pixel (px, py).

    Equivalent to THREE.js `new Vector3(ndc_x, ndc_y, 0.5).unproject(camera).sub(camera.position).normalize()`
    for a default-orientation camera. The standard pinhole-ray form: the near/far
    planes cancel out once you normalize, so we don't need to model them.
    """
    aspect = frame_w / frame_h
    tan_half = math.tan(math.radians(fov_deg) / 2.0)
    ndc_x = (px / frame_w) * 2.0 - 1.0
    ndc_y = -((py / frame_h) * 2.0 - 1.0)
    rx = ndc_x * aspect * tan_half
    ry = ndc_y * tan_half
    rz = -1.0
    n = math.sqrt(rx * rx + ry * ry + rz * rz)
    return np.array([rx / n, ry / n, rz / n])


def _depth_scale(t: np.ndarray, o: np.ndarray, i: float) -> float:
    """The `l = (sqrt(i²+1) − 1) / (a/r)` formula. Returns the depth scale to
    apply to `t` so that `t · l`'s vertical/horizontal displacement from
    `o · scale_to_match_z` is consistent with the known 3D length `i`.
    """
    tz_over_oz = t[2] / o[2] if abs(o[2]) > 1e-9 else 0.0
    dx = t[0] - o[0] * tz_over_oz
    dy = t[1] - o[1] * tz_over_oz
    a = math.sqrt(dx * dx + dy * dy)
    if a < 1e-9:
        return 1.0
    s = dx + dy
    r = s / a
    # The `(sqrt(i²+1) - 1)` is the reference's specific scaling — keeps the formula
    # well-behaved for both small and large 3D lengths.
    return (math.sqrt(i * i + 1.0) - 1.0) / (a / r)


def _fov_correction(fov_deg: float, hip_depth_scale_percent: float = 100.0) -> float:
    """the reference's `d = c * hip_depth_scale_percent/100` with `c = 1 + (2·tan(fov/2)/REF − 1)·0.5`."""
    c = 2.0 * math.tan(math.radians(fov_deg) / 2.0) / FOV_CORRECTION_REF
    c = 1.0 + (c - 1.0) * 0.5
    return c * hip_depth_scale_percent / 100.0


def _solve_depth(sh_px_pair, hp_px_pair, sh_mid_3d, hp_mid_3d,
                    frame_w, frame_h, fov_deg: float,
                    spine_length_ref: float,
                    body_tilt_rad: float = 0.0) -> float:
    """Port of the reference's main depth-computation step from `t()` in the `at` IIFE."""
    # 2D segment: point2D[0] = shoulder-mid (with tilt comp), point2D[1] = hip-mid.
    sh_mid_2d = (np.asarray(sh_px_pair[0], dtype=float)
                 + np.asarray(sh_px_pair[1], dtype=float)) * 0.5
    hp_mid_2d = (np.asarray(hp_px_pair[0], dtype=float)
                 + np.asarray(hp_px_pair[1], dtype=float)) * 0.5
    s_2d = float(np.linalg.norm(hp_mid_2d - sh_mid_2d))
    p0 = hp_mid_2d.copy()
    p0[1] -= s_2d / max(abs(math.cos(body_tilt_rad)), 1e-3)
    p1 = hp_mid_2d.copy()

    # 3D scale reference: |shoulder_midpoint_3D in pose_world| × 2 × spine_length_ref.
    # In MediaPipe's pose_world, the HIP midpoint sits at the origin (anatomy-
    # centered), so |hip_mid| ≈ 0 and is useless as a scale reference. The
    # SHOULDER midpoint, however, sits ~0.45m above (negative Y) the origin —
    # that's the actual spine length. the reference's `else` branch (`d=keypoints3D[5/6]`)
    # uses exactly this when `keypoints3D_raw` isn't available.
    spine_length_3d = float(np.linalg.norm(np.asarray(sh_mid_3d)))
    data3D_length = spine_length_ref * (spine_length_3d * 2.0)

    # Unproject endpoints to unit rays; solve for depth scale `l`.
    t = _unproject_ray(float(p0[0]), float(p0[1]), frame_w, frame_h, fov_deg)
    o = _unproject_ray(float(p1[0]), float(p1[1]), frame_w, frame_h, fov_deg)
    l = _depth_scale(t, o, data3D_length)
    d = _fov_correction(fov_deg)
    # the reference: `p = -t.z * 1.5 * d`. After multiplyScalar(l), t.z became l·rz (still
    # negative since rz = -1 in camera-forward convention). So `-t.z = positive l`,
    # i.e. depth into scene. the reference's coord convention has Z increasing INTO the scene
    # for the avatar pos, but BVH writers typically expect negative Z for "behind
    # camera." Negate to match the BVH convention seen in out1.bvh.
    return -((-t[2] * l) * DEPTH_MAGNITUDE_MULT * d)


def solve_root(hip_px, shoulder_px, frame_w, frame_h, world_shoulder_width,
               *, xy_scale=None, fov_deg: float = DEFAULT_FOV_DEG,
               pose_world_hp_l=None, pose_world_hp_r=None,
               pose_world_sh_l=None, pose_world_sh_r=None,
               spine_length_ref: float = SPINE_LENGTH_REF_DEFAULT,
               body_tilt_rad: float = 0.0,
               leg_length_offset: float = 0.0,
               z_floor: float | None = None,
               z_smoothed: float | None = None):
    """Compute 全ての親 [x, y, z] from the subject's 2D hip + pose_world data.

    the reference's `function i(e)` pipeline (the reference solver:78700-80500):
      1. Compute scene depth `h` (= z_smoothed from camera_depth filter).
      2. Unproject the 2D hip-midpoint to a unit ray through the virtual camera.
      3. Scale the ray to live at depth `h`: `C *= -h / C.z`.
      4. Read scaled (C.x, C.y, C.z) as the hip's 3D scene position.

    This gives X/Y motion that scales with depth — when the subject is far from
    the camera the same screen-pixel offset corresponds to a larger world-space
    offset, exactly the way the reference's avatar moves more broadly on distant subjects.
    Replaces the previous "constant median-shoulder scale" XY formula which
    under-magnified XY motion by 60-75% vs the reference on the same clip.

    z_smoothed: pre-smoothed depth from `camera_depth` OneEuro. When supplied,
        used as `h` in the ray scaling. When None we compute a per-frame depth
        from `_solve_depth` directly (no smoothing — only useful when the
        caller is going to filter pos[2] itself).
    """
    sw = max(shoulder_width_px(shoulder_px), 0.02 * max(frame_w, frame_h))

    # Always compute the per-frame depth from the reference's data3D formula (or the simple
    # fallback when pose_world is missing).
    if all(p is not None for p in (pose_world_hp_l, pose_world_hp_r,
                                   pose_world_sh_l, pose_world_sh_r)):
        sh_mid_3d = (np.asarray(pose_world_sh_l) + np.asarray(pose_world_sh_r)) * 0.5
        hp_mid_3d = (np.asarray(pose_world_hp_l) + np.asarray(pose_world_hp_r)) * 0.5
        depth_z = _solve_depth(shoulder_px, (hip_px, hip_px), sh_mid_3d, hp_mid_3d,
                                  frame_w, frame_h, fov_deg, spine_length_ref,
                                  body_tilt_rad=body_tilt_rad)
    else:
        focal_px = (frame_h / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        depth_z = -focal_px * world_shoulder_width / sw * DEPTH_MAGNITUDE_MULT

    # X/Y from unprojected hip-midpoint ray scaled to depth `h`. Use z_smoothed
    # when available (matches the reference's `h = z_smoothed`).
    h = abs(z_smoothed) if z_smoothed is not None else abs(depth_z)
    ray = _unproject_ray(float(hip_px[0]), float(hip_px[1]), frame_w, frame_h, fov_deg)
    if abs(ray[2]) > 1e-9:
        scale_to_h = -h / ray[2]
        x = ray[0] * scale_to_h
        y = ray[1] * scale_to_h
    else:
        # Degenerate: ray points sideways. Fall back to old constant-scale formula.
        sc = xy_scale if xy_scale is not None else world_shoulder_width / sw
        x = (hip_px[0] - frame_w / 2.0) * sc
        y = -(hip_px[1] - frame_h / 2.0) * sc

    z = depth_z - leg_length_offset
    if z_floor is not None and z < z_floor:
        z = z_floor
    return [float(x), float(y), float(z)]


# Back-compat: a few callers still import this. Kept as a thin helper.
def depth_for_shoulder(sw_px: float, frame_h: int, world_shoulder_width: float,
                       fov_deg: float = DEFAULT_FOV_DEG) -> float:
    """Simple back-projection depth — `depth = focal_px · world_size / pixel_size`."""
    focal_px = (frame_h / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    return focal_px * world_shoulder_width / max(sw_px, 1e-6)
