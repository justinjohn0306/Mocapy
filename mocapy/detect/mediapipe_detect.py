"""MediaPipe Tasks detection over a video, matching the reference engine's config.

Pose : pose_landmarker_full.task | VIDEO | num_poses=1 | conf 0.5
Hands: hand_landmarker.task       | VIDEO | num_hands=2 | conf 0.5
Face : face_landmarker.task       | VIDEO | num_faces=1 | conf 0.5
       + facial_transformation_matrixes + face_blendshapes (52 weights)
Smoothing: OneEuroFilter(30,1,1,2,type=3) per pose landmark, with beta & dCutOff
scaled by filter_factor = clamp(max(w,h)/shoulder_width) (the reference detector:810-828).

Runtime: needs the `mediapipe` package (use the `mocapy` conda env). Uses CPU delegate
by default for portability; the JS uses GPU (negligible numeric difference, but not
bit-identical).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .filters import OneEuroFilter

# MediaPipe pose landmark indices for shoulders (the reference engine's keypoints 5/6).
_L_SHOULDER, _R_SHOULDER = 11, 12

# Lower-body chain (hips, knees, ankles, heels, toes) — the landmarks that go
# noisy / snap onto the opposite leg when legs cross or collide. These are the
# only ones the confidence gate touches.
_LOWER_BODY_IDX = (23, 24, 25, 26, 27, 28, 29, 30, 31, 32)

# Confidence-gate thresholds (MediaPipe `visibility`, 0..1). At/above VIS_HI the
# gate is a pure pass-through. Between VIS_LO and VIS_HI the landmark may still move
# to its detected position *exactly* (no lag) as long as its per-frame step stays
# within a confidence-scaled budget; only jumps that EXCEED the budget (the
# left/right swap teleport) get clamped back. STEP_* are that budget as a fraction
# of shoulder width: STEP_MAX near full confidence, STEP_MIN at VIS_LO (a leg that
# is heavily occluded can still creep but cannot snap onto the other leg).
_GATE_VIS_LO, _GATE_VIS_HI = 0.5, 0.85
_GATE_STEP_MIN, _GATE_STEP_MAX = 0.12, 0.8

from mocapy._paths import MODELS  # noqa: E402  (project models dir)


@dataclass
class FrameResult:
    index: int
    timestamp_ms: int
    pose: np.ndarray | None          # (33,3) normalized x,y,z (smoothed)
    pose_world: np.ndarray | None    # (33,3) world meters (smoothed)
    pose_raw: np.ndarray | None      # (33,3) normalized, pre-filter
    hands: dict = field(default_factory=dict)         # {'Left'|'Right': (21,3) normalized}
    face: np.ndarray | None = None                    # (478,3) normalized face landmarks (468 mesh + 10 iris)
    face_matrix: np.ndarray | None = None             # (4,4) facial_transformation_matrix
    face_blendshapes: dict | None = None              # {name: weight} 52 ARKit-style blendshape weights
    # Pose-guided crop bounding boxes (pixel-space x0,y0,w,h) used for the
    # respective detectors this frame — None when the crop wasn't used (no pose,
    # or full-frame fallback path). Carried for preview-overlay drawing.
    face_bbox: tuple | None = None
    hand_bbox: tuple | None = None


class Detector:
    def __init__(self, *, pose_model="full", num_hands=2,
                 pose_conf=0.5, hand_conf=0.5, face=True, face_conf=0.5,
                 delegate="CPU", leg_stabilize=True, hand_stabilize=True):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        deleg = (python.BaseOptions.Delegate.GPU if delegate.upper() == "GPU"
                 else python.BaseOptions.Delegate.CPU)

        pose_path = MODELS / f"pose_landmarker_{'heavy' if pose_model == 'heavy' else 'full'}.task"
        hand_path = MODELS / "hand_landmarker.task"
        face_path = MODELS / "face_landmarker.task"
        for p in (pose_path, hand_path):
            if not p.exists():
                raise FileNotFoundError(p)
        self._face_enabled = bool(face) and face_path.exists()

        self._vision = vision
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(pose_path), delegate=deleg),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=pose_conf,
                min_pose_presence_confidence=pose_conf,
                min_tracking_confidence=pose_conf,
            )
        )
        # the reference engine runs TWO hand landmarkers in parallel — one at the standard 0.5
        # confidence threshold (close subjects) and one at 0.1 (distant subjects whose
        # hands appear small). The active detector is chosen per-frame by shoulder-
        # width ratio: when max(w,h)/shoulder_width > 7.5 the person is far enough
        # that we switch to the lower-threshold detector to actually find the hands.
        # This is the difference between "no fingers in the preview" and "fingers
        # tracked across the whole room". See the reference detector:605-660.
        def _mk_hands(conf):
            return vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(hand_path), delegate=deleg),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=num_hands,
                    min_hand_detection_confidence=conf,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=conf,
                )
            )
        self._hands_conf_levels = (max(hand_conf, 0.5), 0.1)  # (high, low)
        self._hands_pair = [_mk_hands(c) for c in self._hands_conf_levels]
        self._hands_active = 0  # index into _hands_pair
        self._hands_last_switch_ms = -10_000
        # `self.hands` always points at the currently active detector so existing
        # callers (detect_for_video) keep working without code changes.
        self.hands = self._hands_pair[0]
        # IMAGE-mode landmarker for pose-guided WRIST CROP detection — the reference's
        # `get_hand_canvas(pose)` trick (the reference detector:1276-1360). When the
        # subject is far from the camera, hands are tiny in the full frame and the
        # full-frame detector misses them. Cropping a tight window around the pose's
        # wrist landmarks makes the hand appear large to the detector. IMAGE mode
        # (instead of VIDEO) because crops have no useful temporal continuity
        # between frames; confidence 0.1 because the crop is the "permissive" path.
        self.hands_crop = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(hand_path), delegate=deleg),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=num_hands,
                min_hand_detection_confidence=0.1,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.1,
            )
        )
        # Face landmarker runs in IMAGE mode on a CROP from pose head landmarks (nose +
        # eyes + ears) — distant faces in full-body shots are too small for the built-in
        # face detector at full frame, so we crop to where the pose model says the head
        # is. This is what the reference engine's legacy mp.solutions.face_mesh did internally
        # (its built-in face detector preamble), except we use higher-quality crop
        # coordinates from the already-running pose landmarker.
        self.face = None
        if self._face_enabled:
            self.face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(face_path), delegate=deleg),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=face_conf,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                )
            )

        # One filter per landmark for normalized + world, type 3 (vector).
        self._f_lm = [OneEuroFilter(30, 1, 1, 2, 3) for _ in range(33)]
        self._f_world = [OneEuroFilter(30, 1, 1, 2, 3) for _ in range(33)]
        # Confidence-gate the lower-body chain against occlusion glitches when
        # legs cross/collide (no-op at high visibility — see _stabilize).
        self.leg_stabilize = bool(leg_stabilize)
        self._lower_body = set(_LOWER_BODY_IDX)

        # Hand-landmark preprocessing (port of the reference detector:1179-1255):
        # depth recovery for foreshortened/collapsed fingers + heavy palm-relative
        # temporal smoothing. the reference runs this on every hand before the finger solver;
        # our solver was validated against the reference's *processed* hand points, so without
        # it production fed the solver raw (noisier, depth-collapsed) landmarks.
        # One persistent OneEuro(30, 1, 1/1000, 1, vector) per landmark per side —
        # beta≈0 makes it a strong, near-velocity-independent low-pass.
        self.hand_stabilize = bool(hand_stabilize)
        self._hand_filters = {
            side: [OneEuroFilter(30, 1, 1.0 / 1000, 1, 3) for _ in range(21)]
            for side in ("Left", "Right")
        }

    @classmethod
    def for_processing(cls, *, leg_stabilize=True, hand_stabilize=True):
        """A Detector that has the filtering/gate/hand-prep state but loads NO
        MediaPipe models — for applying our exact post-detection processing to raw
        landmarks acquired elsewhere (e.g. the node `@mediapipe/tasks-vision`
        bridge). See `mocapy.detect.bridge`."""
        self = cls.__new__(cls)
        self._f_lm = [OneEuroFilter(30, 1, 1, 2, 3) for _ in range(33)]
        self._f_world = [OneEuroFilter(30, 1, 1, 2, 3) for _ in range(33)]
        self.leg_stabilize = bool(leg_stabilize)
        self._lower_body = set(_LOWER_BODY_IDX)
        self.hand_stabilize = bool(hand_stabilize)
        self._hand_filters = {
            side: [OneEuroFilter(30, 1, 1.0 / 1000, 1, 3) for _ in range(21)]
            for side in ("Left", "Right")
        }
        return self

    def filter_pose(self, pose_raw, world, vis, ts, w, h, *, zonly=False):
        """Apply the leg-collision gate + per-landmark OneEuro to one frame's raw
        pose. `pose_raw`/`world` are (33,3) float arrays (mutated in place by the
        gate); `vis` is the (33,) visibility array; returns the filtered
        (pose_normalized, pose_world). Shared by the cv2 and bridge backends so both
        smooth identically.

        zonly=True reproduces the reference's exact landmark filtering (the reference detector:390-
        404): the OneEuro is applied to ONLY the z component of both the normalized
        and world landmarks — x/y pass through unfiltered. Use for the closest match
        to the GUI (`--zonly`); our default filters full XYZ."""
        wh = max(w, h)
        if self.leg_stabilize:
            sw_norm = float(np.hypot(pose_raw[_L_SHOULDER, 0] - pose_raw[_R_SHOULDER, 0],
                                     pose_raw[_L_SHOULDER, 1] - pose_raw[_R_SHOULDER, 1]))
            if sw_norm <= 1e-6:
                sw_norm = 0.2
            sw_world = (float(np.linalg.norm(world[_L_SHOULDER] - world[_R_SHOULDER]))
                        if world is not None else 0.3) or 0.3
            for i in _LOWER_BODY_IDX:
                pose_raw[i] = self._stabilize(pose_raw[i], self._f_lm[i].x.s, vis[i], sw_norm)
                if world is not None:
                    world[i] = self._stabilize(world[i], self._f_world[i].x.s, vis[i], sw_world)
        # Dynamic filter factor from shoulder width (px).
        px = pose_raw.copy(); px[:, 0] *= w; px[:, 1] *= h; px[:, 2] *= w
        d = px[_L_SHOULDER] - px[_R_SHOULDER]; d[2] /= 3.0
        sw = float(np.linalg.norm(d))
        ff = 1.0
        if sw > 0:
            ff = wh / sw
            ff = 1.0 if ff < 5 else min(ff / 5.0, 3.0)
        for i in range(33):
            self._f_lm[i].beta = ff; self._f_lm[i].d_cutoff = 2 * ff
            self._f_world[i].beta = ff; self._f_world[i].d_cutoff = 2 * ff
        if zonly:
            # the reference-exact: filter only z; x/y untouched (the reference detector:390-404).
            pose = pose_raw.copy()
            for i in range(33):
                pose[i, 2] = self._f_lm[i].filter([0.0, 0.0, float(pose_raw[i, 2])], ts)[2]
            pose_world = None
            if world is not None:
                pose_world = world.copy()
                for i in range(33):
                    pose_world[i, 2] = self._f_world[i].filter([0.0, 0.0, float(world[i, 2])], ts)[2]
            return pose, pose_world
        pose = np.array([self._f_lm[i].filter(list(pose_raw[i]), ts) for i in range(33)])
        pose_world = None
        if world is not None:
            pose_world = np.array([self._f_world[i].filter(list(world[i]), ts) for i in range(33)])
        return pose, pose_world

    @staticmethod
    def _to_np(landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)

    def _prep_hand(self, h, label, ts):
        """Port of the reference's per-hand landmark prep (the reference detector:1179-1255).

        Operates in the landmarker's native (crop-local, x/y/z-consistent) space —
        i.e. BEFORE any crop→full-frame remap — because all three stages compare
        x/y/z magnitudes against each other. `h` is the (21,3) hand array, mutated
        and returned. MediaPipe hand topology: 0=wrist, then thumb/index/middle/
        ring/pinky as 4 joints each (idx f*4+1 .. f*4+4).

        Stage 1 — palm aspect-ratio Z-correction: MediaPipe's hand depth collapses
          when the palm tilts toward/away from camera. Restore plausible palm
          proportions by rescaling every landmark's z.
        Stage 2 — per-bone minimum-length Z recovery: a finger segment shorter than
          a fraction of the reference length is pushed out along z (it + every joint
          beyond it) so the bone reaches a believable length instead of folding flat.
        Stage 3 — palm-relative heavy OneEuro smoothing: subtract the wrist, filter,
          add it back, so finger SHAPE is smoothed without the hand's global motion
          leaking into (or lagging) the smoother.
        """
        # --- Stage 1: palm aspect-ratio z-correction ---
        pw = h[1] - h[17]          # palm width  (index-MCP -> pinky-MCP)
        ph = h[0] - h[9]           # palm height (wrist -> middle-MCP)
        w_palm = float(np.linalg.norm(pw))
        h_palm = float(np.linalg.norm(ph))
        if w_palm > 1e-9 and h_palm > 1e-9:
            ratio = h_palm / w_palm
            # Correct only when the palm aspect is abnormal (too wide or too tall);
            # inside the normal band [1.25, 1.75] the reference leaves it alone.
            a = 1.25 if ratio < 1.25 else (1.75 if ratio > 1.75 else 1.0)
            if a != 1.0:
                adjust_max = max(abs(ph[2] / h_palm), abs(pw[2] / w_palm))
                s = a * a
                num = (ph[0] ** 2 + ph[1] ** 2) / s - (pw[0] ** 2 + pw[1] ** 2)
                den = pw[2] ** 2 - ph[2] ** 2 / s
                val = abs(num / den) if abs(den) > 1e-12 else float("inf")
                a2 = min(math.sqrt(val), 1.5 + 1.5 * adjust_max)
                h[:, 2] *= a2

        # --- Stage 2: per-bone minimum-length z recovery ---
        palm0 = h[0]
        for f_idx in range(5):
            finger = [h[f_idx * 4 + 1 + idx] for idx in range(4)]  # views into h
            dx = finger[0][0] - palm0[0]
            dy = finger[0][1] - palm0[1]
            # NOTE: upstream uses [1] (y) here, not [2] (z) — an the reference quirk we keep so
            # our processed landmarks match the ones the solver was validated on.
            dz = finger[0][1] - palm0[1]
            ref_length = math.sqrt(dx * dx + dy * dy + dz * dz) * (2.0 if f_idx == 0 else 0.75) * 0.5
            for i in range(3):
                f1 = finger[i + 1] - finger[i]
                min_length = ref_length * (0.4 if i < 2 else 0.2)
                if float(f1[0] ** 2 + f1[1] ** 2 + f1[2] ** 2) < min_length * min_length:
                    z_mod = np.sign(f1[2]) * math.sqrt(max(min_length * min_length
                                                           - (f1[0] ** 2 + f1[1] ** 2), 0.0))
                    for j in range(i + 1, 4):
                        finger[j][2] += z_mod

        # --- Stage 3: palm-relative heavy OneEuro smoothing ---
        filt = self._hand_filters.get(label)
        if filt is not None:
            palm0 = h[0].copy()
            for idx in range(21):
                sm = filt[idx].filter(list(h[idx] - palm0), ts)
                h[idx] = np.asarray(sm, dtype=np.float64) + palm0
        return h

    @staticmethod
    def _stabilize(raw, prev, vis, scale):
        """Confidence-gate one landmark before it enters the OneEuro filter.

        `raw`  : detected landmark this frame (3-vector).
        `prev` : last *smoothed* output of this landmark's filter (or None at start).
        `vis`  : MediaPipe visibility 0..1 for this landmark.
        `scale`: body scale in the same space (shoulder width) for the jump clamp.

        The landmark is allowed to track its detected position exactly (no lag) as
        long as its per-frame step is within a budget that scales with confidence;
        only a step that exceeds the budget is clamped back along its own direction.
        So ordinary leg motion — even at moderate confidence — passes through
        untouched, and the gate bites only on the implausible jump that marks an
        occlusion glitch or a left/right swap. A no-op at/above VIS_HI.
        """
        if prev is None:
            return raw
        w = (vis - _GATE_VIS_LO) / (_GATE_VIS_HI - _GATE_VIS_LO)
        if w >= 1.0:
            return raw                       # confident -> exact pass-through
        w = max(w, 0.0)
        prev = np.asarray(prev, dtype=np.float64)
        step = np.asarray(raw, dtype=np.float64) - prev
        dist = float(np.linalg.norm(step))
        max_step = (_GATE_STEP_MIN + (_GATE_STEP_MAX - _GATE_STEP_MIN) * w) * scale
        if 0.0 < max_step < dist:
            return prev + step * (max_step / dist)   # clamp the teleport
        return raw                           # within budget -> trust detection

    def _maybe_switch_hands(self, shoulder_px: float, frame_max: float, ts_ms: int) -> None:
        """Pick high- vs low-confidence hand detector by subject distance.

        Mirrors the reference detector's `set_score`: when max(w,h)/shoulder_width > 7.5
        the subject is far enough that the standard 0.5 detector misses the hands —
        switch to the 0.1 detector. 1-second debounce keeps it from flapping.
        """
        if shoulder_px <= 0:
            return
        ratio = frame_max / shoulder_px
        # `s` in [0,1]; ceil(s) maps {<=7.5 → 0, >7.5 → 1}.
        want = 0 if ratio <= 7.5 else 1
        if want != self._hands_active and (ts_ms - self._hands_last_switch_ms) > 1000:
            self._hands_active = want
            self._hands_last_switch_ms = ts_ms
            self.hands = self._hands_pair[want]

    # Pose landmarks that bound the head: nose, eye-inner, eye-outer, ear (both sides).
    _HEAD_POSE_IDX = (0, 1, 2, 3, 4, 5, 6, 7, 8)

    @classmethod
    def _face_crop(cls, rgb, pose_raw, w, h, *, pad=2.0):
        """Return (crop_rgb, (x0,y0,cw,ch)) for the head region, or None if too small.

        Builds a square-ish crop around the pose head landmarks with `pad` * span margin
        so the face mesh detector sees the face at a reasonable scale even when the
        person is far from the camera."""
        idx = cls._HEAD_POSE_IDX
        hp = pose_raw[list(idx)].copy()
        hp[:, 0] *= w; hp[:, 1] *= h
        cx = float(np.mean(hp[:, 0])); cy = float(np.mean(hp[:, 1]))
        span = float(max(hp[:, 0].max() - hp[:, 0].min(),
                         hp[:, 1].max() - hp[:, 1].min()))
        if span < 5:
            return None
        half = max(span * pad, 80.0)
        x0 = max(int(cx - half), 0); x1 = min(int(cx + half), w)
        y0 = max(int(cy - half * 1.2), 0); y1 = min(int(cy + half * 0.9), h)
        cw, ch = x1 - x0, y1 - y0
        if cw < 64 or ch < 64:
            return None
        return rgb[y0:y1, x0:x1], (x0, y0, cw, ch)

    # Pose landmark indices for the wrists (MediaPipe BlazePose 33).
    _L_WRIST_IDX, _R_WRIST_IDX = 15, 16

    @classmethod
    def _hand_crop(cls, rgb, pose_raw, w, h):
        """Crop a region covering both wrists (with shoulder-width padding) for
        pose-guided hand detection. Returns (crop_rgb, (x0, y0, cw, ch)) or None
        when the wrists are clipped off-screen or the resulting crop is too small.
        """
        lw = pose_raw[cls._L_WRIST_IDX]
        rw = pose_raw[cls._R_WRIST_IDX]
        sh_l = pose_raw[_L_SHOULDER]
        sh_r = pose_raw[_R_SHOULDER]
        # Normalized shoulder width — gives a rough body scale for padding.
        sw_norm = float(np.hypot(sh_l[0] - sh_r[0], sh_l[1] - sh_r[1]))
        pad = max(sw_norm * 0.8, 0.08)  # never smaller than ~8% of frame
        cx_min = min(lw[0], rw[0]) - pad
        cx_max = max(lw[0], rw[0]) + pad
        cy_min = min(lw[1], rw[1]) - pad
        cy_max = max(lw[1], rw[1]) + pad
        x0 = max(int(cx_min * w), 0)
        y0 = max(int(cy_min * h), 0)
        x1 = min(int(cx_max * w), w)
        y1 = min(int(cy_max * h), h)
        cw, ch = x1 - x0, y1 - y0
        if cw < 64 or ch < 64:
            return None
        return rgb[y0:y1, x0:x1], (x0, y0, cw, ch)

    def process(self, video_path, *, max_frames=None, fps=None, on_frame=None,
                target_size=None, mirror=False):
        """Run detection over a video file. Returns list[FrameResult].

        on_frame: optional `(frame_bgr, FrameResult, raw_pose_lms, raw_hand_lms,
            raw_face_lms) -> bool` callback fired after each frame's detection;
            return False to abort early. Lets callers drive a live preview
            window without re-implementing the capture loop.

        target_size: optional (W, H) tuple — resize each frame to these
            dimensions before running detection. Pass (1280, 720) to match
            the reference engine Electron CLI's internal canvas size. Defaults to the
            video's native resolution.

        mirror: default False — pass True to horizontally flip every frame
            before MediaPipe inference. the reference's CLI pipeline assumes a mirrored
            ("selfie") source — set this when comparing 1:1 against the reference's
            `the reference output`. For natural mocap where "user's right hand =
            avatar's right hand", leave it off (this is the friendlier default
            for most users; the mirror also swaps which hand drives which set
            of finger bones).
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        src_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if target_size is not None:
            w, h = int(target_size[0]), int(target_size[1])
        else:
            w, h = src_w, src_h
        wh = max(w, h)
        self.frame_size = (w, h)

        import mediapipe as mp

        results: list[FrameResult] = []
        idx = 0
        aborted = False
        while True:
            ok, frame_bgr = cap.read()
            if not ok or (max_frames is not None and idx >= max_frames):
                break
            ts = round(idx * 1000.0 / src_fps)
            if target_size is not None and (src_w, src_h) != (w, h):
                frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
            if mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pr = self.pose.detect_for_video(mp_img, ts)

            pose = pose_world = pose_raw = None
            sw = 0.0
            if pr.pose_landmarks:
                pose_raw = self._to_np(pr.pose_landmarks[0])
                world = self._to_np(pr.pose_world_landmarks[0]) if pr.pose_world_landmarks else None

                # Dynamic filter factor from shoulder width in pixels.
                px = pose_raw.copy()
                px[:, 0] *= w
                px[:, 1] *= h
                px[:, 2] *= w
                d = px[_L_SHOULDER] - px[_R_SHOULDER]
                d[2] /= 3.0
                sw = float(np.linalg.norm(d))
                ff = 1.0
                if sw > 0:
                    ff = wh / sw
                    ff = 1.0 if ff < 5 else min(ff / 5.0, 3.0)
                for i in range(33):
                    self._f_lm[i].beta = ff
                    self._f_lm[i].d_cutoff = 2 * ff
                    self._f_world[i].beta = ff
                    self._f_world[i].d_cutoff = 2 * ff

                # Confidence-gate the lower body so colliding/occluded legs hold
                # steady instead of snapping onto each other. `vis` per landmark;
                # body scale = shoulder width (normalized + world spaces). Gating
                # is a no-op wherever visibility is high, so clean frames are
                # unchanged. Applied to BOTH the stored raw (drives root/hip
                # translation) and the filter inputs (drive every rotation).
                vis = np.array([getattr(lm, "visibility", 1.0)
                                for lm in pr.pose_landmarks[0]], dtype=np.float64)
                if self.leg_stabilize:
                    sw_norm = float(np.hypot(pose_raw[_L_SHOULDER, 0] - pose_raw[_R_SHOULDER, 0],
                                             pose_raw[_L_SHOULDER, 1] - pose_raw[_R_SHOULDER, 1]))
                    if sw_norm <= 1e-6:
                        sw_norm = 0.2
                    sw_world = (float(np.linalg.norm(world[_L_SHOULDER] - world[_R_SHOULDER]))
                                if world is not None else 0.3) or 0.3
                    for i in _LOWER_BODY_IDX:
                        pose_raw[i] = self._stabilize(pose_raw[i], self._f_lm[i].x.s,
                                                      vis[i], sw_norm)
                        if world is not None:
                            world[i] = self._stabilize(world[i], self._f_world[i].x.s,
                                                       vis[i], sw_world)

                pose = np.array([self._f_lm[i].filter(list(pose_raw[i]), ts) for i in range(33)])
                if world is not None:
                    pose_world = np.array([self._f_world[i].filter(list(world[i]), ts) for i in range(33)])

            # Hand detection — pose-guided wrist crop FIRST (the reference's `get_hand_canvas`
            # trick: hands are tiny in distant full-frame shots, but huge in a tight
            # crop around the wrists). If the crop misses (or there's no pose), fall
            # back to full-frame VIDEO detection with the dual-confidence adaptive
            # switch. Crop landmarks get re-mapped back to full-frame normalized
            # coordinates so downstream code sees one consistent space.
            hr = None
            crop_bbox = None
            if pose_raw is not None:
                hc = self._hand_crop(rgb, pose_raw, w, h)
                if hc is not None:
                    crop_rgb, (cx0, cy0, ccw, cch) = hc
                    mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB,
                                       data=np.ascontiguousarray(crop_rgb))
                    hr = self.hands_crop.detect(mp_crop)
                    if hr.hand_landmarks:
                        crop_bbox = (cx0, cy0, ccw, cch)
                    else:
                        hr = None
            if hr is None:
                self._maybe_switch_hands(sw, wh, ts)
                hr = self.hands.detect_for_video(mp_img, ts)

            hands = {}
            if hr.hand_landmarks:
                for lms, handed in zip(hr.hand_landmarks, hr.handedness):
                    label = handed[0].category_name  # 'Left' / 'Right'
                    arr = self._to_np(lms)
                    # Finger depth-recovery + palm-relative smoothing in the
                    # landmarker's native space, BEFORE the crop→full-frame remap
                    # (the prep compares x/y/z magnitudes, so they must share scale).
                    if self.hand_stabilize:
                        arr = self._prep_hand(arr, label, ts)
                    if crop_bbox is not None:
                        cx0, cy0, ccw, cch = crop_bbox
                        arr[:, 0] = (arr[:, 0] * ccw + cx0) / w
                        arr[:, 1] = (arr[:, 1] * cch + cy0) / h
                    hands[label] = arr

            face_lms = face_mtx = face_bs = None
            face_bbox_used = None
            if self.face is not None and pose_raw is not None:
                # Crop the face region using pose head landmarks (nose, eyes, ears).
                fc = self._face_crop(rgb, pose_raw, w, h)
                if fc is not None:
                    crop_rgb, (cx0, cy0, cw, ch) = fc
                    face_bbox_used = (cx0, cy0, cw, ch)
                    mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB,
                                       data=np.ascontiguousarray(crop_rgb))
                    fr = self.face.detect(mp_crop)
                    if fr.face_landmarks:
                        # Re-map normalized crop-relative xy back to FULL-FRAME normalized.
                        face_lms = self._to_np(fr.face_landmarks[0])
                        face_lms[:, 0] = (face_lms[:, 0] * cw + cx0) / w
                        face_lms[:, 1] = (face_lms[:, 1] * ch + cy0) / h
                        # z stays in face-mesh internal scale (proportional to face size)
                        if fr.facial_transformation_matrixes:
                            face_mtx = np.asarray(fr.facial_transformation_matrixes[0], dtype=np.float64)
                        if fr.face_blendshapes:
                            face_bs = {c.category_name: float(c.score)
                                       for c in fr.face_blendshapes[0]}

            fr_result = FrameResult(idx, ts, pose, pose_world, pose_raw, hands,
                                    face_lms, face_mtx, face_bs,
                                    face_bbox=face_bbox_used,
                                    hand_bbox=crop_bbox)
            results.append(fr_result)
            if on_frame is not None:
                # Pass full-frame-normalized arrays so the preview draws regardless
                # of whether the hand result came from the crop or full-frame path.
                # The `preview.draw_overlay` function handles both protobuf objects
                # (.x/.y) and numpy-row shapes interchangeably.
                raw_pose = pr.pose_landmarks[0] if pr.pose_landmarks else None
                raw_hands = list(hands.values()) if hands else None
                if not on_frame(frame_bgr, fr_result, raw_pose, raw_hands, face_lms):
                    aborted = True
                    break
            idx += 1

        cap.release()
        if aborted:
            print(f"  detection aborted at frame {idx} (caller requested stop)")
        return results
