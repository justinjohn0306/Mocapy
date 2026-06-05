"""MediaPipe-style overlay for the live preview window.

Draws the pose skeleton (33 BlazePose connections), the hand skeleton (21-point
connections per hand), and a sparse face-mesh sample on top of a BGR frame.
Shared between `tools/webcam_record.py` and `mocapy-detect --preview` so both
surfaces look identical and the drawing code only lives in one place.

The actual MediaPipe `solutions.drawing_utils` is heavyweight and pulls in extra
dependencies; we just reproduce the same visual style with cv2 primitives.
"""

from __future__ import annotations

import cv2
import numpy as np

# BlazePose 33-keypoint connections (the MediaPipe-canonical edge list).
_POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # left arm + hand
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # right arm + hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)

# MediaPipe 21-keypoint hand connections (standard).
_HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

# A sparse face-mesh sample — drawing all 478 dots clutters the view; a few
# dozen well-chosen points (silhouette + eyes + mouth) read as "face detected"
# without overwhelming the underlying frame.
_FACE_SAMPLE: tuple[int, ...] = (
    # silhouette (subset of the FACE_OVAL connection list, vertex-only)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    # eyes (inner ring)
    33, 133, 362, 263,
    # mouth corners + center
    61, 291, 13, 14,
    # nose tip
    1, 4,
)


def _px(lm, w: int, h: int) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def _px_arr(p, w: int, h: int) -> tuple[int, int]:
    return int(p[0] * w), int(p[1] * h)


def draw_overlay(
    frame_bgr: np.ndarray,
    *,
    pose_landmarks=None,
    hand_landmarks=None,
    face_landmarks=None,
    face_bbox: tuple[int, int, int, int] | None = None,
    hand_bbox: tuple[int, int, int, int] | None = None,
    header: str | None = None,
) -> np.ndarray:
    """Overlay pose + hands + face onto `frame_bgr` (mutates in place; also returns).

    pose_landmarks: MediaPipe `pose.pose_landmarks[0]` style — list of objects with
        .x, .y in [0, 1]. Or a numpy (33, ≥2) array.
    hand_landmarks: list of per-hand landmark lists (same .x/.y shape as pose);
        accepts MediaPipe `hands.hand_landmarks` directly. Each hand gets the 21-
        point skeleton drawn.
    face_landmarks: full 478-point face-mesh, sampled down to a few dozen dots.
        Can be a MediaPipe object or a numpy (478, ≥2) array.
    face_bbox / hand_bbox: optional (x0, y0, w, h) tuples in pixel coords showing
        the pose-guided crop region the detector ran on. Drawn as a thin
        rectangle so the user sees what the detector is "looking at".
    header: optional top-bar text (frame count / fps / hint); rendered with a
        translucent black band so it stays legible on bright frames.
    """
    h, w = frame_bgr.shape[:2]

    def pt(lm):
        """Pixel-coords from a MediaPipe landmark OR a numpy/sequence (x, y)."""
        return _px(lm, w, h) if hasattr(lm, "x") else _px_arr(lm, w, h)

    # Crop boxes (drawn UNDER the skeletons so the lines stay on top).
    # Face = cyan, Hand = magenta — matches the dot/line colour of each.
    if face_bbox is not None:
        x0, y0, cw, ch = face_bbox
        cv2.rectangle(frame_bgr, (x0, y0), (x0 + cw, y0 + ch),
                      (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "FACE", (x0 + 4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1, cv2.LINE_AA)
    if hand_bbox is not None:
        x0, y0, cw, ch = hand_bbox
        cv2.rectangle(frame_bgr, (x0, y0), (x0 + cw, y0 + ch),
                      (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "HANDS", (x0 + 4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 0, 255), 1, cv2.LINE_AA)

    # Pose skeleton (green lines + dots) — the canonical MediaPipe colour scheme.
    if pose_landmarks is not None:
        pts = [pt(lm) for lm in pose_landmarks]
        for a, b in _POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame_bgr, pts[a], pts[b], (0, 220, 0), 2, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame_bgr, (x, y), 3, (0, 255, 80), -1, cv2.LINE_AA)

    # Hands (magenta skeleton — picked for high contrast against typical webcam
    # scenes; the previous orange got lost on skin tones and warm backgrounds).
    if hand_landmarks:
        for one_hand in hand_landmarks:
            pts = [pt(lm) for lm in one_hand]
            for a, b in _HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame_bgr, pts[a], pts[b], (255, 0, 255), 2, cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(frame_bgr, (x, y), 4, (255, 100, 255), -1, cv2.LINE_AA)

    # Face mesh (white dots, sparse)
    if face_landmarks is not None:
        # accept both MediaPipe and numpy-array shapes
        is_arr = not hasattr(face_landmarks[0], "x") if len(face_landmarks) else False
        for i in _FACE_SAMPLE:
            if i >= len(face_landmarks):
                continue
            lm = face_landmarks[i]
            x, y = (_px_arr(lm, w, h) if is_arr else _px(lm, w, h))
            cv2.circle(frame_bgr, (x, y), 1, (240, 240, 240), -1, cv2.LINE_AA)

    # Header bar
    if header:
        cv2.rectangle(frame_bgr, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(frame_bgr, header, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr
