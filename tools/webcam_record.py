"""Webcam mocap capture (Stage A, conda env).

Captures live video from a webcam, runs the same MediaPipe pose + hand + face
detection the file-based pipeline uses, and dumps the frames into the same
frames.json format that validation/solve_bvh.py and mocapy-solve consume. Live
preview shows the MediaPipe predictions overlaid (pose + hand skeletons, face
mesh dots).

  $ python tools/webcam_record.py --device 0 --duration 30 --output frames.json
  $ python validation/solve_bvh.py frames.json assets/AliciaSolid.vrm out.bvh

Press 'q' or ESC in the preview window to stop early.

Notes:
  * Capture is THROTTLED to --target-fps (default 30) regardless of what the
    webcam reports. Webcams that deliver 60fps just drop every other frame so
    timing in the output frames.json matches the 30fps BVH timeline. Without
    throttling, a 60fps webcam would record a 10-second performance into a
    20-second BVH (played at 2x duration).
  * --mirror is DEFAULT ON because most webcam users expect the "looking in a
    mirror" feel — raise right arm, the avatar raises ITS right arm. The flag
    is symmetric: it flips the captured image so MediaPipe's 2D landmarks
    follow the mirrored view. MediaPipe's pose_world (3D anatomy) is anatomy-
    relative so the avatar's bone rotations are well-defined either way; the
    visible difference is the X-direction of root translation. Use
    `--no-mirror` for face-to-face puppet behaviour.
  * Preview window is optional (--no-preview) for headless capture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _to_np(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Webcam mocap capture → frames.json")
    ap.add_argument("--device", type=int, default=0, help="webcam index (default: 0)")
    ap.add_argument("--duration", type=float, default=10.0,
                    help="max recording seconds (default: 10)")
    ap.add_argument("--output", "-o", default="frames.json", help="JSON output path")
    ap.add_argument("--width", type=int, default=1280, help="capture width (default: 1280)")
    ap.add_argument("--height", type=int, default=720, help="capture height (default: 720)")
    ap.add_argument("--target-fps", type=float, default=30.0,
                    help="output frame rate (default: 30); webcam frames delivered "
                         "faster than this are dropped to keep BVH playback at "
                         "real-time speed")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                    help="mirror-image preview & capture (default: ON). "
                         "--no-mirror gives face-to-face puppet behaviour.")
    ap.add_argument("--no-preview", action="store_true", help="run headless (no window)")
    ap.add_argument("--pose-model", choices=("full", "heavy"), default="full")
    args = ap.parse_args()

    # Deferred imports — detector needs the conda env with mediapipe + cv2 installed.
    import cv2
    import mediapipe as mp
    from mocapy.detect.mediapipe_detect import Detector
    from mocapy.detect.preview import draw_overlay

    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)  # DSHOW = faster open on Windows
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"ERROR: cannot open webcam device {args.device}", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ok, first = cap.read()
    if not ok:
        print("ERROR: webcam returned no frames", file=sys.stderr)
        return 2
    h, w = first.shape[:2]
    cam_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    print(f"capturing at {w}x{h}, webcam reports {cam_fps:.1f} fps, "
          f"throttling to {args.target_fps:.1f} fps  "
          f"(device={args.device}, mirror={args.mirror})")

    det = Detector(pose_model=args.pose_model)
    det.frame_size = (w, h)

    out_frames: list = []
    out_path = Path(args.output).resolve()
    win = "mocapy webcam capture"
    if not args.no_preview:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    target_dt = 1.0 / args.target_fps
    t0 = time.time()
    next_capture_t = t0           # wall-clock deadline for the next saved frame
    frame_idx = 0                  # index in the 30fps (target_fps) output timeline
    smoothed_fps = 0.0
    last_save_t = t0
    frame = first

    try:
        while True:
            if frame is None:
                ok, frame = cap.read()
                if not ok:
                    break

            now = time.time()
            # Throttle: if the webcam is delivering frames faster than target_fps,
            # drop them until the next 1/target_fps slot opens. This keeps the
            # frames.json count == args.duration * args.target_fps.
            if now < next_capture_t:
                frame = None
                continue
            next_capture_t = max(next_capture_t + target_dt, now)  # never accumulate lag

            if args.mirror:
                frame = cv2.flip(frame, 1)

            # 30fps virtual timestamp keyed to the output timeline (not wall clock)
            # so OneEuro behaves deterministically and matches detect_dump.py.
            ts = int(round(frame_idx * 1000.0 / args.target_fps))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pr = det.pose.detect_for_video(mp_img, ts)

            # Hand detection: pose-guided wrist crop FIRST (huge boost on distant
            # subjects — the reference's get_hand_canvas trick), then dual-confidence full-
            # frame fallback if the crop missed. See mediapipe_detect.py for the
            # full rationale.
            raw_pose_lms = pr.pose_landmarks[0] if pr.pose_landmarks else None
            hr = None
            crop_bbox = None
            if raw_pose_lms is not None:
                pose_raw_np = _to_np(raw_pose_lms)
                hc = Detector._hand_crop(rgb, pose_raw_np, w, h)
                if hc is not None:
                    crop_rgb, (cx0, cy0, ccw, cch) = hc
                    mp_crop_h = mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=np.ascontiguousarray(crop_rgb))
                    hr = det.hands_crop.detect(mp_crop_h)
                    if hr.hand_landmarks:
                        crop_bbox = (cx0, cy0, ccw, cch)
                    else:
                        hr = None
            if hr is None:
                lsh = raw_pose_lms[11] if raw_pose_lms is not None else None
                rsh = raw_pose_lms[12] if raw_pose_lms is not None else None
                sw_px = 0.0 if lsh is None else float(
                    ((lsh.x - rsh.x) * w) ** 2 + ((lsh.y - rsh.y) * h) ** 2) ** 0.5
                det._maybe_switch_hands(sw_px, max(w, h), ts)
                hr = det.hands.detect_for_video(mp_img, ts)

            # Face landmarker — crop on pose head landmarks for distant subjects.
            face_lms = face_mtx = face_bs = None
            face_bbox_used = None
            if det.face is not None and raw_pose_lms is not None:
                # pose_raw_np was already computed in the hand-crop block above
                # (under the same `raw_pose_lms is not None` guard).
                fc = Detector._face_crop(rgb, pose_raw_np, w, h)
                if fc is not None:
                    crop_rgb, (cx0, cy0, cw, ch) = fc
                    face_bbox_used = (cx0, cy0, cw, ch)
                    mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB,
                                       data=np.ascontiguousarray(crop_rgb))
                    fr_face = det.face.detect(mp_crop)
                    if fr_face.face_landmarks:
                        face_lms = _to_np(fr_face.face_landmarks[0])
                        face_lms[:, 0] = (face_lms[:, 0] * cw + cx0) / w
                        face_lms[:, 1] = (face_lms[:, 1] * ch + cy0) / h
                        if fr_face.facial_transformation_matrixes:
                            face_mtx = np.asarray(fr_face.facial_transformation_matrixes[0],
                                                  dtype=np.float64)
                        if fr_face.face_blendshapes:
                            face_bs = {c.category_name: float(c.score)
                                       for c in fr_face.face_blendshapes[0]}

            world = raw = None
            if pr.pose_landmarks:
                raw = _to_np(pr.pose_landmarks[0])
                if pr.pose_world_landmarks:
                    world = _to_np(pr.pose_world_landmarks[0])
            hands = {}
            if hr.hand_landmarks:
                for lms, handed in zip(hr.hand_landmarks, hr.handedness):
                    arr = _to_np(lms)
                    if crop_bbox is not None:
                        cx0, cy0, ccw, cch = crop_bbox
                        arr[:, 0] = (arr[:, 0] * ccw + cx0) / w
                        arr[:, 1] = (arr[:, 1] * cch + cy0) / h
                    hands[handed[0].category_name] = arr.tolist()

            out_frames.append({
                "world": None if world is None else world.tolist(),
                "raw": None if raw is None else raw.tolist(),
                "hands": hands,
                "face": None if face_lms is None else face_lms.tolist(),
                "face_matrix": None if face_mtx is None else face_mtx.tolist(),
                "face_blendshapes": face_bs,
            })

            dt = now - last_save_t
            last_save_t = now
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
            elapsed = now - t0

            if not args.no_preview:
                header = (f"frame {frame_idx} | {elapsed:5.1f}s | "
                          f"{smoothed_fps:4.1f} fps | q/ESC to stop")
                # Pass remapped numpy hand arrays (full-frame coords) so the preview
                # draws in the right place regardless of crop vs full-frame source,
                # plus the face/hand crop boxes that the reference's preview also showed.
                draw_overlay(
                    frame,
                    pose_landmarks=raw_pose_lms,
                    hand_landmarks=list(hands.values()) if hands else None,
                    face_landmarks=face_lms,
                    face_bbox=face_bbox_used,
                    hand_bbox=crop_bbox,
                    header=header,
                )
                cv2.imshow(win, frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    print("stopped by user")
                    break
            elif frame_idx % 30 == 0:
                print(f"  {elapsed:5.1f}s  frame {frame_idx}  ({smoothed_fps:.1f} fps)")

            if elapsed >= args.duration:
                print(f"duration {args.duration:.1f}s reached")
                break
            frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                print("webcam stream ended"); break
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    out_path.write_text(json.dumps({"frame_size": [w, h], "frames": out_frames}),
                        encoding="utf-8")
    nh = sum(1 for f in out_frames if f["hands"])
    nf = sum(1 for f in out_frames if f["face"])
    print(f"\nwrote {out_path.name}: {len(out_frames)} frames "
          f"(@ ~{args.target_fps:.0f} fps), {nh} with hands, {nf} with face, {w}x{h}")
    print(f"  next: mocapy-solve {out_path.name} <vrm> out.bvh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
