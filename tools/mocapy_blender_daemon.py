"""Mocapy realtime detection daemon — runs in the conda env, emits per-frame
landmark + solved-bone JSON to stdout so a Blender addon (running in Blender's
own Python) can apply pose to an armature.

This sidesteps the Python-version mismatch between Blender's bundled Python
(currently 3.13) and what mediapipe ships wheels for (3.12 max). The daemon
runs in the conda env that ALREADY has mediapipe + opencv + mocapy working;
Blender just consumes its stdout.

Usage (driven by the addon, but works standalone for testing):
  python tools/mocapy_blender_daemon.py --camera 0 --fps 15 --mirror

stdout protocol — one JSON object per line, UTF-8, line-buffered:
  {"type":"hello", "fps":15, "width":1280, "height":720}    (once at startup)
  {"type":"frame", "idx":N, "ts_ms":M, "bones":{"MMD_name":[x,y,z,w], ...}}
  {"type":"frame", "idx":N, "ts_ms":M, "bones":null}        (no pose detected)
  {"type":"error", "message":"..."}                         (then exit nonzero)

The "bones" map uses MMD bone names exactly as `mocapy.pipeline.solve_frame`
returns them; the addon maps those to VRM names for application to the rig.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0,
                    help="cv2.VideoCapture device index (default 0)")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="max sample rate (Hz). Lower = less CPU. Default 15")
    ap.add_argument("--mirror", action="store_true", default=False,
                    help="horizontally flip frames before detection. Default off "
                         "(natural mocap: user's right hand = avatar's right hand). "
                         "Enable only when matching the reference CLI output 1:1.")
    ap.add_argument("--no-mirror", action="store_false", dest="mirror")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    # Make `mocapy` importable when the daemon is run from outside the repo.
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    try:
        import cv2
        import mediapipe as mp
        from mocapy.detect.mediapipe_detect import Detector
        from mocapy.pipeline import solve_frame
    except Exception as e:
        emit({"type": "error", "message": f"import failed: {e!r}"})
        return 2

    try:
        # MediaPipe HandLandmarker requires num_hands>=1, so we keep the default
        # but simply don't call det.hands.* in the realtime loop (the loaded
        # models cost ~50MB RAM and a tiny init time, but no per-frame cost).
        det = Detector(face=False)
    except Exception as e:
        emit({"type": "error", "message": f"detector init failed: {e!r}"})
        return 2

    print("[daemon-debug] detector ready, opening camera...", file=sys.stderr, flush=True)
    cap = cv2.VideoCapture(args.camera)
    print(f"[daemon-debug] camera open returned, isOpened={cap.isOpened()}",
          file=sys.stderr, flush=True)
    if not cap.isOpened():
        emit({"type": "error", "message": f"cannot open camera {args.camera}"})
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print("[daemon-debug] camera configured, emitting hello", file=sys.stderr, flush=True)

    emit({"type": "hello", "fps": args.fps, "width": args.width, "height": args.height,
          "mirror": args.mirror})

    period = 1.0 / max(args.fps, 0.1)
    t0 = time.time()
    idx = 0
    next_t = t0
    try:
        while True:
            now = time.time()
            if now < next_t:
                time.sleep(max(0, next_t - now))
            next_t = now + period

            ok, frame_bgr = cap.read()
            if not ok:
                continue
            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != (args.width, args.height):
                frame_bgr = cv2.resize(frame_bgr, (args.width, args.height),
                                       interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - t0) * 1000)

            pose_res = det.pose.detect_for_video(mp_img, ts_ms)
            bones_out = None
            if pose_res.pose_world_landmarks:
                lms = pose_res.pose_world_landmarks[0]
                kp3 = [{"x": p.x, "y": p.y, "z": p.z} for p in lms]
                try:
                    rots = solve_frame(kp3)
                    bones_out = {name: [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
                                 for name, q in rots.items()}
                except Exception as e:
                    emit({"type": "warn", "idx": idx, "message": f"solve failed: {e!r}"})

            emit({"type": "frame", "idx": idx, "ts_ms": ts_ms, "bones": bones_out})
            idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
