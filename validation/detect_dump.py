"""Stage A (conda env w/ mediapipe): video -> frames.json (pose + hands + face).

Two-stage flow: detection needs the mediapipe conda env; solving/BVH needs base
python (conda numpy crashes in load_skeleton). This dumps everything the solver
needs.

  $ python validation/detect_dump.py [video] [output.json] [--preview]

Defaults: video=fixtures/test.mp4, output=frames.json (in cwd).

--preview opens a live window showing the MediaPipe pose/hand/face overlay as
the video is processed. Press 'q' or ESC to abort mid-run.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy.detect.mediapipe_detect import Detector  # noqa: E402
from mocapy._paths import FIXTURES  # noqa: E402


def _resolve(arg, default: Path) -> Path:
    if not arg:
        return default
    p = Path(arg)
    return p if p.is_absolute() else Path.cwd() / p


def main() -> int:
    ap = argparse.ArgumentParser(description="Video -> MediaPipe pose+hands+face -> frames.json")
    ap.add_argument("video", nargs="?", default=str(FIXTURES / "test.mp4"))
    ap.add_argument("output", nargs="?", default="frames.json")
    ap.add_argument("--preview", action="store_true",
                    help="show live MediaPipe overlay during detection")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--mirror-resize", action="store_true",
                    help="match the reference engine CLI behavior 1:1: resize to 1280x720 + mirror input. "
                         "Use this only when comparing head-to-head against the reference's the reference output; "
                         "for natural mocap (user's right hand = avatar's right hand) leave it off.")
    args = ap.parse_args()

    video = _resolve(args.video, FIXTURES / "test.mp4")
    out = _resolve(args.output, Path.cwd() / args.output)
    det = Detector()

    on_frame = None
    if args.preview:
        import cv2
        from mocapy.detect.preview import draw_overlay

        win = "mocapy detect preview"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

        def on_frame(frame_bgr, fr, raw_pose, raw_hands, face_lms):
            header = f"frame {fr.index} | t={fr.timestamp_ms/1000.0:5.2f}s | q/ESC to stop"
            draw_overlay(frame_bgr,
                         pose_landmarks=raw_pose,
                         hand_landmarks=raw_hands,
                         face_landmarks=face_lms,
                         face_bbox=fr.face_bbox,
                         hand_bbox=fr.hand_bbox,
                         header=header)
            cv2.imshow(win, frame_bgr)
            k = cv2.waitKey(1) & 0xFF
            return k not in (ord("q"), 27)

    # --mirror-resize bundles BOTH the 1280x720 resize AND the horizontal mirror —
    # the reference's CLI applies both unconditionally, so to compare 1:1 you need both.
    # Without --mirror-resize we use the video's native resolution and don't flip
    # (natural mocap convention: user's right hand drives avatar's right hand,
    # finger labels correctly aligned, etc).
    target = (1280, 720) if args.mirror_resize else None
    try:
        results = det.process(video, max_frames=args.max_frames,
                              on_frame=on_frame, target_size=target,
                              mirror=args.mirror_resize)
    finally:
        if args.preview:
            import cv2
            cv2.destroyAllWindows()

    W, H = getattr(det, "frame_size", (1280, 720))
    frames = []
    for r in results:
        frames.append({
            "world": None if r.pose_world is None else r.pose_world.tolist(),
            "raw": None if r.pose_raw is None else r.pose_raw.tolist(),
            "hands": {k: v.tolist() for k, v in (r.hands or {}).items()},
            "face": None if r.face is None else r.face.tolist(),
            "face_matrix": None if r.face_matrix is None else r.face_matrix.tolist(),
            "face_blendshapes": r.face_blendshapes,
        })
    out.write_text(json.dumps({"frame_size": [W, H], "frames": frames}), encoding="utf-8")
    nh = sum(1 for f in frames if f["hands"])
    nf = sum(1 for f in frames if f["face"])
    print(f"wrote {out}: {len(frames)} frames, {nh} with hands, {nf} with face, {W}x{H}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
