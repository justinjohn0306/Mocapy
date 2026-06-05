"""Render a side-by-side demo GIF: raw video | live skeleton overlay.

Runs the default cv2 + MediaPipe detector over a clip and composes each frame as
[ original | pose+hands+face overlay ], then writes an optimized looping GIF for
the README.

Usage:
    python tools/make_demo_gif.py [video] [out.gif] [--frames N] [--stride S]
                                  [--height H] [--fps F]

Defaults render ~6 s of fixtures/test.mp4 to media/demo.gif.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root → import mocapy

from mocapy.detect.mediapipe_detect import Detector
from mocapy.detect.preview import draw_overlay

_SEP = 4          # px width of the divider between panels
_BG = (18, 18, 22)  # dark divider / letterbox colour (BGR)


def _label(panel: np.ndarray, text: str) -> None:
    """Draw a small bottom-left caption with a translucent backing."""
    h, w = panel.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(panel, (0, h - th - 10), (tw + 14, h), (0, 0, 0), -1)
    cv2.putText(panel, text, (7, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", nargs="?", default="fixtures/test.mp4")
    ap.add_argument("out", nargs="?", default="media/demo.gif")
    ap.add_argument("--frames", type=int, default=180, help="max source frames to read")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--height", type=int, default=216, help="panel height in px")
    ap.add_argument("--fps", type=int, default=15, help="GIF playback fps")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    gif_frames: list[np.ndarray] = []
    kept = {"n": 0}

    def on_frame(frame_bgr, fr, raw_pose, raw_hands, face_lms):
        if fr.index % args.stride != 0:
            return True  # keep detecting, just don't store this one

        # Right panel: overlay on a dimmed copy so the teal skeleton pops.
        over = (frame_bgr.astype(np.float32) * 0.55).astype(np.uint8)
        draw_overlay(
            over,
            pose_landmarks=fr.pose,
            hand_landmarks=raw_hands,
            face_landmarks=face_lms,
        )

        h, w = frame_bgr.shape[:2]
        pw = int(round(args.height * w / h))
        left = cv2.resize(frame_bgr, (pw, args.height), interpolation=cv2.INTER_AREA)
        right = cv2.resize(over, (pw, args.height), interpolation=cv2.INTER_AREA)
        _label(left, "input video")
        _label(right, "mocapy")

        sep = np.full((args.height, _SEP, 3), _BG, np.uint8)
        combo = np.hstack([left, sep, right])
        gif_frames.append(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB))
        kept["n"] += 1
        return True

    det = Detector(pose_model="full")
    print(f"detecting {args.video} (<= {args.frames} frames) ...")
    det.process(args.video, max_frames=args.frames, on_frame=on_frame)

    if not gif_frames:
        print("no frames captured — is the video path correct?", file=sys.stderr)
        return 1

    print(f"writing {out}  ({kept['n']} frames @ {args.fps} fps)")
    imageio.mimsave(out, gif_frames, format="GIF", fps=args.fps, loop=0)
    size_mb = out.stat().st_size / 1e6
    print(f"done: {out} ({size_mb:.1f} MB, {gif_frames[0].shape[1]}x{gif_frames[0].shape[0]})")
    if size_mb > 9:
        print("  note: >9 MB — consider --stride 3 or --height 180 to shrink for GitHub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
