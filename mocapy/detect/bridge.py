"""Detection backend using the node `@mediapipe/tasks-vision` bridge — the EXACT
WASM build the reference engine uses — instead of the Python cv2/TFLite
MediaPipe. Produces the same `frames.json` our solver consumes.

Motivation: on a head-to-head against the real GUI app, the reference's WASM build matched the
GUI's legs ~26% better than the Python cv2/TFLite path (per-frame, same solver).
The detector — not the solver — is a real part of the residual, so for a faithful
"match the GUI" run this backend is preferred.

Pipeline:
  1. `node node_bridge/detect.js` runs the three landmarkers (pose + dual-conf hands
     with pose-guided wrist crop + face crop) in headless Chromium against the
     video, mirrored to the reference's selfie convention, and writes RAW landmarks per frame.
  2. We apply our exact post-detection processing (leg-collision gate + per-landmark
     OneEuro with the reference's filter_factor + finger depth-recovery/smoothing) via
     `Detector.for_processing`, so the ONLY thing that differs from the cv2 backend
     is the landmark SOURCE. Crop-local hand landmarks get the finger prep in native
     space and are then remapped to full-frame, matching the cv2 path.

Requires Node.js + the installed `node_bridge/node_modules` (puppeteer pulls a
headless Chromium) and `ffprobe` on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from mocapy._paths import PROJECT_ROOT
from mocapy.detect.mediapipe_detect import Detector


def run_node_bridge(video, raw_out, *, target_size=(1280, 720), max_frames=None,
                    preview=False, holistic=False, pose_model="full", hand_conf=0.5):
    """Run the Chromium tasks-vision bridge -> raw landmark JSON at raw_out.

    preview=True opens a VISIBLE Chromium window drawing the GUI-style blue overlay
    (pose/legs + hands + face mesh + bounding boxes) via the same tasks-vision
    DrawingUtils the GUI uses."""
    bridge_dir = PROJECT_ROOT / "node_bridge"
    script = bridge_dir / "detect.js"
    if not script.exists():
        raise FileNotFoundError(f"node bridge not found: {script}")
    if shutil.which("node") is None:
        raise RuntimeError("`node` not on PATH — the bridge backend needs Node.js")
    args = ["node", str(script), str(Path(video).resolve()), str(Path(raw_out).resolve()),
            "--target-size", f"{int(target_size[0])}x{int(target_size[1])}"]
    if max_frames:
        args += ["--max-frames", str(int(max_frames))]
    if preview:
        args += ["--preview"]
    if holistic:
        args += ["--holistic"]
    args += ["--pose-model", "heavy" if pose_model == "heavy" else "full",
             "--hand-conf", str(float(hand_conf))]
    subprocess.run(args, cwd=str(bridge_dir), check=True)


def process_raw(raw_path, out_path, *, fps=30.0,
                leg_stabilize=True, hand_stabilize=True, zonly=False):
    """Apply our exact post-detection processing to the bridge's raw landmarks.

    zonly=True matches the reference's exact Z-only landmark filtering (for `--zonly`)."""
    d = json.loads(Path(raw_path).read_text("utf-8"))
    W, H = d["frame_size"]
    det = Detector.for_processing(leg_stabilize=leg_stabilize, hand_stabilize=hand_stabilize)
    out = []
    for f, fr in enumerate(d["frames"]):
        ts = f * 1000.0 / fps
        raw, world, vis = fr.get("raw"), fr.get("world"), fr.get("vis")
        o = {"world": None, "raw": None, "hands": {}, "face": fr.get("face"),
             "face_matrix": fr.get("face_matrix"),
             "face_blendshapes": fr.get("face_blendshapes")}
        if raw is not None and world is not None:
            pose_raw = np.asarray(raw, dtype=np.float64)
            wl = np.asarray(world, dtype=np.float64)
            v = np.asarray(vis, dtype=np.float64) if vis is not None else np.ones(33)
            _, pose_world = det.filter_pose(pose_raw, wl, v, ts, W, H, zonly=zonly)
            o["world"] = pose_world.tolist()
            o["raw"] = pose_raw.tolist()          # gated raw (matches cv2 backend)

        crops = fr.get("hand_crops") or {}
        for label, arr in (fr.get("hands") or {}).items():
            a = np.asarray(arr, dtype=np.float64)
            if hand_stabilize and label in ("Left", "Right"):
                a = det._prep_hand(a, label, ts)  # finger prep in native (crop-local) space
            bbox = crops.get(label)
            if bbox is not None:                  # then remap crop x,y -> full-frame
                x0, y0, cw, ch = bbox
                a[:, 0] = (a[:, 0] * cw + x0) / W
                a[:, 1] = (a[:, 1] * ch + y0) / H
            o["hands"][label] = a.tolist()
        out.append(o)
    Path(out_path).write_text(json.dumps({"frame_size": [W, H], "frames": out}),
                              encoding="utf-8")
    nh = sum(1 for o in out if o["hands"])
    nf = sum(1 for o in out if o["face"])
    return W, H, len(out), nh, nf


def detect_via_bridge(video, output, *, target_size=(1280, 720), max_frames=None,
                      leg_stabilize=True, hand_stabilize=True, zonly=False,
                      preview=False, holistic=False, pose_model="full", hand_conf=0.5,
                      keep_raw=False):
    """video -> (tasks-vision WASM bridge) -> processing -> frames.json at `output`."""
    raw_tmp = Path(str(output) + ".raw.json")
    run_node_bridge(video, raw_tmp, target_size=target_size, max_frames=max_frames,
                    preview=preview, holistic=holistic, pose_model=pose_model,
                    hand_conf=hand_conf)
    res = process_raw(raw_tmp, output, leg_stabilize=leg_stabilize,
                      hand_stabilize=hand_stabilize, zonly=zonly)
    if not keep_raw:
        try:
            raw_tmp.unlink()
        except OSError:
            pass
    return res
