"""CLI entrypoints declared in pyproject.toml.

After `pip install -e .` you get these console commands:

    mocapy-detect  <video>  [out.json]   [--backend cv2|bridge] [--zonly]
    mocapy-solve   [frames.json]  [vrm]  [out.bvh]           ('-' = bundled skeleton)
    mocapy-webcam  [--device N] [--duration S] [--output frames.json] [--mirror] ...

Each command is a thin wrapper around the matching script in validation/ or tools/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _resolve(arg, default: Path) -> Path:
    if not arg:
        return default
    p = Path(arg)
    return p if p.is_absolute() else Path.cwd() / p


# ────────────────────────────────────────────────────────────────────────────
# mocapy-detect
# ────────────────────────────────────────────────────────────────────────────
def detect_main(argv=None) -> int:
    """Detect from a VIDEO FILE -> frames.json (Stage A)."""
    ap = argparse.ArgumentParser(prog="mocapy-detect",
                                 description="Video file -> MediaPipe pose+hands+face -> frames.json")
    ap.add_argument("video", help="input video file (e.g. fixtures/test.mp4)")
    ap.add_argument("output", nargs="?", default="frames.json",
                    help="output JSON (default: frames.json in cwd)")
    ap.add_argument("--pose-model", choices=("full", "heavy"), default="full",
                    help="pose model. 'heavy' is markedly more accurate on the limbs "
                         "(legs, arms/forearms) and gives steadier wrists -> better "
                         "finger crops, at ~3x the cost. Works for cv2 and bridge.")
    ap.add_argument("--hand-conf", type=float, default=0.5,
                    help="(bridge) hand-detection confidence; lower (e.g. 0.3) catches "
                         "more/faster hands for stronger finger pickup.")
    ap.add_argument("--strong", action="store_true",
                    help="stronger detection preset: pose-model=heavy + hand-conf=0.3. "
                         "Best limb (leg/arm/forearm) + finger detection; slower.")
    ap.add_argument("--backend", choices=("cv2", "bridge"), default="cv2",
                    help="detection backend. 'cv2' (default) = Python MediaPipe "
                         "(TFLite, native). 'bridge' = the node @mediapipe/tasks-vision "
                         "WASM build the reference engine's GUI uses, run in headless Chromium — "
                         "matches the GUI's landmarks (esp. legs) more closely; needs "
                         "Node.js + node_bridge/node_modules + ffprobe, and always uses "
                         "the reference's 1280x720 + mirror convention.")
    ap.add_argument("--zonly", action="store_true",
                    help="(bridge backend) Z-only landmark filtering: OneEuro on the "
                         "depth (z) component of the pose+world landmarks only, leaving "
                         "x/y unfiltered, and disable the leg-collision gate. A "
                         "stricter, lower-latency filtering profile.")
    ap.add_argument("--holistic", action="store_true",
                    help="(bridge backend) use the reference's fast HolisticLandmarker — ONE "
                         "combined model (pose+face+hands, internal ROI cropping) "
                         "instead of separate pose + dual-hand + face + manual crops. "
                         "Much lighter/faster (runs on low-end GPUs), matches the reference's "
                         "use_holistic_landmarker path. Needs models/holistic_landmarker.task.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="limit detection to first N frames (debug)")
    ap.add_argument("--preview", action="store_true",
                    help="show live MediaPipe overlay (pose+hand+face) while detecting; "
                         "press q or ESC to abort mid-run")
    ap.add_argument("--no-leg-stabilize", action="store_true",
                    help="disable the lower-body confidence gate. By default, when "
                         "legs cross/collide and MediaPipe's leg-landmark confidence "
                         "drops, the occluded knee/ankle is held steady instead of "
                         "snapping onto the other leg (no-op on clearly-visible frames). "
                         "Pass this to get raw MediaPipe landmarks unchanged.")
    ap.add_argument("--no-hand-stabilize", action="store_true",
                    help="disable hand-landmark preprocessing. By default each hand "
                         "gets the reference engine's finger depth-recovery (un-collapses fingers "
                         "pointing toward/away from camera) plus heavy palm-relative "
                         "temporal smoothing, which steadies the fingers and matches the "
                         "processed landmarks the solver was validated on. Pass this for "
                         "raw MediaPipe hand landmarks.")
    args = ap.parse_args(argv)
    # --strong preset: heavy pose + permissive hands (best limbs + fingers).
    if args.strong:
        args.pose_model = "heavy"
        args.hand_conf = min(args.hand_conf, 0.3)

    video = _resolve(args.video, Path.cwd() / args.video)
    out = _resolve(args.output, Path.cwd() / args.output)
    if not video.exists():
        print(f"ERROR: video not found: {video}", file=sys.stderr); return 2

    # Bridge backend: the reference's exact @mediapipe/tasks-vision WASM via headless Chromium.
    if args.backend == "bridge":
        from mocapy.detect.bridge import detect_via_bridge
        t0 = time.time()
        # --zonly uses Z-only landmark filtering and turns off the leg gate.
        leg_stab = (not args.no_leg_stabilize) and (not args.zonly)
        try:
            W, H, n, nh, nf = detect_via_bridge(
                video, out, max_frames=args.max_frames,
                pose_model=args.pose_model, hand_conf=args.hand_conf,
                leg_stabilize=leg_stab,
                hand_stabilize=not args.no_hand_stabilize,
                zonly=args.zonly, preview=args.preview, holistic=args.holistic)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"ERROR: bridge backend: {e}", file=sys.stderr); return 2
        print(f"wrote {out}: {n} frames, {nh} with hands, {nf} with face, "
              f"{W}x{H} [tasks-vision bridge], {time.time()-t0:.1f}s")
        return 0

    from mocapy.detect.mediapipe_detect import Detector
    det = Detector(pose_model=args.pose_model, hand_conf=args.hand_conf,
                   leg_stabilize=not args.no_leg_stabilize,
                   hand_stabilize=not args.no_hand_stabilize)

    on_frame = None
    cv2 = None
    if args.preview:
        import cv2  # type: ignore
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

    t0 = time.time()
    try:
        results = det.process(video, max_frames=args.max_frames, on_frame=on_frame)
    finally:
        if cv2 is not None:
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
    print(f"wrote {out}: {len(frames)} frames, {nh} with hands, {nf} with face, "
          f"{W}x{H}, {time.time()-t0:.1f}s")
    return 0


# ────────────────────────────────────────────────────────────────────────────
# mocapy-solve
# ────────────────────────────────────────────────────────────────────────────
def solve_main(argv=None) -> int:
    """frames.json (+ optional VRM) -> BVH (Stage B)."""
    from mocapy._paths import ASSETS
    ap = argparse.ArgumentParser(prog="mocapy-solve",
                                 description="frames.json + VRM avatar -> BVH motion file")
    ap.add_argument("frames", nargs="?", default="frames.json",
                    help="input frames.json (default: ./frames.json)")
    ap.add_argument("vrm", nargs="?", default=None,
                    help="VRM avatar (optional — omit or pass '-' to use the bundled "
                         "default skeleton shipped inside the package)")
    ap.add_argument("output", nargs="?", default="my_out.bvh",
                    help="output BVH (default: my_out.bvh in cwd)")
    ap.add_argument("--fov", type=float, default=None,
                    help="source camera vertical FOV in degrees (default 50°). "
                         "Tighter FOV (e.g. zoomed/telephoto) → more sensitive Z motion; "
                         "wider FOV (action cams ~90°) → less. If the avatar walks "
                         "too much or too little, tune this.")
    ap.add_argument("--mirror", action="store_true",
                    help="left/right-mirror the output (avatar performs the mirror "
                         "image of the subject — raise your right hand, the avatar "
                         "raises its left). Swaps 左/右 bones, negates root X, and "
                         "reflects every rotation across the sagittal plane.")
    ap.add_argument("--ground", action="store_true",
                    help="enable the reference-style foot grounding: drive the hips Y from the "
                         "planted (lower) foot so the figure drops when it crouches "
                         "instead of floating at constant height (port of the reference's "
                         "auto_grounding). Approximate — closes part of the hips-Y "
                         "vertical-motion gap vs the GUI.")
    ap.add_argument("--no-foot-stabilize", action="store_true",
                    help="disable foot (足首) stabilization. By default a plant-"
                         "adaptive hold approximates the reference's floor grounding: when a "
                         "foot is planted and the raw solve jitters (ambiguous "
                         "heel/ankle/toe landmarks), it's damped toward its previous "
                         "pose instead of swinging — real steps and clean motion pass "
                         "through untouched. Pass this for the raw per-frame foot solve.")
    args = ap.parse_args(argv)

    from mocapy.pipeline import landmarks_to_bvh
    from mocapy.export.morphs import write_morphs
    frames_path = _resolve(args.frames, Path.cwd() / args.frames)
    out = _resolve(args.output, Path.cwd() / args.output)

    # Resolve VRM: explicit arg → use it; "-" or empty → bundled default; nothing
    # → try assets/cj.vrm if present, otherwise fall through to the bundled default.
    vrm = None
    if args.vrm and args.vrm != "-":
        vrm = _resolve(args.vrm, Path.cwd() / args.vrm)
        if not vrm.exists():
            print(f"ERROR: VRM not found: {vrm}", file=sys.stderr); return 2
    elif (ASSETS / "cj.vrm").exists() and args.vrm is None:
        # Convenience for the dev checkout (where assets/ is alongside the source):
        # default to cj.vrm so existing invocations keep working unchanged.
        vrm = ASSETS / "cj.vrm"

    if not frames_path.exists():
        print(f"ERROR: frames file not found: {frames_path}", file=sys.stderr); return 2

    data = json.loads(frames_path.read_text("utf-8"))
    W, H = data["frame_size"]
    fr, fr_px, fr_hands = [], [], []
    fr_face, fr_fmtx, fr_bs = [], [], []
    for f in data["frames"]:
        fr_hands.append((f["hands"] or None))
        fr_face.append(f.get("face"))
        fr_fmtx.append(f.get("face_matrix"))
        fr_bs.append(f.get("face_blendshapes"))
        if f["world"] is None or f["raw"] is None:
            fr.append(None); fr_px.append(None)
        else:
            fr.append([{"x": p[0], "y": p[1], "z": p[2]} for p in f["world"]])
            fr_px.append([[p[0] * W, p[1] * H, p[2] * W] for p in f["raw"]])

    t0 = time.time()
    morphs: list = []
    txt = landmarks_to_bvh(fr, fr_px, (W, H), vrm, frames_hands=fr_hands,
                           frames_face=fr_face, frames_face_matrix=fr_fmtx,
                           frames_face_blendshapes=fr_bs, morphs_out=morphs,
                           fov_deg=args.fov, mirror=args.mirror,
                           stabilize_feet=not args.no_foot_stabilize,
                           grounding=args.ground)
    out.write_text(txt, encoding="utf-8")
    nframes = int([l for l in txt.splitlines() if l.startswith("Frames:")][0].split(":")[1])
    rig_label = f"vrm={vrm.name}" if vrm is not None else "rig=bundled-default"
    print(f"wrote {out}: {nframes} frames, {len(txt)//1024} KB, {time.time()-t0:.1f}s [{rig_label}]")
    if any(m for m in morphs):
        morph_path = out.with_suffix(out.suffix + ".morphs.json")
        write_morphs(morph_path, morphs)
        nf = sum(1 for m in morphs if m)
        print(f"wrote {morph_path.name}: morph keys for {nf} frames")
    return 0


# ────────────────────────────────────────────────────────────────────────────
# mocapy-webcam — defers to tools/webcam_record.py (kept there so it's easy to
# hack on without reinstalling).
# ────────────────────────────────────────────────────────────────────────────
def webcam_main(argv=None) -> int:
    """Live webcam capture -> frames.json (Stage A-alt)."""
    from mocapy._paths import PROJECT_ROOT
    script = PROJECT_ROOT / "tools" / "webcam_record.py"
    if not script.exists():
        print(f"ERROR: webcam_record.py not found at {script}", file=sys.stderr); return 2
    # Re-exec the script's main() in this process so argparse owns sys.argv cleanly.
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_webcam_record_mod", script)
    mod = importlib.util.module_from_spec(spec)
    if argv is not None:
        sys.argv = ["mocapy-webcam", *argv]
    spec.loader.exec_module(mod)
    return mod.main()
