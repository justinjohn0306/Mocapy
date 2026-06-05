# Mocapy — Setup Guide

End-to-end install + first-run walkthrough. Following this top to bottom should
take ~10 minutes (most of it is conda creating the env) and end with a
working BVH on disk that you can play in Blender.

---

## 1. Prerequisites

| Requirement | Why | Where to get it |
|---|---|---|
| Python **3.11** (recommended) | `mediapipe 0.10.x` wheels exist for 3.10–3.12; we develop on 3.11 | Conda / Miniconda installs it for you |
| **Miniconda** or **Anaconda** | Isolated env so `mediapipe` and its native deps don't clash with system Python | <https://docs.conda.io/projects/miniconda/> |
| **Webcam** (optional) | Only for live capture via `tools/webcam_record.py` | — |
| ~150 MB free disk | Repo + models + sample fixtures + env | — |

> **Windows, macOS, Linux:** `mediapipe` ships wheels for all three; everything
> below works identically. Path separators in shell snippets use Unix style
> (`/`); on Windows PowerShell, swap `/` for `\`.

---

## 2. Create the conda environment

Mocapy ships an `environment.yml` that pins the working set. One command:

```bash
conda env create -f environment.yml
conda activate mocapy
```

That installs Python 3.11 and the runtime deps:

| Package | Pinned to | Used for |
|---|---|---|
| `numpy` | `>=1.26,<2.0` | Math everywhere. **Pinned <2.0** because numpy 2.x crashes inside `vrm.skeleton.load_skeleton` on some builds. |
| `opencv-python` | `>=4.8` | Video / webcam decoding + the live-preview window |
| `mediapipe` | `>=0.10.21` | Pose + hand landmarker (the actual ML) |
| `matplotlib` | `>=3.7` | Used only by `tools/build_pdf.py` (the story PDF) — optional at runtime |

If `environment.yml` is missing or you'd rather build it by hand:

```bash
conda create -n mocapy python=3.11 -y
conda activate mocapy
pip install "numpy>=1.26,<2.0" "opencv-python>=4.8" "mediapipe>=0.10.21" "matplotlib>=3.7"
```

> **Why one env, not two?** The original work used two envs (conda for detect,
> system Python for solve) because of a numpy-2 crash inside `load_skeleton`.
> Pinning `numpy<2.0` in the `mocapy` env fixes that. **One env, two stages, no
> interpreter juggling.**

### Verify the env

```bash
python -c "import numpy, cv2, mediapipe; print(numpy.__version__, cv2.__version__, mediapipe.__version__)"
# expect something like: 1.26.4 4.10.0 0.10.21
```

---

## 3. Install Mocapy itself

Editable install so your edits to `mocapy/` are picked up immediately:

```bash
pip install -e .
```

This wires up `import mocapy` and registers the project's own CLI entrypoints
(declared in `pyproject.toml`):

| Entrypoint | What it does |
|---|---|
| `mocapy-detect <video> <out.json>` | Same as `python validation/detect_dump.py` |
| `mocapy-solve <frames.json> <vrm> <out.bvh>` | BVH + `.morphs.json` sidecar (pass `-` for the bundled skeleton) |
| `mocapy-webcam [--device 0] [--duration N] ...` | Same as `python tools/webcam_record.py` |

(Old `python validation/<name>.py` forms still work too — the entrypoints just save
the typing and don't require you to be at the repo root.)

### Verify Mocapy

Run the test suite — should print **12 × PASS**:

```bash
for t in test_three_math test_bvh_math \
         validate_hierarchy validate_motion \
         validate_solver_center validate_solver_torso validate_solver_leg \
         validate_fingers validate_wrist validate_pipeline \
         validate_face validate_morphs; do
    python validation/$t.py >/dev/null 2>&1 && echo "PASS $t" || echo "FAIL $t"
done
```

If any FAIL, see *Troubleshooting* at the bottom.

---

## 4. First run — VIDEO → BVH

Mocapy ships with `fixtures/test.mp4` and two VRMs (`assets/cj.vrm`,
`assets/AliciaSolid.vrm`) so you can do an end-to-end run with zero inputs of
your own.

```bash
# Stage A — detect (takes ~1 min on the 38-second sample)
mocapy-detect fixtures/test.mp4 frames.json
# wrote frames.json: 1135 frames, 105 with hands, size 1920x1080

# Stage B-bvh — solve + write BVH + .morphs.json sidecar (~1.5 sec)
mocapy-solve frames.json assets/AliciaSolid.vrm samples/my_first.bvh
# wrote samples/my_first.bvh: 1134 frames, ~1.5s
# wrote my_first.bvh.morphs.json: morph keys for ~1000 frames
```

Drop `samples/my_first.bvh` into Blender (File → Import → Motion Capture (.bvh))
for general 3D work; the `.morphs.json` sidecar carries the per-frame facial
morph weights alongside it.

---

## 5. First run — WEBCAM → BVH (the cool one)

The webcam capture writes the **same** `frames.json` format, so Stage B is
unchanged.

```bash
# Stage A — live capture (preview window opens; press 'q' or ESC to stop)
mocapy-webcam --device 0 --duration 30 --mirror --output frames.json

# Stage B — same as before, just point at the live capture
mocapy-solve frames.json assets/AliciaSolid.vrm samples/webcam.bvh
```

### What you'll see in the preview

* Green dots = MediaPipe pose landmarks (33 of them)
* Orange dots = hand landmarks (21 per hand, when visible)
* Top bar: frame count, elapsed seconds, current FPS, stop hint

### Webcam tips for clean captures

| Tip | Why |
|---|---|
| Stand 2–3 m back so the full body is in frame | The pose model needs to see hips & feet for grounding |
| Use bright, even lighting | The hand landmarker is fragile in low light |
| Plain background | Reduces phantom detections |
| Hands fully open and visible to record finger motion | Closed fists or occluded hands → identity fingers |
| Don't wear baggy long sleeves at the wrist | The wrist basis (pinky/index/wrist) breaks down |
| Use `--mirror` for a forward-facing camera | The solver assumes the mirrored convention (character 左 ← image RIGHT) |
| Stop with **'q' or ESC** rather than Ctrl-C | Ensures `frames.json` is flushed |

### Webcam CLI reference

```text
$ mocapy-webcam --help
  --device       INT      webcam index (default 0; try 1 or 2 if you have multiple cams)
  --duration     SECONDS  max recording length (default 10)
  --output       PATH     where to write frames.json (default ./frames.json)
  --width        PX       capture width (default 1280)
  --height       PX       capture height (default 720)
  --target-fps   FPS      throttle to this rate (default 30); 60fps cams just drop
                          every other frame so the BVH plays at real-time speed
  --mirror / --no-mirror  mirror-image preview & capture (DEFAULT: ON). Off gives
                          face-to-face puppet behaviour. Only affects 2D image
                          coords; the avatar's bone rotations come from pose_world
                          (anatomy-relative) and are well-defined either way.
  --no-preview            run headless (no window) — for scripts/servers
  --pose-model   {full,heavy}   `heavy` is more accurate but ~3× slower
```

The preview window shows the **MediaPipe predictions overlaid on the webcam feed**:
green pose skeleton (33 BlazePose keypoints), magenta hand skeletons (21 each when
visible), white face-mesh dots when the face is in range. `mocapy-detect --preview`
applies the same overlay to file-based detection.

---

## 6. Plug in your own VRM

Any VRM 0.x or VRM 1.x with the standard humanoid bone names works — drop it
into `assets/` and point Stage B at it:

```bash
mocapy-solve frames.json assets/MyAvatar.vrm samples/me_as_avatar.bvh
```

Coverage notes:
* Bones the rig doesn't expose (e.g. some VRMs skip the upper-chest 上半身2 or the
  thumb proximal) just stay identity in the BVH — no error, no warning.
* Full-finger VRMs get all 24 finger bones + the 3 thumb bones (24+3 × 2 hands).
* Rigs without 手捩 (forearm twist) lose only the forearm twist; the rest still
  exports cleanly.

---

## 7. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ImportError: No module named mediapipe` | You're not in the `mocapy` conda env. `conda activate mocapy` and retry. |
| `ModuleNotFoundError: No module named 'mocapy'` | You didn't `pip install -e .`, or you're running from a different dir. From the repo root, run `pip install -e .` once. |
| `cv2.VideoCapture` returns no frames | Wrong `--device`. Try `--device 1` or `2`. On Windows, some virtual cameras need DirectShow (we use it by default). |
| Crash inside `load_skeleton` (numpy float scalar) | Your `numpy` is 2.x. Run `pip install "numpy>=1.26,<2.0"` inside the env. |
| BVH plays but body is "marching in place" | You probably skipped `--mirror`, or `frames.json` was captured without it. Re-capture with `--mirror`. |
| Hands don't curl | Hands weren't in frame for most of the clip, or they were too far / dark. Get closer to the camera and brighten the room. |
| `UnicodeEncodeError` printing 手捩 etc. | On Windows: `set PYTHONIOENCODING=utf-8` once per shell. |
| `validate_*` tests FAIL after `git pull` | Re-run `pip install -e .` (an import path may have moved). |

---

## 8. What you can ship with this

The `mocapy` package itself is < 200 KB of pure Python with three runtime deps
(`numpy`, `opencv-python`, `mediapipe`). The bundled `models/` (~50 MB) and
`fixtures/` (~57 MB, optional — only the test suite needs them) make up the
bulk. For a shipping distribution, the leanest set is:

```
mocapy/                 ← source
models/*.task           ← runtime models (mandatory)
assets/*.vrm            ← whichever avatars you ship
pyproject.toml          ← installable
README.md SETUP.md      ← docs
```

Drop `fixtures/`, `samples/`, `validation/`, `tools/build_pdf.py`, and `docs/` for
end-user builds — none of them are needed at runtime.
