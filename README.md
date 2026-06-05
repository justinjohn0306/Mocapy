<div align="center">

# 🎭 Mocapy

### Markerless motion capture — video or webcam → animated VRM avatar

Turn **any video or live webcam feed** into full-body animation — body, fingers, head,
eye gaze, and facial expression — and export it as **BVH** (Blender / Unity) plus a
per-frame **facial-morph sidecar**.

**No game engine. No plugins. No GPU required.**

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.x-00C4B4.svg)](https://developers.google.com/mediapipe)
[![License](https://img.shields.io/badge/license-see%20LICENSE-green.svg)](./LICENSE)

<br>

![Mocapy demo — input video on the left, live pose + hands + face-mesh skeleton on the right](media/demo.gif)

<sub>Left: source footage · Right: Mocapy's live pose + hand + face-mesh solve</sub>

</div>

---

```
  🎥 video / webcam
        │
        ▼
  🧠 MediaPipe        pose (33) + hands (21×2) + face mesh (478) + 52 blendshapes
        │
        ▼
  🦴 solver           bone quaternions + facial morphs
        │
        ▼
  📦 export           BVH  ·  .morphs.json     → any humanoid VRM rig
```

Works on **any VRM** (0.x or 1.x) with standard humanoid bone names. Or skip the rig
entirely with `-` and use the bundled default skeleton.

---

## ✨ Features

| | |
|---|---|
| 🦴 **Full-body solve** | Torso, arms, legs, head/neck, 24 finger bones + thumbs per hand |
| 👀 **Face & gaze** | Iris-driven eye gaze + 52 ARKit blendshapes → facial morph weights + 12 VRM presets |
| 🎯 **Two detection backends** | `cv2` + MediaPipe Tasks, or a WASM `tasks-vision` bridge in headless Chromium for higher-fidelity legs |
| 💻 **Holistic (CPU-only) mode** | One combined pose+face+hands model that runs with no GPU — works on low-end machines |
| 🟦 **Live skeleton preview** | Realtime overlay window (pose + hands + face mesh + bounding boxes) |
| 🛡️ **Robustness layers** | Leg-collision gating, finger depth-recovery + smoothing, plant-adaptive foot grounding, torso de-wobble |
| 📤 **Exports** | **BVH** + a per-frame `.morphs.json` facial sidecar |
| 🔁 **Detect once, retarget many** | A shared `frames.json` lets you re-solve one capture onto many rigs without re-detecting |

---

## 🚀 Quick start

```bash
# 1. install (editable)
pip install -e .

# 2. download the MediaPipe models into models/  (see Installation below)

# 3a. detect:  video → frames.json   (add --preview for the live overlay window)
mocapy-detect  fixtures/test.mp4  frames.json

# 3b. solve:   frames.json → BVH     ( '-' = bundled default skeleton, no VRM needed )
mocapy-solve   frames.json  assets/AliciaSolid.vrm  out.bvh
mocapy-solve   frames.json  -  out.bvh
```

Drop `out.bvh` into Blender (*File → Import → Motion Capture (.bvh)*) and hit play. ▶️

---

## 📦 Installation

### 1. Prerequisites

| Requirement | Why |
|---|---|
| **Python 3.10 – 3.12** (3.11 recommended) | `mediapipe 0.10.x` ships wheels for these versions |
| **Miniconda / Anaconda** *(recommended)* | Isolates `mediapipe`'s native deps from system Python |
| **Webcam** *(optional)* | Only for live capture |
| ~150 MB free disk | Source + models + sample fixtures |

> 🪟🍎🐧 **Windows · macOS · Linux** all work identically. Shell snippets use Unix-style
> paths; on Windows PowerShell, swap `/` → `\`.

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate mocapy
```

<details>
<summary>Prefer plain venv / pip instead of conda?</summary>

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install "numpy>=1.26,<2.0" "opencv-python>=4.8" "mediapipe>=0.10.21"
```
</details>

> ⚠️ **numpy must be `<2.0`.** numpy 2.x crashes inside the VRM skeleton loader on some
> builds. Both `environment.yml` and `pyproject.toml` pin this for you.

### 3. Install Mocapy

```bash
pip install -e .
```

This registers three CLI entrypoints: `mocapy-detect`, `mocapy-solve`, and
`mocapy-webcam`.

### 4. Download the MediaPipe models

The `.task` model files (~50 MB total) are **not committed**. Download them from
MediaPipe's official model repository into a `models/` folder at the repo root:

```
models/
├── pose_landmarker_full.task
├── pose_landmarker_heavy.task     # optional — only for --pose-model heavy / --strong
├── hand_landmarker.task
├── face_landmarker.task
└── holistic_landmarker.task       # optional — only for --holistic
```

### 5. *(Optional)* Set up the WASM bridge backend

The higher-fidelity `--backend bridge` path runs the browser-grade `tasks-vision`
models in headless Chromium. It needs **Node.js** + **`ffprobe`** (from FFmpeg) on
your `PATH`, plus a one-time install:

```bash
cd node_bridge
npm install        # pulls puppeteer's headless Chromium + @mediapipe/tasks-vision
```

> The default `cv2` backend needs **none** of this — bridge is purely opt-in.

---

## 🎬 Usage

Mocapy runs in **two stages** around a shared `frames.json`. This split lets you
detect once and re-solve onto many avatars.

### Stage A — Detect (`mocapy-detect`)

```bash
mocapy-detect <video> [frames.json] [options]
```

| Flag | Description |
|---|---|
| `--backend {cv2,bridge}` | Detector. `cv2` (default) or `bridge` (WASM in Chromium — best legs) |
| `--pose-model {full,heavy}` | `heavy` is steadier on legs, ~3× slower |
| `--hand-conf FLOAT` | Hand-detection confidence (bridge). Lower (e.g. `0.3`) catches faster hands |
| `--strong` | Preset = `--pose-model heavy --hand-conf 0.3` (best limbs + fingers) |
| `--holistic` | One combined CPU-only model (bridge) — for machines with no GPU |
| `--preview` | Open a live overlay window (pose + hands + face mesh + boxes) |
| `--max-frames N` | Stop after N frames |
| `--no-leg-stabilize` | Disable leg-collision gating |
| `--no-hand-stabilize` | Disable finger depth-recovery + smoothing |

```bash
# fast, no extra deps
mocapy-detect  fixtures/test.mp4  frames.json

# highest-fidelity legs, with a live preview window
mocapy-detect  fixtures/test.mp4  frames.json  --backend bridge --strong --preview

# CPU-only, low-end machine
mocapy-detect  fixtures/test.mp4  frames.json  --backend bridge --holistic
```

### Stage A (live) — Webcam (`mocapy-webcam`)

Writes the **same** `frames.json`, so Stage B is unchanged.

```bash
mocapy-webcam --device 0 --duration 30 --output frames.json
```

| Flag | Default | Description |
|---|---|---|
| `--device INT` | `0` | Webcam index (try `1`/`2` for multiple cams) |
| `--duration SEC` | `10` | Max recording length |
| `--output PATH` | `frames.json` | Where to write the capture |
| `--width / --height` | `1280 / 720` | Capture resolution |
| `--target-fps FPS` | `30` | Throttle rate (60 fps cams drop every other frame) |
| `--mirror / --no-mirror` | **on** | Mirror-image capture (selfie convention) |
| `--no-preview` | — | Headless (no window) — for scripts/servers |
| `--pose-model {full,heavy}` | `full` | `heavy` is more accurate, ~3× slower |

> Stop a webcam capture with **`q` or `ESC`** (not Ctrl-C) so `frames.json` is flushed.

### Stage B — Solve to BVH (`mocapy-solve`)

```bash
mocapy-solve <frames.json> <vrm|-> <out.bvh> [options]
```

Pass a VRM path to retarget onto that rig, or `-` to use the **bundled default
skeleton** (no VRM file needed). Writes the BVH plus a `<out>.bvh.morphs.json` sidecar.

| Flag | Description |
|---|---|
| `--fov FLOAT` | Camera field of view for depth back-projection (tune root Z motion) |
| `--mirror` | Left/right-mirror the solved output (retarget to opposite handedness) |
| `--ground` | Enable foot-grounding (drives hips-Y from the planted foot) |
| `--no-foot-stabilize` | Disable plant-adaptive foot jitter hold |

```bash
mocapy-solve  frames.json  assets/AliciaSolid.vrm  out.bvh           # onto a rig
mocapy-solve  frames.json  -  out.bvh                                 # bundled skeleton
mocapy-solve  frames.json  assets/cj.vrm  out.bvh  --mirror --ground  # mirrored + grounded
```

---

## 📤 Outputs

| File | Plays in | Contains |
|---|---|---|
| `.bvh` | Blender, Unity, Maya, MotionBuilder | Full bone hierarchy + per-frame rotations + root translation |
| `.bvh.morphs.json` | (sidecar) | Per-frame facial morph weights to pair with the BVH |

The first BVH frame is a clean rest pose (identity rotations), so the rig holds its
A-pose until you press play — convenient for Mixamo-style retargeting.

---

## 🗂️ Project layout

```
mocapy/                the library
├── pipeline.py          orchestration: solve_frame → solve_sequence → BVH
├── detect/              MediaPipe Tasks + OneEuro filters + WASM bridge backend
├── solve/               torso · arms · legs · head · wrist · fingers · eyes · morphs · blend
├── vrm/skeleton.py      VRM / glTF rest-pose parser
└── export/              bvh.py · morphs.py
node_bridge/           tasks-vision (WASM) detection bridge + realtime preview
tools/                 webcam capture · Blender add-on · demo-GIF generator
models/                MediaPipe .task files (download separately, ~50 MB)
assets/                example VRM avatars
fixtures/              sample video + reference captures
media/                 README demo GIF
```

Set `MOCAPY_ROOT` to relocate `fixtures/` / `assets/` / `models/`.

---

## 🩹 Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named mediapipe` | Activate the env: `conda activate mocapy` |
| `No module named 'mocapy'` | Run `pip install -e .` from the repo root |
| Crash inside `load_skeleton` (float scalar) | numpy is 2.x → `pip install "numpy>=1.26,<2.0"` |
| Webcam returns no frames | Wrong `--device` — try `1` or `2` |
| Body "marches in place" | Capture was un-mirrored — re-record (webcam mirror is on by default) |
| Hands don't curl | Hands were out of frame / too far / too dark — get closer, brighten the room |
| `UnicodeEncodeError` printing 手捩 | Windows: `set PYTHONIOENCODING=utf-8` |
| `--backend bridge` fails to start | Need Node.js + `ffprobe` on PATH and `npm install` in `node_bridge/` |

---

## 📄 License

See [`LICENSE`](./LICENSE). Built on Google's [MediaPipe](https://developers.google.com/mediapipe)
for landmark detection.

---

<div align="center">

**Made with 🦴 and a lot of quaternions.**

</div>
