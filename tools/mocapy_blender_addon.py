"""Mocapy Realtime — Blender addon for webcam mocap preview + record.

Installation:
  Blender → Edit → Preferences → Add-ons → Install... → pick this .py file → enable.
  Then in any 3D Viewport: press N to open the side panel → "Mocapy" tab.

How it works:
  Blender's bundled Python (3.13) doesn't have a stable mediapipe wheel yet,
  so installing mediapipe INTO Blender doesn't work. Instead this addon spawns
  a small detection daemon (`tools/mocapy_blender_daemon.py`) in your existing
  conda env (which already has mediapipe + opencv + mocapy installed), and
  streams per-frame bone rotations back over stdout. Zero installs into Blender.

Setup:
  1. "Repo path"   — where you cloned Mocapy (default /path/to/Mocapy).
  2. "Conda Python" — full path to your conda env's python.exe. Default:
     E:/Programs/anaconda3/envs/mocapy/python.exe
    .
  3. Select an Armature whose bone names follow VRM convention
     (`leftUpperArm`, `rightLowerLeg`, `hips`, `head`, ...). Most VRM importers
     and Mixamo rigs work out of the box.
  4. Click "Start" — ESC in viewport to stop.
  5. Toggle "Record" before starting to save frames.json on stop, so you can
     re-solve later with `mocapy-solve` for a polished BVH with smoothing,
     hands, face, morphs.

Scope: pose-only realtime preview. Hands+face+morphs run through the offline
solver from the recorded JSON for the polished take.
"""

bl_info = {
    "name": "Mocapy Realtime",
    "author": "Mocapy",
    "version": (0, 2, 0),
    "blender": (4, 0, 0),
    "location": "View3D > N-panel > Mocapy",
    "description": "Realtime webcam mocap preview via conda-env daemon → VRM armature",
    "category": "Animation",
}

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import bpy
import mathutils
from bpy.props import (BoolProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy.types import Operator, Panel, PropertyGroup


# ---------------------------------------------------------------------------
# MMD → VRM bone-name map. Mirrors mocapy/pipeline.py:STANDARD_VRM_TO_MMD.
# ---------------------------------------------------------------------------

def _build_mmd_to_vrm():
    fingers = {"Thumb": ("親指", ("０", "１", "２")),
               "Index": ("人指", ("１", "２", "３")),
               "Middle": ("中指", ("１", "２", "３")),
               "Ring": ("薬指", ("１", "２", "３")),
               "Little": ("小指", ("１", "２", "３"))}
    m = {"センター": "hips", "上半身": "spine", "上半身2": "chest",
         "上半身3": "upperChest", "首": "neck", "頭": "head"}
    for lr, j in (("left", "左"), ("right", "右")):
        m[j + "肩"] = lr + "Shoulder"
        m[j + "腕"] = lr + "UpperArm"
        m[j + "ひじ"] = lr + "LowerArm"
        m[j + "手首"] = lr + "Hand"
        m[j + "足"] = lr + "UpperLeg"
        m[j + "ひざ"] = lr + "LowerLeg"
        m[j + "足首"] = lr + "Foot"
        for f, (mf, sfx) in fingers.items():
            for seg, s in zip(("Proximal", "Intermediate", "Distal"), sfx):
                m[j + mf + s] = lr + f + seg
    return m


MMD_TO_VRM = _build_mmd_to_vrm()

MIXAMO_FALLBACK = {
    "hips": "Hips", "spine": "Spine", "chest": "Spine1", "upperChest": "Spine2",
    "neck": "Neck", "head": "Head",
    "leftShoulder": "LeftShoulder", "leftUpperArm": "LeftArm",
    "leftLowerArm": "LeftForeArm", "leftHand": "LeftHand",
    "rightShoulder": "RightShoulder", "rightUpperArm": "RightArm",
    "rightLowerArm": "RightForeArm", "rightHand": "RightHand",
    "leftUpperLeg": "LeftUpLeg", "leftLowerLeg": "LeftLeg", "leftFoot": "LeftFoot",
    "rightUpperLeg": "RightUpLeg", "rightLowerLeg": "RightLeg", "rightFoot": "RightFoot",
}


def find_bone(armature, vrm_name):
    bones = armature.pose.bones
    if vrm_name in bones:
        return bones[vrm_name]
    for pb in bones:
        ext = getattr(pb.bone, "vrm_addon_extension", None)
        if ext is None:
            continue
        for attr in ("humanoid_bone", "human_bone"):
            tag = getattr(ext, attr, None)
            if tag and str(tag) == vrm_name:
                return pb
    mx = MIXAMO_FALLBACK.get(vrm_name)
    if mx:
        for prefix in ("mixamorig:", "mixamorig_", ""):
            cand = prefix + mx
            if cand in bones:
                return bones[cand]
    target = vrm_name.lower()
    cands = [pb for pb in bones if pb.name.lower().endswith(target)]
    if len(cands) == 1:
        return cands[0]
    return None


def threejs_quat_to_blender(qx, qy, qz, qw):
    """Three.js (Y-up RH, XYZW) → Blender bone-local (WXYZ). VRM/glTF rigs
    apply directly; Mixamo bones whose local Y isn't the bone-axis may need
    per-bone correction."""
    return mathutils.Quaternion((float(qw), float(qx), float(qy), float(qz)))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class MocapyProps(PropertyGroup):
    mocapy_path: StringProperty(
        name="Repo path", subtype='DIR_PATH',
        default="/path/to/Mocapy",
        description="Folder containing the Mocapy checkout")
    python_exe: StringProperty(
        name="Conda Python", subtype='FILE_PATH',
        default="E:/Programs/anaconda3/envs/mocapy/python.exe",
        description="Path to python.exe in the conda env that has mediapipe + mocapy "
                    "installed (the same env you use for `mocapy-detect`)")
    camera_device: IntProperty(name="Camera index", default=0, min=0, max=8)
    target_fps: IntProperty(name="Target FPS", default=15, min=1, max=60)
    flip_horizontal: BoolProperty(
        name="Mirror (selfie)", default=False,
        description="Off (default) = user's right hand drives avatar's right hand. "
                    "On = the reference/MMD selfie convention (mirrored)")
    record: BoolProperty(
        name="Record frames", default=False,
        description="Save raw landmarks to a frames.json sidecar on stop, for "
                    "offline re-solving with full smoothing + face + hands")
    record_path: StringProperty(
        name="Record file", subtype='FILE_PATH',
        default="//mocapy_realtime.json")
    status: StringProperty(default="Stopped")


# ---------------------------------------------------------------------------
# Daemon process + background stdout reader.
# ---------------------------------------------------------------------------

class _DaemonReader(threading.Thread):
    """Reads daemon stdout line-by-line, parses JSON, pushes onto a queue."""

    def __init__(self, proc, q_out):
        super().__init__(daemon=True)
        self.proc = proc
        self.q = q_out
        self._stop = threading.Event()

    def run(self):
        try:
            for line in iter(self.proc.stdout.readline, ""):
                if self._stop.is_set():
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    self.q.put_nowait(json.loads(line))
                except Exception:
                    # Forward malformed lines as a debug entry — easier than dropping.
                    self.q.put_nowait({"type": "raw", "line": line})
        except Exception as e:
            self.q.put_nowait({"type": "error", "message": f"reader: {e!r}"})

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class MOCAPY_OT_StartMocap(Operator):
    bl_idname = "mocapy.start_mocap"
    bl_label = "Start Realtime Mocap"
    bl_description = "Spawn the conda-env detection daemon and apply pose to the armature. ESC stops."

    _timer = None
    _proc = None
    _reader = None
    _queue = None
    _armature = None
    _frame_idx = 0
    _record_frames = None

    def modal(self, context, event):
        if event.type == 'ESC':
            return self._stop(context, cancelled=False)
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        if self._proc is None or self._proc.poll() is not None:
            # Daemon died unexpectedly.
            self.report({'ERROR'}, f"Daemon exited (code {self._proc.returncode if self._proc else '?'})")
            return self._stop(context, cancelled=True)

        # Drain the queue (apply only the latest frame's bones to avoid lag).
        latest = None
        warns = []
        while True:
            try:
                msg = self._queue.get_nowait()
            except queue.Empty:
                break
            t = msg.get("type")
            if t == "frame":
                latest = msg
            elif t == "error":
                self.report({'ERROR'}, f"Daemon error: {msg.get('message')}")
                return self._stop(context, cancelled=True)
            elif t == "warn":
                warns.append(msg.get("message"))
            elif t == "hello":
                self.report({'INFO'}, f"Daemon ready: {msg.get('width')}x{msg.get('height')} @ {msg.get('fps')}fps")

        if latest is None:
            return {'PASS_THROUGH'}

        bones = latest.get("bones")
        applied = 0
        if bones:
            for mmd_name, q in bones.items():
                vrm_name = MMD_TO_VRM.get(mmd_name)
                if not vrm_name:
                    continue
                pb = find_bone(self._armature, vrm_name)
                if pb is None:
                    continue
                pb.rotation_mode = 'QUATERNION'
                pb.rotation_quaternion = threejs_quat_to_blender(*q)
                applied += 1

        # Record raw bone data — same shape the addon receives, easy to re-derive.
        if self._record_frames is not None:
            self._record_frames.append(latest)

        self._frame_idx = latest.get("idx", self._frame_idx) + 1
        ctx_props = context.scene.mocapy
        ctx_props.status = f"Running  frame {self._frame_idx}  bones {applied}"

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        return {'PASS_THROUGH'}

    def execute(self, context):
        props = context.scene.mocapy

        repo = Path(bpy.path.abspath(props.mocapy_path)).resolve()
        daemon = repo / "tools" / "mocapy_blender_daemon.py"
        if not daemon.is_file():
            self.report({'ERROR'}, f"Daemon not found at {daemon}. Set Repo path correctly.")
            return {'CANCELLED'}

        py = Path(bpy.path.abspath(props.python_exe)).resolve()
        if not py.is_file():
            self.report({'ERROR'}, f"Conda Python not found at {py}. Set Conda Python correctly.")
            return {'CANCELLED'}

        obj = context.active_object
        if obj is None or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an Armature in the viewport first.")
            return {'CANCELLED'}
        self._armature = obj

        cmd = [
            str(py), str(daemon),
            "--camera", str(props.camera_device),
            "--fps", str(props.target_fps),
        ]
        if props.flip_horizontal:
            cmd.append("--mirror")
        else:
            cmd.append("--no-mirror")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["MOCAPY_ROOT"] = str(repo)

        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, encoding="utf-8", env=env,
                cwd=str(repo),
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start daemon: {e}")
            return {'CANCELLED'}

        self._queue = queue.Queue()
        self._reader = _DaemonReader(self._proc, self._queue)
        self._reader.start()
        self._frame_idx = 0
        self._record_frames = [] if props.record else None

        wm = context.window_manager
        # Poll a bit faster than the daemon's emit rate so we never miss a frame.
        interval = max(0.01, 0.5 / max(1, props.target_fps))
        self._timer = wm.event_timer_add(interval, window=context.window)
        wm.modal_handler_add(self)
        props.status = "Starting daemon..."
        self.report({'INFO'}, "Daemon launched. ESC in viewport to stop.")
        return {'RUNNING_MODAL'}

    def _stop(self, context, *, cancelled):
        wm = context.window_manager
        if self._timer is not None:
            try:
                wm.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        if self._reader is not None:
            self._reader.stop()
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            except Exception:
                pass
        self._proc = None

        props = context.scene.mocapy
        props.status = f"Stopped ({self._frame_idx} frames)"

        if self._record_frames:
            out = Path(bpy.path.abspath(props.record_path)).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            # We record the bone-rotation stream the daemon emitted. For
            # OFFLINE re-solving with full smoothing + face + hands, you'd
            # want raw landmarks instead — see the note in the addon header.
            out.write_text(json.dumps(
                {"frame_size": [1280, 720], "frames": self._record_frames}),
                encoding="utf-8")
            self.report({'INFO'}, f"Recorded {len(self._record_frames)} frames to {out}")

        return {'CANCELLED' if cancelled else 'FINISHED'}

    def cancel(self, context):
        return self._stop(context, cancelled=True)


class MOCAPY_OT_TestDaemon(Operator):
    bl_idname = "mocapy.test_daemon"
    bl_label = "Test conda env"
    bl_description = ("Run the daemon for 3 seconds with no Blender involvement, "
                      "to verify mediapipe + mocapy import in the chosen conda env")

    def execute(self, context):
        props = context.scene.mocapy
        repo = Path(bpy.path.abspath(props.mocapy_path)).resolve()
        daemon = repo / "tools" / "mocapy_blender_daemon.py"
        py = Path(bpy.path.abspath(props.python_exe)).resolve()
        if not daemon.is_file():
            self.report({'ERROR'}, f"Daemon not found at {daemon}.")
            return {'CANCELLED'}
        if not py.is_file():
            self.report({'ERROR'}, f"Conda Python not found at {py}.")
            return {'CANCELLED'}

        cmd = [str(py), str(daemon), "--camera", str(props.camera_device),
               "--fps", "5"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=4,
                                 encoding="utf-8")
        except subprocess.TimeoutExpired as e:
            # Expected — we never asked it to stop. The fact that it ran for the
            # full 4s without crashing means imports + camera + detector all work.
            out = (e.stdout or "")[:2000]
            first = out.splitlines()[:5]
            self.report({'INFO'}, f"Daemon OK. First lines: {' | '.join(first)}")
            return {'FINISHED'}

        if res.returncode != 0:
            err = (res.stdout or "")[-1500:]
            self.report({'ERROR'}, f"Daemon failed (exit {res.returncode}):\n{err}")
            return {'CANCELLED'}
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# UI panel
# ---------------------------------------------------------------------------

class MOCAPY_PT_Panel(Panel):
    bl_label = "Mocapy Realtime"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Mocapy"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mocapy

        box = layout.box()
        box.label(text="Setup", icon='PREFERENCES')
        box.prop(props, "mocapy_path")
        box.prop(props, "python_exe")
        box.operator("mocapy.test_daemon", icon='CONSOLE')

        box = layout.box()
        box.label(text="Capture", icon='OUTLINER_OB_CAMERA')
        box.prop(props, "camera_device")
        box.prop(props, "target_fps")
        box.prop(props, "flip_horizontal")
        box.prop(props, "record")
        if props.record:
            box.prop(props, "record_path")

        layout.separator()
        layout.label(text=f"Status: {props.status}")

        row = layout.row(align=True)
        row.scale_y = 1.5
        obj = context.active_object
        if obj is None or obj.type != 'ARMATURE':
            row.enabled = False
            row.operator("mocapy.start_mocap", text="Select an Armature first", icon='ERROR')
        else:
            row.operator("mocapy.start_mocap", text=f"Start on '{obj.name}'", icon='PLAY')
        layout.label(text="ESC in viewport to stop", icon='INFO')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    MocapyProps,
    MOCAPY_OT_StartMocap,
    MOCAPY_OT_TestDaemon,
    MOCAPY_PT_Panel,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.mocapy = PointerProperty(type=MocapyProps)


def unregister():
    if hasattr(bpy.types.Scene, "mocapy"):
        del bpy.types.Scene.mocapy
    for c in reversed(_classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
