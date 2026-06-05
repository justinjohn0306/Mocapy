"""Stage B (base python): frames.json + VRM -> BVH (full pipeline w/ fingers).

  $ python validation/solve_bvh.py [frames.json] [vrm] [out.bvh]

Defaults: frames=cwd/frames.json, vrm=assets/cj.vrm, out=cwd/my_out.bvh.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocapy.pipeline import landmarks_to_bvh  # noqa: E402
from mocapy.export.morphs import write_morphs  # noqa: E402
from mocapy._paths import ASSETS  # noqa: E402


def _resolve(arg: str | None, default: Path) -> Path:
    if not arg:
        return default
    p = Path(arg)
    return p if p.is_absolute() else Path.cwd() / p


def main() -> int:
    frames_path = _resolve(sys.argv[1] if len(sys.argv) > 1 else None, Path.cwd() / "frames.json")
    vrm = _resolve(sys.argv[2] if len(sys.argv) > 2 else None, ASSETS / "cj.vrm")
    out = _resolve(sys.argv[3] if len(sys.argv) > 3 else None, Path.cwd() / "my_out.bvh")

    data = json.loads(frames_path.read_text("utf-8"))
    W, H = data["frame_size"]
    frames, frames_px, frames_hands = [], [], []
    frames_face, frames_face_matrix, frames_face_bs = [], [], []
    for f in data["frames"]:
        hands = {k: v for k, v in f["hands"].items()} or None
        frames_hands.append(hands)
        frames_face.append(f.get("face"))
        frames_face_matrix.append(f.get("face_matrix"))
        frames_face_bs.append(f.get("face_blendshapes"))
        if f["world"] is None or f["raw"] is None:
            frames.append(None)
            frames_px.append(None)
        else:
            frames.append([{"x": p[0], "y": p[1], "z": p[2]} for p in f["world"]])
            frames_px.append([[p[0] * W, p[1] * H, p[2] * W] for p in f["raw"]])
    t0 = time.time()
    morphs: list = []
    txt = landmarks_to_bvh(frames, frames_px, (W, H), vrm, frames_hands=frames_hands,
                           frames_face=frames_face,
                           frames_face_matrix=frames_face_matrix,
                           frames_face_blendshapes=frames_face_bs,
                           morphs_out=morphs)
    out.write_text(txt, encoding="utf-8")
    nframes = int([l for l in txt.splitlines() if l.startswith("Frames:")][0].split(":")[1])
    # Sidecar — write only if we collected any morph data.
    if any(m for m in morphs):
        morph_path = out.with_suffix(out.suffix + ".morphs.json")
        write_morphs(morph_path, morphs)
        with_face = sum(1 for m in morphs if m)
        print(f"wrote {out}: {nframes} frames, {time.time()-t0:.1f}s")
        print(f"wrote {morph_path.name}: morph keys for {with_face} frames")
    else:
        print(f"wrote {out}: {nframes} frames, {time.time()-t0:.1f}s (no face data, no morphs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
