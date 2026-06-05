"""Morph (blendshape) sidecar writer — BVH carries bones only, this carries faces.

BVH has no morph track, so the per-frame face expression weights ride alongside
the .bvh as a `<basename>.morphs.json`:

    {
        "fps": 30,
        "frames": <int>,
        "format": "mocapy-morphs/1",
        "naming": "MMD + VRM 0.x preset",
        "morph_names_mmd": ["あ", "い", ...],
        "morph_names_vrm": ["a", "i", "joy", ...],
        "weights_mmd": [[a0, i0, ...], [a1, i1, ...], ...],
        "weights_vrm": [[a0, i0, joy0, ...], ...]
    }

Two-row weights tables (one row per frame, one column per morph) so the file is
compact and trivially loadable in any language. Both MMD and VRM-preset views are
included so a downstream tool can pick whichever its target rig exposes without
re-doing the mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

from mocapy.solve.face_morphs import to_vrm_expressions


def write_morphs(out_path: Path, per_frame_mmd: list[dict[str, float] | None],
                 *, fps: float = 30.0) -> Path:
    """Write the morph sidecar. per_frame_mmd is one MMD-morph dict per frame
    (or None for frames without face data). Returns the path written."""
    # Collect the union of MMD morph names across frames, in a stable order.
    mmd_names: list[str] = []
    seen: set[str] = set()
    for d in per_frame_mmd:
        if not d:
            continue
        for n in d:
            if n not in seen:
                seen.add(n); mmd_names.append(n)
    # VRM preset columns are the union after mapping.
    vrm_names: list[str] = []
    seen = set()
    for d in per_frame_mmd:
        if not d:
            continue
        for n in to_vrm_expressions(d).keys():
            if n not in seen:
                seen.add(n); vrm_names.append(n)

    nframes = len(per_frame_mmd)
    w_mmd: list[list[float]] = []
    w_vrm: list[list[float]] = []
    for d in per_frame_mmd:
        if not d:
            w_mmd.append([0.0] * len(mmd_names))
            w_vrm.append([0.0] * len(vrm_names))
        else:
            w_mmd.append([float(d.get(n, 0.0)) for n in mmd_names])
            v = to_vrm_expressions(d)
            w_vrm.append([float(v.get(n, 0.0)) for n in vrm_names])

    payload = {
        "format": "mocapy-morphs/1",
        "naming": "MMD + VRM 0.x preset (max-aggregated)",
        "fps": float(fps),
        "frames": nframes,
        "morph_names_mmd": mmd_names,
        "morph_names_vrm": vrm_names,
        "weights_mmd": w_mmd,
        "weights_vrm": w_vrm,
    }
    out_path = Path(out_path)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return out_path
