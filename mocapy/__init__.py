"""Mocapy — markerless motion capture: video / webcam → VRM avatar bones → BVH.

A Python reimplementation of a browser-grade mocap solver. The solver and exporters
are pure Python; an optional Node/WASM bridge mirrors the browser detector for
higher-fidelity legs (see SETUP.md).

Quick usage (single-process, when one env has both mediapipe + the VRM tools):

    from mocapy.pipeline import video_to_bvh
    bvh_text = video_to_bvh("input.mp4", "avatar.vrm")
    open("out.bvh", "w", encoding="utf-8").write(bvh_text)

For the standard two-stage CLI (recommended), see SETUP.md.
"""

from mocapy._paths import (  # noqa: F401  (public re-exports)
    PROJECT_ROOT, MODELS, FIXTURES, ASSETS, SAMPLES, DOCS,
)

__version__ = "0.1.0"
__all__ = ["PROJECT_ROOT", "MODELS", "FIXTURES", "ASSETS", "SAMPLES", "DOCS", "__version__"]
