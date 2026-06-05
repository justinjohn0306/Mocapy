"""Project paths — single source of truth for where everything lives.

The Mocapy layout:

    Mocapy/
      mocapy/        ← this package (PKG_ROOT)
      models/        ← MediaPipe .task files (MODELS)
      fixtures/      ← golden captures + test.mp4 (FIXTURES)
      assets/        ← VRM avatars (ASSETS)
      samples/       ← BVH outputs (SAMPLES)
      validation/, tools/, docs/

Importing this from anywhere (`from mocapy._paths import FIXTURES, ASSETS, MODELS`)
gives you absolute paths that work regardless of cwd. Override the project root by
setting the MOCAPY_ROOT environment variable (for tests that run on a copy of the data).
"""

from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = Path(os.environ.get("MOCAPY_ROOT", str(PKG_ROOT.parent))).resolve()

MODELS: Path = PROJECT_ROOT / "models"
FIXTURES: Path = PROJECT_ROOT / "fixtures"
ASSETS: Path = PROJECT_ROOT / "assets"
SAMPLES: Path = PROJECT_ROOT / "samples"
DOCS: Path = PROJECT_ROOT / "docs"
