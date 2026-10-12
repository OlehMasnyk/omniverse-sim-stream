"""Process environment that must be set before ovrtx / ovstage are imported."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def configure() -> None:
    os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

    cache = PROJECT_ROOT / ".cache"
    for sub in ("cuda", "gl", "warp"):
        (cache / sub).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("CUDA_CACHE_PATH", str(cache / "cuda"))
    os.environ.setdefault("__GL_SHADER_DISK_CACHE_PATH", str(cache / "gl"))

    spec = importlib.util.find_spec("ovrtx")
    if spec and spec.origin and "OVRTX_BIN_PATH" not in os.environ:
        os.environ["OVRTX_BIN_PATH"] = str(Path(spec.origin).parent / "bin")
