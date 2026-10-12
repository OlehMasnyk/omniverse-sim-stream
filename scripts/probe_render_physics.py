"""Smoke test: one shared ovstage Stage, ovphysx simulates, ovrtx renders.

Writes out/frame_000.png (before simulation) and out/frame_060.png (after 1 s).
If the bodies moved between the two images, physics output reaches the renderer.

    .venv\\Scripts\\python.exe scripts\\probe_render_physics.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import env

env.configure()

import numpy as np  # noqa: E402
import ovstage  # noqa: E402
from ovrtx import Device, Renderer, RendererConfig  # noqa: E402
from PIL import Image  # noqa: E402

from server.physics import PhysicsSession, register_schemas  # noqa: E402
from server.scene import RENDER_PRODUCT_PATH, SimulationParams, write_scene  # noqa: E402

WIDTH, HEIGHT = 1280, 720
OUT = env.PROJECT_ROOT / "out"


def save_frame(products, path: Path | None) -> bool:
    """Save LdrColor if present; return whether a frame was available."""

    def grab(ctx) -> bool:
        if RENDER_PRODUCT_PATH not in ctx:
            print("render product missing; outputs:", list(ctx.keys()))
            return False
        for frame in ctx[RENDER_PRODUCT_PATH].frames:
            key = next((k for k in frame.render_vars.keys() if k.rsplit("/", 1)[-1] == "LdrColor"), None)
            if key is None:
                print("LdrColor missing; render vars:", list(frame.render_vars.keys()))
                return False
            if path is None:
                return True
            with frame.render_vars[key].map(device=Device.CPU) as rv:
                pixels = np.from_dlpack(rv).copy()
            Image.fromarray(pixels[..., :3]).save(path)
            print(f"saved {path.name}: shape={pixels.shape} mean={pixels[..., :3].mean():.1f}")
            return True
        return False

    if hasattr(products, "__enter__"):
        with products as ctx:
            return grab(ctx)
    return grab(products)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    composite = write_scene(SimulationParams(bodies=12, seed=7), WIDTH, HEIGHT, name="probe")
    print("composite:", composite)

    register_schemas()
    stage = ovstage.Stage("probe")
    ordinal = 1
    ovstage.population.open_usd(stage, str(composite), ordinal=ordinal, domains=ovstage.PopulationDomain.ALL)
    stage.advance_write_floor(ordinal=ordinal).wait()

    t0 = time.time()
    renderer = Renderer(config=RendererConfig(sync_mode=True, keep_system_alive=True))
    renderer.attach_ovstage(stage)
    print(f"renderer ready in {time.time() - t0:.1f}s, version={renderer.version}")

    physics = PhysicsSession(stage, read_ordinal=ordinal)

    t0 = time.time()
    ready_at = None
    for i in range(600):  # first steps load the scene and compile shaders
        products = renderer.step({RENDER_PRODUCT_PATH}, 1 / 60, ordinal=ordinal)
        if save_frame(products, None):
            ready_at = ready_at if ready_at is not None else i
            if i - ready_at >= 30:  # let the path tracer converge a little
                break
    print(f"first LdrColor at step {ready_at}")
    save_frame(renderer.step({RENDER_PRODUCT_PATH}, 1 / 60, ordinal=ordinal), OUT / "frame_000.png")
    print(f"first frames took {time.time() - t0:.1f}s")

    dt = 1 / 60
    t0 = time.time()
    for _ in range(60):
        ordinal += 1
        bodies = physics.step_and_write(dt, output_ordinal=ordinal)
        stage.advance_write_floor(ordinal=ordinal).wait()
        products = renderer.step({RENDER_PRODUCT_PATH}, dt, ordinal=ordinal)
    print(f"60 sim+render frames took {time.time() - t0:.2f}s ({bodies} bodies)")
    save_frame(products, OUT / "frame_060.png")

    physics.close()
    renderer.detach_ovstage()
    del renderer
    stage.destroy()
    print("OK")


if __name__ == "__main__":
    main()
