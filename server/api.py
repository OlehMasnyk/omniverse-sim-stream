"""REST API for controlling the simulation. Interactive docs at http://localhost:8000/docs."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import robot
from .runtime import SimRuntime
from .scene import SimulationParams

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
COMMAND_TIMEOUT_S = 60.0


class SimulationRequest(BaseModel):
    bodies: int = Field(24, ge=0, le=500, description="Number of falling rigid bodies")
    drop_height: float = Field(6.0, ge=0.5, le=50, description="Height (m) of the lowest layer of bodies")
    spread: float = Field(3.0, ge=0.5, le=20, description="Half-width (m) of the drop area")
    sphere_ratio: float = Field(0.3, ge=0, le=1, description="Fraction of bodies that are spheres")
    seed: int | None = Field(None, description="Random seed; omit for a random layout")
    autostart: bool = Field(True, description="Start simulating immediately after loading")
    robot: bool = Field(True, description="Add the drivable rover to the scene")


class StartRequest(BaseModel):
    duration: float | None = Field(None, gt=0, description="Simulated seconds to run, then auto-pause")


class DriveRequest(BaseModel):
    linear: float = Field(0.0, ge=-robot.MAX_LINEAR, le=robot.MAX_LINEAR, description="Forward speed (m/s); negative reverses")
    angular: float = Field(0.0, ge=-robot.MAX_ANGULAR, le=robot.MAX_ANGULAR, description="Turn rate (rad/s); positive turns left")
    duration: float | None = Field(None, gt=0, description="Simulated seconds to drive, then stop; omit to keep driving")


class GotoRequest(BaseModel):
    x: float = Field(..., ge=-19, le=19, description="Target X (m) on the ground")
    z: float = Field(..., ge=-19, le=19, description="Target Z (m) on the ground")


class FollowRequest(BaseModel):
    enabled: bool = True


def create_app(runtime: SimRuntime, signal_port: int) -> FastAPI:
    app = FastAPI(
        title="Omniverse Sim Stream",
        description="Launch an ovphysx rigid-body simulation and watch it live over WebRTC.",
    )

    async def run(command: str, **kwargs):
        future = runtime.submit(command, **kwargs)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), COMMAND_TIMEOUT_S)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/healthz", tags=["system"])
    def healthz():
        status = runtime.status()
        ready = status.get("state") not in ("starting", "error")
        return JSONResponse({"ready": ready}, status_code=200 if ready else 503)

    @app.get("/api/config", tags=["system"])
    def config():
        cfg = runtime.config
        return {"signal_port": signal_port, "width": cfg.width, "height": cfg.height, "fps": cfg.fps}

    @app.get("/api/simulation", tags=["simulation"])
    def get_simulation():
        """Current state, simulated time, render FPS and whether a WebRTC viewer is connected."""
        return runtime.status()

    @app.post("/api/simulation", tags=["simulation"])
    async def launch_simulation(req: SimulationRequest):
        """Build a new scene from the parameters and (by default) start simulating it."""
        params = SimulationParams(
            bodies=req.bodies, drop_height=req.drop_height, spread=req.spread,
            sphere_ratio=req.sphere_ratio, seed=req.seed, robot=req.robot,
        )
        return await run("load", params=params, autostart=req.autostart)

    @app.post("/api/simulation/start", tags=["simulation"])
    async def start(req: StartRequest | None = None):
        return await run("start", duration=req.duration if req else None)

    @app.post("/api/simulation/pause", tags=["simulation"])
    async def pause():
        return await run("pause")

    @app.post("/api/simulation/reset", tags=["simulation"])
    async def reset():
        """Reload the current scene (same seed) back to its initial state."""
        return await run("reset")

    @app.post("/api/robot/drive", tags=["robot"])
    async def robot_drive(req: DriveRequest):
        """Drive the rover at a fixed speed and turn rate, optionally for a duration."""
        return await run("robot_drive", linear=req.linear, angular=req.angular, duration=req.duration)

    @app.post("/api/robot/goto", tags=["robot"])
    async def robot_goto(req: GotoRequest):
        """Drive the rover autonomously to a point on the ground (it turns, then drives)."""
        return await run("robot_goto", x=req.x, z=req.z)

    @app.post("/api/robot/stop", tags=["robot"])
    async def robot_stop():
        return await run("robot_stop")

    @app.post("/api/camera/reset", tags=["camera"])
    async def reset_camera():
        return await run("reset_camera")

    @app.post("/api/camera/follow", tags=["camera"])
    async def camera_follow(req: FollowRequest):
        """Keep the orbit camera centred on the rover."""
        return await run("camera_follow", enabled=req.enabled)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
