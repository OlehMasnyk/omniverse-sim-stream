"""The simulation runtime: one render thread that owns everything GPU-side.

Per frame the render thread:
  1. runs queued commands (load / start / pause / reset ...) from the HTTP API,
  2. if running, steps PhysX and writes body poses into the ovstage Stage,
  3. writes the camera transform if the user moved it,
  4. seals that ordinal, renders it with ovrtx and hands the BGRA CUDA buffer to ovstream.

Nothing else touches the Stage, the renderer or PhysX; other threads talk to the
runtime only through ``submit()`` (a command queue) and ``status()`` (a snapshot).
"""

from __future__ import annotations

import logging
import math
import queue
import random
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

import ovrtx
import ovstage
import ovstream
import ovstream_utils

from . import robot
from .camera import CameraWriter, OrbitCamera
from .physics import PhysicsSession, register_schemas
from .scene import CAMERA_PATH, RENDER_PRODUCT_PATH, SimulationParams, write_scene

log = logging.getLogger("runtime")

_VK_C, _VK_F = 0x43, 0x46  # camera keys: reset view, toggle follow-the-rover


@dataclass
class RuntimeConfig:
    width: int = 1280
    height: int = 720
    fps: int = 60
    signal_port: int = 49100
    public_ip: str | None = None
    render_mode: str = "RealTimePathTracing"


class SimRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.camera = OrbitCamera()
        self.robot = robot.DriveController()
        self._commands: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue()
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {"state": "starting"}
        self._started = Future()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="render", daemon=True)

        # Owned by the render thread only.
        self._stage: ovstage.Stage | None = None
        self._renderer: ovrtx.Renderer | None = None
        self._physics: PhysicsSession | None = None
        self._camera_writer: CameraWriter | None = None
        self._server: ovstream.Server | None = None
        self._ordinal = 0
        self._generation = 0
        self._params = SimulationParams()
        self._running = False
        self._stop_at_time: float | None = None
        self._sim_time = 0.0
        self._steps = 0
        self._bodies = 0
        self._wheel_targets: tuple[float, float] | None = None
        self._robot_pose: robot.Pose | None = None

    # ------------------------------------------------------------------ public API
    def start(self, timeout: float = 900.0) -> None:
        """Start the render thread and block until the first frame is streamable."""
        self._thread.start()
        self._started.result(timeout=timeout)

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def submit(self, fn_name: str, **kwargs: Any) -> Future:
        """Queue a command for the render thread; returns a Future with its result."""
        fn = getattr(self, f"_cmd_{fn_name}")
        future: Future = Future()
        self._commands.put((lambda: fn(**kwargs), future))
        return future

    # ------------------------------------------------------------------ commands (render thread)
    def _cmd_load(self, params: SimulationParams, autostart: bool = False) -> dict:
        params = params.validated()
        if params.seed is None:
            params = replace(params, seed=random.randrange(1_000_000))
        self._load_scene(params)
        self._running = autostart
        return self._snapshot()

    def _cmd_start(self, duration: float | None = None) -> dict:
        self._running = True
        self._stop_at_time = self._sim_time + duration if duration else None
        return self._snapshot()

    def _cmd_pause(self) -> dict:
        self._running = False
        self.robot.stop()
        return self._snapshot()

    def _cmd_reset(self) -> dict:
        self._load_scene(self._params)
        self._running = False
        return self._snapshot()

    def _cmd_reset_camera(self) -> dict:
        self.camera.reset()
        return self._snapshot()

    def _cmd_camera_follow(self, enabled: bool) -> dict:
        self.camera.follow = enabled and self._params.robot
        return self._snapshot()

    def _cmd_robot_drive(self, linear: float, angular: float, duration: float | None = None) -> dict:
        self._require_robot()
        self.robot.drive(linear, angular, duration, now=self._sim_time)
        return self._snapshot()

    def _cmd_robot_goto(self, x: float, z: float) -> dict:
        self._require_robot()
        self.robot.goto(x, z)
        return self._snapshot()

    def _cmd_robot_stop(self) -> dict:
        self.robot.stop()
        return self._snapshot()

    def _require_robot(self) -> None:
        if not self._params.robot:
            raise ValueError("this simulation has no robot; launch one with robot=true")

    # ------------------------------------------------------------------ scene lifecycle
    def _load_scene(self, params: SimulationParams) -> None:
        """Build a fresh Stage for ``params`` and swap the renderer + PhysX onto it."""
        cfg = self.config
        self._generation += 1
        composite = write_scene(params, cfg.width, cfg.height, f"sim_{self._generation % 2}", cfg.render_mode)

        stage = ovstage.Stage(f"sim-{self._generation}")
        try:
            ovstage.population.open_usd(stage, str(composite), ordinal=1, domains=ovstage.PopulationDomain.ALL)
            stage.advance_write_floor(ordinal=1).wait()
        except Exception:
            stage.destroy()
            raise

        self._teardown_scene()
        if self._renderer is None:
            log.info("Creating ovrtx renderer (first run compiles shaders, can take minutes)...")
            self._renderer = ovrtx.Renderer(config=ovrtx.RendererConfig(sync_mode=True, keep_system_alive=True))
        self._renderer.attach_ovstage(stage)
        self._stage = stage
        self._physics = PhysicsSession(stage, read_ordinal=1, scales=robot.BODY_SCALES)
        self._camera_writer = CameraWriter(stage, CAMERA_PATH)
        self.camera.dirty = True
        self.robot.stop()
        self._wheel_targets = None
        self._robot_pose = robot.Pose(robot.SPAWN[0], robot.CHASSIS_Y, robot.SPAWN[1], 0.0) if params.robot else None
        if not params.robot:
            self.camera.follow = False

        self._params = params
        self._ordinal = 1
        self._sim_time = 0.0
        self._steps = 0
        self._bodies = params.bodies + (robot.LINK_COUNT if params.robot else 0)
        self._stop_at_time = None
        log.info("Loaded scene generation %d: %s", self._generation, params)

    def _teardown_scene(self) -> None:
        if self._physics is not None:
            self._physics.close()
            self._physics = None
        if self._camera_writer is not None:
            self._camera_writer.release()
            self._camera_writer = None
        if self._stage is not None:
            self._renderer.detach_ovstage()
            self._stage.destroy()
            self._stage = None

    # ------------------------------------------------------------------ render loop
    def _run(self) -> None:
        try:
            register_schemas()
            self._cmd_load(self._params)
            self._warm_up()
            self._start_stream()
        except BaseException as exc:  # surface startup failures to start()
            log.exception("Startup failed")
            self._set_status(state="error", last_error=str(exc))
            self._started.set_exception(exc)
            return
        self._started.set_result(True)

        cfg = self.config
        dt = 1.0 / cfg.fps
        try:
            with ovstream_utils.Loop(ovstream_utils.LoopConfig(fps_target=cfg.fps)) as loop:
                while not self._stop.is_set():
                    tick = loop.tick()
                    self._drain_commands()
                    try:
                        self._frame(dt)
                    except Exception:
                        log.exception("Frame failed")
                    self._publish_status(fps=tick.stats.fps_current)
        finally:
            self._shutdown_gpu()

    def _frame(self, dt: float) -> None:
        if self._params.robot and self.robot.mode != "idle":
            self._running = True  # driving the rover wakes a paused simulation
        if self._running:
            self._drive_robot()
        if self._running or self.camera.dirty:
            self._ordinal += 1
            if self._running:
                self._bodies = self._physics.step_and_write(dt, self._ordinal)
                self._steps += 1
                self._sim_time += dt
                self._update_robot_pose()
                if self._stop_at_time is not None and self._sim_time >= self._stop_at_time:
                    self._running = False
                    self._stop_at_time = None
            if self.camera.dirty:
                self._camera_writer.write(self.camera.matrix(), self._ordinal)
            self._stage.advance_write_floor(ordinal=self._ordinal).wait()

        products = self._renderer.step({RENDER_PRODUCT_PATH}, dt, ordinal=self._ordinal)
        if RENDER_PRODUCT_PATH not in products.keys():
            return
        for frame in products[RENDER_PRODUCT_PATH].frames:
            key = _find_var(frame.render_vars.keys(), "BgraColor")
            if key is None:
                continue
            mapping = frame.render_vars[key].map(device=ovrtx.Device.CUDA)
            try:
                # Stream unconditionally: the WebRTC session only completes its
                # start handshake once frames are flowing, so gating on
                # is_client_connected would deadlock.
                if self._server is not None:
                    video = ovstream.VideoFrame(
                        buffer=int(mapping.data),
                        width=self.config.width,
                        height=self.config.height,
                        pitch_bytes=self.config.width * 4,
                        sync=ovstream.CudaSync(wait_event=int(mapping.wait_event or 0)),
                    )
                    try:
                        self._server.stream_video(video)
                    except ovstream.OvstreamError:
                        log.debug("Dropped frame (no client yet?)", exc_info=True)
            finally:
                mapping.unmap()
        del products

    def _drive_robot(self) -> None:
        """Turn the controller's wish into wheel motor targets before the physics step."""
        if not self._params.robot:
            return
        linear, angular = self.robot.body_velocity(self._robot_pose, self._sim_time)
        targets = robot.wheel_speeds(linear, angular)
        if targets == self._wheel_targets:
            return
        left, right = targets
        self._physics.set_joint_velocity_targets({
            f"{robot.ROBOT_PATH}/Joint{name[5:]}": (left if dz < 0 else right)
            for name, (_dx, dz) in robot.WHEELS.items()
        })
        self._wheel_targets = targets

    def _update_robot_pose(self) -> None:
        link = self._physics.link_poses.get(robot.CHASSIS_PATH)
        if link is None:
            return
        self._robot_pose = robot.Pose.from_physics(*link)
        if self.camera.follow:
            self.camera.set_target((self._robot_pose.x, 0.5, self._robot_pose.z))

    def _on_input(self, event) -> None:
        """ovstream input callback (SDK thread): keys drive the rover, the mouse orbits the camera."""
        if event.type == ovstream.InputEventType.MOUSE:
            self.camera.on_mouse(event.mouse)
        elif event.type == ovstream.InputEventType.KEYBOARD:
            key = event.keyboard.key_code
            down = event.keyboard.key_state == ovstream.KeyState.DOWN
            log.debug("key %#04x %s", key, "down" if down else "up")
            if self._params.robot and self.robot.on_key(key, down):
                return
            if down and key == _VK_C:
                self.camera.reset()
            elif down and key == _VK_F and self._params.robot:
                self.camera.follow = not self.camera.follow

    def _warm_up(self) -> None:
        """Step until the BGRA output exists and check it matches the configured size."""
        for _ in range(600):
            products = self._renderer.step({RENDER_PRODUCT_PATH}, 1 / 60, ordinal=self._ordinal)
            if RENDER_PRODUCT_PATH in products.keys():
                frame = products[RENDER_PRODUCT_PATH].frames[0]
                key = _find_var(frame.render_vars.keys(), "BgraColor")
                if key is not None:
                    mapping = frame.render_vars[key].map(device=ovrtx.Device.CUDA)
                    shape = tuple(mapping.shape)
                    mapping.unmap()
                    expected = (self.config.height, self.config.width, 4)
                    if shape != expected:
                        raise RuntimeError(f"BgraColor shape {shape} != expected {expected}")
                    log.info("First BGRA frame ready: %dx%d", self.config.width, self.config.height)
                    return
            del products
        raise RuntimeError("Renderer never produced the BgraColor output")

    def _start_stream(self) -> None:
        cfg = self.config
        stream_log = logging.getLogger("ovstream")
        ovstream.initialize(
            log_fn=lambda level, channel, msg, _ts: stream_log.info("[%s] %s: %s", level.name, channel, msg),
            log_min_severity=ovstream.LogLevel.WARNING,
        )
        server = ovstream.Server(ovstream.ServerType.WEBRTC)
        server.on_connection = lambda connected: log.info("WebRTC client %s", "connected" if connected else "disconnected")
        server.on_input = self._on_input
        server_cfg = ovstream.ServerConfig(
            width=cfg.width, height=cfg.height, target_fps=cfg.fps, cuda_device=0,
            webrtc_signal_port=cfg.signal_port,
        )
        if cfg.public_ip:
            server_cfg.webrtc_public_ip = cfg.public_ip
        server.start(server_cfg)
        self._server = server
        log.info("WebRTC signaling on port %d", cfg.signal_port)

    def _shutdown_gpu(self) -> None:
        if self._server is not None:
            try:
                self._server.stop()
            except ovstream.OvstreamError:
                pass
            self._server.close()
            ovstream.shutdown()
        self._teardown_scene()
        if self._renderer is not None:
            self._renderer.destroy()

    # ------------------------------------------------------------------ helpers
    def _drain_commands(self) -> None:
        while True:
            try:
                fn, future = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                future.set_result(fn())
            except Exception as exc:
                log.warning("Command failed: %s", exc)
                future.set_exception(exc)

    def _snapshot(self) -> dict:
        snapshot = {
            "state": "running" if self._running else ("ready" if self._steps == 0 else "paused"),
            "generation": self._generation,
            "sim_time": round(self._sim_time, 3),
            "steps": self._steps,
            "bodies": self._bodies,
            "params": asdict(self._params),
            "camera_follow": self.camera.follow,
        }
        if self._params.robot:
            pose = self._robot_pose
            snapshot["robot"] = {
                **self.robot.status(),
                "pose": None if pose is None else {
                    "x": round(pose.x, 3), "z": round(pose.z, 3),
                    "heading_deg": round(math.degrees(pose.heading), 1),
                },
            }
        return snapshot

    def _publish_status(self, fps: float) -> None:
        connected = bool(self._server and self._server.is_client_connected)
        self._set_status(**self._snapshot(), render_fps=round(fps, 1), client_connected=connected)

    def _set_status(self, **values: Any) -> None:
        with self._lock:
            self._status.update(values)


def _find_var(keys, name: str) -> str | None:
    """Render var keys are prim paths in this ovrtx build; match on the last segment."""
    return next((k for k in keys if k.rsplit("/", 1)[-1] == name), None)
