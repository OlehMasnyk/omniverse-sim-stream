"""Orbit camera driven by mouse drag / wheel events streamed from the browser.

Input callbacks arrive on an ovstream SDK thread; the render thread reads the
matrix. A lock keeps the two consistent.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import ovstage
import ovstream

TARGET = np.array([0.0, 1.0, 0.0])
YAW_GAIN = 0.005  # radians per pixel
PITCH_GAIN = 0.005
ZOOM_GAIN = 0.8  # metres per wheel notch
PITCH_LIMIT = math.radians(85.0)


class OrbitCamera:
    def __init__(self, radius: float = 15.5, yaw: float = 0.0, pitch: float = math.radians(16.0)) -> None:
        self._lock = threading.Lock()
        self._home = (radius, yaw, pitch)
        self._radius, self._yaw, self._pitch = self._home
        self._target = TARGET.copy()
        self._dragging = False
        self._last: tuple[int, int] | None = None
        self.follow = False
        self.dirty = True

    def reset(self) -> None:
        with self._lock:
            self._radius, self._yaw, self._pitch = self._home
            self._target = TARGET.copy()
            self.follow = False
            self.dirty = True

    def set_target(self, point) -> None:
        """Move the orbit centre (used to follow the rover)."""
        point = np.asarray(point, dtype=np.float64)
        with self._lock:
            if np.linalg.norm(point - self._target) > 1e-3:
                self._target = point
                self.dirty = True

    def on_mouse(self, mouse) -> None:
        with self._lock:
            if mouse.type == ovstream.MouseEventType.BUTTON and mouse.data == ovstream.MouseButton.LEFT:
                self._dragging = mouse.button_state == ovstream.KeyState.DOWN
                self._last = (mouse.x, mouse.y) if self._dragging else None
            elif mouse.type == ovstream.MouseEventType.MOVE and self._dragging:
                if self._last is not None:
                    dx, dy = mouse.x - self._last[0], mouse.y - self._last[1]
                    self._yaw -= dx * YAW_GAIN
                    self._pitch = max(-0.05, min(PITCH_LIMIT, self._pitch + dy * PITCH_GAIN))
                    self.dirty = True
                self._last = (mouse.x, mouse.y)
            elif mouse.type == ovstream.MouseEventType.WHEEL:
                self._radius = max(2.0, min(80.0, self._radius - mouse.scroll_y * ZOOM_GAIN))
                self.dirty = True

    def matrix(self) -> np.ndarray:
        """Camera-to-world transform, (4, 4) row-major, Y-up, camera looks down local -Z."""
        with self._lock:
            radius, yaw, pitch, target = self._radius, self._yaw, self._pitch, self._target
            self.dirty = False
        eye = target + radius * np.array(
            [math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)]
        )
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        m = np.eye(4)
        m[0, :3], m[1, :3], m[2, :3], m[3, :3] = right, up, -forward, eye
        return m


class CameraWriter:
    """Writes the camera's ``omni:xform`` into the stage at a given ordinal."""

    def __init__(self, stage: ovstage.Stage, prim_path: str) -> None:
        self._stage = stage
        self._paths = ovstage.PathDictionary(stage)
        self._path_list = self._paths.create_path_list_from_strings([prim_path])
        self._query = stage.query_from_path_list(self._path_list)
        self._attribute = self._paths.intern_token("omni:xform")
        self._dtype = ovstage.numpy_to_dldatatype(np.dtype(np.float64), lanes=16)

    def write(self, matrix: np.ndarray, ordinal: int) -> None:
        tensor = ovstage.make_dltensor(matrix.reshape(1, 4, 4), dtype=self._dtype, shape=[1], ndim=1)
        self._stage.write_attribute(
            self._query, self._attribute, ordinal=ordinal, tensors=tensor, is_array=False
        ).wait()

    def release(self) -> None:
        self._query.release().wait()
        self._paths.destroy_path_list(self._path_list)
        self._paths.destroy()
