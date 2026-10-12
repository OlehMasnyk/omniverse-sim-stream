"""A four-wheeled skid-steer rover: its USD description and its drive controller.

The rover is a PhysX *articulation*: a floating-base tree of rigid links joined by
joints. The chassis is the root link; four wheels hang off it on revolute joints
with velocity drives (motors), and a lidar puck sits on top on a fixed joint.

Driving works like a tank: wheels on the same side always spin together, and the
rover turns by spinning the left and right sides at different speeds.

Conventions (Y-up world, metres): the rover's forward axis is local +X, its
left side is -Z and its right side is +Z. Heading is measured counter-clockwise
around +Y, so heading 0 faces +X and heading +90 degrees faces -Z.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

ROBOT_PATH = "/World/Rover"
CHASSIS_PATH = f"{ROBOT_PATH}/Chassis"

CHASSIS_SIZE = (1.0, 0.25, 0.7)  # length (X), height (Y), width (Z)
CHASSIS_Y = 0.45
WHEEL_RADIUS = 0.25
WHEEL_WIDTH = 0.16
WHEEL_X = 0.32  # front/back offset of the axles from the chassis centre
WHEEL_Z = 0.45  # left/right offset of the wheels from the chassis centre
TRACK_WIDTH = 2 * WHEEL_Z

# name -> (x, z) offset in the chassis frame; z < 0 is the left side
WHEELS = {
    "WheelFL": (WHEEL_X, -WHEEL_Z),
    "WheelFR": (WHEEL_X, WHEEL_Z),
    "WheelRL": (-WHEEL_X, -WHEEL_Z),
    "WheelRR": (-WHEEL_X, WHEEL_Z),
}

LINK_COUNT = len(WHEELS) + 2  # wheels + chassis + lidar

MAX_LINEAR = 2.5  # m/s
MAX_ANGULAR = 2.0  # rad/s
SPAWN = (-7.0, 0.0)  # (x, z), facing +X towards the drop zone

# Windows virtual-key codes, as sent by the browser client.
_VK_LEFT, _VK_UP, _VK_RIGHT, _VK_DOWN = 0x25, 0x26, 0x27, 0x28
_VK_W, _VK_A, _VK_S, _VK_D, _VK_SPACE = 0x57, 0x41, 0x53, 0x44, 0x20
_FORWARD_KEYS = {_VK_UP, _VK_W}
_BACK_KEYS = {_VK_DOWN, _VK_S}
_LEFT_KEYS = {_VK_LEFT, _VK_A}
_RIGHT_KEYS = {_VK_RIGHT, _VK_D}
DRIVE_KEYS = _FORWARD_KEYS | _BACK_KEYS | _LEFT_KEYS | _RIGHT_KEYS | {_VK_SPACE}


def rover_usda() -> str:
    """USD for the rover, to be placed inside /World."""
    sx, sy, sz = CHASSIS_SIZE
    x0, z0 = SPAWN
    wheel_y = WHEEL_RADIUS
    wheels = []
    for name, (dx, dz) in WHEELS.items():
        # Joint frames are in each body's local space, which includes the
        # chassis scale, so the chassis-side anchor is divided by it.
        anchor = (dx / sx, (wheel_y - CHASSIS_Y) / sy, dz / sz)
        wheels.append(f"""
        def Cylinder "{name}" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]
        )
        {{
            double radius = {WHEEL_RADIUS}
            double height = {WHEEL_WIDTH}
            uniform token axis = "Z"
            float3[] extent = [(-{WHEEL_RADIUS}, -{WHEEL_RADIUS}, -{WHEEL_WIDTH / 2}), ({WHEEL_RADIUS}, {WHEEL_RADIUS}, {WHEEL_WIDTH / 2})]
            color3f[] primvars:displayColor = [(0.06, 0.06, 0.07)]
            float physics:mass = 1.0
            double3 xformOp:translate = ({x0 + dx}, {wheel_y}, {z0 + dz})
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}

        def PhysicsRevoluteJoint "Joint{name[5:]}" (
            prepend apiSchemas = ["PhysicsDriveAPI:angular"]
        )
        {{
            rel physics:body0 = <{CHASSIS_PATH}>
            rel physics:body1 = <{ROBOT_PATH}/{name}>
            uniform token physics:axis = "Z"
            point3f physics:localPos0 = ({anchor[0]:.5f}, {anchor[1]:.5f}, {anchor[2]:.5f})
            point3f physics:localPos1 = (0, 0, 0)
            uniform token drive:angular:physics:type = "force"
            float drive:angular:physics:stiffness = 0
            float drive:angular:physics:damping = 1000
            float drive:angular:physics:maxForce = 80
            float drive:angular:physics:targetVelocity = 0
        }}
""")
    lidar_anchor = (0.15 / sx, (sy / 2 + 0.06) / sy, 0.0)
    return f"""
    def Xform "Rover"
    {{
        def Cube "Chassis" (
            prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI",
                                  "PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]
        )
        {{
            bool physxArticulation:enabledSelfCollisions = 0
            double size = 1
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            color3f[] primvars:displayColor = [(0.46, 0.73, 0.0)]
            float physics:mass = 10.0
            double3 xformOp:translate = ({x0}, {CHASSIS_Y}, {z0})
            float3 xformOp:scale = ({sx}, {sy}, {sz})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}

        def Cylinder "Lidar" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        )
        {{
            double radius = 0.12
            double height = 0.12
            uniform token axis = "Y"
            float3[] extent = [(-0.12, -0.06, -0.12), (0.12, 0.06, 0.12)]
            color3f[] primvars:displayColor = [(0.9, 0.9, 0.92)]
            float physics:mass = 0.3
            double3 xformOp:translate = ({x0 + 0.15}, {CHASSIS_Y + sy / 2 + 0.06}, {z0})
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}

        def PhysicsFixedJoint "JointLidar"
        {{
            rel physics:body0 = <{CHASSIS_PATH}>
            rel physics:body1 = <{ROBOT_PATH}/Lidar>
            point3f physics:localPos0 = ({lidar_anchor[0]:.5f}, {lidar_anchor[1]:.5f}, {lidar_anchor[2]:.5f})
            point3f physics:localPos1 = (0, 0, 0)
        }}
{"".join(wheels)}
    }}
"""


# Per-prim scale for bodies that author one; the physics write-back needs it to
# rebuild full world matrices from position + orientation.
BODY_SCALES = {CHASSIS_PATH: CHASSIS_SIZE}


@dataclass
class Pose:
    x: float
    y: float
    z: float
    heading: float  # radians, CCW around +Y; 0 = facing +X

    @classmethod
    def from_physics(cls, position: np.ndarray, quat_xyzw: np.ndarray) -> "Pose":
        qx, qy, qz, qw = (float(c) for c in quat_xyzw)
        fx = 1 - 2 * (qy * qy + qz * qz)  # rotated local +X
        fz = 2 * (qx * qz - qy * qw)
        return cls(float(position[0]), float(position[1]), float(position[2]), math.atan2(-fz, fx))


class DriveController:
    """Turns driver intent (keys, REST commands, a goal point) into wheel speeds.

    Keyboard callbacks run on an ovstream thread and REST commands on the render
    thread, so state is guarded by a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: set[int] = set()
        self._linear = 0.0
        self._angular = 0.0
        self._until: float | None = None
        self._goal: tuple[float, float] | None = None
        self.mode = "idle"

    # ---------------------------------------------------------------- commands
    def drive(self, linear: float, angular: float, duration: float | None, now: float) -> None:
        with self._lock:
            self._linear = _clamp(linear, MAX_LINEAR)
            self._angular = _clamp(angular, MAX_ANGULAR)
            self._until = now + duration if duration else None
            self._goal = None
            self.mode = "drive"

    def goto(self, x: float, z: float) -> None:
        with self._lock:
            self._goal = (x, z)
            self.mode = "goto"

    def stop(self) -> None:
        with self._lock:
            self._linear = self._angular = 0.0
            self._until = self._goal = None
            self.mode = "idle"

    def on_key(self, key_code: int, down: bool) -> bool:
        """Track held drive keys. Returns False for keys the rover doesn't use."""
        if key_code not in DRIVE_KEYS:
            return False
        with self._lock:
            if key_code == _VK_SPACE:
                self._keys.clear()
                self._linear = self._angular = 0.0
                self._until = self._goal = None
                self.mode = "idle"
            elif down:
                self._keys.add(key_code)
                self._goal = None
                self.mode = "keys"
            else:
                self._keys.discard(key_code)
        return True

    # ---------------------------------------------------------------- control
    def body_velocity(self, pose: Pose | None, now: float) -> tuple[float, float]:
        """Desired (linear m/s, angular rad/s) for this step."""
        with self._lock:
            if self.mode == "keys":
                fwd = bool(self._keys & _FORWARD_KEYS) - bool(self._keys & _BACK_KEYS)
                turn = bool(self._keys & _LEFT_KEYS) - bool(self._keys & _RIGHT_KEYS)
                return fwd * MAX_LINEAR * 0.8, turn * MAX_ANGULAR * 0.7
            if self.mode == "drive":
                if self._until is not None and now >= self._until:
                    self._linear = self._angular = 0.0
                    self._until = None
                    self.mode = "idle"
                return self._linear, self._angular
            if self.mode == "goto" and pose is not None and self._goal is not None:
                dx, dz = self._goal[0] - pose.x, self._goal[1] - pose.z
                distance = math.hypot(dx, dz)
                if distance < 0.35:
                    self._goal = None
                    self.mode = "idle"
                    return 0.0, 0.0
                error = _wrap(math.atan2(-dz, dx) - pose.heading)
                angular = _clamp(3.0 * error, MAX_ANGULAR)
                # Only drive forward once roughly facing the goal.
                linear = min(MAX_LINEAR, 1.2 * distance) * max(0.0, math.cos(error)) ** 2
                return linear, angular
            return 0.0, 0.0

    def status(self) -> dict:
        with self._lock:
            goal = {"x": self._goal[0], "z": self._goal[1]} if self._goal else None
            return {"mode": self.mode, "goal": goal}


def wheel_speeds(linear: float, angular: float) -> tuple[float, float]:
    """Body velocity -> (left, right) wheel joint velocity targets in degrees/s.

    Wheels spin about +Z; rolling towards +X needs a negative rotation about +Z.
    """
    left = linear - angular * TRACK_WIDTH / 2
    right = linear + angular * TRACK_WIDTH / 2
    to_deg = -180.0 / (math.pi * WHEEL_RADIUS)
    return left * to_deg, right * to_deg


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
