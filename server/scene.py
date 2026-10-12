"""Generates the USD files the simulation runs on.

Two layers are written per simulation:

* ``<name>.usda`` - the physics scene (ground, rigid bodies, lights).
* ``_composite_<name>.usda`` - viewer-owned layer that sublayers the scene and adds
  the camera, RenderProduct, RenderVar and RenderSettings that ovrtx needs.

Rigid bodies are authored directly on the Cube/Sphere gprims (no child geometry):
physics publishes a world matrix per rigid-body prim and does not propagate it to
descendants.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .robot import rover_usda

CAMERA_PATH = "/OVCamera"
RENDER_PRODUCT_PATH = "/Render/OVServer/ViewportTexture0"
CAMERA_HORIZONTAL_APERTURE = 20.955

SCENES_DIR = Path(__file__).resolve().parent.parent / "scenes" / "_generated"


@dataclass(frozen=True)
class SimulationParams:
    bodies: int = 24
    drop_height: float = 6.0
    spread: float = 3.0
    sphere_ratio: float = 0.3
    seed: int | None = None
    robot: bool = True

    def validated(self) -> "SimulationParams":
        if not 0 <= self.bodies <= 500:
            raise ValueError("bodies must be between 0 and 500")
        if not 0.5 <= self.drop_height <= 50:
            raise ValueError("drop_height must be between 0.5 and 50")
        if not 0.5 <= self.spread <= 20:
            raise ValueError("spread must be between 0.5 and 20")
        if not 0.0 <= self.sphere_ratio <= 1.0:
            raise ValueError("sphere_ratio must be between 0 and 1")
        return self


def _body_usda(index: int, rng: random.Random, params: SimulationParams) -> str:
    x = rng.uniform(-params.spread, params.spread)
    z = rng.uniform(-params.spread, params.spread)
    y = params.drop_height + (index // 9) * 1.2 + rng.uniform(0.0, 0.4)
    color = (rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0))
    # Random initial orientation (w, x, y, z), normalised.
    q = [rng.uniform(-1, 1) for _ in range(4)]
    norm = sum(c * c for c in q) ** 0.5 or 1.0
    w, qx, qy, qz = (c / norm for c in q)

    is_sphere = rng.random() < params.sphere_ratio
    prim_type = "Sphere" if is_sphere else "Cube"
    shape_attrs = (
        "double radius = 0.4\n        float3[] extent = [(-0.4, -0.4, -0.4), (0.4, 0.4, 0.4)]"
        if is_sphere
        else "double size = 0.8\n        float3[] extent = [(-0.4, -0.4, -0.4), (0.4, 0.4, 0.4)]"
    )
    return f"""
    def {prim_type} "Body_{index:03d}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]
    )
    {{
        {shape_attrs}
        color3f[] primvars:displayColor = [({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})]
        float physics:mass = 1.0
        double3 xformOp:translate = ({x:.4f}, {y:.4f}, {z:.4f})
        quatd xformOp:orient = ({w:.5f}, {qx:.5f}, {qy:.5f}, {qz:.5f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}
"""


def physics_scene_usda(params: SimulationParams) -> str:
    rng = random.Random(params.seed)
    bodies = "".join(_body_usda(i, rng, params) for i in range(params.bodies))
    if params.robot:
        bodies += rover_usda()
    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Y"
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, -1, 0)
        float physics:gravityMagnitude = 9.81
    }}

    def DomeLight "Sky"
    {{
        float inputs:intensity = 250
        color3f inputs:color = (0.85, 0.9, 1.0)
    }}

    def DistantLight "Sun"
    {{
        float inputs:intensity = 1200
        float inputs:angle = 1.0
        quatd xformOp:orient = (0.8536, -0.3536, 0.3536, 0.1464)
        uniform token[] xformOpOrder = ["xformOp:orient"]
    }}

    def Cube "Ground" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        double size = 1
        float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
        color3f[] primvars:displayColor = [(0.35, 0.36, 0.38)]
        double3 xformOp:translate = (0, -0.1, 0)
        double3 xformOp:scale = (40, 0.2, 40)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
{bodies}
}}
"""


def composite_usda(scene_basename: str, width: int, height: int, render_mode: str = "RealTimePathTracing") -> str:
    vertical_aperture = CAMERA_HORIZONTAL_APERTURE * float(height) / float(width)
    return f"""#usda 1.0
(
    subLayers = [
        @./{scene_basename}@
    ]
)

def Camera "OVCamera"
{{
    float2 clippingRange = (0.05, 10000)
    float focalLength = 18.15
    float horizontalAperture = {CAMERA_HORIZONTAL_APERTURE:.3f}
    float verticalAperture = {vertical_aperture:.4f}
    token projection = "perspective"
    double3 xformOp:translate = (0, 5, 15)
    float3 xformOp:rotateXYZ = (-12, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
}}

def "Render"
{{
    def "OVServer"
    {{
        def RenderProduct "ViewportTexture0" (
            prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1", "OmniRtxSettingsPtAdvancedAPI_1", "OmniRtxSettingsRtAdvancedAPI_1"]
        )
        {{
            token omni:rtx:rendermode = "{render_mode}"
            uniform int2 resolution = ({width}, {height})
            uint[] deviceIds = [0]
            rel camera = <{CAMERA_PATH}>
            rel orderedVars = [
                <LdrColor>,
                <BgraColor>,
            ]

            def RenderVar "LdrColor"
            {{
                uniform string sourceName = "LdrColor"
                opaque omni:rtx:aov
            }}

            # ovstream wants BGRA8; this SPG node swaps R and B on the GPU so the
            # mapped CUDA buffer can be streamed as-is.
            def RenderVar "BgraColor"
            {{
                uniform string sourceName = "BgraColor"
                opaque omni:rtx:aov.connect = <../Swizzle.outputs:Output>
            }}

            def Shader "Swizzle"
            {{
                uniform token info:implementationSource = "id"
                uniform token info:id = "spg:rtx.spg.stdlib/Swizzle"
                token inputs:swizzle = "zyxw"
                opaque inputs:Input.connect = <../LdrColor.omni:rtx:aov>
                opaque outputs:Output
            }}
        }}
    }}

    def RenderSettings "OVRenderSettings"
    {{
        rel products = [<{RENDER_PRODUCT_PATH}>]
    }}
}}
"""


def write_scene(
    params: SimulationParams,
    width: int,
    height: int,
    name: str = "sim",
    render_mode: str = "RealTimePathTracing",
) -> Path:
    """Write the physics scene and its composite; return the composite path."""
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    scene_path = SCENES_DIR / f"{name}.usda"
    composite_path = SCENES_DIR / f"_composite_{name}.usda"
    scene_path.write_text(physics_scene_usda(params), encoding="utf-8")
    composite_path.write_text(composite_usda(scene_path.name, width, height, render_mode), encoding="utf-8")
    return composite_path
