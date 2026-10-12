"""ovphysx <-> ovstage bridge.

Physics and rendering share one ovstage Stage. After each physics step the poses of
rigid bodies and articulation links (robot parts) are read back and published as
``omni:fabric:worldMatrix`` at a fresh
"output" ordinal, which ovrtx then renders. Output ordinals are never drained back
into physics (``update_from_ovstage``), so physics never re-ingests its own output.

The pose -> matrix conversion is done with NumPy rather than
``ovphysx.utils.step_and_write_to_ovstage`` because that helper launches Warp CPU
kernels, which need ``warp-clang.dll`` - a DLL Windows Smart App Control blocks.
"""

from __future__ import annotations

import numpy as np
import ovphysx
import ovstage
from ovphysx import PhysX
from ovphysx.types import ObjectScope, SimObjectType

_WORLD_MATRIX = "omni:fabric:worldMatrix"
_schemas_registered = False


def register_schemas() -> None:
    """Register ovrtx + PhysX USD schemas. Must run before the first ovstage.Stage()."""
    global _schemas_registered
    if _schemas_registered:
        return
    import ovrtx

    ovrtx.register_schema_paths()
    ovstage.population.register_usd_schemas([str(ovphysx.codeless_schema_root())])
    _schemas_registered = True


def pose_to_row_major(positions: np.ndarray, quats_xyzw: np.ndarray, scales: np.ndarray | None = None) -> np.ndarray:
    """[N,3] + [N,4] (+ optional [N,3] scale) -> [N,16] float64 matrices, USD row-vector convention."""
    x, y, z, w = (quats_xyzw[:, i].astype(np.float64) for i in range(4))
    n = len(positions)
    m = np.zeros((n, 4, 4), dtype=np.float64)
    # Rows of the row-vector matrix are the rotated basis vectors (R transposed).
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y + z * w)
    m[:, 0, 2] = 2 * (x * z - y * w)
    m[:, 1, 0] = 2 * (x * y - z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z + x * w)
    m[:, 2, 0] = 2 * (x * z + y * w)
    m[:, 2, 1] = 2 * (y * z - x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    if scales is not None:
        m[:, :3, :3] *= scales[:, :, None]  # scale applies before rotation: row i scaled by s_i
    m[:, 3, :3] = positions
    m[:, 3, 3] = 1.0
    return m.reshape(n, 16)


class PhysicsSession:
    """Owns a PhysX instance attached to an already-populated, sealed Stage.

    ``scales`` maps prim paths to the scale they author (e.g. a box made from a
    scaled Cube); physics only reports position + orientation, so the scale is
    re-applied when building the world matrix.
    """

    def __init__(self, stage: ovstage.Stage, read_ordinal: int, scales: dict[str, tuple] | None = None) -> None:
        self._stage = stage
        self._scales = scales or {}
        self._paths = ovstage.PathDictionary(stage)
        self._physx = PhysX()
        self._physx.attach_ovstage(stage, read_ordinal=read_ordinal)
        self._physx.warmup()
        self._physx.wait_all()
        # Latest articulation-link poses by prim path: (position[3], quat_xyzw[4]).
        self.link_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def step_and_write(self, dt: float, output_ordinal: int) -> int:
        """Step once and write poses at ``output_ordinal`` (caller seals it). Returns body count."""
        self._physx.step_sync(dt)
        written = self._write_poses(SimObjectType.RIGID_BODY, output_ordinal)
        written += self._write_poses(SimObjectType.ARTICULATION_LINK, output_ordinal, track=True)
        return written

    def set_joint_velocity_targets(self, targets: dict[str, float]) -> None:
        """Set drive velocity targets (degrees/s for revolute joints) by joint prim path.

        Every articulation joint gets a value; joints missing from ``targets`` get 0.
        """
        with self._physx.write(SimObjectType.ARTICULATION_JOINT, "jointVelocityTarget") as session:
            for group in session.groups:
                names = self._paths.get_path_strings(group.prim_list)
                for name, tensor in zip(names, group.tensors):
                    tensor.numpy()[...] = targets.get(name, 0.0)
                session.commit(group)

    def _write_poses(self, kind: SimObjectType, ordinal: int, track: bool = False) -> int:
        written = 0
        with self._physx.read(kind, ["position", "orientation"], scope=ObjectScope.ALL) as result:
            partitions: dict[int, dict[str, np.ndarray]] = {}
            for group in result.groups:
                slot = partitions.setdefault(group.prim_list, {})
                for tensor in group.tensors:
                    column = tensor.numpy()
                    slot["position" if column.shape[1] == 3 else "orientation"] = column
            for prim_list, cols in partitions.items():
                scales = None
                if self._scales or track:
                    names = self._paths.get_path_strings(prim_list)
                    if track:
                        for name, pos, quat in zip(names, cols["position"], cols["orientation"]):
                            self.link_poses[name] = (pos.copy(), quat.copy())
                    if any(n in self._scales for n in names):
                        scales = np.array([self._scales.get(n, (1.0, 1.0, 1.0)) for n in names], dtype=np.float64)
                matrices = pose_to_row_major(cols["position"], cols["orientation"], scales)
                tensor = ovstage.make_dltensor(
                    matrices,
                    dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 64, 16),
                    shape=[len(matrices)],
                )
                query = self._stage.query_from_path_list(prim_list)
                try:
                    self._stage.write_attribute(
                        query,
                        _WORLD_MATRIX,
                        ordinal,
                        tensor,
                        is_array=False,
                        semantic=ovstage.AttributeSemantic.MATRIX,
                    ).wait()
                finally:
                    self._stage.release_query(query).wait()
                written += len(matrices)
        return written

    def close(self) -> None:
        if self._physx is not None:
            self._physx.detach_ovstage()
            self._physx.destroy()
            self._physx = None
        if self._paths is not None:
            self._paths.destroy()
            self._paths = None
