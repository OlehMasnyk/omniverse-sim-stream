"""Physics-only probe: print read/write layouts for the rover and check that it drives."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import env  # noqa: E402

env.configure()

import numpy as np  # noqa: E402
import ovphysx  # noqa: E402
import ovstage  # noqa: E402
from ovphysx import PhysX  # noqa: E402
from ovphysx.types import ObjectScope, SimObjectType  # noqa: E402

from server.scene import SimulationParams, physics_scene_usda  # noqa: E402

out = Path(__file__).resolve().parent.parent / "scenes" / "_generated" / "probe_robot.usda"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(physics_scene_usda(SimulationParams(bodies=3, seed=1)), encoding="utf-8")

ovstage.population.register_usd_schemas([str(ovphysx.codeless_schema_root())])
stage = ovstage.Stage("probe")
ovstage.population.open_usd(stage, str(out), ordinal=1, domains=ovstage.PopulationDomain.ALL)
stage.advance_write_floor(ordinal=1).wait()
paths = ovstage.PathDictionary(stage)

physx = PhysX()
physx.attach_ovstage(stage, read_ordinal=1)
physx.warmup()
physx.wait_all()


def dump_read(kind, attrs):
    with physx.read(kind, attrs, scope=ObjectScope.ALL) as result:
        print(f"READ {kind.name} {attrs}: {len(result.groups)} groups")
        for g in result.groups:
            names = paths.get_path_strings(g.prim_list)
            print("  prim_list", g.prim_list, "attr", g.attribute, "names", names)
            for t in g.tensors:
                print("    tensor", t.shape, t.dtype, t.device, np.round(t.numpy()[:6], 3).tolist())


dump_read(SimObjectType.RIGID_BODY, ["position", "orientation"])
dump_read(SimObjectType.ARTICULATION_LINK, ["position", "orientation"])
dump_read(SimObjectType.ARTICULATION, ["rootPosition", "rootOrientation"])
dump_read(SimObjectType.ARTICULATION_JOINT, ["jointVelocityTarget", "jointPosition"])

with physx.write(SimObjectType.ARTICULATION_JOINT, "jointVelocityTarget") as w:
    print(f"WRITE jointVelocityTarget: {len(w.groups)} groups")
    for g in w.groups:
        names = paths.get_path_strings(g.prim_list)
        print("  names", names, "tensors", [(t.shape, t.device) for t in g.tensors])
        for t in g.tensors:
            view = t.numpy()
            view[...] = -400.0  # every wheel forward at ~1.75 m/s
        w.commit(g)

for i in range(180):
    physx.step_sync(1 / 60)

dump_read(SimObjectType.ARTICULATION, ["rootPosition", "rootOrientation"])
dump_read(SimObjectType.ARTICULATION_JOINT, ["jointVelocity"])
dump_read(SimObjectType.RIGID_BODY, ["position"])

physx.detach_ovstage()
physx.destroy()
paths.destroy()
stage.destroy()
