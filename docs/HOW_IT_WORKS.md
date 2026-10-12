# How it works

This document explains every piece of the project: what each NVIDIA library does,
how the pieces are wired together, and what happens on every frame. It assumes you
know Python but have never touched Omniverse.

## 1. The big picture

One Python process does everything. Inside it there are three kinds of threads:

| Thread | Owner | Job |
| --- | --- | --- |
| HTTP (uvicorn / asyncio) | `server/api.py` | Receives REST calls and turns them into commands |
| **Render thread** | `server/runtime.py` | Owns the GPU: physics, the scene, rendering and video encoding |
| ovstream SDK threads | inside `ovstream` | WebRTC networking; delivers mouse and keyboard events |

```
 Browser                                   Python process
┌──────────────────────┐   HTTP :8000    ┌────────────────────────────────────────────┐
│ control panel        │ ──────────────► │ FastAPI ── submit("load", ...) ──┐         │
│ (buttons, status)    │ ◄── JSON ────── │                                  │ queue   │
│                      │                 │                                  ▼         │
│                      │                 │  ┌───────────── render thread ──────────┐  │
│                      │                 │  │ drain commands                       │  │
│                      │                 │  │ ovphysx.write(wheel motor targets)   │  │
│                      │                 │  │ ovphysx.step()  ──► poses            │  │
│                      │                 │  │ ovstage.write(worldMatrix, ordinal N)│  │
│                      │                 │  │ ovstage.seal(N)                      │  │
│                      │                 │  │ ovrtx.render(N) ──► BGRA in VRAM     │  │
│ <video> element      │ ◄─ WebRTC ───── │  │ ovstream.stream_video(CUDA pointer)  │  │
│   mouse / keys       │ ── data chan ─► │  └──────────────────────────────────────┘  │
└──────────────────────┘ TCP 49100 sig.  │   ▲ _on_input: mouse→camera, keys→rover    │
                         UDP 47998 media └────────────────────────────────────────────┘
```

The key design rule is this: **only the render thread touches GPU objects.** Every
other thread talks to it through a command queue (API calls) or a lock-protected
object (camera and rover input). That avoids a whole class of threading bugs in libraries that
were never designed to be called from several threads.

## 2. The NVIDIA libraries

These are standalone pip packages from `pypi.nvidia.com`. You don't need the
Omniverse Kit app framework or Isaac Sim; each library does one job.

| Package | What it is | Role here |
| --- | --- | --- |
| `ovstage` | An in-memory, GPU-friendly scene database built from USD | The single source of truth both physics and the renderer read from |
| `ovphysx` | NVIDIA PhysX 5 with a Python API | Simulates the falling bodies and the rover |
| `ovrtx` | A headless RTX renderer (path tracing or ray-traced lighting) | Turns the stage into an image in GPU memory |
| `ovstream` | A streaming server (WebRTC, built on NVIDIA StreamSDK and the NVENC hardware encoder) | Encodes frames to H.264 and sends them to the browser |
| `ovstream_utils` | Helpers for ovstream | `Loop`, which keeps the frame loop at a steady fps |
| `@nvidia/ov-web-rtc` | A JavaScript client library | The browser side of the WebRTC connection |

### 2.1 USD in one paragraph

USD (Universal Scene Description, from Pixar) is a file format and object model for
3D scenes. A scene is a tree of **prims**, each with a path like `/World/Body_003`.
Prims have a type (`Cube`, `Camera`, `DistantLight`, and so on) and **attributes**
(`size`, `xformOp:translate`). **API schemas** such as `PhysicsRigidBodyAPI` attach
extra behaviour to a prim: they tell PhysX that this cube is a dynamic body. Files
can include other files through **sublayers**, which is how this project layers
rendering setup on top of the physics scene.

### 2.2 ovstage: the shared stage and its "ordinals"

`ovstage.Stage` loads USD into flat, columnar, GPU-accessible storage, roughly a
database table per attribute. Two ideas matter:

- **Ordinals are version numbers.** Every write is tagged with an integer ordinal,
  for example "the world matrix of these 24 bodies at ordinal 57." Several versions
  can live side by side.
- **The write floor seals versions.** `stage.advance_write_floor(ordinal=N)` says
  "ordinal N is complete; readers may use it." The renderer is then asked to render
  ordinal N specifically, so it never sees a half-written frame.

This lets physics and rendering share one scene without copying data between them.
Physics writes poses, the frame is sealed, and the renderer reads it.

## 3. Startup sequence

`run.ps1` runs `python -m server.main`. Here is what happens, in order.

### 3.1 `server/env.py`: environment, before any NVIDIA import

Some settings are only read when the native libraries load, so they must be set
first:

- `OVRTX_SKIP_USD_CHECK=1` skips a check for a conflicting `pxr` (USD) install.
- `CUDA_CACHE_PATH`, `__GL_SHADER_DISK_CACHE_PATH` and `XDG_CACHE_HOME` point into
  `.cache/`, so compiled GPU shaders persist between runs. This is why only the first
  start takes about 3 minutes.
- `OVRTX_BIN_PATH` tells ovrtx where its native binaries are.

### 3.2 `server/main.py`: arguments and the order of things

1. Parses the command-line flags (`--http-port`, `--signal-port`, `--width`,
   `--height`, `--fps`, `--render-mode`, `--public-ip`).
2. Creates a `SimRuntime` and calls `runtime.start()`, which **blocks** until the
   first frame can be streamed. If the GPU side fails, the process exits instead of
   serving an API that can't work.
3. Starts uvicorn with the FastAPI app on port 8000.

### 3.3 `SimRuntime._run`: what the render thread does first

```
register_schemas()   # teach USD about the RTX and PhysX schemas
_cmd_load(params)    # build scene, stage, renderer and physics
_warm_up()           # render until the BGRA output exists and has the right size
_start_stream()      # start the WebRTC server
loop forever: commands -> _frame()
```

`register_schemas()` (in `physics.py`) must run **before the first `ovstage.Stage`
is created**. Without it, USD doesn't recognise attributes such as
`omni:rtx:rendermode` or the PhysX schemas, and they would be silently dropped.

## 4. The scene: `server/scene.py`

Each launch writes two USD text files into `scenes/_generated/`.

### 4.1 The physics scene: `sim_N.usda`

```
/World
  PhysicsScene      gravity 9.81 m/s² along -Y
  Sky               DomeLight (ambient light from all directions)
  Sun               DistantLight (sharp shadows)
  Ground            Cube scaled 40 x 0.2 x 40, PhysicsCollisionAPI only (static)
  Body_000..N       Cube or Sphere with RigidBody + Mass + Collision APIs
  Rover             the drivable robot (an articulation, section 8)
    Chassis         scaled Cube, articulation root
    Lidar           Cylinder on a fixed joint
    WheelFL..RR     Cylinders on driven revolute joints JointFL..RR
```

- **Static versus dynamic:** the ground has only `PhysicsCollisionAPI`, so things
  collide with it but it never moves. The bodies also have `PhysicsRigidBodyAPI`,
  which makes them dynamic: gravity and collisions move them.
- **Layout:** each body gets a random x/z position within `±spread`, a height of
  `drop_height`, raised 1.2 m for every 9 bodies so they don't start overlapping,
  plus a random tilt and colour. `random.Random(seed)` makes a layout reproducible:
  the same seed always gives the same scene.
- **Why rigid-body APIs sit directly on the Cube/Sphere:** PhysX publishes a world
  matrix for each rigid-body prim, and the renderer does not pass it on to child
  prims. If the body were an `Xform` with a child `Cube`, the physics would move but
  the visible cube would stay put.

### 4.2 The composite: `_composite_sim_N.usda`

This layer sublayers the physics scene and adds what the **renderer** needs, which
keeps the physics scene free of viewer concerns.

- **`/OVCamera`:** a perspective camera. Its starting transform is overwritten each
  time the orbit camera moves (section 7).
- **`/Render/OVServer/ViewportTexture0`:** a `RenderProduct`, which is "an image the
  renderer should produce": its camera, resolution, GPU and render mode.
- **RenderVars** are the outputs (AOVs) of that product:
  - `LdrColor` is the final tone-mapped 8-bit RGBA image.
  - `BgraColor` is the same image with red and blue swapped. A tiny GPU shader node
    (`spg:rtx.spg.stdlib/Swizzle` with `"zyxw"`) does the swap. ovstream's encoder
    expects BGRA, so doing it on the GPU means the frame never has to visit the CPU.
- **`RenderSettings`** points at the product.

`omni:rtx:rendermode` selects between:
- `RealTimePathTracing`: nicer soft shadows and bounce light (about 41 fps here).
- `RaytracedLighting`: cheaper; use `--render-mode RaytracedLighting`.

## 5. Loading a scene: `SimRuntime._load_scene`

This runs at startup and for every `POST /api/simulation` or `/reset`:

1. **Write the USD files.** Names alternate between `sim_0` and `sim_1`, so a new
   scene is never written over the files the still-attached old stage was built from.
2. **Create and populate a new stage:**
   ```python
   stage = ovstage.Stage("sim-<gen>")
   ovstage.population.open_usd(stage, composite, ordinal=1, domains=ALL)
   stage.advance_write_floor(ordinal=1)          # seal the initial scene as ordinal 1
   ```
3. **Tear down the old scene:** close PhysX, release the camera writer, detach the
   renderer from the old stage and destroy that stage.
4. **Create the renderer once** (`ovrtx.Renderer`) and attach it to the new stage.
   The renderer is kept for the whole process lifetime, because creating it is the
   expensive part (shader compilation). Only the stage under it is swapped.
5. **Attach PhysX** (`PhysicsSession(stage, read_ordinal=1, scales=...)`). PhysX
   reads the initial scene at ordinal 1, builds its own internal simulation from it,
   and warms up. `scales` lists the bodies that author a scale (the rover chassis);
   section 6.1 explains why.
6. Reset counters: ordinal 1, `sim_time` 0. Stop the rover controller, and mark the
   camera as dirty so it gets written on the next frame.

**Reset** reloads the same parameters, including the same seed, so you get the
exact same starting layout.

## 6. The frame loop: `SimRuntime._frame`

`ovstream_utils.Loop(fps_target=60)` paces the loop: each `loop.tick()` sleeps just
long enough to hold about 60 iterations per second and reports the measured fps.
Each iteration:

```python
drain_commands()                           # run queued API commands (section 9)

if rover is being driven:
    running = True                         # driving wakes a paused simulation
if running:
    drive_robot()                          # wheel motor targets for this step (section 8)

if running or camera.dirty:
    ordinal += 1                           # a new version of the world
    if running:
        physics.step_and_write(dt, ordinal)    # section 6.1
        update_robot_pose()                # chassis pose for the controller / follow camera
    if camera.dirty:
        camera_writer.write(matrix, ordinal)   # section 7
    stage.advance_write_floor(ordinal)     # seal: this version is complete

products = renderer.step({RENDER_PRODUCT}, dt, ordinal=ordinal)   # section 6.2
bgra = products[...].render_vars[".../BgraColor"].map(device=CUDA)
server.stream_video(VideoFrame(buffer=bgra.data, ..., sync=CudaSync(...)))   # section 6.3
bgra.unmap()
```

When the simulation is paused and the camera is still, no new ordinal is created
and the renderer simply re-renders the last one. With the path tracer, that is still
useful: it keeps refining the image.

### 6.1 Physics: `server/physics.py`

`PhysicsSession.step_and_write(dt, ordinal)`:

1. `physx.step_sync(dt)` advances the simulation by 1/60 s, handling gravity,
   contacts and friction.
2. `physx.read(kind, ["position", "orientation"])` returns poses as arrays:
   positions `[N,3]` and quaternions `[N,4]` in xyzw order. It runs twice, once
   for `RIGID_BODY` (the falling boxes) and once for `ARTICULATION_LINK` (the rover's
   parts). PhysX keeps them in separate categories, and the box read doesn't include
   robot links. Results come in groups, each with a `prim_list` that says which prims
   the rows belong to.
3. `pose_to_row_major` turns each position and quaternion into a 4×4 matrix. USD
   uses the **row-vector** convention (`p' = p · M`), so the rotation is stored
   transposed and the translation sits in the bottom row.
4. The matrices are written to the stage as `omni:fabric:worldMatrix` at this frame's
   ordinal. That is the attribute the renderer reads for an object's final placement.

**Scale.** Physics only knows position and orientation, but the world matrix
replaces the prim's whole transform. A box made from a scaled `Cube`, like the rover
chassis (1.0 × 0.25 × 0.7), would render as a 1 m cube. So `PhysicsSession` takes a
`scales` map (`robot.BODY_SCALES`), looks up each row's prim path with
`PathDictionary.get_path_strings(prim_list)`, and multiplies the rotation rows by
the scale.

**Prim-list handles are per read.** The `prim_list` number changes on every read
(305, 306, ... in `scripts/probe_robot_layout.py`), so rows are matched to prims
by path string each frame rather than by caching the handle.

Two more important details:

- **Output ordinals never flow back into physics.** PhysX keeps its own state
  internally. Ordinal 1 is its input, and everything after that is its output for
  the renderer. Feeding the matrices back in would make physics re-import its own
  results every frame.
- **Why not `ovphysx.utils.step_and_write_to_ovstage`?** That helper does the same
  thing but uses NVIDIA Warp CPU kernels, which need `warp-clang.dll`. Windows Smart
  App Control blocks that DLL, so the conversion is done in NumPy instead. For a few
  hundred bodies, the cost is negligible.

### 6.2 Rendering: ovrtx

`renderer.step({product}, dt, ordinal=N)` renders the sealed version N:

- The renderer was created with `sync_mode=True`, so the call returns when the frame
  is ready, which makes the loop simple and deterministic.
- The result maps render-var names to GPU buffers. In this ovrtx build the keys are
  full prim paths (`/Render/OVServer/ViewportTexture0/BgraColor`), so `_find_var`
  matches on the last path segment.
- `.map(device=CUDA)` gives a **CUDA device pointer** to the pixels, plus a CUDA event
  that fires when the GPU has actually finished writing them.

### 6.3 Streaming: ovstream

`server.stream_video(VideoFrame(...))` hands that same GPU pointer to ovstream:

- **Zero copy:** the frame goes straight from the renderer's buffer to the NVENC
  hardware encoder on the same GPU. Pixels never cross to system memory.
- **`CudaSync(wait_event=...)`:** the encoder waits on the renderer's CUDA event, so
  it never encodes a half-finished image, and the Python thread doesn't have to block.
- After `stream_video` returns, `unmap()` gives the buffer back to the renderer.

Frames are sent **even when no viewer is connected.** That isn't waste: the WebRTC
session only completes its start handshake once frames are flowing. An earlier
version only sent frames after `is_client_connected` became true, and the connection
never finished because each side was waiting on the other.

## 7. The camera: `server/camera.py`

### 7.1 Input path

1. The browser library captures mouse and keyboard events over the `<video>` element,
   but **only while the video has focus**: it switches input on when its container
   gets a `focusin` event. The page focuses the video on connect and whenever the
   mouse enters it. Clicking a sidebar button moves focus away, so keys stop working
   until the mouse comes back over the video.
2. It sends them to the server over a **WebRTC data channel**, inside the same
   connection that carries the video.
3. ovstream calls `server.on_input(event)` on one of its own threads, which goes to
   `SimRuntime._on_input`. That routes the event:
   - mouse events go to `OrbitCamera.on_mouse`
   - drive keys (WASD, arrows, Space) go to the rover controller (section 8)
   - `C` resets the camera and `F` toggles follow mode

Key events carry Windows virtual-key codes (`0x57` is `W`, `0x25`–`0x28` are the
arrow keys), with a `KeyState.DOWN` or `UP` state for each press and release.

### 7.2 `OrbitCamera`

The camera is described by three numbers around a target point, `(0, 1, 0)` by
default:

- **yaw**, the angle around the vertical axis, changed by horizontal drag
- **pitch**, the angle above the ground, changed by vertical drag and clamped
  between about 0° and 85°
- **radius**, the distance to the target, changed by the wheel and clamped to 2–80 m

In **follow mode** the render thread calls `set_target(rover position)` after every
physics step, so the orbit centre moves with the rover while you can still drag
around it. Because input arrives on an SDK thread while the render thread reads the
camera, all fields are protected by a lock. Any change sets `dirty = True`.

`matrix()` turns yaw, pitch and radius into a camera-to-world transform:

```
eye     = target + radius · (cos p · sin y,  sin p,  cos p · cos y)
forward = normalize(target - eye)
right   = normalize(forward × up)
up'     = right × forward
rows    = [right, up', -forward, eye]       # USD cameras look down their local -Z
```

### 7.3 `CameraWriter`

Writes that matrix to `/OVCamera`'s `omni:xform` attribute at the current ordinal.
It prepares the query handle, the interned attribute name and the data type once in
`__init__`, so each frame's write is cheap. The camera is written only when it is
dirty, which is why the main loop checks `camera.dirty`.

## 8. The robot: `server/robot.py`

The rover is a small four-wheeled skid-steer vehicle, driven like a tank: both wheels
on a side always spin together, and it turns by running the two sides at different
speeds. It shows the three things any simulated robot needs: a **body made of linked
parts**, **motors** in its joints, and a **controller** that decides what the motors
should do.

### 8.1 Articulations: robots in PhysX

A plain rigid body (like the falling boxes) moves freely. A robot is several rigid
bodies (**links**) connected by **joints** that restrict how they move relative to
each other. You could simulate that as separate bodies glued by constraints, but the
joints then stretch and jitter under load. PhysX instead offers **articulations**:
a tree of links solved in *reduced coordinates*. The solver tracks only the allowed
motion, such as each wheel's spin angle, so joints never come apart. Robot arms,
legged robots and wheeled robots in Isaac Sim are all articulations.

In USD it takes three things:

- `PhysicsArticulationRootAPI` on the root link. On the chassis it makes a
  **floating-base** articulation, free to drive around. Putting it on a fixed joint
  to the world would make a fixed-base robot, like a bolted-down arm.
- Rigid-body links: the chassis (a scaled `Cube`), four `Cylinder` wheels and a
  `Lidar` cylinder.
- Joints connecting them:
  - `PhysicsRevoluteJoint` (a hinge) for each wheel, spinning about the local Z axis
  - `PhysicsFixedJoint` for the lidar, which simply rides on the chassis

```
Chassis (root, 10 kg) ──JointFL (revolute, Z)──► WheelFL (1 kg)
                      ──JointFR ──► WheelFR      ──JointRL ──► WheelRL
                      ──JointRR ──► WheelRR      ──JointLidar (fixed) ──► Lidar
```

**Joint frames and scale.** `physics:localPos0` is the anchor point on body 0 (the
chassis) in that body's *local* space, which includes its scale. The front-left
wheel sits 0.32 m forward, 0.2 m down and 0.45 m left of the chassis centre, so the
authored anchor is that offset divided by the chassis scale (1.0, 0.25, 0.7). The
probe script confirmed the wheels end up exactly where they were placed.

`physxArticulation:enabledSelfCollisions = 0` stops the wheels from colliding with
the chassis they touch.

### 8.2 Motors: joint drives

Each wheel joint has a `PhysicsDriveAPI:angular`, a PD controller that applies
torque:

\[ \tau = \text{stiffness}\cdot(\theta_\text{target}-\theta) + \text{damping}\cdot(\omega_\text{target}-\omega), \quad |\tau| \le \text{maxForce} \]

With `stiffness = 0` and `damping = 1000` it becomes a pure **velocity motor**: it
pushes the wheel's spin rate toward `targetVelocity`, limited to 80 N·m. Angular
values in PhysX USD are in **degrees** (and degrees per second), not radians.

At runtime the targets are changed with the session **write** API:

```python
with physx.write(SimObjectType.ARTICULATION_JOINT, "jointVelocityTarget") as session:
    for group in session.groups:
        names = paths.get_path_strings(group.prim_list)    # ["/World/Rover/JointFL", ...]
        for name, tensor in zip(names, group.tensors):     # one tensor per joint
            tensor.numpy()[...] = targets.get(name, 0.0)   # CPU sim: write in place
        session.commit(group)                              # publish the whole group
```

The write rules, from the ovphysx docs:
- A write is refused before the first `warmup()` or step.
- One attribute per session.
- Every tensor in a group must be filled before `commit`, because commit publishes
  the whole group.

A write only happens when the targets actually change, not every frame.

### 8.3 From "drive forward and turn" to wheel speeds

The controller thinks in **body velocity**: `linear` (m/s forward) and `angular`
(rad/s, positive turns left). `wheel_speeds()` converts that with the standard
differential-drive formula:

```
v_left  = linear - angular · track/2          track = 0.9 m (left-right wheel distance)
v_right = linear + angular · track/2
ω_wheel = -v / r  (rad/s) → degrees/s        r = 0.25 m
```

The minus sign comes from geometry. The wheels spin about +Z, and with Y up, a wheel
rolling toward +X has to rotate *negatively* about +Z. The probe script confirmed
it: −400°/s on all wheels drove the rover toward +X.

**Skid steering slips.** To turn, the wheels have to scrub sideways over the ground,
so the rover turns slower than the formula predicts. A 2-second spin at
1.5 rad/s gives about 70°, not 170°. That is realistic for skid-steer robots, and
it's why the go-to controller below uses feedback instead of trusting the formula.

### 8.4 The controller: `DriveController`

`DriveController.body_velocity(pose, sim_time)` runs every frame before the physics
step, and its result depends on the **mode**:

| Mode | Set by | Behaviour |
| --- | --- | --- |
| `keys` | holding WASD or the arrows in the viewport | Forward/back and left/right from the set of held keys; releasing them stops the rover |
| `drive` | `POST /api/robot/drive` | Fixed `linear`/`angular`; with `duration` it stops when `sim_time` reaches it |
| `goto` | `POST /api/robot/goto` | Closed-loop drive to a point (below) |
| `idle` | `Space`, `/api/robot/stop`, pause, reset, arrival | Wheels held at zero speed |

**Go-to-point** is a simple feedback controller, the same idea used on real mobile
robots:

```
heading_error = wrap(atan2(-dz, dx) - heading)          # which way to turn
angular = clamp(3.0 · heading_error)                     # turn harder when far off
linear  = min(2.5, 1.2 · distance) · max(0, cos(error))² # drive only when roughly facing it
stop when distance < 0.35 m
```

The `cos²` term makes the rover turn on the spot first when the goal is behind it,
then speed up as it lines up, then slow down as it gets close. It is *closed-loop*:
every frame it reads the real chassis pose from physics, so slipping, bumping into
boxes or being pushed off course all get corrected automatically.

**Pose.** After each step, the chassis row of the articulation-link read (section
6.1) becomes a `Pose`: x, z, and a heading computed from the quaternion. The heading
is the angle of the chassis's local +X axis, counter-clockwise around +Y, with 0
facing +X. The status endpoint reports it, the controller steers by it, and follow
mode centres the camera on it.

**Threading.** Key events arrive on an ovstream thread, while REST commands and
`body_velocity` run on the render thread, so the controller guards its state with a
lock. Any non-idle mode also **wakes a paused simulation**, so driving after a Reset
just works. Pause stops the rover as well.

## 9. Commands: how an HTTP request reaches the GPU

GPU objects belong to the render thread, so API handlers never call them directly.
Here is the life of `POST /api/simulation`:

```
FastAPI handler (asyncio)
  └─ runtime.submit("load", params=..., autostart=True)
       ├─ looks up method _cmd_load
       ├─ puts (lambda, Future) on a queue.Queue
       └─ returns the Future
  └─ await asyncio.wrap_future(future)      # the HTTP request waits here (60 s timeout)

render thread, at the start of the next frame
  └─ _drain_commands()
       └─ result = _cmd_load(...)  ──► future.set_result(result)
                                       (or set_exception, which becomes HTTP 400)
```

`concurrent.futures.Future` is the bridge between the two worlds:
`asyncio.wrap_future` lets async FastAPI await something completed by an ordinary
thread.

The commands are:

| Command | Effect |
| --- | --- |
| `load(params, autostart)` | New scene. Picks a random seed if none is given, so the status always reports a reproducible seed |
| `start(duration)` | `running = True`; with a duration it auto-pauses when `sim_time` reaches the target |
| `pause` | `running = False` and the rover stops; rendering and streaming continue |
| `reset` | Reloads the same parameters and seed, paused |
| `robot_drive(linear, angular, duration)` | Rover in `drive` mode (section 8.4) |
| `robot_goto(x, z)` | Rover in `goto` mode |
| `robot_stop` | Rover idle |
| `reset_camera` | Camera back to the home view (also turns follow off) |
| `camera_follow(enabled)` | Orbit centre tracks the rover |

Robot commands on a scene launched with `robot: false` raise `ValueError`, which the
API turns into HTTP 400 with a clear message.

**Status** goes the other way without a queue. Each frame the render thread writes a
snapshot dict under a lock (`_publish_status`), and `GET /api/simulation` just copies
it, so status calls never wait for the render thread.

## 10. The REST API: `server/api.py`

A thin FastAPI layer:

- **Validation:** Pydantic models (`SimulationRequest`, `StartRequest`,
  `DriveRequest`, `GotoRequest`, `FollowRequest`) reject bad input with HTTP 422
  before anything reaches the render thread. For example, speeds above the rover's
  limits or goals off the 40 m ground plane are rejected.
  `SimulationParams.validated()` checks the same ranges again, in case the runtime
  is used without the API.
- **Errors:** `ValueError` or `RuntimeError` from a command becomes HTTP 400 with the
  message.
- **`/api/config`** tells the browser which signaling port, resolution and fps to use,
  so the page has no hard-coded settings.
- **`/healthz`** returns 503 until the runtime is up, which suits container probes.
- **`/`** and **`/static`** serve the web page and the vendored JS client.
- **`/docs`** is FastAPI's automatic Swagger UI. Every endpoint can be tried there.

## 11. WebRTC in detail

WebRTC is the browser's built-in real-time media stack. Getting video from the
server to the page takes three steps.

### 11.1 Signaling (TCP 49100, WebSocket)

Before media can flow, the two sides exchange **SDP** (Session Description Protocol)
messages over a WebSocket. Through them they agree on codecs (H.264 here), encryption
keys and a data channel. The server sends an **offer** and the browser replies with
an **answer**. ovstream runs this signaling server itself; you only choose the port.

### 11.2 ICE (finding a network path)

**ICE** (Interactive Connectivity Establishment) is how the two peers find an address
pair that works. Each side lists **candidates**, which are addresses it can be
reached on, and the browser probes them with **STUN** packets. ovstream runs in
*ICE-lite* mode: it publishes its host candidates and answers the browser's probes.

### 11.3 Media (UDP 47998)

Once a path is found, encrypted RTP packets (SRTP) carry the H.264 video over UDP.
Mouse and keyboard events go the other way over an SCTP data channel on the same
port.

### 11.4 `--public-ip` and why it is off by default

Setting `webrtc_public_ip` tells ovstream "clients will reach me at exactly this
address," and **turns ICE off**. That is right for a cloud VM with a fixed public IP,
but wrong locally. Chrome doesn't send media over the loopback interface, so a pinned
`127.0.0.1` gives you working signaling and no video. The browser reports "Client
sent STUN requests but did not receive any responses from the server." With the
option unset, ICE finds a working local path on its own.

For a cloud deployment: set `--public-ip <VM IP>` and open TCP 49100 and UDP 47998.

### 11.5 The browser client: `web/index.html`

```js
const streamer = new OVWebRTC.AppStreamer();
streamer.connect({
  streamSource: StreamType.DIRECT,          // talk to ovstream directly, no cloud broker
  streamConfig: {
    videoElementId: 'remote-video',
    signalingServer: location.hostname,     // same host that served the page
    signalingPort: cfg.signal_port,         // from GET /api/config
    maxReconnects: 5, reconnectDelay: 3000,
    onStart, onStop, ...                    // drive the "Live" / error overlay
  },
});
```

The library handles signaling, ICE, decoding into the `<video>` element, and
forwarding input events. The rest of the page is plain `fetch()` calls to the REST
API, plus a status poll every 500 ms.

ovstream's WebRTC server accepts **one viewer at a time**. A second tab will fail to
connect until the first one closes.

## 12. A full round trip

Suppose you click **Launch simulation** with 40 bodies:

1. The page sends `POST /api/simulation {"bodies": 40, ...}`.
2. FastAPI validates the request and queues `load`, and the request waits.
3. At the start of its next frame (at most about 16 ms later), the render thread
   writes the new USD files, builds a new stage, swaps the renderer and PhysX onto
   it, and sets `running = True`.
4. The Future completes and the HTTP response returns the new state, including the seed.
5. On every following frame, PhysX steps, 40 world matrices are written at a new
   ordinal, the ordinal is sealed, RTX renders it, and NVENC encodes it straight
   from GPU memory.
6. H.264 packets travel over UDP to the browser, which decodes them into `<video>`.
   You see the bodies fall, a few tens of milliseconds behind the simulation.
7. You drag on the video. Events travel back over the data channel, the camera
   becomes dirty, and the next frame writes a new camera matrix alongside the poses.
8. You hold `W`. The key event reaches `_on_input`, and the controller switches to
   `keys` mode. On the next frame `wheel_speeds()` produces −458°/s for every wheel
   (80% of the top speed), and one `jointVelocityTarget` write publishes it. PhysX's
   drives then spin the wheels, friction pushes the rover forward, and the
   articulation-link read carries the new chassis and wheel poses to the renderer.

## 13. Tools and files

| File | Purpose |
| --- | --- |
| `setup.ps1` | Creates `.venv`, installs the pinned wheels, works around Smart App Control, prints GPU and package versions |
| `run.ps1` | Starts the server and copies output to `logs/server.log` |
| `requirements.txt` | Version pins: `ovphysx 0.6.3` needs `ovstage 0.2.0`, and `ovrtx 0.5.0.377615` matches its omniclient |
| `scripts/probe_render_physics.py` | Physics plus rendering with no streaming; saves `out/frame_000.png` and `out/frame_060.png` so you can check the pipeline in isolation |
| `scripts/probe_robot_layout.py` | Physics only: prints what each read and write returns for the rover (groups, prim names, tensor shapes), then drives it for 3 s. Run it whenever you change the robot |
| `scenes/_generated/` | The USD files from the latest launches. Open them in a text editor to see exactly what the renderer and physics receive |
| `logs/server.log` | Server log, including ovstream warnings |

## 14. Ideas for experiments

- **Change physics:** add `float physics:restitution` through a
  `PhysicsMaterialAPI` for bouncy spheres, or change the gravity in `PhysicsScene`.
- **Add a REST endpoint:** for example `POST /api/simulation/impulse` that pushes
  every body upward. Add a `_cmd_impulse` to `SimRuntime` and write `linearVelocity`
  for `RIGID_BODY` with the same write API the wheels use.
- **Waypoints:** make `goto` accept a list of points and patrol them, or a "push
  the nearest box to the origin" behaviour that reads box positions from the
  `RIGID_BODY` read.
- **A real sensor:** add a second camera on the lidar puck as another RenderProduct,
  for a robot's-eye view. Or cast rays with ovphysx scene queries to make a fake lidar.
- **A robot arm:** a fixed-base articulation with position drives
  (`stiffness > 0`) controlled through `jointPositionTarget`.
- **Better turning:** lower the wheels' friction, or switch to two driven wheels plus
  a frictionless caster, and compare how closely the rover follows the commanded
  turn rate.
- **Compare render modes:** run with `--render-mode RaytracedLighting` and watch
  `render_fps` in the status panel.
- **Stream a different AOV:** for depth or normals, add a RenderVar to the composite
  and swizzle it into `BgraColor`.
- **Go remote:** run on a cloud GPU VM with `--public-ip`, and open the two ports.
