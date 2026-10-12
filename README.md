# omniverse-sim-stream

A small learning project: a PhysX simulation, with falling boxes and a drivable
four-wheeled rover, that you control over a REST API. It is rendered with RTX and
streamed live to the browser over WebRTC.
It runs natively on Windows and uses NVIDIA's standalone Omniverse Python libraries
(no Kit or Isaac Sim needed).

```
           HTTP :8000 (FastAPI)                       WebRTC
browser ─── POST /api/simulation ───┐        ┌──── signaling TCP :49100
   ▲                                ▼        │     media     UDP :47998
   │                  ┌──────── render thread ───────────┐
   │                  │ ovphysx  step physics            │
   │                  │    │ write omni:fabric:worldMatrix
   │                  │    ▼                             │
   │                  │ ovstage  shared stage (ordinals) │
   │                  │    │                             │
   │                  │    ▼                             │
   │                  │ ovrtx    render BGRA on the GPU  │
   │                  │    │ CUDA pointer (no copy)      │
   │                  │    ▼                             │
   └──── H.264 ───────┤ ovstream NVENC + WebRTC          │
                      └──────────────────────────────────┘
```

## Requirements

- Windows 10/11 with an NVIDIA RTX GPU and a recent driver
- Python 3.10+ (developed on 3.14)

## Setup and run

```powershell
.\setup.ps1          # creates .venv and installs the NVIDIA wheels from pypi.nvidia.com
.\run.ps1            # starts the server; open http://localhost:8000
```

The **first** start compiles RTX shaders, which takes about 3 minutes. Later starts
take about 5 seconds. The shader cache is stored in `.cache/`.

## REST API

Interactive docs are served at <http://localhost:8000/docs>.

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/api/simulation` | Build a new scene and (by default) start it |
| `POST` | `/api/simulation/start` | Run, optionally for `{"duration": seconds}` |
| `POST` | `/api/simulation/pause` | Pause physics (the stream keeps going) |
| `POST` | `/api/simulation/reset` | Return the bodies to their spawn poses |
| `POST` | `/api/robot/drive` | Drive the rover: `{"linear": m/s, "angular": rad/s, "duration": s}` |
| `POST` | `/api/robot/goto` | Drive the rover autonomously to `{"x": .., "z": ..}` |
| `POST` | `/api/robot/stop` | Stop the rover |
| `POST` | `/api/camera/reset` | Reset the orbit camera |
| `POST` | `/api/camera/follow` | `{"enabled": true}` keeps the camera centred on the rover |
| `GET` | `/api/simulation` | Status: state, sim time, fps, rover pose, whether a viewer is connected |

```powershell
curl.exe -X POST localhost:8000/api/simulation -H "Content-Type: application/json" `
  -d '{\"bodies\": 40, \"drop_height\": 8, \"sphere_ratio\": 0.5, \"seed\": 7}'
curl.exe -X POST localhost:8000/api/simulation/pause
curl.exe -X POST localhost:8000/api/simulation/start -H "Content-Type: application/json" -d '{\"duration\": 2}'
curl.exe localhost:8000/api/simulation

# the rover
curl.exe -X POST localhost:8000/api/camera/follow -H "Content-Type: application/json" -d '{\"enabled\": true}'
curl.exe -X POST localhost:8000/api/robot/goto -H "Content-Type: application/json" -d '{\"x\": 5, \"z\": -4}'
curl.exe -X POST localhost:8000/api/robot/drive -H "Content-Type: application/json" `
  -d '{\"linear\": 0, \"angular\": 1.5, \"duration\": 2}'
```

In the viewport, click the video first, then:

| Input | Effect |
| --- | --- |
| `W` `A` `S` `D` / arrow keys (hold) | Drive the rover |
| `Space` | Stop the rover |
| `F` | Toggle the follow camera |
| `C` | Reset the camera |
| Left-drag / wheel | Orbit / zoom |

Mouse and keyboard events travel back to the server over the WebRTC data channel.
The client forwards input only while the video has focus. Clicking a sidebar button
takes focus away; moving the mouse back over the video restores it.

## Code tour

For a full walkthrough of every component and the per-frame data flow, see
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

- `server/scene.py`: generates the USD scene (ground, lights, and random cubes and
  spheres with rigid-body physics), plus a composite layer containing the camera
  and the RTX render product. A GPU swizzle shader in the composite layer outputs
  BGRA, which is the format ovstream encodes.
- `server/physics.py`: wraps ovphysx. Each step reads the poses of the bodies and
  the rover's links and writes them to the shared stage as world matrices. It also
  writes the wheel motor targets.
- `server/robot.py`: the rover. Its USD (an articulation: chassis, four driven wheels
  and a lidar), and a drive controller for keys, timed commands and go-to-point.
- `server/camera.py`: orbit camera driven by WebRTC mouse events; it can follow the rover.
- `server/runtime.py`: the render thread. It owns the stage, renderer, physics and
  stream server. API calls are queued as commands, so all GPU work stays on one thread.
- `server/api.py`: the FastAPI endpoints.
- `web/index.html`: the control panel plus the WebRTC viewer, using NVIDIA's
  `@nvidia/ov-web-rtc` client, vendored as `web/ov-web-rtc.umd.cjs`.
- `scripts/probe_render_physics.py`: a smoke test of physics and rendering without
  streaming. It writes two PNGs to `out/`.
- `scripts/probe_robot_layout.py`: physics only. It prints how ovphysx lays out
  rover reads and writes, then drives the rover for 3 seconds.

## Gotchas found while building this

- **Version pins:** `ovphysx 0.6.3` requires `ovstage 0.2.0`. Both libraries
  must load the same omniclient build, so `ovrtx` is pinned to `0.5.0.377615`.
  Newer ovrtx builds fail with "IOmniClient v2.75 requested".
- **Smart App Control** blocks Warp's `warp-clang.dll`. The project avoids Warp CPU
  kernels (the physics write-back is plain NumPy), and `setup.ps1` moves the DLL out
  of the way when Smart App Control is on.
- **Keep `--public-ip` unset for local use.** Setting `webrtc_public_ip` turns ICE
  off and pins the media endpoint. Chrome doesn't send media from the loopback
  interface, so a pinned `127.0.0.1` makes signaling succeed while no video arrives.
  Set it only on a cloud VM with a fixed public IP, and open UDP 47998 there.
- **Send frames before the client connects.** The WebRTC session completes its
  start handshake only once frames are flowing, so gating `stream_video` on
  `is_client_connected` deadlocks.
- Only one viewer can be connected at a time. That is a limitation of ovstream's
  WebRTC server.
