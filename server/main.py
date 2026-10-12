"""Entry point: python -m server.main"""

from __future__ import annotations

import argparse
import logging
import sys

from . import env

env.configure()

import uvicorn  # noqa: E402

from .api import create_app  # noqa: E402
from .runtime import RuntimeConfig, SimRuntime  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="ovphysx simulation streamed over WebRTC")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--signal-port", type=int, default=49100, help="WebRTC signaling port")
    parser.add_argument("--public-ip", default=None,
                        help="Pin the advertised WebRTC media IP (disables ICE). Leave unset for local/LAN "
                             "use; set it for a cloud VM with a fixed public IP and open UDP 47998")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--render-mode", default="RealTimePathTracing",
                        choices=["RealTimePathTracing", "RaytracedLighting"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    runtime = SimRuntime(RuntimeConfig(
        width=args.width, height=args.height, fps=args.fps,
        signal_port=args.signal_port, public_ip=args.public_ip or None, render_mode=args.render_mode,
    ))
    logging.info("Starting renderer, physics and WebRTC server...")
    try:
        runtime.start()
    except Exception as exc:
        logging.error("Runtime failed to start: %s", exc)
        sys.exit(1)

    logging.info("Open http://localhost:%d  (API docs: http://localhost:%d/docs)", args.http_port, args.http_port)
    try:
        uvicorn.run(create_app(runtime, args.signal_port), host="0.0.0.0", port=args.http_port, log_level="warning")
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
