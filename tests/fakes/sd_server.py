"""A fake sd-server process, launched the way the Engines screen launches the real one.

It takes sd-server's command line (``--listen-ip``, ``--listen-port``, ``--model`` or
``--diffusion-model``; it ignores the rest), waits ``--load-delay`` seconds before it opens
its port as the real one does while it loads, then serves the fake ``/sdcpp/v1`` API (see
fake_sdcpp.py). It reports the main file it was given as its model, and colours its images
by it, so the images of an engine's model card and its edit card tell apart.

    tests/fakes/sd-server.cmd  # what config/dev-swap.yaml launches
"""

import argparse
import os
import threading
import time
import zlib
from pathlib import PurePath

import psutil  # type: ignore[import-untyped]
import uvicorn

from tests.fakes.fake_sdcpp import FakeSdCpp


def _exit_with_the_engine() -> None:
    """Started through a wrapper (sd-server.cmd, then the venv's python launcher), this is a
    grandchild of the process the Engines screen stops: when that dies, this must too, or
    it holds the log pipe open and the stop never ends."""
    ancestors = psutil.Process().parents()[:2]  # the venv launcher, the wrapper
    while all(p.is_running() for p in ancestors):
        time.sleep(0.3)
    os._exit(0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-ip", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=1234)
    parser.add_argument("--model")
    parser.add_argument("--diffusion-model")
    parser.add_argument("--load-delay", type=float, default=3.0, help="seconds to 'load'")
    parser.add_argument("--step-delay", type=float, default=0.3, help="seconds per fake step")
    args, _ = parser.parse_known_args(argv)
    threading.Thread(target=_exit_with_the_engine, daemon=True).start()
    name = PurePath(args.diffusion_model or args.model or "model.gguf").name
    print(f"loading {name}", flush=True)  # noqa: T201
    time.sleep(args.load_delay)
    url = f"http://{args.listen_ip}:{args.listen_port}"
    sd = FakeSdCpp(
        url=url,
        step_delay_s=args.step_delay,
        model_name=name,
        color_salt=zlib.crc32(name.encode()) % 997,
    )
    print(f"loaded {name}; listening on: {url}", flush=True)  # noqa: T201
    uvicorn.run(sd.app(), host=args.listen_ip, port=args.listen_port, log_level="warning")


if __name__ == "__main__":
    main()
