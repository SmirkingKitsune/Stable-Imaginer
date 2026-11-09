"""Run the fake engines for manual testing: ``uv run python -m tests.fakes``.

Ports match ``config/dev.yaml``. Then start the server with ``uv run imaginer serve``.
"""

import argparse
import threading

import uvicorn

from tests.fakes.fake_llama import FakeLlama
from tests.fakes.fake_sdcpp import FakeSdCpp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-port", type=int, default=18080)
    parser.add_argument("--sd-port", type=int, default=11234)
    parser.add_argument("--sd-step-delay", type=float, default=0.3, help="seconds per fake step")
    parser.add_argument(
        "--sd-loras",
        default="draft-8-steps.safetensors,styles/ink-wash.safetensors,styles/film-grain.safetensors",
        help="comma-separated LoRA paths the fake sd-server lists ('' for none)",
    )
    args = parser.parse_args()

    llama = FakeLlama(url=f"http://127.0.0.1:{args.llama_port}")
    sd = FakeSdCpp(url=f"http://127.0.0.1:{args.sd_port}", step_delay_s=args.sd_step_delay)
    sd.loras = [path for path in args.sd_loras.split(",") if path]
    sd_server = uvicorn.Server(
        uvicorn.Config(sd.app(), host="127.0.0.1", port=args.sd_port, log_level="warning")
    )
    threading.Thread(target=sd_server.run, daemon=True).start()
    print(f"fake sd-server on {sd.url}")  # noqa: T201
    print(f"fake llama-server on {llama.url}")  # noqa: T201
    uvicorn.run(llama.app(), host="127.0.0.1", port=args.llama_port, log_level="warning")


if __name__ == "__main__":
    main()
