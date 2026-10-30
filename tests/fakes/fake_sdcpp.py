"""A fake ``sd-server`` speaking the native ``/sdcpp/v1`` async API.

Shapes follow sd.cpp's examples/server (async_jobs.cpp ``make_async_job_json``,
routes_sdcpp.cpp): ``img_gen`` answers 202 with a job id; jobs go
``queued -> generating -> completed|failed|cancelled``; previews carry step progress;
each result image has ``generation_info`` with the seed as a decimal string.

Images are deterministic per seed. A prompt using Qwen-Image-2.1's RGBA phrasing yields a
transparent background.
"""

import asyncio
import base64
import io
import itertools
import threading
from dataclasses import dataclass, field
from typing import Any

from PIL import Image
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

RGBA_MARKER = "This is an RGBA image with transparency."
_ids = itertools.count(1)


@dataclass
class FakeSdCpp:
    url: str = ""
    state: str = "ready"  # ready | loading | empty | error
    origin: str = "command_line"
    ref_images_feature: bool = True
    preview_modes: list[str] = field(default_factory=lambda: ["none", "proj", "tae", "vae"])
    default_steps: int = 3
    step_delay_s: float = 0.0
    fail_next: str | None = None
    fail_at_request: int | None = None  # 1-based: fail exactly that request
    hold: threading.Event | None = None  # generating jobs pause after step 1 until set
    queue_full: bool = False
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    requests: list[dict[str, Any]] = field(default_factory=list)
    cancels: list[str] = field(default_factory=list)
    started: threading.Event = field(default_factory=threading.Event)  # a job began generating

    def reset(self) -> None:
        self.state = "ready"
        self.origin = "command_line"
        self.ref_images_feature = True
        self.preview_modes = ["none", "proj", "tae", "vae"]
        self.default_steps = 3
        self.step_delay_s = 0.0
        self.fail_next = None
        self.fail_at_request = None
        self.hold = None
        self.queue_full = False
        self.jobs.clear()
        self.requests.clear()
        self.cancels.clear()
        self.started = threading.Event()

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/sdcpp/v1/capabilities", self._capabilities),
                Route("/sdcpp/v1/img_gen", self._img_gen, methods=["POST"]),
                Route("/sdcpp/v1/jobs/{job_id}", self._job),
                Route("/sdcpp/v1/jobs/{job_id}/cancel", self._cancel, methods=["POST"]),
            ]
        )

    async def _capabilities(self, request: Request) -> Response:
        features = {"ref_images": self.ref_images_feature, "init_image": True, "preview": True}
        return JSONResponse(
            {
                "model": {
                    "name": "qwen_image_2.1-Q4_K.gguf",
                    "stem": "qwen_image_2.1-Q4_K",
                    "path": "/models/qwen_image_2.1-Q4_K.gguf",
                },
                "runtime": {"state": self.state, "origin": self.origin},
                "current_mode": "img_gen",
                "supported_modes": ["img_gen"] if self.state == "ready" else [],
                "features_by_mode": {"img_gen": features},
                "features": features,
                "limits": {
                    "min_width": 64,
                    "max_width": 4096,
                    "min_height": 64,
                    "max_height": 4096,
                },
                "samplers": ["euler", "euler_a"],
                "schedulers": ["discrete"],
                "preview_modes": self.preview_modes,
            }
        )

    async def _img_gen(self, request: Request) -> Response:
        if self.queue_full:
            return JSONResponse({"error": "job queue is full"}, status_code=429)
        body = await request.json()
        if not body.get("prompt"):
            return JSONResponse({"error": "invalid generation parameters"}, status_code=400)
        self.requests.append(body)
        job_id = f"job-{next(_ids)}"
        job: dict[str, Any] = {
            "id": job_id,
            "kind": "img_gen",
            "status": "queued",
            "cancel_requested": False,
            "created": 0,
            "started": None,
            "completed": None,
            "queue_position": 1,
            "result": None,
            "error": None,
        }
        self.jobs[job_id] = job
        asyncio.get_running_loop().create_task(self._execute(job, body))
        out = {key: job[key] for key in ("id", "kind", "status", "created")}
        return JSONResponse({**out, "poll_url": f"/sdcpp/v1/jobs/{job_id}"}, status_code=202)

    async def _job(self, request: Request) -> Response:
        job = self.jobs.get(request.path_params["job_id"])
        if job is None:
            return JSONResponse({"error": "job not found"}, status_code=404)
        return JSONResponse(job)

    async def _cancel(self, request: Request) -> Response:
        job_id = request.path_params["job_id"]
        job = self.jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "job not found"}, status_code=404)
        self.cancels.append(job_id)
        if job["status"] == "queued":
            job.update(
                status="cancelled",
                error={"code": "cancelled", "message": "job cancelled by client"},
            )
        elif job["status"] == "generating":
            job["cancel_requested"] = True
        return JSONResponse(job)

    async def _execute(self, job: dict[str, Any], body: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        if job["status"] == "cancelled":
            return
        job.update(status="generating", started=1, queue_position=0)
        self.started.set()
        steps = (body.get("sample_params") or {}).get("sample_steps") or self.default_steps
        for step in range(1, steps + 1):
            if body.get("preview", "none") != "none":
                job["preview"] = {"pass": 1, "step": step, "total_steps": steps, "b64_json": ""}
            if step == 1 and self.hold is not None:
                await asyncio.to_thread(self.hold.wait, 30)
            if self.step_delay_s:
                await asyncio.sleep(self.step_delay_s)
            if job["cancel_requested"]:
                job.pop("preview", None)
                job.update(
                    status="cancelled",
                    error={"code": "cancelled", "message": "job cancelled by client"},
                )
                return
        job.pop("preview", None)
        if self.fail_at_request is not None and len(self.requests) == self.fail_at_request:
            self.fail_next = "planned failure"
        if self.fail_next is not None:
            message, self.fail_next = self.fail_next, None
            job.update(status="failed", error={"code": "generation_failed", "message": message})
            return
        refs = body.get("ref_images") or []
        for ref in refs:  # must be decodable data URLs
            base64.b64decode(ref.split(",", 1)[1], validate=True)
        seed = int(body["seed"])
        data = render(
            seed,
            body["width"],
            body["height"],
            RGBA_MARKER in body["prompt"],
            body.get("output_format", "png"),
        )
        job.update(
            status="completed",
            completed=2,
            result={
                "output_format": body.get("output_format", "png"),
                "images": [
                    {
                        "index": 0,
                        "b64_json": base64.b64encode(data).decode(),
                        "generation_info": {
                            "schema": "sdcpp.generation/v1",
                            "prompt": body["prompt"],
                            "seed": str(seed),
                            "width": body["width"],
                            "height": body["height"],
                            "ref_images": len(refs),
                        },
                    }
                ],
            },
        )


def render(seed: int, width: int, height: int, transparent: bool, fmt: str = "png") -> bytes:
    color = (seed * 37 % 256, seed * 91 % 256, seed * 53 % 256)
    if transparent:
        im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        im.paste((*color, 255), (width // 4, height // 4, 3 * width // 4, 3 * height // 4))
    else:
        im = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    im.save(buf, format={"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}[fmt])
    return buf.getvalue()
