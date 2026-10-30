from pydantic import Field

from imaginer.backends.registry import BackendRegistry
from imaginer.domain.enums import Capability, JobTask
from imaginer.orchestration.jobs import JobManager, JobSpec, random_seed
from imaginer.tools.base import Tool, ToolContext, ToolOutcome
from imaginer.tools.images import (
    ImageOptions,
    apply_user_options,
    run_image_job,
    user_options_note,
    user_params,
)


class GenerateImageArgs(ImageOptions):
    prompt: str = Field(
        min_length=1,
        max_length=4000,
        description=(
            "Detailed visual description of the image: subject, setting, style, lighting, "
            "composition. Quote any text that must appear in the image."
        ),
    )


class GenerateImageTool(Tool):
    name = "generate_image"
    description = (
        "Create new images from a text description. The images are shown to the user and "
        "get labels (e.g. img-4) you can refer to afterwards."
    )
    guidance = (
        "To create an image, call generate_image with a detailed English prompt. "
        "Afterwards, refer to the result by its label and don't describe details you "
        "did not ask for."
    )
    args_model = GenerateImageArgs

    def __init__(self, registry: BackendRegistry, jobs: JobManager) -> None:
        self._registry = registry
        self._jobs = jobs

    def available(self) -> bool:
        return self._registry.offers("t2i")

    async def run(self, args: GenerateImageArgs, ctx: ToolContext) -> ToolOutcome:
        options = ctx.image_options
        args = apply_user_options(args, options)
        params = user_params(options)
        require = {Capability.RGBA} if args.transparent else set()
        lease = await self._registry.image_backend(
            "t2i", options.backend if options else None, require=require
        )
        spec = JobSpec(
            conversation_id=ctx.conversation_id,
            task=JobTask.T2I,
            backend=lease.backend,
            lease=lease,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed if args.seed is not None else random_seed(),
            n=args.n,
            width=args.width,
            height=args.height,
            aspect_ratio=args.aspect_ratio,
            alpha=args.transparent,
            params=params,
            requested={**args.model_dump(exclude_none=True), **params},
            request_message_id=ctx.message_id,
            tool_call_id=ctx.tool_call_id,
        )
        return await run_image_job(self._jobs, spec, ctx, **user_options_note(options))
