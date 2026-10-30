"""``edit_image``: change, combine or restyle images the conversation already has.

Labels resolve only within the calling conversation, so an edit can never read another
conversation's images. Lineage records how each output was made: the first input is the
``edit_source``, further inputs are ``reference`` (ordinal = position), and the mask is
``mask``.
"""

from typing import Annotated

from pydantic import AfterValidator, Field, model_validator

from imaginer.backends.base import ImageInput
from imaginer.backends.registry import BackendRegistry
from imaginer.db.repositories.artifacts import ParentRef
from imaginer.domain.enums import ArtifactKind, Capability, JobTask, LineageRelation
from imaginer.domain.errors import InvalidRequestError, NotFoundError
from imaginer.domain.models import Artifact
from imaginer.orchestration.jobs import JobManager, JobSpec, random_seed
from imaginer.services.artifacts import ArtifactService
from imaginer.tools.base import Tool, ToolContext, ToolOutcome
from imaginer.tools.images import (
    ImageOptions,
    apply_user_options,
    run_image_job,
    user_options_note,
    user_params,
)

AVAILABLE_LIMIT = 30

Label = Annotated[
    str,
    Field(pattern=r"^\s*[A-Za-z]+-\d+\s*$", description="An image label, e.g. img-3"),
    AfterValidator(lambda value: value.strip().lower()),
]


class EditImageArgs(ImageOptions):
    images: list[Label] = Field(
        min_length=1,
        max_length=10,
        description=(
            "Labels of the input images, in order. The first is the image to change; any "
            "others are references to take a person, object or style from."
        ),
    )
    prompt: str = Field(
        min_length=1,
        max_length=4000,
        description=(
            "What to change, as an instruction: e.g. 'replace the sky with a stormy sunset', "
            "'put the hat from image 2 on the cat in image 1', 'remove the background'."
        ),
    )
    mask: Label | None = Field(
        default=None,
        description="Label of a black-and-white mask image: only its white area is changed.",
    )

    @model_validator(mode="after")
    def _distinct_inputs(self) -> "EditImageArgs":
        if len(set(self.images)) != len(self.images):
            raise ValueError("images must not repeat a label")
        if self.mask is not None and self.mask in self.images:
            raise ValueError("mask must not also be one of the images")
        return self


class EditImageTool(Tool):
    name = "edit_image"
    description = (
        "Edit, combine or restyle existing images by their labels. The results are shown "
        "to the user and get new labels."
    )
    guidance = (
        "To change, combine or restyle existing images, call edit_image with their labels "
        "(the image to change first) and an instruction-style prompt. The user's uploads "
        "and earlier results can all be edited."
    )
    args_model = EditImageArgs

    def __init__(
        self, registry: BackendRegistry, jobs: JobManager, artifacts: ArtifactService
    ) -> None:
        self._registry = registry
        self._jobs = jobs
        self._artifacts = artifacts

    def available(self) -> bool:
        return self._registry.offers("edit")

    async def run(self, args: EditImageArgs, ctx: ToolContext) -> ToolOutcome:
        options = ctx.image_options
        args = apply_user_options(args, options)
        if options is not None and options.mask is not None:
            # The user painted this mask for the edit; the model doesn't have to name it.
            args = args.model_copy(update={"mask": options.mask})
        params = user_params(options)
        labels = [*args.images, *([args.mask] if args.mask else [])]
        resolved = await self._resolve(ctx.conversation_id, labels, mask=args.mask)

        require: set[Capability] = set()
        if len(labels) > 1:
            require.add(Capability.MULTI_REF)
        if args.transparent:
            require.add(Capability.RGBA)
        lease = await self._registry.image_backend(
            "edit", options.backend if options else None, require=require
        )
        backend = lease.backend
        try:
            if len(labels) > backend.max_ref_images:
                raise InvalidRequestError(
                    f"at most {backend.max_ref_images} input images (including a mask) "
                    f"are supported, got {len(labels)}"
                )
            inputs = {label: await self._load(artifact) for label, artifact in resolved.items()}
        except BaseException:
            lease.release()  # the job owns it only once submitted
            raise
        first, *others = args.images
        parents = [ParentRef(resolved[first].id, LineageRelation.EDIT_SOURCE, 0)]
        parents += [
            ParentRef(resolved[label].id, LineageRelation.REFERENCE, ordinal)
            for ordinal, label in enumerate(others, start=1)
        ]
        if args.mask:
            parents.append(ParentRef(resolved[args.mask].id, LineageRelation.MASK, 0))

        spec = JobSpec(
            conversation_id=ctx.conversation_id,
            task=JobTask.EDIT,
            backend=backend,
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
            images=tuple(inputs[label] for label in args.images),
            mask=inputs[args.mask] if args.mask else None,
            parents=tuple(parents),
            requested={**args.model_dump(exclude_none=True), **params},
            request_message_id=ctx.message_id,
            tool_call_id=ctx.tool_call_id,
        )
        return await run_image_job(
            self._jobs,
            spec,
            ctx,
            edited=args.images,
            **({"mask": args.mask} if args.mask else {}),
            **user_options_note(options),
        )

    async def _resolve(
        self, conversation_id: str, labels: list[str], *, mask: str | None = None
    ) -> dict[str, Artifact]:
        resolved: dict[str, Artifact] = {}
        unknown: list[str] = []
        for label in labels:
            try:
                artifact = await self._artifacts.get_by_label(conversation_id, label)
            except NotFoundError:
                unknown.append(label)
                continue
            # A painted mask (mask-N) can only be the mask; any image can serve as one too.
            allowed = {ArtifactKind.IMAGE}
            if label == mask:
                allowed.add(ArtifactKind.MASK)
            if artifact.kind not in allowed:
                raise InvalidRequestError(f"{label} is a {artifact.kind.value}, not an image")
            resolved[label] = artifact
        if unknown:
            page = await self._artifacts.list(
                conversation_id, kind=ArtifactKind.IMAGE, limit=AVAILABLE_LIMIT
            )
            available = ", ".join(a.label for a in page.items) or "none yet"
            raise InvalidRequestError(
                f"unknown image label(s) {', '.join(unknown)}; this conversation has: {available}"
            )
        return resolved

    async def _load(self, artifact: Artifact) -> ImageInput:
        _, data = await self._artifacts.read_bytes(artifact.id)
        return ImageInput(
            data=data,
            mime=artifact.mime,
            label=artifact.label,
            width=artifact.width,
            height=artifact.height,
        )
