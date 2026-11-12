"""``set_portrait``: the agent paints a self-portrait and makes it its portrait.

The agent writes its own prompt; the image is painted square on the default image engine
(or the one the user picked), shown in the conversation like any other image, and a square
copy becomes the agent's portrait: its next self-portrait version (agents/library.py).
Offered only to a turn that has an agent.
"""

import dataclasses

from pydantic import BaseModel, ConfigDict, Field

from imaginer.agents.library import AgentLibrary
from imaginer.backends.registry import BackendRegistry
from imaginer.domain.enums import Capability, JobTask
from imaginer.domain.events import EventType, JobUpdateEventData
from imaginer.orchestration.jobs import Emit, JobManager, JobSpec, random_seed
from imaginer.services.artifacts import ArtifactService
from imaginer.tools.base import Tool, ToolContext, ToolOutcome
from imaginer.tools.images import (
    image_prompt,
    loras_requested,
    run_image_job,
    sampler_params,
    user_loras,
    user_preset,
)

PORTRAIT_ASPECT = "1:1"


class SetPortraitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        min_length=1,
        max_length=4000,
        description=(
            "Your self-portrait as you picture yourself: subject, style, colours, framing. "
            "It is painted square and shown small, so keep it simple and bold."
        ),
    )
    negative_prompt: str | None = Field(
        default=None, max_length=2000, description="Things that must not appear."
    )


class SetPortraitTool(Tool):
    name = "set_portrait"
    description = (
        "Paint a square self-portrait from your own prompt and make it your portrait, the "
        "picture shown beside your replies. The painting also appears in this conversation."
    )
    guidance = (
        "You can paint your own portrait with set_portrait. Ask the user before changing it, "
        "unless they asked you to."
    )
    args_model = SetPortraitArgs
    summary = "Paints a self-portrait and makes it the agent's portrait. Asks first unless told to."
    flags = "t2i · 1:1"
    needs_agent = True

    def __init__(
        self,
        registry: BackendRegistry,
        jobs: JobManager,
        artifacts: ArtifactService,
        agents: AgentLibrary,
    ) -> None:
        self._registry = registry
        self._jobs = jobs
        self._artifacts = artifacts
        self._agents = agents

    def available(self) -> bool:
        return self._registry.offers("t2i")

    async def run(self, args: SetPortraitArgs, ctx: ToolContext) -> ToolOutcome:
        agent = self._agents.agent(ctx.agent_id)
        if agent is None:
            return ToolOutcome.error("no_agent", "only an agent has a portrait to set")
        options = ctx.image_options
        preset = user_preset(options, self._registry, "t2i")
        loras = user_loras(options)
        params = sampler_params(ctx)
        lease = await self._registry.image_backend(
            "t2i",
            options.backend if options else None,
            require={Capability.LORA} if loras else set(),
            **preset,
        )
        spec = JobSpec(
            conversation_id=ctx.conversation_id,
            task=JobTask.T2I,
            backend=lease.backend,
            lease=lease,
            prompt=image_prompt(args.prompt),
            negative_prompt=args.negative_prompt,
            seed=random_seed(),
            n=1,
            aspect_ratio=PORTRAIT_ASPECT,
            params=params,
            requested={
                **args.model_dump(exclude_none=True),
                "aspect_ratio": PORTRAIT_ASPECT,
                **params,
                **preset,
                **loras_requested(loras),
            },
            **preset,
            loras=loras,
            request_message_id=ctx.message_id,
            tool_call_id=ctx.tool_call_id,
        )
        self._agents.painting_started(agent.id, ctx.conversation_id)
        try:
            ctx = dataclasses.replace(ctx, emit=self._following(agent.id, ctx.emit))
            outcome = await run_image_job(self._jobs, spec, ctx)
            if not outcome.ok or not outcome.artifact_ids:
                return outcome
            artifact, data = await self._artifacts.read_bytes(outcome.artifact_ids[0])
            version = await self._agents.record_self_portrait(
                agent.id,
                data,
                prompt=args.prompt,
                conversation_id=ctx.conversation_id,
                artifact_id=artifact.id,
                label=artifact.label,
            )
            outcome.content["portrait"] = {"version": version, "label": artifact.label}
            outcome.content["note"] = (
                f"This is your portrait now (self-portrait v{version}). The user can see it."
            )
            return outcome
        finally:
            self._agents.painting_ended(agent.id)

    def _following(self, agent_id: str, emit: Emit) -> Emit:
        """The turn's emitter, also telling the library how the painting goes."""
        agents = self._agents

        async def wrapped(event_type: EventType, data: BaseModel) -> None:
            if event_type is EventType.JOB_UPDATE and isinstance(data, JobUpdateEventData):
                agents.painting_progress(agent_id, data.job)
            await emit(event_type, data)

        return wrapped
