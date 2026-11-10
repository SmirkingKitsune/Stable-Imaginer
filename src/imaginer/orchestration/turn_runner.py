"""One assistant turn: store the user message, run the LLM (and its tools), store results.

``start`` does everything that can fail cheaply (unknown conversation, bad attachments, no
capable or reachable backend, turn already running) before anything is stored, so those
errors come back as plain HTTP errors. Everything after that runs in a background task
and reports through the turn's event channel. The task does not depend on the HTTP client:
if the client disconnects, the turn still completes and is persisted.

The tool loop: stream an assistant message; if it calls tools, run them one by one, store
each result as a ``tool`` message, rebuild the context and stream again. At most
``orchestration.max_tool_iterations`` rounds may call tools; the round after that is
told ``tool_choice: none`` so the turn always ends with an answer.

Branches: every message stored in a turn follows the one stored before it (``parent_id``),
starting from the branch the user message continues. ``retry`` is a turn without a user
message: it stores a new attempt at an earlier tool call beside the old result, runs the
call again with a new seed, and lets the assistant reply from there.

Prompt modes (``MessageCreate.mode``): ``chat`` is the loop above. ``interpret`` and
``curate`` add guidance (``orchestration/modes.py``), and ``interpret`` makes the first round
call a tool. ``raw`` runs no LLM at all: the turn stores an assistant message that calls the
image tool with the user's words as the prompt, runs it, and ends there, so later turns see
an ordinary tool call.

Quality check (``MessageCreate.qa``): after an image tool succeeds, a vision LLM reviews the
images (``orchestration/qa.py``). If it wants them redone, the call runs again with its
changes as a new attempt beside the old result, like ``retry``, at most
``orchestration.qa.max_retries`` times. Each result records the check in ``metadata.qa``,
and in its content so the assistant can tell the user.

LoRA mentions (``@lora:<name>``, ``domain/mentions.py``) in the user's words add LoRAs to
every image call of the turn, on top of ``image_options.loras``. The stored message keeps
them as written; the LLM, the quality check and a raw prompt get the words without them.

Agents (``agents/library.py``): a turn goes to the agent the message names, else the
conversation's, else the one the default chat engine runs. Its engine answers, told the
agent's system prompt, offered only the tools it has on, with its sampling; each assistant
message records the agent, and the conversation remembers it (``metadata.agent``). A message
that names a backend but no agent runs as before, without one.
"""

import asyncio
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from imaginer.agents.library import AgentLibrary, Persona
from imaginer.backends.base import ChatRequest, LLMBackend
from imaginer.backends.registry import BackendRegistry, Lease
from imaginer.core.clock import utcnow
from imaginer.core.config import OrchestrationConfig
from imaginer.core.ids import new_id
from imaginer.core.logging import get_logger
from imaginer.db import models as orm
from imaginer.db.engine import Database
from imaginer.db.repositories import artifacts as art_repo
from imaginer.db.repositories import conversations as conv_repo
from imaginer.db.repositories import messages as msg_repo
from imaginer.domain import mentions
from imaginer.domain import models as dto
from imaginer.domain.enums import ArtifactKind, Capability, MessageStatus, PromptMode, Role
from imaginer.domain.errors import (
    EngineInterruptedError,
    ImaginerError,
    InvalidRequestError,
    NotFoundError,
)
from imaginer.domain.events import (
    DeltaEventData,
    DoneEventData,
    ErrorEventData,
    EventType,
    MessageEventData,
    ToolCallEventData,
)
from imaginer.orchestration.assembler import MessageAssembler
from imaginer.orchestration.context_builder import ContextBuilder
from imaginer.orchestration.event_bus import TurnChannel, TurnManager
from imaginer.orchestration.jobs import Emit
from imaginer.orchestration.modes import MODE_GUIDANCE
from imaginer.orchestration.qa import (
    PINNABLE,
    QA_GUIDANCE,
    QualityChecker,
    Review,
    Subject,
    preset_pins,
)
from imaginer.services import mappers
from imaginer.tools.base import Tool, ToolContext, ToolOutcome, ToolRegistry

log = get_logger(__name__)

IMAGE_TOOLS = frozenset({"generate_image", "edit_image"})  # what the quality check reviews


@dataclass
class _TurnState:
    message_ids: list[str]
    leaf_id: str  # the branch's newest message: the next one stored follows it
    streaming_id: str | None = None  # assistant message currently being streamed
    assembler: MessageAssembler = field(default_factory=MessageAssembler)
    assistant_id: str | None = None  # last assistant message
    pending_calls: list[dict[str, Any]] = field(default_factory=list)  # calls without a result
    image_options: dto.ImageRequestOptions | None = None  # the user's picks, for every tool call
    placeholders: dict[str, str] = field(default_factory=dict)  # call id -> stored result to fill
    mode: PromptMode = PromptMode.CHAT
    raw_call: dict[str, Any] | None = None  # raw mode: the call to store and run, no LLM
    force_tool: bool = False  # interpret mode: the next round must call a tool
    qa: bool = False  # check every image this turn makes
    request: str = ""  # the user's words, for the quality check
    reviewer: LLMBackend | None = None  # the vision LLM that checks; None: checks are skipped
    qa_skip: str | None = None  # why there is no reviewer
    persona: Persona | None = None  # the agent answering, if any
    only_tools: frozenset[str] | None = None  # offer just these tools, and require a call


class TurnRunner:
    def __init__(
        self,
        *,
        db: Database,
        registry: BackendRegistry,
        turns: TurnManager,
        context: ContextBuilder,
        tools: ToolRegistry,
        qa: QualityChecker,
        config: OrchestrationConfig,
    ) -> None:
        self._db = db
        self._registry = registry
        self._turns = turns
        self._context = context
        self._tools = tools
        self._qa = qa
        self._config = config
        self._prompt_hints: Callable[[str | None], str | None] | None = None
        self._agents: AgentLibrary | None = None

    def set_prompt_hints(self, hints: Callable[[str | None], str | None]) -> None:
        """How to prompt the model an image backend runs (None: the default one's): the
        model library's note, added to the system prompt when image tools are offered."""
        self._prompt_hints = hints

    def set_agents(self, agents: AgentLibrary) -> None:
        """Where a turn's agent comes from: its engine, persona, tools and sampling."""
        self._agents = agents

    async def _agent_for(
        self, conversation_agent: str | None, agent_id: str | None, backend: str | None
    ) -> tuple[Persona | None, str | None]:
        """The persona a turn answers with and the chat backend it runs on. Naming a
        backend but no agent runs without one, as before agents."""
        if self._agents is None or (agent_id is None and backend is not None):
            return None, backend
        agent = await self._agents.resolve(agent_id, conversation_agent)
        if agent is None:
            return None, backend
        engine = self._agents.check_runs(agent)
        if backend is not None and backend != engine:
            raise InvalidRequestError(f"{agent.name} runs on {engine}, not {backend}")
        return self._agents.persona(agent), engine

    async def start(
        self,
        conversation_id: str,
        body: dto.MessageCreate,
        *,
        only_tools: frozenset[str] | None = None,
    ) -> TurnChannel:
        """Start a turn. ``only_tools``: offer just these tools and require one to be called
        (an agent painting its portrait)."""
        if not body.content.strip() and not body.attachments:
            raise InvalidRequestError("a message needs content or attachments")
        if len(set(body.attachments)) != len(body.attachments):
            raise InvalidRequestError("attachments contain duplicates")
        if body.mode is PromptMode.RAW and not mentions.strip(body.content):
            raise InvalidRequestError(
                "raw mode sends the message as the image prompt: add text (besides any LoRAs)"
            )
        options = mentions.with_mentions(body.image_options, body.content)

        async with self._db.session() as session, session.begin():
            conversation = await conv_repo.require_active(session, conversation_id)
            conversation_agent = (conversation.metadata_json or {}).get("agent")
            found = await art_repo.get_many_in_conversation(
                session, conversation_id, body.attachments
            )
        missing = sorted(set(body.attachments) - {a.id for a in found})
        if missing:
            raise InvalidRequestError(
                "attachments must be artifacts of this conversation", missing=missing
            )
        by_id = {a.id: a for a in found}
        attached = [by_id[i] for i in body.attachments]

        qa = body.qa if body.qa is not None else self._config.qa.enabled
        raw_call = None
        persona: Persona | None = None
        if body.mode is PromptMode.RAW:
            raw_call = self._raw_call(mentions.strip(body.content), attached)
            lease = await self._optional_reviewer(body.backend) if qa else None
        else:
            persona, backend = await self._agent_for(
                conversation_agent if isinstance(conversation_agent, str) else None,
                body.agent,
                body.backend,
            )
            required = {Capability.CHAT, Capability.STREAM}
            if any(a.kind is ArtifactKind.IMAGE for a in attached):
                # Never drop the user's image silently: a backend that can't see it is an error.
                required.add(Capability.VISION_IN)
            if body.mode is not PromptMode.CHAT:
                if not self._tools.offered():
                    raise InvalidRequestError(
                        f"{body.mode.value} mode makes images, but no image backend is available"
                    )
                required.add(Capability.TOOLS)
            if only_tools is not None:
                if not any(t.name in only_tools for t in self._tools.offered()):
                    raise InvalidRequestError(
                        f"{', '.join(sorted(only_tools))}: no image engine is available"
                    )
                required.add(Capability.TOOLS)
            lease = await self._registry.chat_backend(backend, require=required)
        try:
            if body.image_options is not None and body.image_options.backend is not None:
                self._registry.validate_image_backend(body.image_options.backend)
            if body.image_options is not None and body.image_options.preset is not None:
                self._registry.validate_image_preset(
                    body.image_options.preset, body.image_options.backend
                )
            if body.image_options is not None and body.image_options.mask is not None:
                await self._check_mask(conversation_id, body.image_options.mask)

            channel = self._turns.open(conversation_id)
            try:
                user_message = await self._store_user_message(
                    conversation_id, body, persona.agent_id if persona is not None else None
                )
            except BaseException:
                self._turns.release(channel)
                raise
        except BaseException:
            if lease is not None:
                lease.release()
            raise
        await channel.publish(EventType.MESSAGE, MessageEventData(message=user_message))
        state = _TurnState(
            message_ids=[user_message.id],
            leaf_id=user_message.id,
            image_options=options,
            mode=body.mode,
            raw_call=raw_call,
            force_tool=body.mode is PromptMode.INTERPRET or only_tools is not None,
            request=mentions.for_llm(body.content),
            persona=persona,
            only_tools=only_tools,
        )
        self._set_reviewer(state, lease, qa=qa)
        channel.task = self._start(channel, lease, state)
        return channel

    def _raw_call(self, prompt: str, attached: list[orm.Artifact]) -> dict[str, Any]:
        """The tool call a raw message stands for: an edit of its attached images (in
        order; a painted mask rides in image_options.mask), or else a new image."""
        images = [a.label for a in attached if a.kind is ArtifactKind.IMAGE]
        name = "edit_image" if images else "generate_image"
        if not any(tool.name == name for tool in self._tools.offered()):
            what = "editing" if images else "generation"
            raise InvalidRequestError(f"raw mode: no image {what} backend is available")
        arguments = {"images": images, "prompt": prompt} if images else {"prompt": prompt}
        return {
            "id": f"call_{new_id().replace('-', '')[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }

    async def _optional_reviewer(self, backend: str | None) -> Lease[LLMBackend] | None:
        """A raw turn needs no LLM; it only borrows one to check images, if one is ready."""
        try:
            return await self._registry.chat_backend(
                backend, require={Capability.CHAT, Capability.STREAM, Capability.VISION_IN}
            )
        except ImaginerError as exc:
            log.info("qa.no_reviewer", detail=exc.detail)
            return None

    @staticmethod
    def _set_reviewer(state: _TurnState, lease: Lease[LLMBackend] | None, *, qa: bool) -> None:
        state.qa = qa
        if not qa:
            return
        if lease is None:
            state.qa_skip = "no chat engine that can see images is ready"
        elif Capability.VISION_IN not in lease.backend.capabilities():
            state.qa_skip = f"{lease.backend.name} can't see images"
        else:
            state.reviewer = lease.backend

    async def _check_mask(self, conversation_id: str, label: str) -> None:
        async with self._db.session() as session:
            mask = await art_repo.get_by_label(session, conversation_id, label)
        if mask is None or mask.kind is not ArtifactKind.MASK:
            raise InvalidRequestError(f"image_options.mask: no mask labelled {label!r} here")

    async def retry(self, conversation_id: str, message_id: str) -> TurnChannel:
        """Run the tool call behind the result ``message_id`` again, as a new attempt.

        Same arguments, image options, mode and quality check as the original call, but a
        new seed. The attempt is stored right away (``streaming``, beside the old result)
        and filled in when the tool finishes; then the assistant replies on that new branch
        (not in raw mode, which has no assistant).
        """
        async with self._db.session() as session, session.begin():
            await conv_repo.require_active(session, conversation_id)
            result = await msg_repo.get_in_conversation(session, conversation_id, message_id)
            if result is None:
                raise NotFoundError(
                    f"message {message_id} not found in conversation {conversation_id}"
                )
            if result.role is not Role.TOOL or result.status is MessageStatus.STREAMING:
                raise InvalidRequestError("only a finished tool result can be retried")
            path = await msg_repo.path_to(session, result.id)
        callers = [i for i, m in enumerate(path) if m.role is Role.ASSISTANT]
        caller = path[callers[-1]] if callers else None
        calls = (caller.tool_calls_json if caller is not None else None) or []
        call = next((c for c in calls if c.get("id") == result.tool_call_id), None)
        if caller is None or call is None:
            raise InvalidRequestError("the tool call behind this result is not stored")
        asker = next((m for m in reversed(path[: callers[-1]]) if m.role is Role.USER), None)
        asked_for = (asker.content_text or "") if asker is not None else ""
        options = mentions.with_mentions(_options_without_seed(asker), asked_for)
        asked = (asker.metadata_json if asker is not None else None) or {}
        mode = PromptMode(asked.get("mode", PromptMode.CHAT))
        qa = bool(asked.get("qa", self._config.qa.enabled))
        persona: Persona | None = None
        if mode is PromptMode.RAW:
            lease = await self._optional_reviewer(None) if qa else None
        else:
            caller_agent = caller.agent_id
            if self._agents is not None and self._agents.agent(caller_agent) is None:
                caller_agent = None  # deleted since: the conversation's, or the default
            persona, backend = await self._agent_for(
                await self._conversation_agent(conversation_id), caller_agent, None
            )
            lease = await self._registry.chat_backend(
                backend, require={Capability.CHAT, Capability.STREAM}
            )
        try:
            if options is not None and options.backend is not None:
                self._registry.validate_image_backend(options.backend)

            channel = self._turns.open(conversation_id)
            try:
                attempt = await self._store_tool_placeholder(
                    conversation_id, result.parent_id, call
                )
            except BaseException:
                self._turns.release(channel)
                raise
        except BaseException:
            if lease is not None:
                lease.release()
            raise
        await channel.publish(EventType.MESSAGE, MessageEventData(message=attempt))
        log.info("tool.retry", tool_call_id=call["id"], retry_of=result.id)
        state = _TurnState(
            message_ids=[attempt.id],
            leaf_id=attempt.id,
            assistant_id=caller.id,
            pending_calls=[_reseeded(call)],
            image_options=options,
            placeholders={call["id"]: attempt.id},
            mode=mode,
            request=mentions.for_llm(asked_for),
            persona=persona,
        )
        self._set_reviewer(state, lease, qa=qa)
        channel.task = self._start(channel, lease, state)
        return channel

    # --- the turn -------------------------------------------------------------

    def _start(
        self, channel: TurnChannel, lease: Lease[LLMBackend] | None, state: _TurnState
    ) -> asyncio.Task[None]:
        """Run the turn. It holds the chat backend's lease until it ends, however it ends
        (even cancelled before it starts), so a reconfigure can't close it mid-turn. A raw
        turn may have no chat backend at all."""
        backend = lease.backend if lease is not None else None
        task = asyncio.create_task(
            self._run(channel, backend, state), name=f"turn-{channel.turn_id}"
        )
        if lease is not None:
            task.add_done_callback(lambda _: lease.release())
        return task

    async def _run(
        self, channel: TurnChannel, backend: LLMBackend | None, state: _TurnState
    ) -> None:
        agent_id = state.persona.agent_id if state.persona is not None else None
        if agent_id is not None and self._agents is not None:
            self._agents.turn_started(agent_id)
        with structlog.contextvars.bound_contextvars(
            conversation_id=channel.conversation_id, turn_id=channel.turn_id
        ):
            log.info(
                "turn.started",
                mode=state.mode.value,
                qa=state.qa,
                backend=backend.name if backend else None,
                model=backend.model if backend else None,
                agent=agent_id,
            )
            try:
                if state.mode is PromptMode.RAW:
                    await self._raw(channel, state)
                else:
                    assert backend is not None
                    await self._loop(channel, backend, state)
                await self._done(channel, state, state.assembler.finish_reason)
                log.info("turn.done", finish_reason=state.assembler.finish_reason)
            except asyncio.CancelledError:
                log.info("turn.cancelled")
                try:
                    await self._wind_down(channel, state, MessageStatus.CANCELLED)
                except Exception:  # e.g. the conversation is being deleted
                    log.exception("turn.wind_down_failed")
                await self._done(channel, state, "cancelled")
            except Exception as exc:
                await self._fail(channel, state, exc)
            finally:
                if agent_id is not None and self._agents is not None:
                    self._agents.turn_ended(agent_id)
                await channel.close()
                self._turns.release(channel)

    async def _loop(self, channel: TurnChannel, backend: LLMBackend, state: _TurnState) -> None:
        capabilities = backend.capabilities()
        tools = self._tools.offered() if Capability.TOOLS in capabilities else []
        tools = [t for t in tools if _allowed(t, state)]
        schemas = [tool.schema() for tool in tools]
        guidance = [tool.guidance for tool in tools]
        if tools and state.mode in MODE_GUIDANCE:
            guidance.append(MODE_GUIDANCE[state.mode])
        if tools and state.reviewer is not None:
            guidance.append(QA_GUIDANCE)
        if tools and self._prompt_hints is not None:
            chosen = state.image_options.backend if state.image_options is not None else None
            try:
                hint = self._prompt_hints(chosen)
            except Exception:
                log.exception("turn.prompt_hint_failed")
                hint = None
            if hint:
                guidance.append(f"About the image model: {hint}")
        vision = Capability.VISION_IN in capabilities
        rounds = self._config.max_tool_iterations if tools else 0
        for round_index in range(rounds + 1):
            # Calls from the previous round, or a retry's call, which comes ready-made.
            for call in list(state.pending_calls):
                await self._call_tool(channel, state, call)
            may_call = round_index < rounds
            persona = state.persona
            context = await self._context.build(
                channel.conversation_id,
                leaf_id=state.leaf_id,
                vision=vision,
                tool_guidance=guidance,
                system_prompt=persona.system_prompt if persona is not None else None,
            )
            tool_choice = None if may_call or not schemas else "none"
            if may_call and state.force_tool:
                tool_choice, state.force_tool = "required", False
            request = ChatRequest(
                messages=context,
                tools=schemas or None,
                tool_choice=tool_choice,
                params=dict(persona.sampling) if persona is not None else {},
            )
            await self._stream_assistant(channel, backend, request, state, keep_calls=may_call)
            if not state.pending_calls:
                return

    async def _raw(self, channel: TurnChannel, state: _TurnState) -> None:
        """Raw mode: store the call the message stands for, run it, and end; no LLM reply.
        (A raw retry comes with its call pending and its caller already stored.)"""
        if state.raw_call is not None:
            caller = await self._store_raw_caller(
                channel.conversation_id, state.leaf_id, state.raw_call
            )
            state.assistant_id = state.leaf_id = caller.id
            state.message_ids.append(caller.id)
            state.pending_calls = [state.raw_call]
            await channel.publish(EventType.MESSAGE, MessageEventData(message=caller))
        for call in list(state.pending_calls):
            await self._call_tool(channel, state, call)

    async def _stream_assistant(
        self,
        channel: TurnChannel,
        backend: LLMBackend,
        request: ChatRequest,
        state: _TurnState,
        *,
        keep_calls: bool,
    ) -> None:
        assistant = await self._store_assistant_placeholder(
            channel.conversation_id,
            backend,
            state.leaf_id,
            state.persona.agent_id if state.persona is not None else None,
        )
        state.assistant_id = state.streaming_id = state.leaf_id = assistant.id
        state.assembler = MessageAssembler()
        state.message_ids.append(assistant.id)
        await channel.publish(EventType.MESSAGE, MessageEventData(message=assistant))
        async for delta in backend.chat_stream(request):
            state.assembler.add(delta)
            if delta.content or delta.reasoning:
                await channel.publish(
                    EventType.DELTA,
                    DeltaEventData(
                        message_id=assistant.id, content=delta.content, reasoning=delta.reasoning
                    ),
                )
        if state.assembler.finish_reason is None:
            # A finished answer says why it stopped; one that just ends was cut off, and
            # must not be stored as complete.
            raise EngineInterruptedError(
                f"{backend.name} stopped streaming before finishing the answer",
                backend=backend.name,
            )
        calls = state.assembler.tool_calls() if keep_calls else []
        if not keep_calls and state.assembler.tool_calls():
            log.warning("turn.tool_calls_ignored", reason="tool round limit reached")
        await self._finalize(channel, assistant.id, state.assembler, MessageStatus.COMPLETE, calls)
        state.streaming_id = None
        state.pending_calls = calls

    async def _call_tool(
        self, channel: TurnChannel, state: _TurnState, call: dict[str, Any]
    ) -> None:
        """Run the call and store its result. With the quality check on, an image result
        the check rejects is followed by a new attempt beside it, with the check's changes."""
        sampler: dict[str, Any] = {}
        earlier: list[dict[str, Any]] = []  # what the check found in the rejected attempts
        while True:
            outcome = await self._execute(channel, state, call, sampler)
            result = await self._store_tool_result(channel, state, call, outcome)
            name = call["function"]["name"]
            if not state.qa or name not in IMAGE_TOOLS or not outcome.artifact_ids:
                return
            if not outcome.ok:  # partial images from a failed job are not worth a check
                return
            if state.reviewer is None:
                review = Review(status="skipped", detail=state.qa_skip)
            else:
                await self._publish_qa(channel, result, {"status": "checking"})
                subject = _subject(
                    channel.conversation_id, state, call, outcome, self._preset_pins(state, name)
                )
                review = await self._qa.review(state.reviewer, subject)
            redo = review.status == "redo" and len(earlier) < self._config.qa.max_retries
            await self._record_qa(channel, result, outcome, _qa_record(review, earlier, redo))
            if not redo:
                return
            earlier.append({"issues": review.issues, "changes": review.changes})
            call, sampler = _revised(call, review.changes, outcome, sampler)
            attempt = await self._store_tool_placeholder(
                channel.conversation_id, result.parent_id, call, redo_of=result.id
            )
            state.placeholders[call["id"]] = attempt.id
            state.pending_calls.append(call)  # answered on wind-down if the turn ends now
            state.message_ids.append(attempt.id)
            await channel.publish(EventType.MESSAGE, MessageEventData(message=attempt))
            log.info("qa.redo", tool_call_id=call["id"], attempt=len(earlier) + 1)

    def _preset_pins(self, state: _TurnState, tool: str) -> set[str]:
        """The settings the user's preset fixes, as the backend this tool used has it."""
        options = state.image_options
        if options is None or options.preset is None:
            return set()
        task: Literal["t2i", "edit"] = "edit" if tool == "edit_image" else "t2i"
        return preset_pins(self._registry.image_preset(task, options.backend, options.preset))

    async def _execute(
        self,
        channel: TurnChannel,
        state: _TurnState,
        call: dict[str, Any],
        sampler: dict[str, Any],
    ) -> ToolOutcome:
        assert state.assistant_id is not None
        name, arguments = call["function"]["name"], call["function"]["arguments"]
        await channel.publish(
            EventType.TOOL_CALL,
            ToolCallEventData(
                message_id=state.assistant_id,
                tool_call_id=call["id"],
                name=name,
                arguments=arguments,
            ),
        )
        log.info("tool.called", tool=name, tool_call_id=call["id"])
        ctx = ToolContext(
            conversation_id=channel.conversation_id,
            message_id=state.assistant_id,
            tool_call_id=call["id"],
            emit=_emitter(channel),
            image_options=state.image_options,
            agent_id=state.persona.agent_id if state.persona is not None else None,
            sampler=sampler,
        )
        outcome = await self._tools.execute(name, arguments, ctx)
        log.info("tool.finished", tool=name, ok=outcome.ok)
        return outcome

    async def _publish_qa(
        self, channel: TurnChannel, result: dto.Message, qa: dict[str, Any]
    ) -> None:
        """Show a check's state on a result without storing it (e.g. while it runs)."""
        message = result.model_copy(update={"metadata": {**(result.metadata or {}), "qa": qa}})
        await channel.publish(EventType.MESSAGE, MessageEventData(message=message))

    async def _record_qa(
        self, channel: TurnChannel, result: dto.Message, outcome: ToolOutcome, qa: dict[str, Any]
    ) -> None:
        """Store the check on the result: in its metadata, and in its content for the LLM."""
        async with self._db.session() as session, session.begin():
            row = await msg_repo.get(session, result.id)
            if row is None:  # conversation deleted mid-turn
                return
            row.content_text = json.dumps({**outcome.content, "qa": qa})
            row.metadata_json = {**(row.metadata_json or {}), "qa": qa}
            await session.flush()
            message = mappers.message(row, outcome.artifact_ids)
        await channel.publish(EventType.MESSAGE, MessageEventData(message=message))

    async def _store_tool_result(
        self,
        channel: TurnChannel,
        state: _TurnState,
        call: dict[str, Any],
        outcome: ToolOutcome,
    ) -> dto.Message:
        content = json.dumps(outcome.content)
        metadata = {"tool": call["function"]["name"], "ok": outcome.ok}
        placeholder_id = state.placeholders.pop(call["id"], None)
        async with self._db.session() as session, session.begin():
            if placeholder_id is not None:
                row = await msg_repo.get(session, placeholder_id)
                if row is None:
                    raise NotFoundError(f"conversation {channel.conversation_id} not found")
                row.content_text = content
                row.metadata_json = {**(row.metadata_json or {}), **metadata}  # keeps redo_of
                row.status = MessageStatus.COMPLETE
                await session.flush()
            else:
                now = utcnow()
                seq = await conv_repo.allocate_seq(session, channel.conversation_id, now)
                row = orm.Message(
                    id=new_id(),
                    conversation_id=channel.conversation_id,
                    parent_id=state.leaf_id,
                    seq=seq,
                    role=Role.TOOL,
                    content_text=content,
                    tool_call_id=call["id"],
                    metadata_json=metadata,
                    created_at=now,
                )
                await msg_repo.insert(session, row)
                state.message_ids.append(row.id)
            await msg_repo.add_attachments(session, row.id, outcome.artifact_ids)
        state.pending_calls.remove(call)
        state.leaf_id = row.id
        message = mappers.message(row, outcome.artifact_ids)
        await channel.publish(EventType.MESSAGE, MessageEventData(message=message))
        return message

    # --- endings --------------------------------------------------------------

    async def _wind_down(
        self, channel: TurnChannel, state: _TurnState, status: MessageStatus
    ) -> None:
        """Leave a well-formed history: finish the streaming message, answer open calls."""
        if state.streaming_id is not None:
            await self._finalize(channel, state.streaming_id, state.assembler, status, [])
            state.streaming_id = None
        for call in list(state.pending_calls):
            outcome = ToolOutcome.error(
                status.value, f"the turn ended ({status.value}) before this tool finished"
            )
            await self._store_tool_result(channel, state, call, outcome)

    async def _done(
        self, channel: TurnChannel, state: _TurnState, finish_reason: str | None
    ) -> None:
        await channel.publish(
            EventType.DONE,
            DoneEventData(
                turn_id=channel.turn_id,
                finish_reason=finish_reason,
                message_ids=state.message_ids,
                usage=state.assembler.usage,
            ),
        )

    async def _fail(self, channel: TurnChannel, state: _TurnState, exc: Exception) -> None:
        if isinstance(exc, ImaginerError):
            status, code, title, detail = exc.status, exc.code, exc.title, exc.detail
            log.warning("turn.failed", code=code, detail=detail)
        else:
            status, code, title = 500, "internal_error", "Internal error"
            detail = "the turn failed unexpectedly"
            log.exception("turn.crashed")
        try:
            if state.streaming_id is not None:
                await self._finalize(
                    channel,
                    state.streaming_id,
                    state.assembler,
                    MessageStatus.ERROR,
                    [],
                    error={"code": code, "detail": detail},
                )
                state.streaming_id = None
            await self._wind_down(channel, state, MessageStatus.ERROR)
        except Exception:
            log.exception("turn.wind_down_failed")
        await channel.publish(
            EventType.ERROR,
            ErrorEventData(
                turn_id=channel.turn_id, status=status, code=code, title=title, detail=detail
            ),
        )

    # --- persistence ----------------------------------------------------------

    async def _conversation_agent(self, conversation_id: str) -> str | None:
        async with self._db.session() as session:
            row = await conv_repo.get_active(session, conversation_id)
        agent = (row.metadata_json or {}).get("agent") if row is not None else None
        return agent if isinstance(agent, str) else None

    async def _store_user_message(
        self, conversation_id: str, body: dto.MessageCreate, agent_id: str | None = None
    ) -> dto.Message:
        options = body.image_options.model_dump(exclude_none=True) if body.image_options else {}
        metadata: dict[str, Any] = {"image_options": options} if options else {}
        if body.mode is not PromptMode.CHAT:
            metadata["mode"] = body.mode.value
        if body.qa is not None:
            metadata["qa"] = body.qa
        async with self._db.session() as session, session.begin():
            # Resolved only now, with the turn open, so "the latest" can't move underneath.
            parent_id = await _parent_for(session, conversation_id, body)
            now = utcnow()
            seq = await conv_repo.allocate_seq(session, conversation_id, now)
            row = orm.Message(
                id=new_id(),
                conversation_id=conversation_id,
                parent_id=parent_id,
                seq=seq,
                role=Role.USER,
                content_text=body.content,
                metadata_json=metadata or None,
                created_at=now,
            )
            await msg_repo.insert(session, row)
            await msg_repo.add_attachments(session, row.id, body.attachments)
            await art_repo.claim_uploads(session, body.attachments, row.id)
            if agent_id is not None:
                conversation = await conv_repo.require_active(session, conversation_id)
                metadata = dict(conversation.metadata_json or {})
                if metadata.get("agent") != agent_id:
                    conversation.metadata_json = {**metadata, "agent": agent_id}
        return mappers.message(row, list(body.attachments))

    async def _store_assistant_placeholder(
        self,
        conversation_id: str,
        backend: LLMBackend,
        parent_id: str,
        agent_id: str | None = None,
    ) -> dto.Message:
        async with self._db.session() as session, session.begin():
            now = utcnow()
            seq = await conv_repo.allocate_seq(session, conversation_id, now)
            row = orm.Message(
                id=new_id(),
                conversation_id=conversation_id,
                parent_id=parent_id,
                seq=seq,
                role=Role.ASSISTANT,
                status=MessageStatus.STREAMING,
                backend=backend.name,
                model=backend.model,
                agent_id=agent_id,
                created_at=now,
            )
            await msg_repo.insert(session, row)
        return mappers.message(row)

    async def _store_raw_caller(
        self, conversation_id: str, parent_id: str, call: dict[str, Any]
    ) -> dto.Message:
        """Raw mode's stand-in for the assistant: a finished message that makes ``call``."""
        async with self._db.session() as session, session.begin():
            now = utcnow()
            seq = await conv_repo.allocate_seq(session, conversation_id, now)
            row = orm.Message(
                id=new_id(),
                conversation_id=conversation_id,
                parent_id=parent_id,
                seq=seq,
                role=Role.ASSISTANT,
                status=MessageStatus.COMPLETE,
                tool_calls_json=[call],
                metadata_json={"mode": PromptMode.RAW.value, "finish_reason": "tool_calls"},
                created_at=now,
            )
            await msg_repo.insert(session, row)
        return mappers.message(row)

    async def _store_tool_placeholder(
        self,
        conversation_id: str,
        parent_id: str | None,
        call: dict[str, Any],
        *,
        redo_of: str | None = None,
    ) -> dto.Message:
        """A new, unfinished result for ``call``, following ``parent_id``: a sibling of the
        earlier attempt. ``redo_of``: the attempt the quality check rejected, if it did."""
        metadata = {"tool": call["function"]["name"]}
        if redo_of is not None:
            metadata["redo_of"] = redo_of
        async with self._db.session() as session, session.begin():
            now = utcnow()
            seq = await conv_repo.allocate_seq(session, conversation_id, now)
            row = orm.Message(
                id=new_id(),
                conversation_id=conversation_id,
                parent_id=parent_id,
                seq=seq,
                role=Role.TOOL,
                status=MessageStatus.STREAMING,
                tool_call_id=call["id"],
                metadata_json=metadata,
                created_at=now,
            )
            await msg_repo.insert(session, row)
        return mappers.message(row)

    async def _finalize(
        self,
        channel: TurnChannel,
        message_id: str,
        assembler: MessageAssembler,
        status: MessageStatus,
        tool_calls: list[dict[str, Any]],
        error: dict[str, Any] | None = None,
    ) -> None:
        metadata: dict[str, Any] = {"finish_reason": assembler.finish_reason}
        if assembler.usage:
            metadata["usage"] = assembler.usage
        if error:
            metadata["error"] = error
        async with self._db.session() as session, session.begin():
            row = await msg_repo.get(session, message_id)
            if row is None:  # conversation deleted mid-turn
                return
            row.content_text = assembler.text or None
            row.reasoning_text = assembler.reasoning
            row.tool_calls_json = tool_calls or None
            row.status = status
            row.metadata_json = metadata
            await session.flush()
            message = mappers.message(row)
        await channel.publish(EventType.MESSAGE, MessageEventData(message=message))


async def _parent_for(
    session: AsyncSession, conversation_id: str, body: dto.MessageCreate
) -> str | None:
    """Where a new user message goes: after ``parent_id``, beside ``edit_of`` (sharing its
    parent), or by default after the newest message, continuing the latest branch."""
    anchor_id = body.parent_id or body.edit_of
    if anchor_id is None:
        latest = await msg_repo.latest(session, conversation_id)
        return latest.id if latest is not None else None
    anchor = await msg_repo.get_in_conversation(session, conversation_id, anchor_id)
    field_name = "parent_id" if body.parent_id is not None else "edit_of"
    if anchor is None:
        raise InvalidRequestError(f"{field_name} {anchor_id} is not a message of this conversation")
    if body.edit_of is None:
        return anchor.id
    if anchor.role is not Role.USER:
        raise InvalidRequestError("edit_of must be one of your own messages")
    return anchor.parent_id


def _allowed(tool: Tool, state: _TurnState) -> bool:
    """Whether the turn offers a tool: the agent has it on (a tool for agents needs one),
    and it is one of the turn's only tools, if it has those."""
    if state.only_tools is not None and tool.name not in state.only_tools:
        return False
    if state.persona is None:
        return not tool.needs_agent
    return tool.name not in state.persona.tools_off


def _reseeded(call: dict[str, Any]) -> dict[str, Any]:
    """The same call without its seed, so the tool draws a new one."""
    try:
        arguments = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return call
    if isinstance(arguments, dict):
        arguments.pop("seed", None)
    return {**call, "function": {**call["function"], "arguments": json.dumps(arguments)}}


def _options_without_seed(asker: orm.Message | None) -> dto.ImageRequestOptions | None:
    """The image options of the user message that led to a call, minus a pinned seed."""
    stored = (asker.metadata_json or {}).get("image_options") if asker is not None else None
    if not stored:
        return None
    options = dto.ImageRequestOptions.model_validate({**stored, "seed": None})
    return options if options.model_dump(exclude_none=True) else None


def _subject(
    conversation_id: str,
    state: _TurnState,
    call: dict[str, Any],
    outcome: ToolOutcome,
    fixed_by_preset: Iterable[str] = (),
) -> Subject:
    options = state.image_options
    pinned = {k for k in PINNABLE if options is not None and getattr(options, k) is not None}
    pinned.update(fixed_by_preset)
    return Subject(
        conversation_id=conversation_id,
        request=state.request,
        tool=call["function"]["name"],
        job_id=outcome.content["job_id"],
        artifact_ids=outcome.artifact_ids,
        raw=state.mode is PromptMode.RAW,
        pinned=frozenset(pinned),
    )


def _qa_record(review: Review, earlier: list[dict[str, Any]], redo: bool) -> dict[str, Any]:
    """The check as stored on a result. ``redone``: a new attempt follows beside this one;
    ``flagged``: kept with its issues (no attempts left, or nothing the check may change)."""
    status = {"redo": "redone" if redo else "flagged"}.get(review.status, review.status)
    record: dict[str, Any] = {"status": status, "attempt": len(earlier) + 1}
    if review.issues:
        record["issues"] = review.issues
    if redo:
        record["changes"] = review.changes
    if review.detail:
        record["detail"] = review.detail
    if earlier:
        record["earlier"] = earlier
    return record


def _revised(
    call: dict[str, Any],
    changes: dict[str, Any],
    outcome: ToolOutcome,
    sampler: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The next attempt's call and sampler settings, with the check's changes. Without a
    new seed it keeps the seed it had, so a change of steps or CFG refines the same image."""
    arguments = json.loads(call["function"]["arguments"] or "{}")  # valid: the tool ran
    for key in ("prompt", "negative_prompt"):
        if key in changes:
            arguments[key] = changes[key]
    seed = (outcome.content.get("images") or [{}])[0].get("seed")
    if changes.get("new_seed") or seed is None:
        arguments.pop("seed", None)
    else:
        arguments["seed"] = seed
    sampler = {**sampler, **{k: changes[k] for k in ("steps", "cfg_scale") if k in changes}}
    revised = {**call, "function": {**call["function"], "arguments": json.dumps(arguments)}}
    return revised, sampler


def _emitter(channel: TurnChannel) -> Emit:
    async def emit(event_type: EventType, data: BaseModel) -> None:
        if not channel.closed:
            await channel.publish(event_type, data)

    return emit
