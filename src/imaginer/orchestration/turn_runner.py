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
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import structlog
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

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
from imaginer.domain import models as dto
from imaginer.domain.enums import ArtifactKind, Capability, MessageStatus, Role
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
from imaginer.services import mappers
from imaginer.tools.base import ToolContext, ToolOutcome, ToolRegistry

log = get_logger(__name__)


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


class TurnRunner:
    def __init__(
        self,
        *,
        db: Database,
        registry: BackendRegistry,
        turns: TurnManager,
        context: ContextBuilder,
        tools: ToolRegistry,
        config: OrchestrationConfig,
    ) -> None:
        self._db = db
        self._registry = registry
        self._turns = turns
        self._context = context
        self._tools = tools
        self._config = config

    async def start(self, conversation_id: str, body: dto.MessageCreate) -> TurnChannel:
        if not body.content.strip() and not body.attachments:
            raise InvalidRequestError("a message needs content or attachments")
        if len(set(body.attachments)) != len(body.attachments):
            raise InvalidRequestError("attachments contain duplicates")

        async with self._db.session() as session, session.begin():
            await conv_repo.require_active(session, conversation_id)
            found = await art_repo.get_many_in_conversation(
                session, conversation_id, body.attachments
            )
        missing = sorted(set(body.attachments) - {a.id for a in found})
        if missing:
            raise InvalidRequestError(
                "attachments must be artifacts of this conversation", missing=missing
            )

        required = {Capability.CHAT, Capability.STREAM}
        if any(a.kind is ArtifactKind.IMAGE for a in found):
            # Never drop the user's image silently: a backend that can't see it is an error.
            required.add(Capability.VISION_IN)
        lease = await self._registry.chat_backend(body.backend, require=required)
        try:
            if body.image_options is not None and body.image_options.backend is not None:
                self._registry.validate_image_backend(body.image_options.backend)
            if body.image_options is not None and body.image_options.mask is not None:
                await self._check_mask(conversation_id, body.image_options.mask)

            channel = self._turns.open(conversation_id)
            try:
                user_message = await self._store_user_message(conversation_id, body)
            except BaseException:
                self._turns.release(channel)
                raise
        except BaseException:
            lease.release()
            raise
        await channel.publish(EventType.MESSAGE, MessageEventData(message=user_message))
        state = _TurnState(
            message_ids=[user_message.id],
            leaf_id=user_message.id,
            image_options=body.image_options,
        )
        channel.task = self._start(channel, lease, state)
        return channel

    async def _check_mask(self, conversation_id: str, label: str) -> None:
        async with self._db.session() as session:
            mask = await art_repo.get_by_label(session, conversation_id, label)
        if mask is None or mask.kind is not ArtifactKind.MASK:
            raise InvalidRequestError(f"image_options.mask: no mask labelled {label!r} here")

    async def retry(self, conversation_id: str, message_id: str) -> TurnChannel:
        """Run the tool call behind the result ``message_id`` again, as a new attempt.

        Same arguments and image options as the original call, but a new seed. The attempt
        is stored right away (``streaming``, beside the old result) and filled in when the
        tool finishes; then the assistant replies on that new branch.
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
        options = _options_without_seed(asker)
        lease = await self._registry.chat_backend(
            None, require={Capability.CHAT, Capability.STREAM}
        )
        try:
            if options is not None and options.backend is not None:
                self._registry.validate_image_backend(options.backend)

            channel = self._turns.open(conversation_id)
            try:
                attempt = await self._store_tool_placeholder(conversation_id, result, call)
            except BaseException:
                self._turns.release(channel)
                raise
        except BaseException:
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
        )
        channel.task = self._start(channel, lease, state)
        return channel

    # --- the turn -------------------------------------------------------------

    def _start(
        self, channel: TurnChannel, lease: Lease[LLMBackend], state: _TurnState
    ) -> asyncio.Task[None]:
        """Run the turn. It holds the chat backend's lease until it ends, however it ends
        (even cancelled before it starts), so a reconfigure can't close it mid-turn."""
        task = asyncio.create_task(
            self._run(channel, lease.backend, state), name=f"turn-{channel.turn_id}"
        )
        task.add_done_callback(lambda _: lease.release())
        return task

    async def _run(self, channel: TurnChannel, backend: LLMBackend, state: _TurnState) -> None:
        with structlog.contextvars.bound_contextvars(
            conversation_id=channel.conversation_id, turn_id=channel.turn_id
        ):
            log.info("turn.started", backend=backend.name, model=backend.model)
            try:
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
                await channel.close()
                self._turns.release(channel)

    async def _loop(self, channel: TurnChannel, backend: LLMBackend, state: _TurnState) -> None:
        capabilities = backend.capabilities()
        tools = self._tools.offered() if Capability.TOOLS in capabilities else []
        schemas = [tool.schema() for tool in tools]
        guidance = [tool.guidance for tool in tools]
        vision = Capability.VISION_IN in capabilities
        rounds = self._config.max_tool_iterations if tools else 0
        for round_index in range(rounds + 1):
            # Calls from the previous round, or a retry's call, which comes ready-made.
            for call in list(state.pending_calls):
                await self._call_tool(channel, state, call)
            may_call = round_index < rounds
            context = await self._context.build(
                channel.conversation_id,
                leaf_id=state.leaf_id,
                vision=vision,
                tool_guidance=guidance,
            )
            request = ChatRequest(
                messages=context,
                tools=schemas or None,
                tool_choice=None if may_call or not schemas else "none",
            )
            await self._stream_assistant(channel, backend, request, state, keep_calls=may_call)
            if not state.pending_calls:
                return

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
            channel.conversation_id, backend, state.leaf_id
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
        )
        outcome = await self._tools.execute(name, arguments, ctx)
        log.info("tool.finished", tool=name, ok=outcome.ok)
        await self._store_tool_result(channel, state, call, outcome)

    async def _store_tool_result(
        self,
        channel: TurnChannel,
        state: _TurnState,
        call: dict[str, Any],
        outcome: ToolOutcome,
    ) -> None:
        content = json.dumps(outcome.content)
        metadata = {"tool": call["function"]["name"], "ok": outcome.ok}
        placeholder_id = state.placeholders.pop(call["id"], None)
        async with self._db.session() as session, session.begin():
            if placeholder_id is not None:
                row = await msg_repo.get(session, placeholder_id)
                if row is None:
                    raise NotFoundError(f"conversation {channel.conversation_id} not found")
                row.content_text = content
                row.metadata_json = metadata
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

    async def _store_user_message(
        self, conversation_id: str, body: dto.MessageCreate
    ) -> dto.Message:
        options = body.image_options.model_dump(exclude_none=True) if body.image_options else {}
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
                metadata_json={"image_options": options} if options else None,
                created_at=now,
            )
            await msg_repo.insert(session, row)
            await msg_repo.add_attachments(session, row.id, body.attachments)
            await art_repo.claim_uploads(session, body.attachments, row.id)
        return mappers.message(row, list(body.attachments))

    async def _store_assistant_placeholder(
        self, conversation_id: str, backend: LLMBackend, parent_id: str
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
                created_at=now,
            )
            await msg_repo.insert(session, row)
        return mappers.message(row)

    async def _store_tool_placeholder(
        self, conversation_id: str, attempt: orm.Message, call: dict[str, Any]
    ) -> dto.Message:
        """A new, unfinished result for ``call``: a sibling of the earlier ``attempt``."""
        async with self._db.session() as session, session.begin():
            now = utcnow()
            seq = await conv_repo.allocate_seq(session, conversation_id, now)
            row = orm.Message(
                id=new_id(),
                conversation_id=conversation_id,
                parent_id=attempt.parent_id,
                seq=seq,
                role=Role.TOOL,
                status=MessageStatus.STREAMING,
                tool_call_id=call["id"],
                metadata_json={"tool": call["function"]["name"]},
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


def _emitter(channel: TurnChannel) -> Emit:
    async def emit(event_type: EventType, data: BaseModel) -> None:
        if not channel.closed:
            await channel.publish(event_type, data)

    return emit
