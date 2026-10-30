"""Server-sent events emitted during an assistant turn.

Every event has a per-turn, increasing ``id`` so a client can resume with ``Last-Event-ID``.

| event              | when                                                         |
|--------------------|--------------------------------------------------------------|
| ``message``        | a message was stored or finalised (user, assistant, tool)    |
| ``delta``          | streamed assistant text or reasoning                         |
| ``tool_call``      | the assistant called a tool                                  |
| ``job_update``     | a generation job changed state or made progress              |
| ``artifact_created`` | a job produced an artifact                                 |
| ``done``           | the turn ended; ``finish_reason`` is ``cancelled`` if stopped |
| ``error``          | the turn failed; the body is a problem-details object        |
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from imaginer.domain.models import Artifact, Job, Message


class EventType(StrEnum):
    MESSAGE = "message"
    DELTA = "delta"
    TOOL_CALL = "tool_call"
    JOB_UPDATE = "job_update"
    ARTIFACT_CREATED = "artifact_created"
    DONE = "done"
    ERROR = "error"


class MessageEventData(BaseModel):
    message: Message


class DeltaEventData(BaseModel):
    message_id: str
    content: str | None = None
    reasoning: str | None = None


class ToolCallEventData(BaseModel):
    message_id: str  # the assistant message that made the call
    tool_call_id: str
    name: str
    arguments: str  # raw JSON text, as the model produced it


class JobUpdateEventData(BaseModel):
    job: Job


class ArtifactCreatedEventData(BaseModel):
    artifact: Artifact
    job_id: str | None = None


class DoneEventData(BaseModel):
    turn_id: str
    finish_reason: str | None
    message_ids: list[str]
    usage: dict[str, Any] | None = None


class ErrorEventData(BaseModel):
    turn_id: str
    status: int
    code: str
    title: str
    detail: str


@dataclass(frozen=True, slots=True)
class TurnEvent:
    id: int
    type: EventType
    data: BaseModel
