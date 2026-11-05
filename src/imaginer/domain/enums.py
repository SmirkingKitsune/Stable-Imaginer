from enum import StrEnum


class Capability(StrEnum):
    """What a backend can do. Routing always goes by capability, never by backend type."""

    T2I = "t2i"
    EDIT = "edit"
    MULTI_REF = "multi_ref"
    RGBA = "rgba"
    CHAT = "chat"
    VISION_IN = "vision_in"
    TOOLS = "tools"
    STREAM = "stream"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PromptMode(StrEnum):
    """How a user message reaches the image model."""

    CHAT = "chat"  # the LLM chats, and draws when asked, writing its own prompt
    RAW = "raw"  # no LLM: the message is the image prompt, word for word
    INTERPRET = "interpret"  # the LLM must turn the message into a faithful image prompt
    CURATE = "curate"  # the LLM enriches the prompt, or asks first when it's too unclear


class MessageStatus(StrEnum):
    STREAMING = "streaming"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


class ArtifactKind(StrEnum):
    IMAGE = "image"
    MASK = "mask"  # a black-and-white image painted for an edit: white = change
    AUDIO = "audio"  # reserved
    VIDEO = "video"  # reserved
    FILE = "file"
    TEXT = "text"


class ArtifactSource(StrEnum):
    UPLOAD = "upload"
    GENERATED = "generated"
    EDITED = "edited"
    DERIVED = "derived"


class LineageRelation(StrEnum):
    EDIT_SOURCE = "edit_source"
    REFERENCE = "reference"
    MASK = "mask"


class JobTask(StrEnum):
    T2I = "t2i"
    EDIT = "edit"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackendKind(StrEnum):
    LLM = "llm"
    IMAGE = "image"


class BackendStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"  # reachable but not ready (e.g. model still loading)
    DOWN = "down"  # configured and built, but the engine is unreachable
    UNAVAILABLE = "unavailable"  # disabled, or its adapter could not be built
    UNKNOWN = "unknown"  # not probed yet


LABEL_PREFIX: dict[ArtifactKind, str] = {
    ArtifactKind.IMAGE: "img",
    ArtifactKind.MASK: "mask",
    ArtifactKind.AUDIO: "aud",
    ArtifactKind.VIDEO: "vid",
    ArtifactKind.FILE: "file",
    ArtifactKind.TEXT: "txt",
}
