"""Pydantic DTOs shared by services and the HTTP API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from imaginer.domain.enums import (
    ArtifactKind,
    ArtifactSource,
    BackendKind,
    BackendStatus,
    Capability,
    JobStatus,
    JobTask,
    LineageRelation,
    MessageStatus,
    Role,
)

SEED_MAX = 2**31 - 1  # portable across engines (sd.cpp int64, torch generators)
AspectRatio = Literal["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Output(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None = None


# --- conversations ------------------------------------------------------------


class ConversationCreate(_Input):
    title: str | None = Field(default=None, max_length=500)
    metadata: dict[str, Any] | None = None


class ConversationUpdate(_Input):
    """Partial update. Only fields present in the request body are applied."""

    title: str | None = Field(default=None, max_length=500)
    metadata: dict[str, Any] | None = None
    archived: bool | None = None


class Conversation(_Output):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None
    metadata: dict[str, Any] | None = None


# --- messages -----------------------------------------------------------------


class Message(_Output):
    id: str
    conversation_id: str
    parent_id: str | None = Field(
        default=None,
        description=(
            "The message this one follows; null for a first message. Messages with the same "
            "parent are alternatives, each on its own branch."
        ),
    )
    seq: int = Field(description="Creation order in the conversation, across all branches.")
    role: Role
    status: MessageStatus
    content_text: str | None
    reasoning_text: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    backend: str | None = None
    model: str | None = None
    attachments: list[str] = Field(
        default_factory=list, description="Artifact ids attached to this message, in order."
    )
    metadata: dict[str, Any] | None = None
    created_at: datetime


class MessagePage(BaseModel):
    items: list[Message]
    has_more: bool


class ImageRequestOptions(_Input):
    """Image settings the user picked for one message. Each one that is set overrides what
    the model asks for in every image tool call of that turn."""

    aspect_ratio: AspectRatio | None = None
    width: int | None = Field(
        default=None, ge=256, le=4096, description="Exact size; give height too. Wins over aspect."
    )
    height: int | None = Field(default=None, ge=256, le=4096)
    n: int | None = Field(default=None, ge=1, le=4, description="How many images per call.")
    seed: int | None = Field(
        default=None, ge=0, le=SEED_MAX, description="First seed; further images use seed+1, ..."
    )
    steps: int | None = Field(
        default=None, ge=1, le=150, description="Sampling steps instead of the backend default."
    )
    cfg_scale: float | None = Field(
        default=None, ge=0, le=30, description="Guidance scale instead of the backend default."
    )
    negative_prompt: str | None = Field(
        default=None, max_length=2000, description="Sent with every image call of the turn."
    )
    transparent: bool | None = Field(
        default=None, description="Transparent (RGBA) output on or off, whatever the model asks."
    )
    mask: str | None = Field(
        default=None,
        pattern=r"^mask-\d+$",
        description="Label of a mask artifact (white = change) for every edit_image call.",
    )
    backend: str | None = Field(
        default=None, description="Image backend to use instead of the configured default."
    )

    @model_validator(mode="after")
    def _both_sides(self) -> "ImageRequestOptions":
        if (self.width is None) != (self.height is None):
            raise ValueError("give width and height together")
        return self


class MessageCreate(_Input):
    """A user message. The assistant's reply streams back as server-sent events."""

    content: str = Field(default="", max_length=100_000)
    attachments: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Ids of artifacts in this conversation (e.g. fresh uploads), in order.",
    )
    backend: str | None = Field(
        default=None, description="Chat backend to use instead of the configured default."
    )
    image_options: ImageRequestOptions | None = Field(
        default=None,
        description="Stored on the user message as metadata.image_options when any is set.",
    )
    parent_id: str | None = Field(
        default=None,
        description=(
            "Message to continue from, usually a branch's leaf_id. Omit to continue the latest "
            "branch. Continuing from a message that already has replies starts a new branch."
        ),
    )
    edit_of: str | None = Field(
        default=None,
        description=(
            "Edit & branch: the id of one of your earlier messages. The new message becomes an "
            "alternative to it on a new branch; the original branch stays as it is."
        ),
    )

    @model_validator(mode="after")
    def _one_anchor(self) -> "MessageCreate":
        if self.parent_id is not None and self.edit_of is not None:
            raise ValueError("give parent_id or edit_of, not both")
        return self


class TurnInfo(BaseModel):
    turn_id: str
    conversation_id: str
    started_at: datetime
    running: bool
    last_event_id: int


# --- artifacts ----------------------------------------------------------------


class Artifact(_Output):
    id: str
    conversation_id: str
    message_id: str | None
    kind: ArtifactKind
    source: ArtifactSource
    label: str = Field(description="Stable per-conversation handle, e.g. 'img-3'.")
    mime: str
    sha256: str
    size_bytes: int
    width: int | None = None
    height: int | None = None
    has_alpha: bool | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_url(self) -> str:
        return f"/v1/artifacts/{self.id}/content"


class ConversationSummary(Conversation):
    """A conversation as listed: with what a gallery needs to show it."""

    image_count: int = 0
    cover: Artifact | None = Field(default=None, description="Its most recent image.")
    turn_running: bool = Field(
        default=False, description="A turn is running: attach with GET .../turn/events."
    )


SearchField = Literal["title", "prompt", "reply", "label"]
SEARCH_FIELDS: tuple[SearchField, ...] = ("title", "prompt", "label", "reply")


class SearchMatch(BaseModel):
    field: SearchField = Field(
        description="Where it matched: the title, a prompt, an assistant reply, or a label."
    )
    snippet: str | None = Field(default=None, description="The matching text, shortened.")
    label: str | None = Field(default=None, description="The image whose prompt or label matched.")


class SearchHit(ConversationSummary):
    match: SearchMatch


class SearchPage(BaseModel):
    """One page of conversations searched, in gallery order. Keep asking with next_cursor
    until it is null; ``scanned`` and ``total`` say how far the search has got."""

    items: list[SearchHit]
    scanned: int = Field(description="Conversations searched on this page.")
    total: int = Field(description="Conversations there are to search.")
    next_cursor: str | None = None


class Branch(BaseModel):
    """A path from a first message to a leaf. Continue it with ``parent_id = leaf_id``."""

    leaf_id: str = Field(description="Its newest message. GET .../messages?leaf_id= shows it.")
    name: str = Field(description="'main', 'message 1, edited' or 'retry of img-3'.")
    kind: Literal["main", "edit", "retry"]
    fork_id: str | None = Field(
        description="Its first message of its own, where it leaves its parent branch."
    )
    parent_leaf_id: str | None = Field(description="Leaf of the branch it forked from.")
    origin: str | None = Field(
        description="Where it forked: 'from your first message', 'from assistant reply 1'."
    )
    depth: int = Field(description="0 for main, 1 for a branch of main, and so on.")
    message_count: int = Field(
        description="Your messages plus assistant replies; tool results belong to their reply."
    )
    latest_image: Artifact | None = Field(description="Its newest image.")
    latest: bool = Field(
        description="The latest branch, used when a request gives no leaf_id or parent_id."
    )
    updated_at: datetime


class LineageEdge(BaseModel):
    artifact_id: str
    label: str
    relation: LineageRelation
    ordinal: int


class Lineage(BaseModel):
    artifact_id: str
    parents: list[LineageEdge]
    children: list[LineageEdge]


# --- generation jobs ----------------------------------------------------------


class JobOutputRef(BaseModel):
    artifact_id: str
    ordinal: int
    seed: int | None


class Job(BaseModel):
    """A generation job. ``params`` holds what was requested and, once outputs exist,
    the resolved engine settings, so any output can be regenerated exactly."""

    id: str
    conversation_id: str
    request_message_id: str | None
    tool_call_id: str | None
    task: JobTask
    alpha: bool
    backend: str
    model: str
    params: dict[str, Any]
    seed: int | None
    status: JobStatus
    error: str | None
    engine_job_id: str | None
    progress: dict[str, Any] | None
    outputs: list[JobOutputRef] = Field(default_factory=list)
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


# --- backends / health --------------------------------------------------------


class ImageDefaultsInfo(BaseModel):
    """What an image backend uses when a request leaves a setting open."""

    width: int
    height: int
    steps: int | None
    cfg_scale: float | None
    sampler: str | None
    size_multiple: int


class BackendInfo(BaseModel):
    name: str
    type: str
    kind: BackendKind | None
    enabled: bool
    status: BackendStatus
    capabilities: list[Capability]
    model: str | None = None
    default_for: list[Literal["chat", "t2i", "edit"]] = Field(
        default_factory=list, description="Tasks routed here when a request names no backend."
    )
    detail: str | None = None
    latency_ms: float | None = None
    probed_at: datetime | None = None
    image_defaults: ImageDefaultsInfo | None = Field(
        default=None, description="Image backends: the settings used when a request omits them."
    )


class Health(BaseModel):
    status: str
    version: str


# --- engines (the Engines screen) ---------------------------------------------


class EngineFile(BaseModel):
    """A model file an engine loads, checked on disk."""

    key: str = Field(description="Its launch setting, e.g. 'vae'.")
    flag: str = Field(description="Its command-line flag, e.g. '--vae'.")
    what: str
    path: str | None
    exists: bool = False
    size_bytes: int | None = None
    format: Literal["gguf", "safetensors", "unknown"] | None = None
    problem: str | None = Field(default=None, description="Why it can't be used, if it can't.")


class EngineLogLine(BaseModel):
    at: datetime
    text: str
    error: bool


class EngineProcess(BaseModel):
    """The engine process this server started, if any."""

    state: Literal["stopped", "starting", "running", "stopping", "exited"]
    pid: int | None = None
    exit_code: int | None = None
    started_at: datetime | None = None
    exited_at: datetime | None = None
    summary: str | None = Field(default=None, description="Its last error line, if any.")
    command: list[str] | None = None
    log_path: str | None = None


class EngineRuntime(BaseModel):
    """The managed runtime under <data_root>/engines, from its active-runtime.json."""

    installed: bool
    version: str | None = None
    method: str | None = None
    accelerator: str | None = None
    executable: str | None = None
    installed_at: str | None = None
    directory: str


class EngineInstall(BaseModel):
    state: Literal["running", "succeeded", "failed", "cancelled"]
    method: str
    accelerator: str
    version: str
    step: int
    total: int
    label: str
    fraction: float = Field(description="0..1 within the step, or -1 when unknown.")
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    warning: str | None = None
    executable: str | None = None
    log_path: str | None = None
    output: list[str] = Field(default_factory=list, description="The newest output lines.")


class EngineInfo(BaseModel):
    name: str
    type: str
    kind: BackendKind | None
    enabled: bool
    status: BackendStatus
    detail: str | None = None
    model: str | None = None
    capabilities: list[Capability]
    declared_capabilities: list[Capability] | None = None
    default_for: list[Literal["chat", "t2i", "edit"]]
    base_url: str | None = None
    host: str | None = None
    port: int | None = None
    api_key_env: str | None = None
    loopback: bool = False
    launchable: bool = Field(description="It has launch settings, so it can be started here.")
    files: list[EngineFile] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)
    autostart: bool = False
    executable: str | None = None
    process: EngineProcess | None = None
    runtime: EngineRuntime | None = None
    install: EngineInstall | None = None
    settings_file: str | None = Field(
        default=None, description="Where changes made here are saved."
    )


class EngineLaunchUpdate(_Input):
    files: dict[str, str | None] = Field(
        default_factory=dict, description="Launch setting -> path (null clears it)."
    )
    extra_args: list[str] | None = None
    autostart: bool | None = None
    executable: str | None = None


class EngineUpdate(_Input):
    """Changes from the Engines screen. Saved to the profile's .local.yaml sidecar."""

    host: str | None = Field(default=None, min_length=1, max_length=253)
    port: int | None = Field(default=None, ge=1, le=65535)
    declared_capabilities: list[Capability] | None = None
    launch: EngineLaunchUpdate | None = None


class EngineInstallRequest(_Input):
    method: Literal["auto", "release", "source"] = "auto"
    accelerator: str = Field(default="", max_length=16, description="Empty: detect.")
    version: str = Field(default="latest", max_length=64)
    force: bool = False


# --- linked instances -----------------------------------------------------------------

GalleryMode = Literal["off", "read", "full"]
Task = Literal["chat", "t2i", "edit"]


class EngineAvailability(BaseModel):
    """Whether a shared engine can run its whole model on the instance that shares it."""

    state: Literal["ready", "cold", "unfit", "unavailable"] = Field(
        description="ready: loaded. cold: stopped, but it fits and may be started on first "
        "use. unfit: the model would not fit in free VRAM. unavailable: anything else."
    )
    detail: str | None = None
    remote_start: bool = False
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    need_mb: int | None = Field(default=None, description="Estimated VRAM the model needs.")


class SharedEngine(BaseModel):
    name: str
    type: str
    kind: BackendKind
    model: str | None = None
    status: BackendStatus
    detail: str | None = None
    capabilities: list[Capability]
    max_ref_images: int | None = None
    image_defaults: ImageDefaultsInfo | None = None
    queue_depth: int = Field(default=0, description="Image requests waiting for or using it.")
    availability: EngineAvailability


class PeerStatus(BaseModel):
    """What one instance tells a linked one about itself (GET /v1/link/status)."""

    instance_id: str
    name: str
    api_version: int
    version: str
    gallery: GalleryMode
    engines: list[SharedEngine]


class LinkHello(BaseModel):
    """GET /v1/link/hello: who answers, and a challenge to pair against."""

    instance_id: str
    name: str
    api_version: int
    version: str
    challenge: str
    proof: str | None = Field(
        default=None,
        description="For a linked asker: HMAC of its nonce, keyed with the hash of its token.",
    )


class PairRequest(_Input):
    challenge: str = Field(max_length=128)
    proof: str = Field(
        min_length=64,
        max_length=64,
        description="hex HMAC-SHA256(sha256(code), challenge|requester id|responder id)",
    )
    instance_id: str = Field(max_length=36)
    name: str = Field(min_length=1, max_length=64)
    url: str = Field(max_length=255, pattern=r"^https?://")
    api_version: int
    token: str = Field(
        min_length=32, max_length=128, description="The token to call the requester with."
    )


class PairAccepted(BaseModel):
    instance_id: str
    name: str
    api_version: int
    token: str = Field(description="The token to call the responder with.")


class LinkInfo(BaseModel):
    id: str
    name: str = Field(description="This instance's name for the link; linked backends use it.")
    peer_instance_id: str
    peer_name: str
    url: str
    created_at: datetime
    status: Literal["unknown", "ok", "unreachable", "incompatible", "unauthorized"]
    detail: str | None = None
    last_seen_at: datetime | None = None
    gallery: GalleryMode | None = Field(default=None, description="The peer's gallery mode.")
    engines: list[SharedEngine] = Field(default_factory=list)
    used_by: list[str] = Field(default_factory=list, description="Local backends using it.")


class LinkShareInfo(BaseModel):
    remote_start: bool


class LinksOverview(BaseModel):
    enabled: bool
    instance_id: str
    name: str
    advertise_url: str | None
    discovery: bool
    gallery: GalleryMode
    share: dict[str, LinkShareInfo]
    links: list[LinkInfo]
    settings_file: str | None = None


class DiscoveredInstance(BaseModel):
    instance_id: str
    name: str
    url: str
    api_version: int | None = None
    link_id: str | None = Field(default=None, description="Set when already linked.")


class PairingCode(BaseModel):
    code: str
    expires_at: datetime


class LinkCreate(_Input):
    url: str = Field(max_length=255, pattern=r"^https?://")
    code: str = Field(min_length=4, max_length=32)
    name: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,31}$")


class LinkUse(_Input):
    """Route tasks to an engine the linked instance shares, through a local backend."""

    engine: str = Field(max_length=32)
    tasks: list[Task] = Field(default_factory=list, description="Make it the default for these.")
    backend: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,31}$",
        description="Local backend name. Default: <link>_<engine>.",
    )


class LinkShareUpdate(_Input):
    remote_start: bool = False


class LinksSettingsUpdate(_Input):
    share: dict[str, LinkShareUpdate] | None = Field(
        default=None, description="The complete set of shared engines."
    )
    gallery: GalleryMode | None = None


class Origin(BaseModel):
    """Where a conversation lives. Remote ones are reached through ``base_path``."""

    link_id: str | None = None
    instance_id: str
    name: str
    base_path: str = Field(
        default="", description="Prefix for this conversation's API paths and content_url."
    )


class GalleryItem(ConversationSummary):
    origin: Origin


class GallerySource(BaseModel):
    id: str = Field(description="'local', or a link id.")
    name: str
    local: bool
    status: Literal["ok", "unreachable", "error"]
    detail: str | None = None


class GalleryPage(BaseModel):
    items: list[GalleryItem]
    next_cursor: str | None = None
    sources: list[GallerySource]


class PeerChatRequest(_Input):
    """One chat completion on a shared engine (a linked instance's turn)."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class PeerImage(_Input):
    data_b64: str
    mime: str = Field(max_length=127)
    label: str | None = Field(default=None, max_length=32)
    width: int | None = None
    height: int | None = None


class PeerImageRequest(_Input):
    """One image on a shared engine: the fields of a backend GenRequest / EditRequest."""

    op: Literal["generate", "edit"]
    prompt: str
    seed: int = Field(ge=0, le=SEED_MAX)
    negative_prompt: str | None = None
    width: int | None = Field(default=None, ge=64, le=8192)
    height: int | None = Field(default=None, ge=64, le=8192)
    aspect_ratio: AspectRatio | None = None
    alpha: bool = False
    params: dict[str, Any] = Field(default_factory=dict)
    images: list[PeerImage] = Field(default_factory=list, max_length=32)
    mask: PeerImage | None = None
