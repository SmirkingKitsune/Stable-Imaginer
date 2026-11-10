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
    PromptMode,
    Role,
)

SEED_MAX = 2**31 - 1  # portable across engines (sd.cpp int64, torch generators)
AspectRatio = Literal["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]
PRESET_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,31}$"  # an image backend's preset name


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
    agent: str | None = Field(
        default=None, description="Assistant messages: the agent that wrote it (GET /v1/agents)."
    )
    attachments: list[str] = Field(
        default_factory=list, description="Artifact ids attached to this message, in order."
    )
    metadata: dict[str, Any] | None = None
    created_at: datetime


class MessagePage(BaseModel):
    items: list[Message]
    has_more: bool


MAX_LORA_PICKS = 8


class LoraPick(_Input):
    """A LoRA the user adds to one message's images."""

    path: str = Field(
        min_length=1,
        max_length=1024,
        description="As the image backend lists it (available_loras), or its file name or name.",
    )
    weight: float = Field(
        default=1.0, ge=-10, le=10, description="Its strength; 0 turns off a preset's LoRA."
    )


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
    preset: str | None = Field(
        default=None,
        pattern=PRESET_PATTERN,
        description="One of the image backend's presets (its LoRAs and sampler settings).",
    )
    loras: list[LoraPick] | None = Field(
        default=None,
        max_length=MAX_LORA_PICKS,
        description="LoRAs to apply on top of the preset's (or the backend's default ones). "
        "One the preset already applies gets this weight instead.",
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
    agent: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "The agent to answer (GET /v1/agents): its engine, persona, tools and sampling. "
            "Default: the conversation's (metadata.agent), else the one the default chat "
            "engine runs. Stored as the conversation's metadata.agent."
        ),
    )
    image_options: ImageRequestOptions | None = Field(
        default=None,
        description="Stored on the user message as metadata.image_options when any is set.",
    )
    mode: PromptMode = Field(
        default=PromptMode.CHAT,
        description=(
            "chat: the LLM replies, and draws when asked. raw: no LLM, the content is the "
            "image prompt as written (attached images make it an edit). interpret: the LLM "
            "must turn the content into a faithful image prompt. curate: the LLM enriches "
            "the prompt, or asks questions first when the request is too unclear. Stored as "
            "metadata.mode unless chat."
        ),
    )
    qa: bool | None = Field(
        default=None,
        description=(
            "Quality check: a vision LLM looks at every image and makes it again (more steps, "
            "another seed, a fixed prompt) when it has clear flaws. Default: "
            "orchestration.qa.enabled. Stored as metadata.qa when given."
        ),
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
    name: str = Field(
        description="'main', 'message 1, edited', 'retry of img-3' or 'QA redo of img-3'."
    )
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


class LoraInfo(BaseModel):
    path: str
    weight: float


class ImageDefaultsInfo(BaseModel):
    """What an image backend uses when a request leaves a setting open."""

    width: int
    height: int
    steps: int | None
    cfg_scale: float | None
    sampler: str | None
    size_multiple: int
    loras: list[LoraInfo] = Field(default_factory=list)


class ImagePresetInfo(BaseModel):
    """Settings a message can pick with ``image_options.preset``. Null fields keep the
    backend's defaults; ``loras`` null keeps the default LoRAs."""

    name: str
    description: str
    loras: list[LoraInfo] | None
    steps: int | None
    cfg_scale: float | None
    sampler: str | None


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
    image_presets: list[ImagePresetInfo] = Field(
        default_factory=list, description="Image backends: presets a message can pick."
    )
    available_loras: list[str] | None = Field(
        default=None,
        description="LoRAs the engine can apply, as paths for presets; null if it doesn't say.",
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
    format: Literal["gguf", "safetensors", "json", "unknown"] | None = None
    problem: str | None = Field(default=None, description="Why it can't be used, if it can't.")
    required: bool = Field(default=False, description="The model's family needs it.")
    label: str | None = Field(default=None, description="The family's name for it.")


LoraApplyMode = Literal["auto", "immediately", "at_runtime"]


class EngineLoraFolder(BaseModel):
    """An image engine's LoRA folder, checked on disk: an sd-server's ``--lora-model-dir``,
    a Diffusers backend's ``lora_dir``."""

    path: str | None = Field(description="Null: not set, so the engine applies no LoRAs.")
    exists: bool = False
    count: int | None = Field(
        default=None, description="LoRA files in it, subfolders included (what the engine lists)."
    )
    problem: str | None = Field(default=None, description="Why it can't be used, if it can't.")
    warning: str | None = Field(
        default=None, description="Something in it the engine would wrongly list as a LoRA."
    )
    apply_mode: LoraApplyMode | None = Field(
        default=None, description="sd-server only; null: its auto."
    )
    default_path: str = Field(description="Where Hugging Face installs LoRAs while it isn't set.")


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
    model_card: str | None = Field(
        default=None, description="sd-server engines: the model card it runs (GET /v1/models)."
    )
    agent: str | None = Field(
        default=None, description="llama-server engines: the agent it runs (GET /v1/agents)."
    )
    family: str | None = Field(default=None, description="Its model's family (a recipe id).")
    family_name: str | None = None
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
    loras: EngineLoraFolder | None = Field(
        default=None, description="sd-server engines launched here: the folder of LoRAs."
    )
    model_id: str | None = Field(
        default=None, description="Diffusers backends: a Hub repo id or a local folder."
    )
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
    lora_model_dir: str | None = Field(
        default=None, max_length=1024, description="sd-server's LoRA folder; empty clears it."
    )
    lora_apply_mode: LoraApplyMode | Literal[""] | None = Field(
        default=None, description="sd-server's --lora-apply-mode; empty: its default (auto)."
    )


class EngineUpdate(_Input):
    """Changes from the Engines screen. Saved to the profile's .local.yaml sidecar."""

    host: str | None = Field(default=None, min_length=1, max_length=253)
    port: int | None = Field(default=None, ge=1, le=65535)
    declared_capabilities: list[Capability] | None = None
    launch: EngineLaunchUpdate | None = None
    model_card: str | None = Field(
        default=None,
        max_length=64,
        description="sd-server engines: the model card to run; empty: none (its own files).",
    )
    agent: str | None = Field(
        default=None,
        max_length=64,
        description="llama-server engines: the agent whose model to run; empty: its own files.",
    )
    model_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=1024,
        description="Diffusers backends: a Hub repo id or a local folder.",
    )
    lora_dir: str | None = Field(
        default=None,
        max_length=1024,
        description="Diffusers backends: the LoRA folder, made if it isn't there; empty: none. "
        "(An sd-server's is launch.lora_model_dir.)",
    )
    presets: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Image backends: presets to add or change, by name, each merged over the "
        "one it replaces (a preset's loras list is replaced whole).",
    )
    default_loras: list[LoraPick] | None = Field(
        default=None, description="Image backends: defaults.loras, applied to every image."
    )


class EngineInstallRequest(_Input):
    method: Literal["auto", "release", "source"] = "auto"
    accelerator: str = Field(default="", max_length=16, description="Empty: detect.")
    version: str = Field(default="latest", max_length=64)
    force: bool = False


# --- Hugging Face Hub (docs/huggingface.md) -------------------------------------------


class HubStatus(BaseModel):
    enabled: bool
    endpoint: str
    token: bool = Field(description="Whether a Hub token is set. The token itself never is.")
    token_source: str | None = Field(default=None, description="'$HF_TOKEN' or 'hf auth login'.")
    models_dir: str = Field(description="Where downloads go.")
    hf_cache: str | None = Field(
        default=None, description="The Hugging Face cache searched for copies, if any."
    )


class HubSettingsUpdate(_Input):
    models_dir: str | None = Field(
        default=None, min_length=1, max_length=1024, description="Where downloads go."
    )


class HubRepoSummary(BaseModel):
    id: str
    url: str
    task: str | None = None
    library: str | None = None
    downloads: int | None = None
    likes: int | None = None
    gated: bool = False
    updated_at: str | None = None
    formats: list[Literal["gguf", "diffusers", "safetensors"]] = Field(default_factory=list)


class HubLocal(BaseModel):
    """A copy already on this machine."""

    source: Literal["models_dir", "hf_cache", "lora_dir"] = Field(
        description="lora_dir: a LoRA, installed in the engine's LoRA folder."
    )
    path: str = Field(description="The first file, or the folder for a Diffusers pipeline.")
    in_use: list[str] = Field(
        default_factory=list, description="'<engine>.<setting>' already pointed at it."
    )


class HubFile(BaseModel):
    path: str
    size_bytes: int | None = None


class HubVariant(BaseModel):
    """One loadable model in a repo: a file, or the shards of one."""

    id: str = Field(description="Its first file's path in the repo.")
    name: str
    component: Literal[
        "model", "mmproj", "diffusion_model", "vae", "text_encoder", "lora", "tokenizer",
        "connector", "other",
    ]  # fmt: skip
    format: Literal["gguf", "safetensors", "json"]
    encoder: Literal["clip_l", "clip_g", "t5xxl", "llm"] | None = Field(
        default=None, description="A text encoder's kind, when it can be told."
    )
    tiny: bool = Field(default=False, description="A tiny autoencoder (sd-server's --tae).")
    quant: str | None = None
    size_bytes: int
    shards: int = Field(default=1, description="Weight files; an index isn't counted.")
    files: list[HubFile] = Field(description="Everything to download, the entry first.")
    local: HubLocal | None = None
    partial_bytes: int = Field(default=0, description="Already in the models folder.")
    copy_from: HubLocal | None = Field(
        default=None,
        description="A LoRA not in the LoRA folder yet: a copy elsewhere on this machine, "
        "which installing copies instead of downloading.",
    )


class HubPipeline(BaseModel):
    """A Diffusers pipeline: what a diffusers backend loads, as a whole."""

    pipeline_class: str | None = None
    components: list[str]
    size_bytes: int
    file_count: int
    compatible: bool
    problem: str | None = None
    local: HubLocal | None = None
    partial_bytes: int = 0


class HubRole(BaseModel):
    """One of the engine's model settings, and what in the repo can fill it."""

    key: str = Field(description="The launch setting, e.g. 'mmproj', or 'model_id'.")
    flag: str | None = None
    what: str
    current: str | None = Field(default=None, description="What it is set to now.")
    candidates: list[str] = Field(default_factory=list, description="Variant ids.")
    suggested: str | None = None
    folder: str | None = Field(
        default=None,
        description="The lora role: the folder a LoRA is installed into (the engine's LoRA "
        "folder, or the one it gets on the first install).",
    )


class HubRepo(BaseModel):
    id: str
    url: str
    revision: str
    sha: str | None = None
    task: str | None = None
    library: str | None = None
    gated: bool = False
    downloads: int | None = None
    likes: int | None = None
    base_models: list[str] = Field(default_factory=list)
    kind: Literal["gguf", "diffusers", "comfy", "transformers", "safetensors", "unknown"]
    notes: list[str] = Field(default_factory=list)
    variants: list[HubVariant] = Field(default_factory=list)
    pipeline: HubPipeline | None = None
    engine: str | None = None
    model: str | None = Field(default=None, description="The model card matched to, if any.")
    roles: list[HubRole] = Field(default_factory=list)
    quant: str | None = Field(default=None, description="From a 'repo:QUANT' reference.")
    path: str | None = Field(default=None, description="A file or folder the reference named.")


class HubDownloadRequest(_Input):
    repo: str = Field(min_length=3, max_length=500, description="A repo id or Hub reference.")
    revision: str | None = Field(default=None, max_length=200, description="Default: main.")
    variant: str | None = Field(
        default=None, max_length=1024, description="A variant id; none: the Diffusers pipeline."
    )
    engine: str | None = Field(default=None, description="Set this engine's setting when done.")
    model: str | None = Field(
        default=None, max_length=64, description="Or put it in this model card's slot (key)."
    )
    agent: str | None = Field(
        default=None, max_length=64, description="Or this agent's file (key: model or mmproj)."
    )
    key: str | None = Field(
        default=None, description="The launch setting, 'model_id', or a model card's slot."
    )


class HubDownload(BaseModel):
    id: str
    repo: str
    revision: str
    label: str
    state: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    total_bytes: int
    done_bytes: int
    file_count: int
    file_index: int
    current_file: str | None = None
    target: str
    assign: list[str] = Field(
        default_factory=list,
        description="'<engine>.<setting>', or 'model:<card id>.<slot>' for a model card.",
    )
    speed_bps: float | None = None
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    warning: str | None = None


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
    image_presets: list[ImagePresetInfo] = Field(
        default_factory=list, description="Presets a request can name."
    )
    available_loras: list[str] | None = Field(
        default=None,
        description="LoRAs a request can name (empty for a stopped engine: listed once it "
        "is up). Null: it takes none, or the instance is too old to take them.",
    )
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
    # Sent only when set, so an instance that predates them still takes every other request.
    preset: str | None = Field(default=None, pattern=PRESET_PATTERN)
    loras: list[LoraPick] = Field(
        default_factory=list,
        max_length=MAX_LORA_PICKS,
        description="On top of the preset's (or default) LoRAs, as image_options.loras.",
    )


# --- the LoRA library -------------------------------------------------------------------

CompatState = Literal["ok", "maybe", "no", "unknown"]
CompatCall = Literal["auto", "works", "no"]


class LoraTrigger(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    on: bool = True


class LoraEngineUse(BaseModel):
    """How one engine names a LoRA, and whether it works with that engine's model."""

    engine: str
    path: str = Field(description="As the engine lists it: what presets and requests name.")
    model: str | None = Field(default=None, description="The model's key (LoraModel.key).")
    compat: CompatState


class LoraCompat(BaseModel):
    """The LoRA's standing across the engines that list it."""

    state: CompatState = Field(description="The best of its engines': ok, maybe, unknown, no.")
    text: str = Field(description='"sdcpp, diffusers", "applied on nas", "SDXL".')
    bad: str | None = Field(default=None, description="Why it works with none of them.")


class LoraCover(BaseModel):
    url: str
    source: str = Field(description='"img-12 · Clay foxes in a garden", "Hugging Face".')
    width: int | None = None
    height: int | None = None
    conversation_id: str | None = None
    artifact_id: str | None = None


class LoraSourceInfo(BaseModel):
    repo: str
    revision: str | None = None
    url: str
    has_images: bool = False


class LoraItem(BaseModel):
    id: str = Field(description="The file's SHA-256, or an id for one only listed by name.")
    name: str = Field(description="The display name: the user's, else the file's name.")
    custom_name: bool = False
    file_name: str
    path: str = Field(description="Relative to its LoRA folder, or as its engine lists it.")
    folder: str = Field(description='Its folder in the LoRA folder, "" at the top.')
    root: str | None = Field(default=None, description="The LoRA folder it is in, if here.")
    local: bool = Field(description="Its file is on this machine.")
    origin: str | None = Field(default=None, description="The linked instance it lives on.")
    size_bytes: int | None = None
    sha256: str | None = None
    modified_at: datetime | None = None
    engines: list[LoraEngineUse]
    triggers: list[LoraTrigger]
    default_weight: float = 1.0
    tags: list[str] = Field(default_factory=list)
    pinned: bool = False
    notes: str = ""
    cover: LoraCover | None = None
    source: LoraSourceInfo | None = None
    compat: LoraCompat
    uses: int = Field(default=0, description="Images made with it.")
    conversations: int = 0
    last_used_at: datetime | None = None


class LoraModel(BaseModel):
    """A model one of the image engines runs (or one the user made a call on)."""

    key: str
    name: str
    quant: str | None = None
    engines: list[str] = Field(default_factory=list)
    where: str
    loaded: bool = False
    checkable: bool = Field(default=False, description="Its weights' names can be read here.")


class LoraLibrary(BaseModel):
    items: list[LoraItem]
    models: list[LoraModel]
    tags: list[str] = Field(description="Every tag in use, most used first.")
    folders: list[str]
    roots: list[str] = Field(description="The LoRA folders on this machine.")
    total_bytes: int = 0
    local_count: int = 0
    origins: dict[str, int] = Field(default_factory=dict, description="Linked instance: count.")


class LoraModelCompat(BaseModel):
    key: str
    name: str
    where: str
    loaded: bool = False
    detected: CompatState
    detected_text: str
    call: CompatCall = "auto"
    state: CompatState = Field(description="The user's call, else what was detected.")


class LoraPresetUse(BaseModel):
    engine: str
    name: str
    description: str = ""
    included: bool
    weight: float | None = None
    meta: str = Field(default="", description="Its other LoRAs, steps and CFG.")


class LoraUseImage(BaseModel):
    conversation_id: str
    title: str | None
    images: int
    weight: float | None = None
    last_used_at: datetime
    artifact: Artifact | None = Field(default=None, description="The latest image made with it.")


class LoraFileInfo(BaseModel):
    path: str
    root: str
    engines: list[str]
    size_bytes: int
    hashed_at: datetime | None = None


class LoraDetail(LoraItem):
    compat_models: list[LoraModelCompat]
    lineage: str = ""
    lineage_default: str = ""
    presets: list[LoraPresetUse]
    used_in: list[LoraUseImage] = Field(description="Conversations, newest use first.")
    recent_images: list[Artifact] = Field(description="Images made with it, newest first.")
    files: list[LoraFileInfo]
    format: str | None = None
    rank: int | None = None
    base: str | None = Field(default=None, description="The model it says it was made for.")
    triggers_from: str | None = Field(
        default=None, description="Where its trigger words came from, until edited."
    )
    file_triggers: list[str] = Field(default_factory=list)


class LoraUpdate(_Input):
    """Only fields present are changed. An empty name goes back to the file's;
    reset_triggers goes back to the words the file names."""

    name: str | None = Field(default=None, max_length=120)
    default_weight: float | None = Field(default=None, ge=-10, le=10)
    triggers: list[LoraTrigger] | None = Field(default=None, max_length=24)
    reset_triggers: bool = False
    notes: str | None = Field(default=None, max_length=4000)
    tags: list[str] | None = Field(default=None, max_length=24)
    pinned: bool | None = None
    compat: dict[str, CompatCall] | None = Field(
        default=None, description="Model key: auto (detected), works or no."
    )
    lineage: str | None = Field(default=None, max_length=120)


class LoraMove(_Input):
    folder: str = Field(max_length=512, description='Relative to its LoRA folder; "" for the top.')


class LoraCoverSet(_Input):
    artifact_id: str | None = Field(default=None, max_length=36)
    hub: bool = Field(default=False, description="The first image on its Hub card.")


class LoraPresetSet(_Input):
    engine: str = Field(min_length=1, max_length=64)
    preset: str = Field(pattern=PRESET_PATTERN)
    weight: float | None = Field(
        default=None, ge=-10, le=10, description="Null takes the LoRA out of the preset."
    )
    create: bool = Field(default=False, description="Make the preset (with the defaults' LoRAs).")
    description: str = Field(default="", max_length=200)


# --- the model library --------------------------------------------------------------------

PromptStyle = Literal["natural", "tags", "json"]


class ModelSlot(BaseModel):
    """A component slot of a model family (an sd-server recipe slot)."""

    id: str = Field(description="sd-server's slot name, e.g. diffusion_model or t5xxl.")
    flag: str = Field(description="Its command-line flag, e.g. --t5xxl.")
    what: str
    label: str = Field(description="The recipe's name for it, e.g. Qwen 3 VL 8B Text Encoder.")
    required: bool
    sources: list[str] = Field(
        default_factory=list, description="Hub references to get it from, best first."
    )


class ModelFamilyBrief(BaseModel):
    id: str
    name: str
    group: str | None = None
    hidden: bool = False


class ModelFamily(ModelFamilyBrief):
    """An image model family sd-server runs: its recipe, and Stable-Imaginer's notes."""

    description: str = ""
    model_url: str | None = None
    origin: Literal["builtin", "engine"] = Field(
        description="builtin: the bundled recipe; engine: the one a running sd-server reported."
    )
    slots: list[ModelSlot]
    required_any: list[list[str]] = Field(
        default_factory=list, description="Groups of slots of which one must be filled."
    )
    defaults: dict[str, Any] = Field(description="Generation defaults, in ImageDefaults' names.")
    size_multiple: int
    capabilities: list[Capability]
    features: list[str] = Field(default_factory=list, description="The recipe's features.")
    prompt: PromptStyle
    prompt_hint: str | None = None
    note: str | None = None
    vae_format: str | None = None
    model_args: dict[str, str] = Field(default_factory=dict)


class ModelGuess(BaseModel):
    family: str
    name: str
    score: float = Field(description="0..1: how well the evidence fits.")
    reasons: list[str]


class ModelComponent(BaseModel):
    """One slot of a model card, checked on disk."""

    slot: str
    flag: str
    what: str
    label: str
    required: bool = Field(description="Its family needs it (or one of its group).")
    path: str | None = None
    exists: bool = False
    size_bytes: int | None = None
    format: Literal["gguf", "safetensors", "json", "unknown"] | None = None
    problem: str | None = None
    source: dict[str, Any] | None = Field(
        default=None, description="Where the Hub installed it from: repo, revision, path."
    )
    shared_with: list[str] = Field(
        default_factory=list, description="Other model cards that use the same file."
    )
    sources: list[str] = Field(
        default_factory=list, description="Hub references to get it from, best first."
    )


class ModelCover(BaseModel):
    url: str
    source: str
    width: int | None = None
    height: int | None = None
    conversation_id: str | None = None
    artifact_id: str | None = None


class ModelEngineUse(BaseModel):
    """An sd-server engine running (or set to run) the model."""

    engine: str
    launchable: bool = Field(description="Started here, so switching models restarts it.")
    running: bool = False
    needs_restart: bool = Field(
        default=False, description="Running with other files or launch arguments than these."
    )


class ModelLaunch(BaseModel):
    vae_format: str | None = None
    model_args: dict[str, str] = Field(
        default_factory=dict, description="--model-args over the family's."
    )
    extra_args: list[str] = Field(
        default_factory=list, description="Appended to the engine's own launch arguments."
    )


class ModelCardItem(BaseModel):
    id: str
    name: str
    family: str | None = None
    family_name: str | None = None
    group: str | None = None
    main_file: str | None = Field(default=None, description="Its checkpoint or diffusion model.")
    size_bytes: int = Field(default=0, description="Of its component files on disk.")
    ready: bool = Field(description="It has a family, every component it needs, all usable.")
    missing: list[str] = Field(
        default_factory=list, description="Slots still empty, e.g. t5xxl, or vae or tae."
    )
    problems: list[str] = Field(default_factory=list)
    engines: list[ModelEngineUse] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    pinned: bool = False
    notes: str = ""
    cover: ModelCover | None = None
    uses: int = Field(default=0, description="Images made with it.")
    last_used_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class ModelCardDetail(ModelCardItem):
    components: list[ModelComponent]
    launch: ModelLaunch
    defaults: dict[str, Any] = Field(description="Its ImageDefaults.")
    presets: dict[str, dict[str, Any]] = Field(description="Its presets (ImagePreset), by name.")
    prompt_hint: str | None = Field(default=None, description="Its own, over the family's.")
    family_prompt_hint: str | None = None
    family_note: str | None = None
    capabilities: list[Capability] = Field(default_factory=list)
    detected: list[ModelGuess] = Field(
        default_factory=list, description="Families its main file looks like, best first."
    )
    recent_images: list[Artifact] = Field(default_factory=list)


class SharedComponent(BaseModel):
    """A component file on disk, and the cards that use it."""

    path: str
    slots: list[str]
    cards: list[str] = Field(description="Ids of the model cards that use it.")
    size_bytes: int | None = None
    exists: bool = False


class ModelLibraryEngine(BaseModel):
    name: str
    model_card: str | None = None
    launchable: bool
    running: bool = False


class ModelLibraryView(BaseModel):
    items: list[ModelCardItem]
    families: list[ModelFamilyBrief] = Field(description="Every family a card can have.")
    components: list[SharedComponent]
    engines: list[ModelLibraryEngine] = Field(description="The sd-server engines.")
    tags: list[str] = Field(default_factory=list)


class ModelCreate(_Input):
    """A new model card: from a model file on this machine (path), from a Hub reference
    (ref: its components are downloaded into it next), or empty for a family. Without a
    family, it is detected."""

    name: str | None = Field(default=None, max_length=120)
    family: str | None = Field(default=None, max_length=64)
    path: str | None = Field(default=None, max_length=1024)
    ref: str | None = Field(default=None, max_length=1024)
    components: dict[str, str] = Field(default_factory=dict, description="slot -> path")


class ModelDetect(_Input):
    path: str | None = Field(default=None, max_length=1024)
    ref: str | None = Field(default=None, max_length=1024)


class ModelDetected(BaseModel):
    guesses: list[ModelGuess]
    slot: str | None = Field(
        default=None, description="The slot the file goes in for the best guess."
    )
    name: str | None = Field(default=None, description="A name for a card made from it.")


class ModelLocate(_Input):
    """A file dropped or picked in the browser, which tells its name and size but not its
    folder."""

    name: str = Field(min_length=1, max_length=255)
    size_bytes: int | None = Field(default=None, ge=0)


class ModelLocated(BaseModel):
    paths: list[str] = Field(description="Files with that name here, same size first.")
    searched: list[str] = Field(description="The folders looked in.")


class ModelLaunchUpdate(_Input):
    vae_format: Literal["", "auto", "flux", "sd3", "flux2", "wan"] | None = Field(
        default=None, description="Empty: the family's."
    )
    model_args: dict[str, str] | None = Field(default=None, description="Replaces them whole.")
    extra_args: list[str] | None = None


class ModelUpdate(_Input):
    """Only fields present are changed."""

    name: str | None = Field(default=None, max_length=120)
    family: str | None = Field(default=None, max_length=64)
    components: dict[str, str | None] | None = Field(
        default=None, description="slot -> path; null or empty clears it."
    )
    launch: ModelLaunchUpdate | None = None
    defaults: dict[str, Any] | None = Field(
        default=None, description="Merged over its defaults (ImageDefaults' fields)."
    )
    reset_defaults: bool = Field(default=False, description="Back to its family's defaults.")
    presets: dict[str, dict[str, Any] | None] | None = Field(
        default=None, description="By name: merged over the preset it replaces; null removes it."
    )
    prompt_hint: str | None = Field(default=None, max_length=2000)
    notes: str | None = Field(default=None, max_length=4000)
    tags: list[str] | None = Field(default=None, max_length=24)
    pinned: bool | None = None


class ModelUse(_Input):
    engine: str = Field(min_length=1, max_length=64)
    start: bool = Field(
        default=True,
        description="Start the engine with it if it isn't running (one that is restarts).",
    )


class ModelCoverSet(_Input):
    artifact_id: str | None = Field(default=None, max_length=36)
    hub: bool = Field(default=False, description="The first image on its main file's Hub card.")


# --- agents (docs/agent-library.md) ------------------------------------------------------------

AgentPortraitSource = Literal["hub", "self", "upload"]
AgentViewer = Literal["status", "reasoning"]
AgentFileKey = Literal["model", "mmproj", "chat_template"]


class AgentPortraitImage(BaseModel):
    """A portrait the agent has, in use or not."""

    key: str = Field(description="'hub', 'upload' or 'self-<version>'.")
    source: AgentPortraitSource
    url: str
    version: int | None = Field(default=None, description="A self-portrait's number (v1, v2…).")
    prompt: str | None = Field(default=None, description="A self-portrait's prompt, its own.")
    owner: str | None = Field(default=None, description="From the Hub: whose avatar it is.")
    conversation_id: str | None = Field(default=None, description="Where it was painted.")
    artifact_id: str | None = None
    label: str | None = Field(default=None, description="Its label in that conversation.")
    created_at: datetime | None = None


class AgentPainting(BaseModel):
    """A self-portrait being painted now."""

    conversation_id: str
    job_id: str | None = None
    status: Literal["writing", "queued", "running"] = "writing"
    step: int | None = None
    total_steps: int | None = None


class AgentPortrait(BaseModel):
    key: str | None = Field(
        default=None, description="The portrait in use (see images); null: its monogram."
    )
    source: AgentPortraitSource | None = None
    url: str | None = None
    monogram: str = Field(description="Shown while it has no portrait: its name's first letter.")
    origin: str = Field(default="", description="Where it came from, e.g. 'HUB · QWEN'.")
    hub_owner: str | None = Field(
        default=None, description="The Hub user or organization whose avatar it can use."
    )
    hub_fetched_at: datetime | None = None
    images: list[AgentPortraitImage] = Field(default_factory=list)
    prompt: str | None = Field(default=None, description="Its latest self-portrait's prompt.")
    painting: AgentPainting | None = None


class AgentEngineUse(BaseModel):
    """The chat engine an agent runs on."""

    engine: str
    kind: Literal["local", "remote", "linked"] = Field(
        description="local: started here; remote: started elsewhere; linked: shared by a "
        "linked instance."
    )
    link: str | None = Field(default=None, description="A linked engine's instance.")
    launchable: bool
    runs_it: bool = Field(
        description="It is set to run this agent's model (always, for an agent with no files "
        "of its own)."
    )
    running: bool = Field(description="Up and serving this agent's model now.")
    status: str = Field(description="The engine's status: ok, degraded, down, unknown.")
    default: bool = Field(description="The default chat engine: new prompts go to it.")
    needs_restart: bool = False
    runs_other: str | None = Field(
        default=None, description="The agent it runs instead, if another one."
    )


class AgentFile(BaseModel):
    key: AgentFileKey
    flag: str
    what: str
    path: str | None = None
    exists: bool = False
    size_bytes: int | None = None
    problem: str | None = None
    required: bool = False
    source: dict[str, Any] | None = Field(default=None, description="Where the Hub got it.")


class AgentTool(BaseModel):
    name: str
    description: str
    flags: str = Field(description="What it makes: 't2i', 'edit · mask', 't2i · 1:1'.")
    enabled: bool = Field(description="The agent may call it.")
    available: bool = Field(description="An engine here can serve it now.")


class AgentItem(BaseModel):
    id: str
    name: str
    engine: str | None = Field(description="The chat engine it runs on, as resolved.")
    engine_set: str | None = Field(
        default=None, description="As set on the agent; null: the default chat engine."
    )
    model: str | None = Field(
        default=None, description="Its model file, or the model its engine reports."
    )
    quant: str | None = None
    size_bytes: int = Field(default=0, description="Of its files on disk.")
    family: str | None = Field(default=None, description="Its model's name without the size.")
    vision: bool = False
    tools: bool = Field(default=False, description="It can call tools (and has some on).")
    own_files: bool = Field(description="It has model files of its own.")
    ready: bool = Field(description="Everything it needs is here and usable.")
    missing: list[AgentFileKey] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    use: AgentEngineUse | None = None
    default: bool = Field(default=False, description="New prompts go to it.")
    working: bool = Field(default=False, description="Answering, or painting a portrait, now.")
    portrait: AgentPortrait
    viewer: AgentViewer = "reasoning"
    pinned: bool = False
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    source: dict[str, Any] | None = None
    turns: int = 0
    images: int = 0
    last_used_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class AgentDetail(AgentItem):
    files: list[AgentFile]
    system_prompt: str = Field(description="What it is told, as used.")
    system_prompt_custom: bool = Field(description="Its own, rather than the built-in one.")
    default_system_prompt: str
    tool_list: list[AgentTool]
    sampling: dict[str, Any] = Field(
        default_factory=dict, description="Its own, over its engine's."
    )
    engine_sampling: dict[str, Any] = Field(default_factory=dict)
    context: int | None = None


class AgentLibraryEngine(BaseModel):
    name: str
    kind: Literal["local", "remote", "linked"]
    link: str | None = None
    launchable: bool
    running: bool
    agent: str | None = Field(default=None, description="The agent it is set to run.")
    default: bool
    model: str | None = None


class AgentLibraryView(BaseModel):
    items: list[AgentItem]
    engines: list[AgentLibraryEngine] = Field(description="The chat engines.")
    default: str | None = Field(default=None, description="The agent new prompts go to.")
    tags: list[str] = Field(default_factory=list)


class AgentCreate(_Input):
    """A new agent: from a GGUF on this machine (path), from a Hub reference (ref: its model
    and vision projector are downloaded into it next), or for whatever an engine runs."""

    name: str | None = Field(default=None, max_length=120)
    engine: str | None = Field(default=None, max_length=64)
    path: str | None = Field(default=None, max_length=1024)
    mmproj: str | None = Field(default=None, max_length=1024)
    ref: str | None = Field(default=None, max_length=1024)
    system_prompt: str | None = Field(default=None, max_length=20_000)


class AgentUpdate(_Input):
    """Only fields present are changed."""

    name: str | None = Field(default=None, max_length=120)
    engine: str | None = Field(default=None, max_length=64, description="Empty: the default.")
    files: dict[AgentFileKey, str | None] | None = Field(
        default=None, description="key -> path; null or empty clears it."
    )
    system_prompt: str | None = Field(
        default=None, max_length=20_000, description="Empty: back to the built-in one."
    )
    viewer: AgentViewer | None = None
    tools: dict[str, bool] | None = Field(default=None, description="Turn tools on or off.")
    sampling: dict[str, Any] | None = Field(
        default=None, description="Merged over its own; a null value removes that setting."
    )
    context: int | None = Field(default=None, ge=0, le=1_048_576, description="0: the engine's.")
    notes: str | None = Field(default=None, max_length=4000)
    tags: list[str] | None = Field(default=None, max_length=24)
    pinned: bool | None = None


class AgentUse(_Input):
    start: bool = Field(
        default=True,
        description="Start its engine with it if the engine isn't running (one that is "
        "restarts with its model).",
    )


class AgentPortraitSet(_Input):
    """Which portrait to show: one it has (key), an image from a conversation (artifact_id,
    kept as its upload), its Hub uploader's avatar fetched again (refresh), or none."""

    key: str | None = Field(default=None, max_length=32, pattern=r"^(hub|upload|self-\d+|none)$")
    artifact_id: str | None = Field(default=None, max_length=36)
    refresh: bool = False


class AgentPainted(BaseModel):
    agent: AgentDetail
    conversation_id: str = Field(description="Where it is painting; the image stays there.")
