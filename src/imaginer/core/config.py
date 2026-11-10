"""Configuration: one YAML profile per host, validated by pydantic-settings.

Precedence, highest first: explicit keyword overrides, ``IMAGINER_*`` environment variables
(``__`` separates nesting, e.g. ``IMAGINER_SERVER__PORT=9000``), then ``config/<profile>.yaml``.
Secrets never live in YAML: the server token comes from ``IMAGINER_SERVER__AUTH_TOKEN`` and
engine API keys from the env var each backend names in ``api_key_env``.
"""

import ipaddress
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import platformdirs
import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from imaginer.domain.enums import BackendKind, Capability
from imaginer.domain.models import PRESET_PATTERN

PROFILE_ENV = "IMAGINER_PROFILE"
CONFIG_DIR_ENV = "IMAGINER_CONFIG_DIR"
DEFAULT_PROFILE = "dev"

# Official Qwen-Image-2.1 phrasing for transparent output. Transparency is prompt-driven:
# there is no request flag in either sd.cpp or Diffusers.
DEFAULT_RGBA_PROMPT_TEMPLATE = (
    "This is an RGBA image with transparency. {prompt}. "
    "The image has alpha channel and the background is transparent."
)
# Qwen-Image-2.1 accepts "separate masks" as an extra reference image; the prompt says so.
DEFAULT_MASK_INSTRUCTION = (
    "The last image is a mask: change only the area that is white in the mask and keep "
    "everything else exactly as it is."
)

BackendName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")]


class ConfigError(ValueError):
    """The selected profile could not be found or loaded."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = Field(default=8700, ge=1, le=65535)
    auth_token: SecretStr | None = Field(
        default=None, description="Static bearer token. Set via IMAGINER_SERVER__AUTH_TOKEN."
    )
    cors_origins: list[str] = Field(default_factory=list)


class DatabaseConfig(_Strict):
    url: str | None = Field(
        default=None, description="SQLAlchemy async URL. Default: SQLite under data_root."
    )
    auto_migrate: bool = True
    echo: bool = False


class StorageConfig(_Strict):
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    sweep_interval_s: float = Field(default=600.0, ge=0, description="0 disables periodic sweeps.")
    orphan_grace_s: float = Field(
        default=3600.0, ge=0, description="Unreferenced files younger than this are kept."
    )


class LoggingConfig(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["console", "json"] = "console"


class TimeoutConfig(_Strict):
    connect: float = Field(default=5.0, gt=0)
    read: float = Field(default=300.0, gt=0)
    write: float = Field(default=60.0, gt=0)
    pool: float = Field(default=5.0, gt=0)


class RetryConfig(_Strict):
    attempts: int = Field(default=3, ge=1, le=10)
    backoff_initial_s: float = Field(default=0.5, ge=0)
    backoff_max_s: float = Field(default=8.0, ge=0)


class LoraRef(_Strict):
    """One LoRA to apply. ``path`` names it as the engine lists it (``available_loras``):
    for sd-server, a path from the ``loras`` of ``GET /sdcpp/v1/capabilities`` (relative to
    ``--lora-model-dir``); for Diffusers, its path in ``lora_dir``. Its file name, or its
    name without the extension, also works when only one LoRA has it."""

    path: str = Field(min_length=1, max_length=1024)
    weight: float = Field(
        default=1.0,
        ge=-10,
        le=10,
        description="Its strength: sd.cpp's multiplier, the adapter weight in Diffusers.",
    )


class ImageDefaults(_Strict):
    """Sampler defaults: a model card's, else the backend's. Engines disagree on good values
    for the same model."""

    width: int = Field(default=2048, ge=64, le=8192, description="Qwen-Image-2.1 is native 2K.")
    height: int = Field(default=2048, ge=64, le=8192)
    steps: int | None = Field(default=None, ge=1)
    cfg_scale: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    flow_shift: float | None = None
    sigmas: list[float] | str | None = Field(
        default=None, description="Explicit sigma list, or an engine preset name."
    )
    guidance: float | None = Field(
        default=None,
        description="Distilled guidance, for models with a guidance input (FLUX.1 dev, FLUX.2 "
        "dev): sd.cpp's --guidance. Not CFG.",
    )
    clip_skip: int | None = Field(
        default=None, ge=1, description="CLIP layers to skip (SD 1.x/2.x/SDXL); 1 skips none."
    )
    loras: list[LoraRef] = Field(
        default_factory=list, description="Applied to every image unless a preset says otherwise."
    )
    negative_prompt: str = ""
    output_format: Literal["png", "webp", "jpeg"] = "png"
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific fields passed through verbatim."
    )
    extra_sample: dict[str, Any] = Field(
        default_factory=dict,
        description="More sd-server sample_params, sent as they are (e.g. shifted_timestep).",
    )


class ImagePreset(_Strict):
    """Named settings the user can pick for a message, e.g. a step-distill LoRA together with
    the steps and CFG it was made for. Unset fields keep the backend's defaults, except that
    steps without sigmas drop the default sigmas (a schedule sets the step count). Steps or
    CFG the user sets still win. The quality check leaves the steps and CFG a preset sets."""

    description: str = Field(default="", max_length=200, description="Shown where it's picked.")
    loras: list[LoraRef] | None = Field(
        default=None, description="Instead of defaults.loras ([] for none). Unset: keep them."
    )
    steps: int | None = Field(default=None, ge=1)
    cfg_scale: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    flow_shift: float | None = None
    sigmas: list[float] | str | None = None
    guidance: float | None = None
    clip_skip: int | None = Field(default=None, ge=1)


ImagePresets = dict[Annotated[str, Field(pattern=PRESET_PATTERN)], ImagePreset]


class _HttpBackend(_Strict):
    enabled: bool = True
    base_url: str
    api_key_env: str | None = Field(
        default=None, description="Name of the env var that holds this engine's API key."
    )
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    declared_capabilities: set[Capability] | None = Field(
        default=None,
        description="Upper bound on advertised capabilities. Probing can only remove from it.",
    )

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class _Launch(_Strict):
    """How to start a local engine from the Engines screen (or at startup, with autostart).
    Without it the backend is treated as remote: only its endpoint is managed here."""

    executable: Path | None = Field(
        default=None, description="Default: the runtime installed under <data_root>/engines."
    )
    extra_args: list[str] = Field(default_factory=list, description="Appended verbatim.")
    autostart: bool = Field(default=False, description="Start it when the server starts.")


class LlamaCppLaunch(_Launch):
    model: Path | None = Field(default=None, description="-m: the GGUF model")
    mmproj: Path | None = Field(default=None, description="--mmproj: vision projector")


VaeFormat = Literal["auto", "flux", "sd3", "flux2", "wan"]


class SdCppLaunch(_Launch):
    """The model files, by sd-server's slot names. Which a model needs depends on its family
    (image_models/recipes.py): a checkpoint (``model``) for SD 1.x/SDXL, a diffusion model
    with its text encoders and VAE for most others. A model card fills them when the engine
    has one (``model_card``)."""

    model: Path | None = Field(default=None, description="--model: a full checkpoint")
    diffusion_model: Path | None = Field(default=None, description="--diffusion-model")
    uncond_diffusion_model: Path | None = Field(
        default=None, description="--uncond-diffusion-model (Ideogram 4)"
    )
    vae: Path | None = Field(default=None, description="--vae")
    tae: Path | None = Field(default=None, description="--tae: a tiny autoencoder")
    clip_l: Path | None = Field(default=None, description="--clip_l")
    clip_g: Path | None = Field(default=None, description="--clip_g")
    t5xxl: Path | None = Field(default=None, description="--t5xxl")
    llm: Path | None = Field(default=None, description="--llm: an LLM text encoder")
    llm_vision: Path | None = Field(
        default=None, description="--llm_vision: needed for edit and multi_ref"
    )
    embeddings_connectors: Path | None = Field(default=None, description="--embeddings-connectors")
    tokenizer: Path | None = Field(default=None, description="--tokenizer: a tokenizer.json")
    vae_format: VaeFormat | None = Field(
        default=None, description="--vae-format. Default: the family's, else auto."
    )
    model_args: dict[str, str | bool | int | float] = Field(
        default_factory=dict,
        description="--model-args, over the ones the model's family recommends.",
    )
    lora_model_dir: Path | None = Field(
        default=None,
        description="--lora-model-dir: the folder whose LoRAs (searched recursively) requests "
        "may apply. Keep only LoRAs in it: sd-server lists every weights file there as one.",
    )
    lora_apply_mode: Literal["auto", "immediately", "at_runtime"] | None = Field(
        default=None,
        description="--lora-apply-mode. auto (sd-server's default) applies at runtime when the "
        "model is quantized, else merges the LoRA into the weights.",
    )


class LlamaCppBackendConfig(_HttpBackend):
    type: Literal["llamacpp"]
    launch: LlamaCppLaunch | None = None
    model: str | None = Field(
        default=None, description="Model id sent with requests (needed in router mode)."
    )
    sampling: dict[str, Any] = Field(
        default_factory=dict, description="Passed through to /v1/chat/completions."
    )


class SdCppBackendConfig(_HttpBackend):
    type: Literal["sdcpp"]
    launch: SdCppLaunch | None = None
    model_card: str | None = Field(
        default=None,
        max_length=64,
        description="The model card (GET /v1/models) this engine runs: its files, family, "
        "defaults and presets replace the ones set here.",
    )
    family: str | None = Field(
        default=None,
        max_length=64,
        description="The model's family, an sd-server recipe id (GET /v1/models/families). "
        "A model card sets it; name one for an engine set up by hand.",
    )
    concurrency: int = Field(default=1, ge=1)
    max_ref_images: int = Field(default=10, ge=1)
    size_multiple: int | None = Field(
        default=None,
        ge=1,
        description="Sizes snap to multiples of it. Default: the family's, else 32 "
        "(Qwen-Image-2.1's).",
    )
    defaults: ImageDefaults = Field(default_factory=ImageDefaults)
    presets: ImagePresets = Field(
        default_factory=dict, description="Named settings a message can pick (image_options)."
    )
    rgba_prompt_template: str | None = Field(
        default=None,
        description="How a prompt asks for transparency, with {prompt}. Default: the "
        "family's (Qwen-Image-2.1's without a family).",
    )
    mask_instruction: str | None = Field(
        default=None,
        description="How an edit's mask is explained in the prompt. Default: the family's "
        "(Qwen-Image-2.1's without a family).",
    )
    max_ref_pixels: int | None = Field(
        default=None,
        gt=0,
        description="Larger input images are downscaled first. Default: the default image's "
        "area, at least 1024x1024.",
    )
    poll_interval_s: float = Field(default=0.5, gt=0)
    preview: str = Field(
        default="proj",
        description="none | proj | tae | vae. Previews carry step progress; proj is cheapest.",
    )
    preview_interval: int = Field(default=1, ge=1)


class DiffusersBackendConfig(_Strict):
    type: Literal["diffusers"]
    enabled: bool = True
    model_id: str = "Qwen/Qwen-Image-2.1"
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    cpu_offload: bool = Field(
        default=False, description="Keep weights in RAM and move them in per stage (less VRAM)."
    )
    preload: bool = Field(
        default=False, description="Load the model at startup instead of on first use."
    )
    concurrency: int = Field(default=1, ge=1)
    max_ref_images: int = Field(default=10, ge=1)
    size_multiple: int = Field(default=32, ge=8)
    declared_capabilities: set[Capability] | None = None
    defaults: ImageDefaults = Field(default_factory=lambda: ImageDefaults(steps=40, cfg_scale=1.0))
    presets: ImagePresets = Field(
        default_factory=dict, description="Named settings a message can pick."
    )
    lora_dir: Path | None = Field(
        default=None,
        description="The folder whose LoRAs (.safetensors, subfolders too) requests may "
        "apply, by their path in it. Changing it doesn't reload the model. Needs peft.",
    )
    rgba_prompt_template: str = DEFAULT_RGBA_PROMPT_TEMPLATE
    mask_instruction: str = DEFAULT_MASK_INSTRUCTION
    max_ref_pixels: int | None = Field(default=None, gt=0)


class LinkedBackendConfig(_Strict):
    """An engine that a linked Imaginer instance shares (docs/linked-instances.md).

    The whole model runs on that instance. Its URL and token come from the link, made by
    pairing, so nothing secret is configured here; so do timeouts (a stream stays open for
    as long as the work takes, kept alive by the peer's pings)."""

    type: Literal["linked"]
    enabled: bool = True
    link: BackendName = Field(description="The link's name on this instance.")
    engine: BackendName = Field(description="The backend's name on the linked instance.")
    kind: BackendKind
    concurrency: int = Field(default=1, ge=1, description="Image jobs sent at once.")
    declared_capabilities: set[Capability] | None = Field(
        default=None,
        description="Upper bound on advertised capabilities. The peer can only remove from it.",
    )


BackendConfig = Annotated[
    LlamaCppBackendConfig | SdCppBackendConfig | DiffusersBackendConfig | LinkedBackendConfig,
    Field(discriminator="type"),
]

BACKEND_KINDS: dict[str, BackendKind] = {
    "llamacpp": BackendKind.LLM,
    "sdcpp": BackendKind.IMAGE,
    "diffusers": BackendKind.IMAGE,
}


def backend_kind(config: BackendConfig) -> BackendKind:
    """LLM or image. A linked backend says which; every other type is one or the other."""
    if isinstance(config, LinkedBackendConfig):
        return config.kind
    return BACKEND_KINDS[config.type]


class RoutingConfig(_Strict):
    """Default backend per task. An explicit backend in a request always wins."""

    chat: str | None = None
    t2i: str | None = None
    edit: str | None = None


class QAConfig(_Strict):
    """The quality check: a vision LLM reviews each image and has it made again if flawed."""

    enabled: bool = Field(
        default=True, description="For messages that don't say (MessageCreate.qa)."
    )
    max_retries: int = Field(
        default=2, ge=0, le=5, description="New attempts per tool call before keeping the last."
    )


class OrchestrationConfig(_Strict):
    max_tool_iterations: int = Field(default=4, ge=0, le=32)
    job_timeout_s: float = Field(default=900.0, gt=0)
    feed_outputs_to_vision: bool = False
    probe_ttl_s: float = Field(
        default=15.0, ge=0, description="Re-probe an engine before use if its probe is older."
    )
    qa: QAConfig = Field(default_factory=QAConfig)


class ContextConfig(_Strict):
    max_images: int = Field(
        default=4, ge=0, description="Most recent attached images sent as pixels to a vision LLM."
    )
    image_max_side: int = Field(default=1024, ge=64)
    max_messages: int = Field(
        default=200, ge=1, description="Most recent messages included in the LLM context."
    )
    system_prompt: str | None = Field(
        default=None, description="Replaces the built-in system prompt (artifact list is appended)."
    )


class LinkShare(_Strict):
    remote_start: bool = Field(
        default=False,
        description="Linked instances may start it when it is stopped (if the model fits).",
    )


class LinksConfig(_Strict):
    """Linking with other Imaginer instances on the network (docs/linked-instances.md).

    Off, the instance is standalone: no peer routes, no discovery. On, it can pair with
    other instances, show their conversations in its gallery, use the engines they share
    and share its own."""

    enabled: bool = False
    name: str | None = Field(
        default=None, max_length=64, description="Shown to linked instances. Default: hostname."
    )
    advertise_url: str | None = Field(
        default=None,
        description="How linked instances reach this one. Default: http://<hostname>.local:<port>.",
    )
    discovery: bool = Field(default=True, description="Advertise and browse over mDNS.")
    poll_interval_s: float = Field(default=30.0, gt=0)
    start_timeout_s: float = Field(
        default=300.0, gt=0, description="How long a stopped shared engine may take to load."
    )
    gallery: Literal["off", "read", "full"] = Field(
        default="full",
        description="What linked instances may do with this instance's conversations.",
    )
    share: dict[BackendName, LinkShare | None] = Field(
        default_factory=dict,
        description="Engines linked instances may use. null un-shares one the profile shares.",
    )

    @field_validator("advertise_url")
    @classmethod
    def _http_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("advertise_url must start with http:// or https://")
        return value.rstrip("/") if value else value

    def shared(self) -> dict[str, LinkShare]:
        return {name: share for name, share in self.share.items() if share is not None}


class HubConfig(_Strict):
    """The Hugging Face Hub: search, repo listings and model downloads (docs/huggingface.md).

    The token is never configured here. It comes from ``$HF_TOKEN`` (or the env var
    ``token_env`` names), else from ``hf auth login``'s token file, as the ``hf`` CLI reads it.
    """

    enabled: bool = True
    endpoint: str | None = Field(
        default=None, description="Default: $HF_ENDPOINT, then https://huggingface.co."
    )
    token_env: str = Field(default="HF_TOKEN", description="Env var holding a Hub token.")
    models_dir: Path | None = Field(
        default=None, description="Where downloads go. Default: <data_root>/models."
    )
    use_hf_cache: bool = Field(
        default=True,
        description="Use files that hf, Diffusers or transformers already put in the "
        "Hugging Face cache instead of downloading them again.",
    )

    @field_validator("endpoint")
    @classmethod
    def _http_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("hub.endpoint must start with http:// or https://")
        return value.rstrip("/") if value else value


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _default_data_root() -> Path:
    return platformdirs.user_data_path("imaginer", appauthor=False)


# The YAML file for the Settings instance being built (set by load_settings).
_yaml_path: ContextVar[Path | None] = ContextVar("imaginer_yaml_path", default=None)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IMAGINER_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    profile: str | None = None
    data_root: Path = Field(default_factory=_default_data_root)
    server: ServerConfig = Field(default_factory=ServerConfig)
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    backends: dict[BackendName, BackendConfig] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    links: LinksConfig = Field(default_factory=LinksConfig)
    hub: HubConfig = Field(default_factory=HubConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        yaml_path = _yaml_path.get()
        if yaml_path is not None:
            sources.append(_ProfileSource(settings_cls, yaml_path))
        return tuple(sources)

    @field_validator("data_root", mode="after")
    @classmethod
    def _expand_data_root(cls, value: Path) -> Path:
        return Path(os.path.expandvars(value)).expanduser().resolve()

    @model_validator(mode="after")
    def _check_routing(self) -> "Settings":
        for task, name in (
            ("chat", self.routing.chat),
            ("t2i", self.routing.t2i),
            ("edit", self.routing.edit),
        ):
            expected = BackendKind.LLM if task == "chat" else BackendKind.IMAGE
            if name is None:
                # Without an explicit default, the choice must be unambiguous.
                candidates = self.enabled_backends(expected)
                if len(candidates) > 1:
                    raise ValueError(
                        f"routing.{task} is not set and several {expected.value} backends are "
                        f"enabled ({', '.join(candidates)}); choose one"
                    )
                continue
            backend = self.backends.get(name)
            if backend is None:
                raise ValueError(f"routing.{task} refers to unknown backend '{name}'")
            if backend_kind(backend) is not expected:
                raise ValueError(
                    f"routing.{task} needs an {expected.value} backend, "
                    f"but '{name}' is a {backend_kind(backend).value} backend"
                )
        return self

    @model_validator(mode="after")
    def _check_links(self) -> "Settings":
        for name in self.links.shared():
            backend = self.backends.get(name)
            if backend is None:
                raise ValueError(f"links.share refers to unknown backend '{name}'")
            if isinstance(backend, LinkedBackendConfig):
                raise ValueError(
                    f"links.share can't share '{name}': it already runs on a linked instance"
                )
        if not self.links.enabled:
            return self
        host_local = is_loopback_host(self.server.host)
        advertised = self.links.advertise_url
        if host_local and advertised is None:
            raise ValueError(
                "links.enabled: linked instances can't reach 127.0.0.1; set server.host "
                "(e.g. 0.0.0.0), or links.advertise_url behind a reverse proxy"
            )
        reach_local = host_local and is_loopback_host(urlsplit(advertised or "").hostname or "")
        if not reach_local and self.server.auth_token is None:
            raise ValueError(
                "links.enabled: linking beyond this machine needs a server token; "
                "set IMAGINER_SERVER__AUTH_TOKEN"
            )
        return self

    def enabled_backends(self, kind: BackendKind) -> list[str]:
        return [
            name
            for name, backend in self.backends.items()
            if backend.enabled and backend_kind(backend) is kind
        ]

    def default_backend(self, task: Literal["chat", "t2i", "edit"]) -> str | None:
        """The configured default for a task, or the only enabled backend of the right kind."""
        explicit: str | None = getattr(self.routing, task)
        if explicit is not None:
            return explicit
        kind = BackendKind.LLM if task == "chat" else BackendKind.IMAGE
        candidates = self.enabled_backends(kind)
        return candidates[0] if len(candidates) == 1 else None

    @property
    def models_dir(self) -> Path:
        """Where Hub downloads go. Not resolved: a mapped drive stays a drive letter in the
        paths engines are given."""
        configured = self.hub.models_dir
        if configured is None:
            return self.data_root / "models"
        return Path(os.path.expandvars(configured)).expanduser().absolute()

    @property
    def database_url(self) -> str:
        if self.db.url:
            return self.db.url
        return f"sqlite+aiosqlite:///{(self.data_root / 'imaginer.sqlite3').as_posix()}"


def local_overrides_path(profile_path: Path) -> Path:
    """``config/<profile>.local.yaml``: what the Engines screen saves, over the profile."""
    return profile_path.with_name(f"{profile_path.stem}.local.yaml")


def merge_mappings(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """``over`` on top of ``base``, recursing into nested mappings."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_mappings(out[key], value)
        else:
            out[key] = value
    return out


class _ProfileSource(YamlConfigSettingsSource):
    """The profile YAML with its ``.local.yaml`` sidecar deep-merged on top.

    The sidecar holds settings changed from the UI, so the hand-written profile (and its
    comments) is never rewritten.
    """

    def __init__(self, settings_cls: type[BaseSettings], profile_path: Path) -> None:
        self._profile_path = profile_path
        super().__init__(settings_cls, yaml_file=profile_path)

    def _read_files(self, files: Any, deep_merge: bool = False) -> dict[str, Any]:
        data = super()._read_files(files)
        local = local_overrides_path(self._profile_path)
        if local.is_file():
            overrides = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
            if not isinstance(overrides, dict):
                raise ConfigError(f"{local} must hold a mapping")
            data = merge_mappings(data, overrides)
        return data


def resolve_profile_path(profile: str, config_dir: Path | None = None) -> Path:
    """A profile is a name (``config/<name>.yaml``) or a path to a YAML file."""
    candidate = Path(profile)
    if candidate.suffix in {".yaml", ".yml"}:
        if not candidate.is_file():
            raise ConfigError(f"config file not found: {candidate}")
        return candidate.resolve()
    base = config_dir or Path(os.environ.get(CONFIG_DIR_ENV, "config"))
    path = base / f"{profile}.yaml"
    if not path.is_file():
        raise ConfigError(f"profile '{profile}' not found (looked for {path.resolve()})")
    return path.resolve()


def load_settings(
    profile: str | None = None, *, config_dir: Path | None = None, **overrides: Any
) -> Settings:
    """Load ``config/<profile>.yaml`` with env overrides applied on top.

    The profile defaults to ``$IMAGINER_PROFILE``, then ``dev``.
    """
    name = profile or os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE
    path = resolve_profile_path(name, config_dir)
    token = _yaml_path.set(path)
    try:
        return Settings(profile=name, **overrides)
    finally:
        _yaml_path.reset(token)
