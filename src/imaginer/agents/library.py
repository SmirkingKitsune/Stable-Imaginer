"""The agent library: a chat model plus a persona, listed like the model library.

**What an agent is.** A model as llama-server loads it (its GGUF, vision projector and chat
template, with its context size and sampling) on the chat engine it runs on, and a persona:
a display name, the system prompt it is told before every conversation, a portrait, the
tools it may call and whether the viewer shows its reasoning as it thinks.

**Running one.** A llama-server engine launched here runs the agent its ``agent`` names: its
config as it runs (``effective``) has the agent's files and context size over its own. Using
an agent on such an engine restarts it with the agent's model. An agent with no files of its
own (one on an engine started elsewhere, or shared by a linked instance) uses whatever that
engine runs; its persona still applies.

**Turns.** A message goes to the agent it names, else the conversation's, else the one the
default chat engine runs (``resolve``). The agent's engine answers with its persona: its
system prompt, its tools and its sampling. Each assistant message records the agent.

**Portraits.** By default the avatar of whoever uploaded its GGUF repo to Hugging Face. It
can paint a self-portrait with the ``set_portrait`` tool (the image stays in the conversation
it was painted in; a square copy becomes its portrait), and one can be uploaded or taken from
a conversation. Without any, the UI shows its monogram.

**Moving in.** Every chat engine gets an agent at startup if none runs on it: made from its
model files, or, for one started elsewhere, named after the model it reports.
"""

import asyncio
import contextlib
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, cast

import sqlalchemy as sa
from PIL import UnidentifiedImageError
from sqlalchemy.orm import aliased

from imaginer.agents import portraits
from imaginer.backends.common import expand_path
from imaginer.core.clock import utcnow
from imaginer.core.config import (
    BackendConfig,
    LinkedBackendConfig,
    LlamaCppBackendConfig,
    Settings,
    backend_kind,
)
from imaginer.core.ids import new_id
from imaginer.core.logging import get_logger
from imaginer.db.engine import Database
from imaginer.db.models import (
    AgentRecord,
    Conversation,
    GenerationJob,
    JobOutput,
    Message,
)
from imaginer.domain import models as dto
from imaginer.domain.enums import BackendKind, Capability, Role
from imaginer.domain.errors import (
    ConflictError,
    ImaginerError,
    InvalidRequestError,
    NotFoundError,
)
from imaginer.engines import launch
from imaginer.hub.refs import RefError, parse_ref
from imaginer.loras import files as names
from imaginer.services import covers

if TYPE_CHECKING:
    from imaginer.backends.registry import BackendRegistry
    from imaginer.engines.service import EngineService
    from imaginer.hub.service import HubService
    from imaginer.orchestration.turn_runner import TurnRunner
    from imaginer.services.artifacts import ArtifactService
    from imaginer.services.conversations import ConversationService
    from imaginer.tools.base import ToolRegistry

log = get_logger(__name__)

# (key, llama-server flag, what it is), in launch order.
FILES: tuple[tuple[str, str, str], ...] = (
    ("model", "-m", "model"),
    ("mmproj", "--mmproj", "vision projector"),
    ("chat_template", "--chat-template-file", "chat template"),
)
FILE_KEYS = tuple(key for key, _, _ in FILES)
# Sampling an agent may set, sent with each of its requests (over its engine's sampling).
SAMPLING: dict[str, type] = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "min_p": float,
    "repeat_penalty": float,
    "presence_penalty": float,
    "max_tokens": int,
}
VIEWERS = ("status", "reasoning")
# What a "Paint another" turn asks. Only set_portrait is offered, and it must be called.
PAINT_REQUEST = (
    "Paint yourself a new portrait, something that suits you. Call set_portrait now with a "
    "prompt of your own; you don't need to ask first."
)
ADOPT_WAIT_S = 5.0  # how long a request waits for startup's agents before going on
_CACHE_REPO = re.compile(r"models--([^\\/]+?)--([^\\/]+)")
_ID = re.compile(r"^[0-9a-z-]{1,64}$")


@dataclass(frozen=True, slots=True)
class Agent:
    id: str
    name: str
    engine: str | None
    files: Mapping[str, str]
    source: Mapping[str, Any] | None
    system_prompt: str | None
    viewer: str
    tools_off: frozenset[str]
    sampling: Mapping[str, Any]
    context: int | None
    portrait: Mapping[str, Any]
    notes: str | None
    tags: tuple[str, ...]
    pinned: bool
    created_at: datetime
    updated_at: datetime

    @property
    def own_files(self) -> bool:
        return bool(self.files.get("model"))

    @property
    def monogram(self) -> str:
        found = re.search(r"[A-Za-z0-9]", self.name)
        return found.group(0).upper() if found else "?"


@dataclass(frozen=True, slots=True)
class Persona:
    """What a turn takes from its agent."""

    agent_id: str
    name: str
    system_prompt: str | None  # None: the built-in one
    tools_off: frozenset[str]
    sampling: Mapping[str, Any]


@dataclass(slots=True)
class _Use:
    turns: int = 0
    images: int = 0
    last: datetime | None = None


@dataclass(slots=True)
class _Live:
    """What is happening with agents now, in memory."""

    turns: dict[str, int] = field(default_factory=dict)  # agent id -> running turns
    painting: dict[str, dto.AgentPainting] = field(default_factory=dict)


def _agent(row: AgentRecord) -> Agent:
    tools = row.tools or {}
    return Agent(
        id=row.id,
        name=row.name,
        engine=row.engine,
        files={str(k): str(v) for k, v in (row.files or {}).items() if v},
        source=dict(row.source) if row.source else None,
        system_prompt=row.system_prompt,
        viewer=row.viewer if row.viewer in VIEWERS else "reasoning",
        tools_off=frozenset(str(k) for k, on in tools.items() if on is False),
        sampling=dict(row.sampling or {}),
        context=row.context,
        portrait=dict(row.portrait or {}),
        notes=row.notes,
        tags=tuple(row.tags or ()),
        pinned=row.pinned,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class AgentLibrary:
    def __init__(self, settings: Settings, db: Database) -> None:
        self._settings = settings
        self._db = db
        self._agents: dict[str, Agent] = {}
        self.portraits_dir = settings.data_root / "agent_portraits"
        self._engines: EngineService | None = None
        self._registry: BackendRegistry | None = None
        self._hub: HubService | None = None
        self._artifacts: ArtifactService | None = None
        self._tools: ToolRegistry | None = None
        self._turns: TurnRunner | None = None
        self._conversations: ConversationService | None = None
        self._default_prompt: Callable[[], str] = lambda: ""
        self._lock = asyncio.Lock()  # one change to the agents at a time
        self._live = _Live()
        self._adopted = asyncio.Event()
        self._tasks: set[asyncio.Task[Any]] = set()

    async def load(self) -> None:
        """Read every agent. Done before the registry is built, so engines start out
        running their agents' models."""
        async with self._db.session() as session, session.begin():
            rows = (await session.execute(sa.select(AgentRecord))).scalars().all()
            self._agents = {row.id: _agent(row) for row in rows}

    def attach(
        self,
        *,
        engines: "EngineService",
        registry: "BackendRegistry",
        hub: "HubService",
        artifacts: "ArtifactService",
        tools: "ToolRegistry",
        turns: "TurnRunner",
        conversations: "ConversationService",
        default_prompt: Callable[[], str],
    ) -> None:
        self._engines = engines
        self._registry = registry
        self._hub = hub
        self._artifacts = artifacts
        self._tools = tools
        self._turns = turns
        self._conversations = conversations
        self._default_prompt = default_prompt
        hub.on_agent_install = self.install_file

    @property
    def engines(self) -> "EngineService":
        assert self._engines is not None, "attach() first"
        return self._engines

    def agent(self, agent_id: str | None) -> Agent | None:
        return self._agents.get(agent_id) if agent_id else None

    def _require(self, agent_id: str) -> Agent:
        agent = self._agents.get(agent_id)
        if agent is None:
            raise NotFoundError(f"no agent {agent_id!r}")
        return agent

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # --- engines --------------------------------------------------------------------------

    def effective(self, name: str, config: BackendConfig) -> BackendConfig:
        """An engine's config as it runs: for a llama-server with an agent, the agent's
        model files, chat template and context size over its own. An agent that is gone
        (or has no files) leaves the engine as it is configured."""
        if not isinstance(config, LlamaCppBackendConfig) or not config.agent:
            return config
        agent = self._agents.get(config.agent)
        if agent is None:
            log.warning("agents.agent_missing", engine=name, agent=config.agent)
            return config
        if config.launch is None or not agent.own_files:
            return config
        extra = list(config.launch.extra_args)
        template = agent.files.get("chat_template")
        if template:
            extra += ["--chat-template-file", template]
        if agent.context:
            extra += ["-c", str(agent.context)]
        mmproj = agent.files.get("mmproj")
        launch_cfg = config.launch.model_copy(
            update={
                "model": Path(agent.files["model"]),
                "mmproj": Path(mmproj) if mmproj else None,
                "extra_args": extra,
            }
        )
        return config.model_copy(update={"launch": launch_cfg})

    def _chat_engines(self) -> list[tuple[str, BackendConfig]]:
        out = []
        for name in self.engines.names():
            config = self.engines.own_config(name)
            if config.enabled and backend_kind(config) is BackendKind.LLM:
                out.append((name, config))
        return out

    def _engine_config(self, name: str | None) -> BackendConfig | None:
        if not name or self._engines is None:
            return None
        with contextlib.suppress(NotFoundError):
            return self._engines.own_config(name)
        return None

    def engine_of(self, agent: Agent) -> str | None:
        """The chat engine an agent runs on: its own, else the default."""
        return agent.engine or self._settings.default_backend("chat")

    @staticmethod
    def _launchable(config: BackendConfig | None) -> bool:
        return isinstance(config, LlamaCppBackendConfig) and config.launch is not None

    def runs_it(self, agent: Agent, engine: str | None = None) -> bool:
        """Whether its engine is set to run its model (always, for an agent without files of
        its own, or on an engine started elsewhere)."""
        config = self._engine_config(engine or self.engine_of(agent))
        if config is None:
            return False
        if not agent.own_files or not self._launchable(config):
            return True
        return getattr(config, "agent", None) == agent.id

    def default_agent(self) -> Agent | None:
        """The agent new prompts go to: the one the default chat engine is set to run (its
        ``agent``), else the first made for it."""
        engine = self._settings.default_backend("chat")
        if engine is None:
            return None
        config = self._engine_config(engine)
        chosen = self.agent(getattr(config, "agent", None))
        if chosen is not None:
            return chosen
        candidates = [
            a
            for a in self._agents.values()
            if self.engine_of(a) == engine and (not a.own_files or not self._launchable(config))
        ]
        candidates.sort(key=lambda a: (a.engine is None, a.created_at))
        return candidates[0] if candidates else None

    async def resolve(self, agent_id: str | None, conversation_agent: str | None) -> Agent | None:
        """The agent a turn goes to: the one named, else the conversation's (if it still
        exists), else the default one."""
        await self.wait_adopted()
        if agent_id:
            return self._require(agent_id)
        return self.agent(conversation_agent) or self.default_agent()

    def check_runs(self, agent: Agent) -> str:
        """Its engine, ready to take a turn as this agent: refused when that engine was
        launched here and is set to run another agent's model."""
        engine = self.engine_of(agent)
        if engine is None:
            raise InvalidRequestError(f"{agent.name} has no chat engine to run on")
        config = self._engine_config(engine)
        if config is None:
            raise InvalidRequestError(f"{agent.name} runs on {engine}, which isn't configured")
        if not self.runs_it(agent, engine):
            other = self.agent(getattr(config, "agent", None))
            runs = other.name if other is not None else "its own model"
            raise ConflictError(
                f"{engine} runs {runs}: switch it to {agent.name} first "
                f"(Use on {engine}, POST /v1/agents/{agent.id}/use)"
            )
        return engine

    def persona(self, agent: Agent) -> Persona:
        return Persona(
            agent_id=agent.id,
            name=agent.name,
            system_prompt=agent.system_prompt,
            tools_off=agent.tools_off,
            sampling=dict(agent.sampling),
        )

    def turn_started(self, agent_id: str) -> None:
        self._live.turns[agent_id] = self._live.turns.get(agent_id, 0) + 1

    def turn_ended(self, agent_id: str) -> None:
        left = self._live.turns.get(agent_id, 0) - 1
        if left > 0:
            self._live.turns[agent_id] = left
        else:
            self._live.turns.pop(agent_id, None)

    # --- moving in ------------------------------------------------------------------------

    def start_adopting(self, *, after: asyncio.Future[Any] | None = None) -> None:
        """Adopt the engines in the background, once ``after`` (the startup probe) is done:
        an engine started elsewhere has said what it runs by then."""
        self._spawn(self._adopt_in_background(after))

    async def _adopt_in_background(self, after: asyncio.Future[Any] | None) -> None:
        try:
            if after is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.shield(after), ADOPT_WAIT_S)
            await self.adopt_engines()
        except Exception:
            log.exception("agents.adopt_failed")
        finally:
            self._adopted.set()

    async def wait_adopted(self, limit_s: float = ADOPT_WAIT_S) -> None:
        if self._adopted.is_set():
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._adopted.wait(), limit_s)

    async def adopt_engines(self) -> list[str]:
        """Give every chat engine without an agent one: made from the model files of a
        llama-server launched here (which is then pointed at it), or named after the model
        an engine started elsewhere reports. Returns the agents made."""
        made: list[str] = []
        for name, config in self._chat_engines():
            if self._launchable(config):
                assert isinstance(config, LlamaCppBackendConfig) and config.launch is not None
                if config.agent and config.agent in self._agents:
                    continue
                if config.launch.model is None:
                    continue
                files = {"model": str(config.launch.model)}
                if config.launch.mmproj is not None:
                    files["mmproj"] = str(config.launch.mmproj)
                existing = next(
                    (
                        a
                        for a in self._agents.values()
                        if a.own_files and _norm(a.files["model"]) == _norm(files["model"])
                    ),
                    None,
                )
                agent = existing or await self._adopt(name, files, None)
                if existing is None:
                    made.append(agent.id)
                await self.engines.update(name, dto.EngineUpdate(agent=agent.id))
                log.info("agents.adopted", engine=name, agent=agent.id, name=agent.name)
                continue
            if any(a.engine == name for a in self._agents.values()):
                continue
            if name == self._settings.default_backend("chat") and any(
                a.engine is None and not a.own_files for a in self._agents.values()
            ):
                continue
            model = await self._reported_model(name, config)
            agent = await self._adopt(name, {}, model)
            made.append(agent.id)
            log.info("agents.adopted", engine=name, agent=agent.id, name=agent.name)
        return made

    async def _reported_model(self, name: str, config: BackendConfig) -> str | None:
        """The model an engine started elsewhere says it runs, as last probed."""
        if self._registry is None:
            return getattr(config, "model", None)
        with contextlib.suppress(Exception):
            info = await self._registry.describe_one(name)
            if info.model:
                return info.model
        return getattr(config, "model", None)

    async def _adopt(self, engine: str, files: dict[str, str], model: str | None) -> Agent:
        source = await asyncio.to_thread(_described, files.get("model"), files)
        name = _name_for(source, files.get("model") or model, engine)
        if model is None and not files:
            link = self._engine_config(engine)
            if isinstance(link, LinkedBackendConfig):
                name = f"{name} on {link.link}"
        return await self._insert(
            name=name,
            engine=engine,
            files=files,
            source=source,
            system_prompt=None,
            notes=f"Made from {engine}'s settings.",
        )

    async def _insert(
        self,
        *,
        name: str,
        engine: str | None,
        files: dict[str, str],
        source: dict[str, Any] | None,
        system_prompt: str | None,
        notes: str | None = None,
    ) -> Agent:
        now = utcnow()
        owner = _owner_of(source)
        row = AgentRecord(
            id=new_id(),
            name=name[:120],
            engine=engine,
            files=files or None,
            source=source,
            system_prompt=system_prompt,
            viewer="reasoning",
            tools=None,
            sampling=None,
            context=None,
            portrait={"hub": {"owner": owner}} if owner else None,
            notes=notes,
            pinned=False,
            created_at=now,
            updated_at=now,
        )
        async with self._lock, self._db.session() as session, session.begin():
            session.add(row)
            await session.flush()
            agent = self._agents[row.id] = _agent(row)
        if owner and self._hub is not None and self._settings.hub.enabled:
            self._spawn(self._quietly(self.fetch_hub_portrait(agent.id)))
        return agent

    @staticmethod
    async def _quietly(coro: Any) -> None:
        try:
            await coro
        except ImaginerError as exc:
            log.info("agents.portrait_unavailable", detail=exc.detail)
        except Exception:
            log.exception("agents.portrait_failed")

    # --- views ----------------------------------------------------------------------------

    async def overview(self) -> dto.AgentLibraryView:
        await self.wait_adopted()
        agents = sorted(self._agents.values(), key=lambda a: (not a.pinned, a.name.casefold()))
        usage = await self._usage()
        infos = await self._engine_infos()
        default = self.default_agent()
        items = [await asyncio.to_thread(self._item, a, usage, infos, default) for a in agents]
        engines = []
        for name, config in self._chat_engines():
            info = infos.get(name)
            engines.append(
                dto.AgentLibraryEngine(
                    name=name,
                    kind=_kind(config),
                    link=config.link if isinstance(config, LinkedBackendConfig) else None,
                    launchable=self._launchable(config),
                    running=self._engine_running(name, config, info),
                    agent=getattr(config, "agent", None),
                    default=name == self._settings.default_backend("chat"),
                    model=info.model if info is not None else None,
                )
            )
        tags = sorted({t for a in agents for t in a.tags}, key=str.casefold)
        return dto.AgentLibraryView(
            items=items,
            engines=engines,
            default=default.id if default is not None else None,
            tags=tags,
        )

    async def get(self, agent_id: str) -> dto.AgentDetail:
        agent = self._require(agent_id)
        usage = await self._usage(agent_id)
        infos = await self._engine_infos()
        item = await asyncio.to_thread(self._item, agent, usage, infos, self.default_agent())
        checks = await asyncio.to_thread(self._checks, agent)
        engine = self.engine_of(agent)
        config = self._engine_config(engine)
        tools = []
        if self._tools is not None:
            for tool in self._tools.all():
                tools.append(
                    dto.AgentTool(
                        name=tool.name,
                        description=tool.summary,
                        flags=tool.flags,
                        enabled=tool.name not in agent.tools_off,
                        available=tool.available(),
                    )
                )
        default_prompt = self._default_prompt()
        return dto.AgentDetail(
            **item.model_dump(),
            files=[
                dto.AgentFile(
                    key=cast(dto.AgentFileKey, key),
                    flag=flag,
                    what=what,
                    path=agent.files.get(key),
                    exists=check.exists if check else False,
                    size_bytes=check.size_bytes if check else None,
                    problem=check.problem if check else None,
                    required=key == "model" or (key == "mmproj" and _wants_mmproj(agent)),
                    source=dict(agent.source) if key == "model" and agent.source else None,
                )
                for key, flag, what in FILES
                for check in [checks.get(key)]
            ],
            system_prompt=agent.system_prompt or default_prompt,
            system_prompt_custom=agent.system_prompt is not None,
            default_system_prompt=default_prompt,
            tool_list=tools,
            sampling=dict(agent.sampling),
            engine_sampling=dict(getattr(config, "sampling", None) or {}),
            context=agent.context,
        )

    async def _engine_infos(self) -> dict[str, dto.BackendInfo]:
        if self._registry is None:
            return {}
        out = {}
        for name, _ in self._chat_engines():
            with contextlib.suppress(Exception):
                out[name] = await self._registry.describe_one(name)
        return out

    def _engine_running(
        self, name: str, config: BackendConfig, info: dto.BackendInfo | None
    ) -> bool:
        if self._launchable(config):
            return self.engines.running(name)
        return info is not None and info.status.value == "ok"

    def _checks(self, agent: Agent) -> dict[str, launch.FileCheck]:
        """Its files, checked on disk. Blocking."""
        out = {}
        for key, flag, what in FILES:
            path = agent.files.get(key)
            if not path:
                continue
            if key == "chat_template":  # a Jinja text file, not weights
                found = expand_path(path)
                size = found.stat().st_size if found.is_file() else None
                out[key] = launch.FileCheck(
                    key, flag, what, path, exists=size is not None, size_bytes=size,
                    problem=None if size is not None else "file not found",
                )  # fmt: skip
                continue
            out[key] = launch.check_file(key, flag, what, Path(path), engine="llamacpp")
        return out

    def _item(
        self,
        agent: Agent,
        usage: dict[str, _Use],
        infos: dict[str, dto.BackendInfo],
        default: Agent | None,
    ) -> dto.AgentItem:
        checks = self._checks(agent)
        engine = self.engine_of(agent)
        config = self._engine_config(engine)
        info = infos.get(engine) if engine else None
        problems = [f"{c.flag} {c.path}: {c.problem}" for c in checks.values() if c.problem]
        missing: list[dto.AgentFileKey] = []
        if self._launchable(config) and not agent.own_files:
            missing.append("model")
        if agent.own_files and _wants_mmproj(agent) and not agent.files.get("mmproj"):
            missing.append("mmproj")
        if config is None:
            problems.append(
                f"its engine {engine} isn't configured" if engine else "there is no chat engine"
            )
        runs = self.runs_it(agent, engine)
        caps = set(info.capabilities) if info is not None and runs else set()
        use = None
        if engine is not None and config is not None:
            running = runs and self._engine_running(engine, config, info)
            stale = False
            if running and self._launchable(config):
                now = self.engines.command(engine)
                stale = now is not None and now != self.engines.running_command(engine)
            other = self.agent(getattr(config, "agent", None)) if not runs else None
            use = dto.AgentEngineUse(
                engine=engine,
                kind=_kind(config),
                link=config.link if isinstance(config, LinkedBackendConfig) else None,
                launchable=self._launchable(config),
                runs_it=runs,
                running=running,
                status=info.status.value if info is not None else "unknown",
                default=engine == self._settings.default_backend("chat"),
                needs_restart=stale,
                runs_other=other.name if other is not None else None,
            )
        model_path = agent.files.get("model")
        model = model_path or (info.model if info is not None and runs else None)
        vision = bool(agent.files.get("mmproj")) or Capability.VISION_IN in caps
        has_tools = bool(
            self._tools is not None
            and any(t.name not in agent.tools_off for t in self._tools.all())
            and (not caps or Capability.TOOLS in caps)
        )
        use_stats = usage.get(agent.id, _Use())
        return dto.AgentItem(
            id=agent.id,
            name=agent.name,
            engine=engine,
            engine_set=agent.engine,
            model=model,
            quant=names.model_quant(model) if model else None,
            size_bytes=sum(c.size_bytes or 0 for c in checks.values()),
            family=_family(agent, model),
            vision=vision,
            tools=has_tools,
            own_files=agent.own_files,
            ready=not missing and not problems,
            missing=missing,
            problems=problems,
            use=use,
            default=default is not None and default.id == agent.id,
            working=agent.id in self._live.turns or agent.id in self._live.painting,
            portrait=self._portrait(agent),
            viewer=cast(dto.AgentViewer, agent.viewer),
            pinned=agent.pinned,
            tags=list(agent.tags),
            notes=agent.notes or "",
            source=dict(agent.source) if agent.source else None,
            turns=use_stats.turns,
            images=use_stats.images,
            last_used_at=use_stats.last,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
        )

    def _portrait(self, agent: Agent) -> dto.AgentPortrait:
        data = agent.portrait
        base = f"/v1/agents/{agent.id}/portraits"
        images: list[dto.AgentPortraitImage] = []
        hub = data.get("hub") or {}
        if hub.get("v"):
            images.append(
                dto.AgentPortraitImage(
                    key="hub",
                    source="hub",
                    url=f"{base}/hub?v={hub['v']}",
                    owner=hub.get("owner"),
                    created_at=_when(hub.get("fetched_at")),
                )
            )
        upload = data.get("upload") or {}
        if upload.get("v"):
            images.append(
                dto.AgentPortraitImage(
                    key="upload",
                    source="upload",
                    url=f"{base}/upload?v={upload['v']}",
                    conversation_id=upload.get("conversation_id"),
                    artifact_id=upload.get("artifact_id"),
                    label=upload.get("label"),
                    created_at=_when(upload.get("at")),
                )
            )
        selves = sorted(data.get("self") or [], key=lambda s: int(s.get("version", 0)))
        for entry in selves:
            version = int(entry.get("version", 0))
            images.append(
                dto.AgentPortraitImage(
                    key=f"self-{version}",
                    source="self",
                    url=f"{base}/self-{version}?v={entry.get('v', 0)}",
                    version=version,
                    prompt=entry.get("prompt"),
                    conversation_id=entry.get("conversation_id"),
                    artifact_id=entry.get("artifact_id"),
                    label=entry.get("label"),
                    created_at=_when(entry.get("at")),
                )
            )
        key = data.get("key")
        shown = next((i for i in images if i.key == key), None)
        origin = ""
        if shown is not None and shown.source == "hub":
            origin = f"HUB · {(shown.owner or '').upper()}".strip(" ·")
        elif shown is not None and shown.source == "self":
            origin = "SELF-PORTRAIT"
        elif shown is not None:
            origin = "UPLOADED"
        return dto.AgentPortrait(
            key=shown.key if shown else None,
            source=shown.source if shown else None,
            url=shown.url if shown else None,
            monogram=agent.monogram,
            origin=origin,
            hub_owner=hub.get("owner"),
            hub_fetched_at=_when(hub.get("fetched_at")),
            images=images,
            prompt=selves[-1].get("prompt") if selves else None,
            painting=self._live.painting.get(agent.id),
        )

    def portrait_file(self, agent_id: str, key: str) -> Path | None:
        path = portraits.path(self.portraits_dir, agent_id, key)
        return path if path is not None and path.is_file() else None

    async def _usage(self, agent_id: str | None = None) -> dict[str, _Use]:
        """Turns each agent answered (its first reply to each of your messages) and the
        images its calls made."""
        parent = aliased(Message)
        turns = (
            sa.select(Message.agent_id, sa.func.count(), sa.func.max(Message.created_at))
            .join(parent, parent.id == Message.parent_id)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(
                Message.agent_id.is_not(None),
                Message.role == Role.ASSISTANT,
                parent.role == Role.USER,
                Conversation.deleted_at.is_(None),
            )
            .group_by(Message.agent_id)
        )
        images = (
            sa.select(Message.agent_id, sa.func.count(JobOutput.artifact_id))
            .join(GenerationJob, GenerationJob.request_message_id == Message.id)
            .join(JobOutput, JobOutput.job_id == GenerationJob.id)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(Message.agent_id.is_not(None), Conversation.deleted_at.is_(None))
            .group_by(Message.agent_id)
        )
        if agent_id is not None:
            turns = turns.where(Message.agent_id == agent_id)
            images = images.where(Message.agent_id == agent_id)
        out: dict[str, _Use] = {}
        async with self._db.session() as session, session.begin():
            for aid, count, last in (await session.execute(turns)).all():
                use = out.setdefault(str(aid), _Use())
                use.turns, use.last = int(count), last
            for aid, count in (await session.execute(images)).all():
                out.setdefault(str(aid), _Use()).images = int(count)
        return out

    # --- changing agents ------------------------------------------------------------------

    async def create(self, body: dto.AgentCreate) -> dto.AgentDetail:
        engine = (body.engine or "").strip() or None
        if engine is not None:
            config = self._engine_config(engine)
            if config is None or backend_kind(config) is not BackendKind.LLM:
                raise InvalidRequestError(f"{engine} is not a chat engine")
        files: dict[str, str] = {}
        source: dict[str, Any] | None = None
        if body.path:
            files["model"] = str(expand_path(body.path.strip()))
            if body.mmproj:
                files["mmproj"] = str(expand_path(body.mmproj.strip()))
            source = await asyncio.to_thread(_described, files["model"], files)
        elif body.ref:
            try:
                ref = parse_ref(body.ref)
            except RefError as exc:
                raise InvalidRequestError(str(exc)) from exc
            source = {"repo": ref.repo_id, "revision": ref.revision, "path": ref.path}
            if ref.quant:
                source["quant"] = ref.quant
        name = " ".join((body.name or "").split()) or _name_for(
            source, files.get("model"), engine or "Agent"
        )
        prompt = (body.system_prompt or "").strip() or None
        agent = await self._insert(
            name=name, engine=engine, files=files, source=source, system_prompt=prompt
        )
        log.info("agents.created", agent=agent.id, name=agent.name, engine=engine)
        return await self.get(agent.id)

    async def update(self, agent_id: str, body: dto.AgentUpdate) -> dto.AgentDetail:
        self._require(agent_id)
        fields = body.model_fields_set
        async with self._lock, self._db.session() as session, session.begin():
            row = await self._row(session, agent_id)
            if "name" in fields and body.name is not None:
                text = " ".join(body.name.split())
                if not text:
                    raise InvalidRequestError("an agent needs a name")
                row.name = text
            if "engine" in fields:
                engine = (body.engine or "").strip() or None
                if engine is not None:
                    config = self._engine_config(engine)
                    if config is None or backend_kind(config) is not BackendKind.LLM:
                        raise InvalidRequestError(f"{engine} is not a chat engine")
                row.engine = engine
            if body.files is not None:
                merged = dict(row.files or {})
                for key, value in body.files.items():
                    text = (value or "").strip()
                    if text:
                        merged[key] = str(expand_path(text))
                    else:
                        merged.pop(key, None)
                if merged.get("model") != (row.files or {}).get("model"):
                    row.source = _source_of(merged.get("model")) or (
                        row.source if merged.get("model") else None
                    )
                row.files = merged or None
            if "system_prompt" in fields:
                row.system_prompt = (body.system_prompt or "").strip() or None
            if body.viewer is not None:
                row.viewer = body.viewer
            if body.tools is not None:
                known = {t.name for t in self._tools.all()} if self._tools is not None else set()
                unknown = sorted(set(body.tools) - known)
                if unknown:
                    raise InvalidRequestError(f"no tool named {', '.join(unknown)}")
                tools = dict(row.tools or {})
                for tool_name, on in body.tools.items():
                    if on:
                        tools.pop(tool_name, None)
                    else:
                        tools[tool_name] = False
                row.tools = tools or None
            if body.sampling is not None:
                row.sampling = _merged_sampling(dict(row.sampling or {}), body.sampling) or None
            if "context" in fields:
                row.context = body.context or None
            if "notes" in fields:
                row.notes = (body.notes or "").strip() or None
            if body.tags is not None:
                row.tags = _clean_tags(body.tags) or None
            if body.pinned is not None:
                row.pinned = body.pinned
            row.updated_at = utcnow()
            await session.flush()
            agent = self._agents[agent_id] = _agent(row)
        self._refresh_engines(agent_id)
        if _owner_of(agent.source) and not (agent.portrait.get("hub") or {}).get("v"):
            self._spawn(self._quietly(self.fetch_hub_portrait(agent_id)))
        return await self.get(agent_id)

    def _users(self, agent_id: str) -> list[str]:
        return [
            name
            for name, config in self._chat_engines()
            if isinstance(config, LlamaCppBackendConfig) and config.agent == agent_id
        ]

    def _refresh_engines(self, agent_id: str) -> None:
        for name in self._users(agent_id):
            self.engines.refresh(name)

    async def delete(self, agent_id: str) -> None:
        """Forget an agent (its files, and the messages it wrote, stay). Not one an engine
        runs."""
        self._require(agent_id)
        users = self._users(agent_id)
        if users:
            raise ConflictError(
                f"{', '.join(users)} runs it: give {'them' if len(users) > 1 else 'it'} "
                "another agent first"
            )
        async with self._lock, self._db.session() as session, session.begin():
            row = await session.get(AgentRecord, agent_id)
            if row is not None:
                await session.delete(row)
            self._agents.pop(agent_id, None)
        folder = self.portraits_dir / agent_id
        if _ID.fullmatch(agent_id):
            await asyncio.to_thread(_remove_folder, folder)

    async def use(self, agent_id: str, body: dto.AgentUse) -> dto.AgentDetail:
        """Run the agent's model on its engine. One launched here is restarted with it (or
        started, with ``start``); refused while a turn is using it. An agent with no files of
        its own, or on an engine started elsewhere, needs nothing done."""
        agent = self._require(agent_id)
        engine = self.engine_of(agent)
        config = self._engine_config(engine)
        if engine is None or config is None:
            raise InvalidRequestError(f"{agent.name} has no chat engine to run on")
        if not self._launchable(config) or not agent.own_files:
            # Nothing to load: on a llama-server it becomes the agent the engine answers as.
            if isinstance(config, LlamaCppBackendConfig) and config.agent != agent_id:
                await self.engines.update(engine, dto.EngineUpdate(agent=agent_id))
            if body.start and self._launchable(config) and not self.engines.running(engine):
                await self.engines.start(engine)
            return await self.get(agent_id)
        if self.engines.busy(engine):
            raise ConflictError(f"{engine} is answering: try again when it's done")
        item = await self.get(agent_id)
        if not item.ready:
            gaps = [*item.missing, *item.problems]
            raise InvalidRequestError(f"{agent.name} isn't ready: {'; '.join(gaps)}")
        running = self.engines.running(engine)
        switching = getattr(config, "agent", None) != agent_id
        if running and switching:
            await self.engines.stop(engine)
        if switching:
            await self.engines.update(engine, dto.EngineUpdate(agent=agent_id))
        if (running and switching) or (body.start and not self.engines.running(engine)):
            await self.engines.start(engine)
        log.info("agents.used", agent=agent_id, engine=engine, restarted=running and switching)
        return await self.get(agent_id)

    async def stop(self, agent_id: str) -> dto.AgentDetail:
        """Stop its engine, if it was launched here and runs it."""
        agent = self._require(agent_id)
        engine = self.engine_of(agent)
        config = self._engine_config(engine)
        if engine is None or not self._launchable(config):
            raise InvalidRequestError(f"{agent.name}'s engine isn't started here")
        if not self.runs_it(agent, engine):
            raise ConflictError(f"{engine} runs another agent")
        await self.engines.stop(engine)
        return await self.get(agent_id)

    async def install_file(
        self, agent_id: str, key: str, path: Path, source: dict[str, Any]
    ) -> None:
        """A file the Hub downloaded for one of an agent's files."""
        if agent_id not in self._agents:
            raise NotFoundError(f"no agent {agent_id!r}: it was deleted meanwhile")
        if key not in FILE_KEYS:
            raise InvalidRequestError(f"agents have no file {key!r}")
        async with self._lock, self._db.session() as session, session.begin():
            row = await self._row(session, agent_id)
            row.files = {**(row.files or {}), key: str(path)}
            if key == "model":
                previous = dict(row.source or {})
                row.source = {**previous, **source}
            row.updated_at = utcnow()
            await session.flush()
            self._agents[agent_id] = _agent(row)
        self._refresh_engines(agent_id)

    # --- portraits ------------------------------------------------------------------------

    async def set_portrait(self, agent_id: str, body: dto.AgentPortraitSet) -> dto.AgentDetail:
        agent = self._require(agent_id)
        if body.refresh:
            await self.fetch_hub_portrait(agent_id, use=True)
        elif body.artifact_id:
            assert self._artifacts is not None
            artifact, data = await self._artifacts.read_bytes(body.artifact_id)
            await self._store_portrait(
                agent_id,
                "upload",
                data,
                {
                    "conversation_id": artifact.conversation_id,
                    "artifact_id": artifact.id,
                    "label": artifact.label,
                },
            )
        elif body.key:
            key = None if body.key == "none" else body.key
            if key is not None and key not in {i.key for i in self._portrait(agent).images}:
                raise InvalidRequestError(f"{agent.name} has no portrait {key!r}")
            await self._update_portrait(agent_id, lambda p: {**p, "key": key})
        else:
            raise InvalidRequestError("say which portrait: key, artifact_id, or refresh")
        return await self.get(agent_id)

    async def upload_portrait(self, agent_id: str, data: bytes) -> dto.AgentDetail:
        self._require(agent_id)
        if len(data) > portraits.PORTRAIT_MAX_BYTES:
            raise InvalidRequestError("that image is too large for a portrait")
        await self._store_portrait(agent_id, "upload", data, {})
        return await self.get(agent_id)

    async def fetch_hub_portrait(self, agent_id: str, *, use: bool = False) -> None:
        """The avatar of whoever uploaded the agent's model repo to Hugging Face. It becomes
        the portrait when the agent has none yet (or with ``use``)."""
        agent = self._require(agent_id)
        owner = _owner_of(agent.source)
        if owner is None:
            raise InvalidRequestError(f"{agent.name}'s model didn't come from Hugging Face")
        if self._hub is None:
            raise InvalidRequestError("Hugging Face isn't available here")
        data = await self._hub.avatar(owner, portraits.PORTRAIT_MAX_BYTES)
        if data is None:
            raise InvalidRequestError(f"{owner} has no avatar on Hugging Face")
        await self._store_portrait(
            agent_id,
            "hub",
            data,
            {"owner": owner, "fetched_at": utcnow().isoformat()},
            use=use or not agent.portrait.get("key"),
        )

    async def record_self_portrait(
        self,
        agent_id: str,
        data: bytes,
        *,
        prompt: str,
        conversation_id: str,
        artifact_id: str,
        label: str,
    ) -> int:
        """A self-portrait the agent painted: kept as its next version, and shown."""
        self._require(agent_id)
        self._live.painting.pop(agent_id, None)
        try:
            image = await asyncio.to_thread(portraits.square, data)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidRequestError(f"the painting can't be read: {exc}") from exc
        async with self._lock:
            agent = self._require(agent_id)
            version = 1 + max(
                (int(s.get("version", 0)) for s in agent.portrait.get("self") or []), default=0
            )
            target = portraits.path(self.portraits_dir, agent_id, f"self-{version}")
            assert target is not None
            await asyncio.to_thread(_write, target, image)
            entry = {
                "version": version,
                "v": _stamp(),
                "prompt": prompt,
                "conversation_id": conversation_id,
                "artifact_id": artifact_id,
                "label": label,
                "at": utcnow().isoformat(),
            }

            def change(p: dict[str, Any]) -> dict[str, Any]:
                return {**p, "self": [*(p.get("self") or []), entry], "key": f"self-{version}"}

            await self._write_portrait(agent_id, change)
        log.info("agents.self_portrait", agent=agent_id, version=version)
        return version

    async def _store_portrait(
        self,
        agent_id: str,
        key: str,
        data: bytes,
        meta: dict[str, Any],
        *,
        use: bool = True,
    ) -> None:
        try:
            image = await asyncio.to_thread(portraits.square, data)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidRequestError(f"that image can't be read: {exc}") from exc
        target = portraits.path(self.portraits_dir, agent_id, key)
        assert target is not None
        await asyncio.to_thread(_write, target, image)
        stamp = _stamp()
        if key == "upload":
            meta = {**meta, "at": utcnow().isoformat()}

        def change(p: dict[str, Any]) -> dict[str, Any]:
            out = {**p, key: {**meta, "v": stamp}}
            if use:
                out["key"] = key
            return out

        await self._update_portrait(agent_id, change)

    async def _update_portrait(
        self, agent_id: str, change: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        async with self._lock:
            await self._write_portrait(agent_id, change)

    async def _write_portrait(
        self, agent_id: str, change: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        async with self._db.session() as session, session.begin():
            row = await self._row(session, agent_id)
            row.portrait = change(dict(row.portrait or {}))
            row.updated_at = utcnow()
            await session.flush()
            self._agents[agent_id] = _agent(row)

    # --- painting a self-portrait ---------------------------------------------------------

    def painting_started(self, agent_id: str, conversation_id: str) -> None:
        self._live.painting[agent_id] = dto.AgentPainting(conversation_id=conversation_id)

    def painting_progress(self, agent_id: str, job: dto.Job) -> None:
        painting = self._live.painting.get(agent_id)
        if painting is None:
            return
        progress = job.progress or {}
        painting.job_id = job.id
        painting.status = "queued" if job.status.value == "queued" else "running"
        painting.step = progress.get("step")
        painting.total_steps = progress.get("total_steps")

    def painting_ended(self, agent_id: str) -> None:
        self._live.painting.pop(agent_id, None)

    async def paint(self, agent_id: str) -> dto.AgentPainted:
        """Have the agent paint another self-portrait: a new conversation in which it is asked
        to, and may only call set_portrait. The painting stays in that conversation."""
        agent = self._require(agent_id)
        if agent_id in self._live.painting:
            raise ConflictError(f"{agent.name} is painting a portrait already")
        assert self._conversations is not None and self._turns is not None
        self.check_runs(agent)
        conversation = await self._conversations.create(
            dto.ConversationCreate(
                title=f"Self-portrait · {agent.name}", metadata={"agent": agent_id}
            )
        )
        try:
            await self._turns.start(
                conversation.id,
                dto.MessageCreate(content=PAINT_REQUEST, agent=agent_id),
                only_tools=frozenset({"set_portrait"}),
            )
        except BaseException:
            with contextlib.suppress(Exception):
                await self._conversations.delete(conversation.id)
            raise
        self._live.painting.setdefault(agent_id, dto.AgentPainting(conversation_id=conversation.id))
        return dto.AgentPainted(agent=await self.get(agent_id), conversation_id=conversation.id)

    async def _row(self, session: Any, agent_id: str) -> AgentRecord:
        row = await session.get(AgentRecord, agent_id)
        if row is None:
            raise NotFoundError(f"no agent {agent_id!r}")
        assert isinstance(row, AgentRecord)
        return row


# --- helpers ------------------------------------------------------------------------------


def _kind(config: BackendConfig) -> Any:
    if isinstance(config, LinkedBackendConfig):
        return "linked"
    if isinstance(config, LlamaCppBackendConfig) and config.launch is not None:
        return "local"
    return "remote"


def _source_of(path: str | None) -> dict[str, Any] | None:
    """The Hub repo a model file is in, when its path says: the Hugging Face cache
    (``models--<owner>--<name>``) or our models folder (``<owner>/<name>/…``, told by the
    repo-style name)."""
    if not path:
        return None
    text = str(path)
    found = _CACHE_REPO.search(text)
    if found:
        return {"repo": f"{found.group(1)}/{found.group(2)}", "path": PurePath(text).name}
    parts = PurePath(text.replace("\\", "/")).parts
    for i in range(len(parts) - 2):
        if re.search(r"-gguf$", parts[i + 1], re.IGNORECASE):
            return {"repo": f"{parts[i]}/{parts[i + 1]}", "path": "/".join(parts[i + 2 :])}
    return None


def _described(model: str | None, files: Mapping[str, str]) -> dict[str, Any] | None:
    """Where a model file came from (``_source_of``), and whether it sees images: a vision
    projector (``mmproj*.gguf``) beside it says so. Blocking."""
    source = _source_of(model)
    if model and not files.get("mmproj"):
        with contextlib.suppress(OSError):
            folder = expand_path(model).parent
            if any(
                f.name.lower().startswith("mmproj") and f.suffix.lower() == ".gguf"
                for f in folder.iterdir()
            ):
                source = {**(source or {}), "vision": True}
    return source


def _owner_of(source: Mapping[str, Any] | None) -> str | None:
    repo = str((source or {}).get("repo") or "")
    owner = repo.split("/", 1)[0] if "/" in repo else ""
    return owner or None


def _name_for(source: Mapping[str, Any] | None, model: str | None, fallback: str) -> str:
    """A name for an agent: its repo's name without "-GGUF", else its model file's."""
    repo = str((source or {}).get("repo") or "")
    if "/" in repo:
        name = re.sub(r"[-_]gguf$", "", repo.split("/", 1)[1], flags=re.IGNORECASE)
        if name:
            return name
    if model:
        return names.model_name(model)
    return fallback


def _family(agent: Agent, model: str | None) -> str | None:
    """Its model's family as a tag: "Qwen3-VL-8B-Instruct" -> "Qwen3-VL"."""
    base = _name_for(agent.source, model, "") if (agent.source or model) else ""
    found = re.match(r"^([A-Za-z]+[\d.]*(?:[-_ ]?(?:VL|Vision|Coder|Math|A\d+B))?)", base)
    return found.group(1) if found else None


def _wants_mmproj(agent: Agent) -> bool:
    """Whether its model is known to need a vision projector: its repo offers one."""
    return bool((agent.source or {}).get("vision"))


def _merged_sampling(current: dict[str, Any], given: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in given.items():
        kind = SAMPLING.get(key)
        if kind is None:
            raise InvalidRequestError(
                f"unknown sampling setting {key!r}: one of {', '.join(SAMPLING)}"
            )
        if value is None or value == "":
            current.pop(key, None)
            continue
        try:
            current[key] = kind(value)
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(f"sampling.{key}: {value!r} is not a number") from exc
    return current


def _clean_tags(tags: Iterable[str]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        text = " ".join(str(tag).split())[:32]
        if text and text.casefold() not in {t.casefold() for t in out}:
            out.append(text)
    return out


def _norm(path: str) -> str:
    return str(expand_path(path)).replace("\\", "/").casefold()


def _stamp() -> int:
    return int(time.time() * 1000)


def _when(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    with contextlib.suppress(ValueError):
        return datetime.fromisoformat(value)
    return None


def _write(path: Path, data: bytes) -> None:
    covers.write_atomic(path, data)


def _remove_folder(folder: Path) -> None:
    if not folder.is_dir():
        return
    for child in folder.iterdir():
        with contextlib.suppress(OSError):
            child.unlink()
    with contextlib.suppress(OSError):
        folder.rmdir()


__all__ = [
    "FILES",
    "PAINT_REQUEST",
    "SAMPLING",
    "Agent",
    "AgentLibrary",
    "Persona",
]
