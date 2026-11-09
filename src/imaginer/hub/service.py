"""The Hub, as the API, the Engines screen and the CLI use it.

A repo is shown for one engine: its variants are matched to that engine's model settings,
each setting with a suggestion (the quant a ``repo:QUANT`` reference asks for, else the one
the setting uses now, else a good default), and every variant says whether a copy is on
this machine already and which settings point at it. A download sets the engine setting it
was started for once it's done, the same way an edit on the Engines screen does.

An sd-server launched here and a Diffusers backend also get a ``lora`` role: the repo's
LoRAs (only ``.safetensors`` ones for Diffusers), installed into the engine's LoRA folder as
``<owner>/<name>/<path in the repo>``, where it lists them for requests without a restart.
An engine with no LoRA folder yet gets one in the models folder
(``launch.DEFAULT_LORA_FOLDER``) with its first LoRA; an sd-server then needs a restart to
load it. A LoRA already on this machine (the models folder, the HF cache) is copied in
rather than downloaded again.
"""

import asyncio
import contextlib
import os
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import httpx

from imaginer.backends import diffusers_inproc
from imaginer.backends.diffusers_inproc import PIPELINE_CLASS
from imaginer.core.config import (
    BackendConfig,
    DiffusersBackendConfig,
    LlamaCppBackendConfig,
    SdCppBackendConfig,
    Settings,
)
from imaginer.domain import models as dto
from imaginer.domain.errors import HubError, ImaginerError, InvalidRequestError, NotFoundError
from imaginer.engines import launch
from imaginer.engines.service import EngineService
from imaginer.hub.client import (
    HubClient,
    RepoInfo,
    RepoSummary,
    find_token,
    hub_cache_dir,
    resolve_endpoint,
)
from imaginer.hub.downloads import DownloadJob, DownloadManager
from imaginer.hub.layout import (
    Pipeline,
    RepoLayout,
    Variant,
    classify,
    is_adapter,
    loras_in,
    quant_of,
    ranked,
    roles_for,
    shard_base,
    suggest,
)
from imaginer.hub.local import LocalCopy, find_copy, partial_bytes, repo_dir, safe_relpath
from imaginer.hub.refs import HubRef, RefError, parse_ref

MODEL_ID = "model_id"  # the "launch setting" of a Diffusers backend
LORA = "lora"  # the role of an engine's LoRAs: installed into its LoRA folder
# The setting a LoRA install sets when the engine has no LoRA folder yet, by engine type.
LORA_FOLDERS = {"sdcpp": "lora_model_dir", "diffusers": "lora_dir"}
_LOCAL_TYPES = (LlamaCppBackendConfig, SdCppBackendConfig, DiffusersBackendConfig)
HubFormat = Literal["gguf", "diffusers", "safetensors"]
FORMATS: tuple[HubFormat, ...] = ("gguf", "diffusers", "safetensors")
SORTS = {"downloads": "downloads", "trending": "trendingScore", "likes": "likes",
         "updated": "lastModified"}  # fmt: skip


class HubService:
    def __init__(
        self,
        settings: Settings,
        engines: EngineService,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._engines = engines
        self.client = HubClient(
            resolve_endpoint(settings.hub.endpoint), self._token, transport=transport
        )
        self.downloads = DownloadManager(self.client, on_success=self._assign)
        # Told where a LoRA from the Hub goes, and what its card says (the LoRA library).
        self.on_lora_install: Callable[[Path, dict[str, Any]], Awaitable[None]] | None = None

    def _token(self) -> str | None:
        return find_token(self._settings.hub.token_env).token

    @property
    def models_dir(self) -> Path:
        return self._settings.models_dir

    @property
    def cache_dir(self) -> Path | None:
        return hub_cache_dir() if self._settings.hub.use_hf_cache else None

    async def update(self, body: dto.HubSettingsUpdate) -> dto.HubStatus:
        """Change where downloads go; saved to the profile's .local.yaml like the Engines
        screen's edits. Downloads already running finish where they started."""
        if body.models_dir is not None:
            path = await asyncio.to_thread(_models_folder, body.models_dir)
            self._settings.hub.models_dir = path
            await self._engines.save_settings({"hub": {"models_dir": str(path)}})
        return self.status()

    def _require_enabled(self) -> None:
        if not self._settings.hub.enabled:
            raise InvalidRequestError("Hugging Face Hub access is off (hub.enabled: false)")

    async def aclose(self) -> None:
        await self.downloads.aclose()
        await self.client.aclose()

    # --- status and search ------------------------------------------------------------

    def status(self) -> dto.HubStatus:
        credentials = find_token(self._settings.hub.token_env)
        cache = self.cache_dir
        return dto.HubStatus(
            enabled=self._settings.hub.enabled,
            endpoint=self.client.endpoint,
            token=credentials.token is not None,
            token_source=credentials.source,
            models_dir=str(self.models_dir),
            hf_cache=str(cache) if cache is not None else None,
        )

    async def search(
        self,
        query: str = "",
        *,
        fmt: str | None = None,
        base_model: str | None = None,
        relation: str | None = None,
        task: str | None = None,
        sort: str = "downloads",
        limit: int = 30,
    ) -> list[dto.HubRepoSummary]:
        self._require_enabled()
        filters: list[str] = []
        if fmt in FORMATS:
            filters.append(fmt)
        if base_model:
            try:
                base = parse_ref(base_model).repo_id
            except RefError as exc:
                raise InvalidRequestError(str(exc)) from exc
            filters.append(f"base_model:{relation}:{base}" if relation else f"base_model:{base}")
        found = await self.client.search(
            query.strip(),
            filters=filters,
            pipeline_tag=task,
            sort=SORTS.get(sort, "downloads"),
            limit=limit,
        )
        return [self._summary(r) for r in found]

    def _summary(self, repo: RepoSummary) -> dto.HubRepoSummary:
        tags = set(repo.tags)
        return dto.HubRepoSummary(
            id=repo.id,
            url=self.client.repo_url(repo.id),
            task=repo.pipeline_tag,
            library=repo.library,
            downloads=repo.downloads,
            likes=repo.likes,
            gated=repo.gated,
            updated_at=repo.updated_at,
            formats=[f for f in FORMATS if f in tags],
        )

    # --- one repo ---------------------------------------------------------------------------

    async def repo(self, reference: str, *, engine: str | None = None) -> dto.HubRepo:
        self._require_enabled()
        ref = _parse(reference)
        config = self._engine_config(engine) if engine else None
        info, layout = await self._load(ref)
        return await asyncio.to_thread(self._repo_view, ref, info, layout, engine, config)

    async def _load(self, ref: HubRef) -> tuple[RepoInfo, RepoLayout]:
        info = await self.client.repo(ref.repo_id, ref.revision)
        model_index = None
        if any(f.path == "model_index.json" for f in info.files):
            with contextlib.suppress(HubError, NotFoundError):
                found = await self.client.json_file(
                    ref.repo_id, info.sha or ref.revision, "model_index.json"
                )
                model_index = found if isinstance(found, dict) else None
        layout = classify(
            info.files,
            repo_id=ref.repo_id,
            pipeline_tag=info.summary.pipeline_tag,
            model_index=model_index,
            adapter=is_adapter(info.summary.tags),
        )
        return info, layout

    def _repo_view(
        self,
        ref: HubRef,
        info: RepoInfo,
        layout: RepoLayout,
        engine: str | None,
        config: BackendConfig | None,
    ) -> dto.HubRepo:
        in_use = self._in_use()
        lora_dir = self._lora_dir(config)
        variants = [
            self._lora_variant(ref.repo_id, v, lora_dir, f"{engine}.{LORA}")
            if lora_dir is not None and v.component == "lora"
            else self._variant(ref.repo_id, info.sha, v, in_use)
            for v in layout.variants
        ]
        pipeline = (
            self._pipeline(ref.repo_id, info.sha, layout.pipeline, in_use)
            if layout.pipeline is not None
            else None
        )
        roles = self._roles(ref, layout, config) if config is not None else []
        notes = list(layout.notes)
        # (A transformers checkpoint's own note already says what to look for instead.)
        nothing = not any(r.candidates for r in roles) and layout.kind != "transformers"
        if isinstance(config, LlamaCppBackendConfig | SdCppBackendConfig) and nothing:
            notes.append(_nothing_fits(config, layout))
        summary = info.summary
        return dto.HubRepo(
            id=ref.repo_id,
            url=self.client.repo_url(ref.repo_id),
            revision=ref.revision,
            sha=info.sha,
            task=summary.pipeline_tag,
            library=summary.library,
            gated=summary.gated,
            downloads=summary.downloads,
            likes=summary.likes,
            base_models=list(info.base_models),
            kind=layout.kind,
            notes=notes,
            variants=variants,
            pipeline=pipeline,
            engine=engine,
            roles=roles,
            quant=ref.quant,
            path=ref.path,
        )

    def _variant(
        self, repo_id: str, sha: str | None, v: Variant, in_use: dict[str, list[str]]
    ) -> dto.HubVariant:
        copy = find_copy(
            repo_id, v.files, models_dir=self.models_dir, cache_dir=self.cache_dir, commit=sha
        )
        return dto.HubVariant(
            id=v.id,
            name=v.name,
            component=v.component,
            format=v.format,
            quant=v.quant,
            size_bytes=v.size,
            shards=len(v.files) - v.indexed,
            files=[dto.HubFile(path=f.path, size_bytes=f.size) for f in v.files],
            local=_local(copy, copy.path if copy else None, in_use),
            partial_bytes=0
            if copy
            else partial_bytes(repo_id, v.files, models_dir=self.models_dir),
        )

    def _lora_variant(self, repo_id: str, v: Variant, lora_dir: Path, role: str) -> dto.HubVariant:
        """A LoRA: what counts is its copy in the engine's LoRA folder. A copy anywhere
        else is what an install copies in."""
        view = self._variant(repo_id, None, v, {})
        copy = find_copy(repo_id, v.files, models_dir=lora_dir, cache_dir=None)
        local = None
        if copy is not None and copy.source == "models_dir":
            local = dto.HubLocal(source="lora_dir", path=str(copy.path), in_use=[role])
        return view.model_copy(
            update={
                "local": local,
                "copy_from": None if local else view.local,
                "partial_bytes": 0
                if local
                else partial_bytes(repo_id, v.files, models_dir=lora_dir),
            }
        )

    def _lora_dir(self, config: BackendConfig | None) -> Path | None:
        """Where an engine that takes LoRAs from a folder here gets them, or would: an
        sd-server launched here, a Diffusers backend."""
        if isinstance(config, DiffusersBackendConfig) or (
            isinstance(config, SdCppBackendConfig) and config.launch is not None
        ):
            return launch.lora_dir(config, self._settings)
        return None

    def _pipeline(
        self, repo_id: str, sha: str | None, p: Pipeline, in_use: dict[str, list[str]]
    ) -> dto.HubPipeline:
        problem = _pipeline_problem(p)
        copy = find_copy(
            repo_id, p.files, models_dir=self.models_dir, cache_dir=self.cache_dir, commit=sha
        )
        return dto.HubPipeline(
            pipeline_class=p.pipeline_class,
            components=list(p.components),
            size_bytes=p.size,
            file_count=len(p.files),
            compatible=problem is None,
            problem=problem,
            local=_local(copy, copy.root if copy else None, in_use),
            partial_bytes=0
            if copy
            else partial_bytes(repo_id, p.files, models_dir=self.models_dir),
        )

    def _roles(self, ref: HubRef, layout: RepoLayout, config: BackendConfig) -> list[dto.HubRole]:
        if isinstance(config, DiffusersBackendConfig):
            pipeline = dto.HubRole(key=MODEL_ID, what="Diffusers pipeline", current=config.model_id)
            return [pipeline, self._lora_role(layout, config)]
        fitting = roles_for(config.type, layout)
        named = PurePosixPath(ref.path) if ref.path else None
        roles = []
        for check in launch.check_files(config):
            candidates = ranked(fitting.get(check.key, []), key=check.key)
            current_quant = (
                quant_of(shard_base(Path(check.path).name)) if check.path is not None else None
            )
            pick = next(
                (v for v in candidates if named is not None and PurePosixPath(v.id) == named), None
            ) or suggest(candidates, key=check.key, quant=ref.quant or current_quant)
            roles.append(
                dto.HubRole(
                    key=check.key,
                    flag=check.flag,
                    what=check.what,
                    current=check.path,
                    candidates=[v.id for v in candidates],
                    suggested=pick.id if pick else None,
                )
            )
        if isinstance(config, SdCppBackendConfig) and config.launch is not None:
            roles.append(self._lora_role(layout, config))
        return roles

    def _lora_role(
        self, layout: RepoLayout, config: SdCppBackendConfig | DiffusersBackendConfig
    ) -> dto.HubRole:
        current = launch.configured_lora_dir(config)
        return dto.HubRole(
            key=LORA,
            flag="--lora-model-dir" if isinstance(config, SdCppBackendConfig) else "lora_dir",
            what="LoRAs",
            current=str(current) if current is not None else None,
            candidates=[v.id for v in _loras_for(config, layout)],
            folder=str(launch.lora_dir(config, self._settings)),
        )

    def _in_use(self) -> dict[str, list[str]]:
        """Resolved path -> the '<engine>.<setting>'s pointed at it."""
        found: dict[str, list[str]] = defaultdict(list)
        for name, config in self._settings.backends.items():
            if isinstance(config, DiffusersBackendConfig):
                found[_key(config.model_id)].append(f"{name}.{MODEL_ID}")
            for check in launch.check_files(config) if launch.file_specs(config) else []:
                if check.path is not None:
                    found[_key(check.path)].append(f"{name}.{check.key}")
        return found

    def _engine_config(self, name: str) -> BackendConfig:
        config = self._settings.backends.get(name)
        if config is None:
            raise NotFoundError(f"no backend named {name!r}")
        if not isinstance(config, _LOCAL_TYPES):
            raise InvalidRequestError(f"{name} runs on a linked instance; its models live there")
        return config

    # --- downloads --------------------------------------------------------------------------

    async def download(self, body: dto.HubDownloadRequest) -> dto.HubDownload:
        self._require_enabled()
        ref = _parse(body.repo)
        if body.revision:
            ref = HubRef(ref.repo_id, body.revision, ref.path, ref.quant)
        assign: tuple[str, str] | None = None
        source: Path | None = None  # a copy on this machine to copy instead of downloading
        config = None
        if body.engine is not None:
            config = self._engine_config(body.engine)
            if body.key is None:
                raise InvalidRequestError("say which setting to set: key")
            assign = (body.engine, body.key)
        info, layout = await self._load(ref)
        commit = info.sha or ref.revision
        root = repo_dir(self.models_dir, ref.repo_id)
        if body.variant is not None:
            variant = layout.variant(body.variant)
            if variant is None:
                raise NotFoundError(f"{ref.repo_id} has no model file {body.variant!r}")
            if config is not None and body.key == LORA:
                lora_dir = self._lora_dir(config)
                if lora_dir is None:
                    raise InvalidRequestError(
                        "only an sd-server launched here or a Diffusers backend takes LoRAs "
                        "from the Hub"
                    )
                assert isinstance(config, SdCppBackendConfig | DiffusersBackendConfig)
                assert body.engine is not None
                if variant not in _loras_for(config, layout):
                    what = "sd-server" if isinstance(config, SdCppBackendConfig) else "Diffusers"
                    raise InvalidRequestError(f"{variant.id} is not a LoRA {what} can apply")
                root = repo_dir(lora_dir, ref.repo_id)
                has_folder = launch.configured_lora_dir(config) is not None
                assign = None if has_folder else (body.engine, LORA_FOLDERS[config.type])
                copy = await asyncio.to_thread(
                    find_copy,
                    ref.repo_id,
                    variant.files,
                    models_dir=self.models_dir,
                    cache_dir=self.cache_dir,
                )
                if copy is not None and copy.root != root:
                    source = copy.root
                if self.on_lora_install is not None:
                    await self.on_lora_install(
                        root / safe_relpath(variant.id),
                        {
                            "repo": ref.repo_id,
                            "revision": commit,
                            "words": info.instance_prompt,
                            "base_models": list(info.base_models),
                        },
                    )
            elif config is not None:
                _check_fit(config, body.key, layout, variant)
            files, label, target = variant.files, variant.name, root / safe_relpath(variant.id)
        else:
            if layout.pipeline is None:
                raise InvalidRequestError(
                    f"{ref.repo_id} is not a Diffusers pipeline: name a variant to download"
                )
            if config is not None:
                if body.key != MODEL_ID or not isinstance(config, DiffusersBackendConfig):
                    raise InvalidRequestError(
                        f"a Diffusers pipeline can only be a diffusers backend's {MODEL_ID}"
                    )
                problem = _pipeline_problem(layout.pipeline)
                if problem is not None:
                    raise InvalidRequestError(f"{ref.repo_id} is {problem}")
            files, label, target = layout.pipeline.files, f"{ref.repo_id} (Diffusers)", root
        job = self.downloads.start(
            repo_id=ref.repo_id,
            revision=commit,
            label=label,
            files=files,
            root=root,
            target=target,
            assign=assign,
            source=source,
        )
        return download_view(job)

    async def card(self, repo_id: str, revision: str = "main") -> RepoInfo:
        """A repo's card, as the Hub reports it."""
        self._require_enabled()
        return await self.client.repo(repo_id, revision)

    async def fetch(self, repo_id: str, revision: str, path: str, max_bytes: int) -> bytes:
        """A small file from a repo (a card's image), up to ``max_bytes``."""
        self._require_enabled()
        body = bytearray()
        async with self.client.open(repo_id, revision, path) as resp:
            async for chunk in resp.aiter_bytes():
                body += chunk
                if len(body) > max_bytes:
                    raise InvalidRequestError(f"{path} in {repo_id} is over {max_bytes} bytes")
        return bytes(body)

    def list_downloads(self) -> list[dto.HubDownload]:
        return [download_view(j) for j in self.downloads.jobs()]

    def cancel(self, job_id: str) -> dto.HubDownload:
        return download_view(self.downloads.cancel(job_id))

    async def _assign(self, job: DownloadJob) -> None:
        """Point each setting the download was asked for at it; one that can't be set (its
        engine was removed meanwhile) doesn't stop the others."""
        failed = []
        for engine, key in job.assign:
            if key == MODEL_ID:
                update = dto.EngineUpdate(model_id=str(job.target))
            elif key in LORA_FOLDERS.values():
                # A LoRA's root is <LoRA folder>/<owner>/<name>.
                folder = str(job.root.parents[1])
                if key == "lora_dir":
                    update = dto.EngineUpdate(lora_dir=folder)
                else:
                    update = dto.EngineUpdate(launch=dto.EngineLaunchUpdate(lora_model_dir=folder))
            else:
                update = dto.EngineUpdate(
                    launch=dto.EngineLaunchUpdate(files={key: str(job.target)})
                )
            try:
                await self._engines.update(engine, update)
            except ImaginerError as exc:
                failed.append(f"{engine}.{key}: {exc.detail}")
        if failed:
            raise InvalidRequestError("; ".join(failed))


def download_view(job: DownloadJob) -> dto.HubDownload:
    return dto.HubDownload(
        id=job.id,
        repo=job.repo_id,
        revision=job.revision,
        label=job.label,
        state=job.state,
        total_bytes=job.total_bytes,
        done_bytes=job.done_bytes,
        file_count=len(job.files),
        file_index=job.file_index,
        current_file=job.current,
        target=str(job.target),
        assign=[f"{engine}.{key}" for engine, key in job.assign],
        speed_bps=job.speed_bps,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        warning=job.warning,
    )


def _loras_for(
    config: SdCppBackendConfig | DiffusersBackendConfig, layout: RepoLayout
) -> list[Variant]:
    """The repo's LoRAs the engine can apply: Diffusers loads only safetensors files."""
    if isinstance(config, DiffusersBackendConfig):
        allowed = diffusers_inproc.LORA_EXTENSIONS
        return [v for v in loras_in(layout) if PurePosixPath(v.id).suffix.lower() in allowed]
    return loras_in(layout)


def _pipeline_problem(p: Pipeline) -> str | None:
    """Why the diffusers backend can't load this pipeline, if it can't."""
    if p.pipeline_class != PIPELINE_CLASS:
        kind = p.pipeline_class or "Diffusers"
        return f"a {kind} pipeline; the diffusers backend runs {PIPELINE_CLASS}"
    if p.weights_only_as_variant:
        return "its weights are only published as a Diffusers variant (e.g. fp16)"
    return None


def _models_folder(text: str) -> Path:
    path = Path(os.path.expandvars(text.strip())).expanduser()
    if not path.is_absolute():
        raise InvalidRequestError("the models folder must be an absolute path")
    if path.exists() and not path.is_dir():
        raise InvalidRequestError(f"{path} is a file, not a folder")
    return path


def _parse(reference: str) -> HubRef:
    try:
        return parse_ref(reference)
    except RefError as exc:
        raise InvalidRequestError(str(exc)) from exc


def _key(path: str | Path) -> str:
    """A path as comparable text: expanded, with links and mapped drives followed (so ``N:\\``
    and its ``\\\\server\\share`` match, as do a cache snapshot and its blob), case-folded
    where the OS is."""
    expanded = Path(os.path.expandvars(str(path))).expanduser()
    return os.path.normcase(os.path.realpath(expanded))


def _local(
    copy: LocalCopy | None, shown: Path | None, in_use: dict[str, list[str]]
) -> dto.HubLocal | None:
    if copy is None or shown is None:
        return None
    return dto.HubLocal(source=copy.source, path=str(shown), in_use=in_use.get(_key(shown), []))


def _check_fit(
    config: BackendConfig, key: str | None, layout: RepoLayout, variant: Variant
) -> None:
    if isinstance(config, DiffusersBackendConfig):
        raise InvalidRequestError(
            f"a diffusers backend loads a whole pipeline; {variant.id} is a single file"
        )
    specs = {k: (flag, what) for k, flag, what in launch.file_specs(config)}
    if key not in specs:
        raise InvalidRequestError(f"{config.type} engines have no launch setting {key!r}")
    if getattr(config, "launch", None) is None:
        raise InvalidRequestError("set up launch for this engine first")
    if variant not in roles_for(config.type, layout).get(key, []):
        flag, what = specs[key]
        raise InvalidRequestError(f"{variant.id} can't be loaded as {flag} ({what})")


def _nothing_fits(config: BackendConfig, layout: RepoLayout) -> str:
    if isinstance(config, LlamaCppBackendConfig):
        if layout.kind in {"transformers", "safetensors", "diffusers"}:
            return "llama.cpp loads GGUF files only: look at this repo's quantized versions."
        return "Nothing here loads in llama.cpp."
    return "Nothing here loads in sd-server."
