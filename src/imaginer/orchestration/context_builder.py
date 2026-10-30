"""Turns one branch of a conversation's stored history into OpenAI chat messages for the LLM.

* The history is the path from the first message to ``leaf_id`` (by default the newest
  message). Other branches are invisible to the model.
* The system prompt ends with a manifest of the branch's artifacts by label (plus uploads
  no message has attached yet), so the model can refer to "img-3" without seeing every
  image. Labels stay unique across branches.
* With a vision-capable backend, the most recent attached images (``context.max_images``)
  are sent as pixels, downscaled. Older ones appear as labels only.
* Only the last ``context.max_messages`` messages are sent, and a cut never leaves a tool
  result without the assistant message that requested it. Likewise a tool call whose result
  was never stored (crash, cancellation) is dropped, so chat templates stay valid.
* With ``orchestration.feed_outputs_to_vision``, images a tool produced are shown to a vision
  model right after the tool result, so it can check its own work.
"""

import asyncio
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from imaginer.core.config import ContextConfig
from imaginer.db import models as orm
from imaginer.db.engine import Database
from imaginer.db.repositories import artifacts as art_repo
from imaginer.db.repositories import messages as msg_repo
from imaginer.domain.enums import ArtifactKind, ArtifactSource, MessageStatus, Role
from imaginer.services.media import encode_for_llm
from imaginer.storage.base import ArtifactStore

DEFAULT_SYSTEM_PROMPT = (
    "You are Imaginer, the assistant in a private, local multimodal chat app.\n"
    "Every image in this conversation has a short label such as img-1. When you talk about "
    "an image, refer to it by its label."
)
MANIFEST_LIMIT = 50
_SOURCE_WORDS = {
    ArtifactSource.UPLOAD: "uploaded",
    ArtifactSource.GENERATED: "generated",
    ArtifactSource.EDITED: "edited",
    ArtifactSource.DERIVED: "derived",
}
_IMAGE_CACHE_SIZE = 64


class ContextBuilder:
    def __init__(
        self,
        db: Database,
        store: ArtifactStore,
        config: ContextConfig,
        *,
        feed_outputs_to_vision: bool = False,
    ) -> None:
        self._db = db
        self._store = store
        self._config = config
        self._feed_outputs = feed_outputs_to_vision
        self._image_cache: OrderedDict[tuple[str, int], str] = OrderedDict()

    async def build(
        self,
        conversation_id: str,
        *,
        leaf_id: str | None = None,
        vision: bool,
        tool_guidance: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        async with self._db.session() as session, session.begin():
            if leaf_id is None:
                leaf = await msg_repo.latest(session, conversation_id)
                leaf_id = leaf.id if leaf is not None else None
            path = await msg_repo.path_to(session, leaf_id) if leaf_id is not None else []
            attachments = await msg_repo.attachments_for(session, [m.id for m in path])
            referenced = {aid for ids in attachments.values() for aid in ids}
            rows = await art_repo.get_many_in_conversation(
                session, conversation_id, sorted(referenced)
            )
            loose = await art_repo.unattached(session, conversation_id)
        known = {a.id: a for a in rows}
        # The branch's artifacts, plus any no message has attached yet (on every branch).
        manifest = sorted([*rows, *loose], key=lambda a: a.id)[-MANIFEST_LIMIT:]
        history = path[-self._config.max_messages :]

        history = _trim_orphan_tool_results(
            [m for m in history if m.status is not MessageStatus.STREAMING]
        )
        answered = _answered_calls(history)
        pixels = self._images_to_show(history, attachments, known) if vision else set()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(manifest, tool_guidance)}
        ]
        for message in history:
            if message.role is Role.TOOL and message.tool_call_id not in answered:
                continue
            attached = attachments.get(message.id, [])
            messages.append(await self._render(message, attached, known, pixels, answered))
            if message.role is Role.TOOL and self._feed_outputs:
                shown = [known[a] for a in attached if a in known and a in pixels]
                if shown:
                    messages.append(await self._tool_images(shown))
        return messages

    def _system_prompt(self, manifest: list[orm.Artifact], tool_guidance: Sequence[str]) -> str:
        prompt = self._config.system_prompt or DEFAULT_SYSTEM_PROMPT
        if tool_guidance:
            prompt += "\n\n" + "\n".join(tool_guidance)
        if not manifest:
            return prompt
        lines = [_describe(a) for a in manifest]
        return f"{prompt}\n\nArtifacts in this conversation (oldest first):\n" + "\n".join(lines)

    def _images_to_show(
        self,
        history: list[orm.Message],
        attachments: dict[str, list[str]],
        known: dict[str, orm.Artifact],
    ) -> set[str]:
        """Ids of the newest attached images, up to ``max_images``."""
        chosen: list[str] = []
        for message in reversed(history):
            if message.role is Role.TOOL and not self._feed_outputs:
                continue
            for artifact_id in reversed(attachments.get(message.id, [])):
                artifact = known.get(artifact_id)
                if artifact is None or artifact.kind is not ArtifactKind.IMAGE:
                    continue
                if len(chosen) >= self._config.max_images:
                    return set(chosen)
                chosen.append(artifact_id)
        return set(chosen)

    async def _render(
        self,
        message: orm.Message,
        attached: list[str],
        known: dict[str, orm.Artifact],
        pixels: set[str],
        answered: set[str],
    ) -> dict[str, Any]:
        text = message.content_text or ""
        if message.role is Role.TOOL:
            return {"role": "tool", "tool_call_id": message.tool_call_id or "", "content": text}
        if message.role is Role.ASSISTANT:
            rendered: dict[str, Any] = {"role": "assistant", "content": text}
            calls = [c for c in message.tool_calls_json or [] if c.get("id") in answered]
            if calls:
                rendered["tool_calls"] = calls
            return rendered

        artifacts = [known[a] for a in attached if a in known]
        if not artifacts:
            return {"role": message.role.value, "content": text}
        labels = ", ".join(a.label for a in artifacts)
        note = f"[Attached: {labels}]"
        shown = [a for a in artifacts if a.id in pixels]
        if not shown:
            return {"role": message.role.value, "content": f"{text}\n\n{note}".strip()}
        parts: list[dict[str, Any]] = [{"type": "text", "text": f"{text}\n\n{note}".strip()}]
        for artifact in shown:
            parts.append({"type": "text", "text": f"{artifact.label}:"})
            parts.append(
                {"type": "image_url", "image_url": {"url": await self._data_url(artifact)}}
            )
        return {"role": message.role.value, "content": parts}

    async def _tool_images(self, artifacts: list[orm.Artifact]) -> dict[str, Any]:
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": "Images produced by the tool call above:"}
        ]
        for artifact in artifacts:
            parts.append({"type": "text", "text": f"{artifact.label}:"})
            parts.append(
                {"type": "image_url", "image_url": {"url": await self._data_url(artifact)}}
            )
        return {"role": "user", "content": parts}

    async def _data_url(self, artifact: orm.Artifact) -> str:
        key = (artifact.id, self._config.image_max_side)
        cached = self._image_cache.get(key)
        if cached is not None:
            self._image_cache.move_to_end(key)
            return cached
        data = await self._store.read_bytes(artifact.conversation_id, artifact.storage_key)
        url = await asyncio.to_thread(encode_for_llm, data, self._config.image_max_side)
        self._image_cache[key] = url
        if len(self._image_cache) > _IMAGE_CACHE_SIZE:
            self._image_cache.popitem(last=False)
        return url


def _describe(artifact: orm.Artifact) -> str:
    parts = [f"{_SOURCE_WORDS[artifact.source]} {artifact.kind.value}"]
    if artifact.width and artifact.height:
        parts.append(f"{artifact.width}x{artifact.height}")
    if artifact.has_alpha:
        parts.append("transparent background")
    filename = (artifact.metadata_json or {}).get("original_filename")
    if filename:
        parts.append(f'file "{filename}"')
    return f"- {artifact.label}: {', '.join(parts)}"


def _answered_calls(history: list[orm.Message]) -> set[str]:
    """Tool call ids that have both a request and a stored result in ``history``."""
    requested = {
        call.get("id")
        for m in history
        if m.role is Role.ASSISTANT
        for call in m.tool_calls_json or []
    }
    results = {m.tool_call_id for m in history if m.role is Role.TOOL}
    return {cid for cid in requested & results if cid}


def _trim_orphan_tool_results(history: list[orm.Message]) -> list[orm.Message]:
    """Drop leading tool messages whose requesting assistant message was cut off."""
    start = 0
    while start < len(history) and history[start].role is Role.TOOL:
        start += 1
    return history[start:]
