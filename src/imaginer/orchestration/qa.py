"""The quality check: a vision LLM looks at the images a tool made and decides whether to keep them.

The turn runner calls ``review`` after an image tool succeeds, when the turn has QA on. The
reviewer gets the user's request, the prompt and settings the image model got, the result,
and for an edit the image being edited. It answers with a JSON verdict: ``pass``, or
``redo`` with what to change (prompt, negative prompt, steps, CFG scale, seed). Its changes
never touch what the user fixed: their image options and, in raw mode, the prompt itself.

A check that can't run or can't be read never fails the turn; it comes back ``skipped``.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from imaginer.backends.base import ChatRequest, LLMBackend
from imaginer.core.logging import get_logger
from imaginer.db import models as orm
from imaginer.db.engine import Database
from imaginer.db.repositories import artifacts as art_repo
from imaginer.domain.errors import EngineInterruptedError, ImaginerError
from imaginer.orchestration.assembler import MessageAssembler
from imaginer.orchestration.context_builder import ContextBuilder
from imaginer.orchestration.jobs import JobManager
from imaginer.tools.base import json_schema

log = get_logger(__name__)

# Told to the chat model when its images are checked, so it doesn't redo them itself.
QA_GUIDANCE = (
    "Every image is checked automatically and made again when it has clear flaws; the tool "
    "result's qa field says what happened. Don't call the tool again just to fix quality."
)
REVIEW_PROMPT = (
    "You check images from an image model before the user sees them. You get the user's "
    "request, the prompt and settings the image model got, and the result.\n"
    "Pass the result unless it has a clear problem:\n"
    "- the main subject, or something the request asks for, is missing or wrong\n"
    "- text the request asks for is missing or misspelled\n"
    "- distorted anatomy (faces, hands, limbs) or broken objects\n"
    "- heavy noise, blur or smearing, or an unfinished look\n"
    "- washed out, or burnt and oversaturated\n"
    "Matters of taste are not problems. With several images, redo only if most of them "
    "have a problem. For an edit, check that the change was made and the rest was kept.\n"
    "To redo, choose the change most likely to fix the problem:\n"
    "- noise, blur or unfinished detail: more steps\n"
    "- washed out: a higher cfg_scale; burnt or oversaturated: a lower one\n"
    "- wrong or missing content: a revised prompt that keeps everything the user asked for\n"
    "- a bad composition or a one-off glitch: a new seed\n"
    'Answer with one JSON object and nothing else: {"verdict": "pass" or "redo", '
    '"issues": [short phrases], "prompt": string or null, "negative_prompt": string or null, '
    '"steps": integer or null, "cfg_scale": number or null, "new_seed": true or false}. '
    "Use null for whatever stays as it is."
)
PINNABLE = ("steps", "cfg_scale", "seed", "negative_prompt")  # image options the check obeys
MAX_ISSUES = 8
_SETTINGS = ("steps", "cfg_scale", "sampler", "width", "height")


class Verdict(BaseModel):
    """The reviewer's answer. Bounds are applied afterwards, so a slightly off answer counts."""

    verdict: Literal["pass", "redo"]
    issues: list[str] = []
    prompt: str | None = None
    negative_prompt: str | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    new_seed: bool = False


_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "verdict", "schema": json_schema(Verdict)},
}


@dataclass(frozen=True, kw_only=True)
class Review:
    status: Literal["passed", "redo", "flagged", "skipped"]
    issues: list[str] = field(default_factory=list)
    # What the next attempt changes: prompt, negative_prompt, steps, cfg_scale, new_seed.
    changes: dict[str, Any] = field(default_factory=dict)
    detail: str | None = None  # why the check was skipped


@dataclass(frozen=True, kw_only=True)
class Subject:
    """What one check looks at."""

    conversation_id: str
    request: str  # the user's words
    tool: str
    job_id: str
    artifact_ids: list[str]
    raw: bool  # the prompt is the user's own and stays as written
    pinned: frozenset[str]  # PINNABLE settings the user fixed


class QualityChecker:
    def __init__(self, db: Database, context: ContextBuilder, jobs: JobManager) -> None:
        self._db = db
        self._context = context
        self._jobs = jobs

    async def review(self, backend: LLMBackend, subject: Subject) -> Review:
        try:
            job = await self._jobs.get(subject.job_id)
            requested = job.params.get("requested") or {}
            resolved = job.params.get("resolved") or {}
            request = ChatRequest(
                messages=[
                    {"role": "system", "content": _instructions(subject)},
                    {"role": "user", "content": await self._show(subject, requested, resolved)},
                ],
                params={"response_format": _RESPONSE_FORMAT, "temperature": 0.2},
            )
            assembler = MessageAssembler()
            async for delta in backend.chat_stream(request):
                assembler.add(delta)
            if assembler.finish_reason is None:
                raise EngineInterruptedError(
                    f"{backend.name} stopped before finishing the check", backend=backend.name
                )
        except ImaginerError as exc:
            log.warning("qa.failed", code=exc.code, detail=exc.detail)
            return Review(status="skipped", detail=f"the check failed: {exc.detail}")
        except Exception:
            log.exception("qa.crashed")
            return Review(status="skipped", detail="the check failed unexpectedly")
        verdict = parse(assembler.text)
        if verdict is None:
            log.warning("qa.unreadable", answer=assembler.text[:200])
            return Review(status="skipped", detail="the check's answer could not be read")
        review = _normalized(verdict, subject, requested, resolved)
        log.info("qa.reviewed", status=review.status, issues=review.issues, changes=review.changes)
        return review

    async def _show(
        self, subject: Subject, requested: dict[str, Any], resolved: dict[str, Any]
    ) -> list[dict[str, Any]]:
        settings = [f"{k} {resolved[k]}" for k in _SETTINGS if resolved.get(k) is not None]
        lines = [
            f"Request: {subject.request.strip() or '(none)'}",
            f"Tool: {subject.tool}",
            f"Prompt: {requested.get('prompt', '')}",
            f"Negative prompt: {resolved.get('negative_prompt') or '(none)'}",
            f"Settings: {', '.join(settings) or 'engine defaults'}",
        ]
        async with self._db.session() as session:
            outputs = await art_repo.get_many_in_conversation(
                session, subject.conversation_id, subject.artifact_ids
            )
            source_label = (requested.get("images") or [None])[0]
            source = (
                await art_repo.get_by_label(session, subject.conversation_id, source_label)
                if source_label
                else None
            )
        parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
        if source is not None:
            parts.append({"type": "text", "text": f"The image being edited ({source.label}):"})
            parts.append(await self._image(source))
        for artifact in sorted(outputs, key=lambda a: subject.artifact_ids.index(a.id)):
            parts.append({"type": "text", "text": f"Result {artifact.label}:"})
            parts.append(await self._image(artifact))
        return parts

    async def _image(self, artifact: orm.Artifact) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": await self._context.data_url(artifact)}}


def parse(text: str) -> Verdict | None:
    """The verdict in an answer, also when it is wrapped in prose or a code fence."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        return Verdict.model_validate_json(match.group(0))
    except ValidationError:
        return None


def _instructions(subject: Subject) -> str:
    rules = [REVIEW_PROMPT]
    if subject.raw:
        rules.append(
            "The prompt is the user's own wording and is used exactly as written: "
            "leave prompt and negative_prompt null."
        )
    fixed = [k for k in ("steps", "cfg_scale", "negative_prompt") if k in subject.pinned]
    if fixed:
        rules.append(f"The user fixed these settings, leave them null: {', '.join(fixed)}.")
    if "seed" in subject.pinned:
        rules.append("The user fixed the seed: new_seed must be false.")
    return "\n\n".join(rules)


def _normalized(
    verdict: Verdict, subject: Subject, requested: dict[str, Any], resolved: dict[str, Any]
) -> Review:
    """Only changes the check may make, within the bounds the image tools accept."""
    issues = [i.strip()[:200] for i in verdict.issues if i.strip()][:MAX_ISSUES]
    if verdict.verdict == "pass":
        return Review(status="passed", issues=issues)
    changes: dict[str, Any] = {}
    if not subject.raw:
        prompt = (verdict.prompt or "").strip()[:4000]
        if prompt and prompt != requested.get("prompt"):
            changes["prompt"] = prompt
        negative = (verdict.negative_prompt or "").strip()[:2000]
        unpinned = "negative_prompt" not in subject.pinned
        if negative and unpinned and negative != resolved.get("negative_prompt"):
            changes["negative_prompt"] = negative
    if verdict.steps is not None and "steps" not in subject.pinned:
        steps = min(150, max(1, verdict.steps))
        if steps != resolved.get("steps"):
            changes["steps"] = steps
    if verdict.cfg_scale is not None and "cfg_scale" not in subject.pinned:
        cfg = round(min(30.0, max(0.0, verdict.cfg_scale)), 2)
        if cfg != resolved.get("cfg_scale"):
            changes["cfg_scale"] = cfg
    if "seed" not in subject.pinned and (verdict.new_seed or not changes):
        # Redoing with nothing changed would repeat the image; a new seed at least varies it.
        changes["new_seed"] = True
    if not changes:
        return Review(status="flagged", issues=issues)
    return Review(status="redo", issues=issues, changes=changes)
