"""LoRA mentions: ``@lora:<name>`` in a user's message, or ``@lora:<name>:<weight>``, picks a
LoRA for that message's images, as ``image_options.loras`` does. The name is as the image
backend lists it (``available_loras``), or its file name, or its name without the
extension. One with characters other than letters, digits, ``_``, ``-``, ``/`` and inner
dots is quoted: ``@lora:"my style":0.8``. The web UI shows a mention as a chip and
writes the same text (``web/js/mentions.js`` has the same grammar).

Only the user picks LoRAs: they are read from the user's message, on top of the message's
``image_options.loras`` (a mention's weight wins for a LoRA named in both), and taken out
of what the image engine reads. A mention in a prompt the LLM writes is dropped, never
applied.

The LLM reads each mention as the LoRA's display name, and a note after the message lists
the LoRAs it picks (either way) with what the LoRA library knows of each: its trigger
words, tags and notes. ``LLM_GUIDANCE`` in the system prompt says what the note means, so
the LLM writes prompts that suit the LoRAs instead of taking their names for stray text.
"""

import re
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from imaginer.domain.errors import InvalidRequestError
from imaginer.domain.models import MAX_LORA_PICKS, ImageRequestOptions, LoraPick

_NAME = r"[\w\-/]+(?:\.[\w\-/]+)*"  # dots only inside, so a sentence's full stop stays out
MENTION = re.compile(
    r"(?<!\w)@lora:(?:\"(?P<quoted>[^\"\n]{1,1024})\"|(?P<bare>" + _NAME + r"))"
    r"(?::(?P<weight>-?\d+(?:\.\d+)?))?"
)
WEIGHT_FILES = frozenset({".safetensors", ".gguf", ".pt", ".pth", ".ckpt", ".sft"})


@dataclass(frozen=True, slots=True)
class Mention:
    name: str
    weight: float
    text: str  # as written

    @property
    def shown(self) -> str:
        """The name a person would call it: its file's name without the extension."""
        path = PurePosixPath(self.name.replace("\\", "/"))
        return path.stem if path.suffix.lower() in WEIGHT_FILES else path.name


def find(text: str) -> list[Mention]:
    found = []
    for m in MENTION.finditer(text):
        weight = float(m["weight"]) if m["weight"] is not None else 1.0
        if not -10 <= weight <= 10:
            raise InvalidRequestError(f"{m[0]}: a LoRA's weight is between -10 and 10")
        found.append(Mention(name=m["quoted"] or m["bare"], weight=weight, text=m[0]))
    return found


_SPACED = re.compile(r"(?P<pre>[ \t]*)(?:" + MENTION.pattern + r")(?P<post>[ \t]*)")


def strip(text: str) -> str:
    """``text`` without its LoRA mentions, and without the spaces around them unless one
    is needed between the words on either side."""
    if "@lora:" not in text:
        return text.strip()
    return _SPACED.sub(lambda m: " " if m["pre"] and m["post"] else "", text).strip()


def for_llm(text: str, names: Mapping[str, str] | None = None) -> str:
    """The user's words as the LLM (and the quality check) reads them: each mention as the
    LoRA's name, as ``names`` has it (the library's display names), else as a person would
    say its file's."""
    if "@lora:" not in text:
        return text

    def named(m: re.Match[str]) -> str:
        mention = Mention(m["quoted"] or m["bare"], 1.0, m[0])
        return (names or {}).get(mention.name) or mention.shown

    return MENTION.sub(named, text)


@dataclass(frozen=True, slots=True)
class LoraBrief:
    """What the LLM is told about a LoRA a message picks, from the LoRA library."""

    name: str  # its display name
    words: tuple[str, ...] = ()  # the trigger words the image engine adds to the prompt
    tags: tuple[str, ...] = ()
    notes: str | None = None


# A LoRA as a message names it, on the image backend the message picked (None: the default).
LoraKey = tuple[str, str | None]
LoraLookup = Callable[[Collection[LoraKey]], Awaitable[Mapping[LoraKey, LoraBrief]]]

LLM_GUIDANCE = (
    "Some of the user's messages pick LoRAs: add-ons for the image model that give its "
    "images a style, subject or look. A note after such a message lists them. The image "
    "tools apply them to every image made for that message, and add their trigger words to "
    "the prompt, by themselves. So describe what the user asked for in a way that suits "
    "them, and leave the LoRAs' names and trigger words out of your prompts."
)
_NOTES_SHOWN = 200
_NOTE_HEAD = "[LoRAs the user picked for this message's images:"
_NOTE = re.compile(r"\s*" + re.escape(_NOTE_HEAD) + r"(?:\n- [^\n]*)+\]")


def picked(text: str, options: Mapping[str, Any] | None) -> list[tuple[str, float]]:
    """The LoRAs a stored user message picks, with their weights: its
    ``image_options.loras``, then its mentions (a LoRA named twice keeps its last weight)."""
    loras: dict[str, float] = {}
    for lora in (options or {}).get("loras") or []:
        if isinstance(lora, Mapping) and isinstance(lora.get("path"), str):
            loras[lora["path"]] = float(lora.get("weight", 1.0))
    try:
        found = find(text) if "@lora:" in text else []
    except InvalidRequestError:
        found = []
    for mention in found:
        loras[mention.name] = mention.weight
    return list(loras.items())


def llm_note(loras: Sequence[tuple[str, float]], briefs: Mapping[str, LoraBrief]) -> str:
    """The note after a user message that picks LoRAs: each by its display name, with its
    weight and what the library knows of it. One named twice (picked as ``styles/ink`` and
    mentioned as ``ink``) is listed once, with its last weight, as the engine applies it."""
    shown: dict[str, tuple[LoraBrief, float]] = {}
    for name, weight in loras:
        brief = briefs.get(name) or LoraBrief(Mention(name, weight, "").shown)
        shown[brief.name] = (brief, weight)
    lines = [_NOTE_HEAD]
    for brief, weight in shown.values():
        about = [brief.name, f"weight {weight:g}"]
        if brief.words:
            about.append("trigger words " + ", ".join(f'"{w}"' for w in brief.words))
        if brief.tags:
            about.append("tags " + ", ".join(brief.tags))
        notes = " ".join((brief.notes or "").split())
        if notes:
            cut = notes if len(notes) <= _NOTES_SHOWN else notes[: _NOTES_SHOWN - 3] + "..."
            about.append(f'notes "{cut}"')
        lines.append("- " + ", ".join(about))
    return "\n".join(lines) + "]"


def without_note(prompt: str) -> str:
    """``prompt`` without a LoRA note an LLM copied into it from the user's message."""
    return _NOTE.sub("", prompt).strip() if _NOTE_HEAD in prompt else prompt


def with_mentions(options: ImageRequestOptions | None, text: str) -> ImageRequestOptions | None:
    """``options`` with the LoRAs ``text`` mentions added after its own."""
    mentions = find(text)
    if not mentions:
        return options
    picked = [LoraPick(path=m.name, weight=m.weight) for m in mentions]
    own = (options.loras or []) if options is not None else []
    loras = [*own, *picked]
    if len({x.path for x in loras}) > MAX_LORA_PICKS:
        raise InvalidRequestError(f"at most {MAX_LORA_PICKS} LoRAs per message")
    base = options if options is not None else ImageRequestOptions()
    return base.model_copy(update={"loras": loras})
