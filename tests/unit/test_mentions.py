"""LoRA mentions in a user's message: "@lora:<name>[:<weight>]"."""

import pytest

from imaginer.domain import mentions
from imaginer.domain.errors import InvalidRequestError
from imaginer.domain.models import ImageRequestOptions, LoraPick


def test_mentions_are_found_with_their_weights() -> None:
    text = 'a @lora:clay:0.8 fox, @lora:"my style" and @lora:styles/ink-wash.safetensors.'
    found = [(m.name, m.weight, m.shown) for m in mentions.find(text)]
    assert found == [
        ("clay", 0.8, "clay"),
        ("my style", 1.0, "my style"),
        ("styles/ink-wash.safetensors", 1.0, "ink-wash"),  # the full stop ends the sentence
    ]


@pytest.mark.parametrize(
    "text",
    ["mail me@lora:x", "see @lora: nothing", "an image @img-1", "@lora:"],
    ids=["glued to a word", "no name", "an image", "bare"],
)
def test_what_isnt_a_mention(text: str) -> None:
    assert mentions.find(text) == []
    assert mentions.strip(text) == text.strip()


def test_a_weight_out_of_range_is_refused() -> None:
    with pytest.raises(InvalidRequestError, match="between -10 and 10"):
        mentions.find("a fox @lora:clay:12")
    assert mentions.find("@lora:clay:-1.5")[0].weight == -1.5


@pytest.mark.parametrize(
    ("text", "stripped"),
    [
        ("a fox in the style of @lora:ink-wash.", "a fox in the style of."),
        ("@lora:ink:0.6 a heron at dawn", "a heron at dawn"),
        ("a @lora:clay fox", "a fox"),
        ("a fox\n@lora:clay", "a fox"),
        ("two @lora:a @lora:b words", "two words"),
    ],
)
def test_the_prompt_loses_its_mentions(text: str, stripped: str) -> None:
    assert mentions.strip(text) == stripped


def test_the_llm_reads_the_loras_names() -> None:
    read = mentions.for_llm("draw a fox in @lora:clay:0.8 style, @lora:styles/ink.safetensors")
    assert read == "draw a fox in clay style, ink"
    assert mentions.for_llm("nothing to see") == "nothing to see"
    # The library's display names, where it has them.
    assert mentions.for_llm("a @lora:clay:0.8 fox", {"clay": "Clay Sculpture"}) == (
        "a Clay Sculpture fox"
    )


def test_a_messages_loras_are_its_picks_then_its_mentions() -> None:
    options = {"loras": [{"path": "ink", "weight": 0.5}, {"path": "lightning"}]}
    assert mentions.picked("a @lora:clay:0.8 fox @lora:ink", options) == [
        ("ink", 1.0),  # the mention's weight wins
        ("lightning", 1.0),
        ("clay", 0.8),
    ]
    assert mentions.picked("no LoRAs", None) == []


def test_the_note_lists_each_lora_with_what_the_library_knows() -> None:
    clay = mentions.LoraBrief("Clay Sculpture", ("clay sculpture",), ("style",), "x " * 150)
    note = mentions.llm_note([("clay", 0.8), ("styles/ink.safetensors", 1.0)], {"clay": clay})
    assert note.splitlines() == [
        "[LoRAs the user picked for this message's images:",
        '- Clay Sculpture, weight 0.8, trigger words "clay sculpture", tags style, '
        f'notes "{("x " * 99)[:197]}..."',
        "- ink, weight 1]",
    ]
    # A LoRA named twice is listed once, with its last weight.
    assert mentions.llm_note([("styles/ink.safetensors", 1.0), ("ink", 0.5)], {}) == (
        "[LoRAs the user picked for this message's images:\n- ink, weight 0.5]"
    )
    # Should an LLM copy it into a prompt, the image engine doesn't get it.
    assert mentions.without_note(f"a clay fox\n\n{note}") == "a clay fox"
    assert mentions.without_note("a fox [in brackets]") == "a fox [in brackets]"


def test_mentions_go_on_top_of_the_picked_loras() -> None:
    picked = ImageRequestOptions(loras=[LoraPick(path="ink", weight=1.0)], steps=4)
    options = mentions.with_mentions(picked, "a fox @lora:ink:0.5 @lora:clay")
    assert options is not None and options.steps == 4
    assert [(x.path, x.weight) for x in options.loras or []] == [
        ("ink", 1.0),
        ("ink", 0.5),  # the engine keeps the first place and takes this, the last, weight
        ("clay", 1.0),
    ]
    assert mentions.with_mentions(picked, "no mentions") is picked
    alone = mentions.with_mentions(None, "@lora:clay")
    assert alone is not None and [x.path for x in alone.loras or []] == ["clay"]
    many = " ".join(f"@lora:style{n}" for n in range(9))
    with pytest.raises(InvalidRequestError, match="at most 8 LoRAs"):
        mentions.with_mentions(None, many)
