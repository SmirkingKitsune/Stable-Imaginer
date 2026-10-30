from collections.abc import AsyncIterator

import pytest

from imaginer.backends.base import ChatDelta, ToolCallFragment
from imaginer.backends.openai_http.chat import iter_chat_deltas, iter_sse_data, parse_chunk
from imaginer.domain.errors import EngineError, EngineInterruptedError
from imaginer.orchestration.assembler import MessageAssembler


async def _lines(*lines: str) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def test_sse_framing() -> None:
    lines = _lines(
        ": keep-alive",
        "data: one",
        "",
        "event: x",
        "data: two-a",
        "data:two-b",
        "\r",
        "data: tail-without-blank-line",
    )
    assert [d async for d in iter_sse_data(lines)] == [
        "one",
        "two-a\ntwo-b",
        "tail-without-blank-line",
    ]


async def test_chat_deltas_stop_at_done() -> None:
    lines = _lines(
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        "",
        "data: [DONE]",
        "",
        'data: {"choices":[{"delta":{"content":"ignored"}}]}',
        "",
    )
    deltas = [d async for d in iter_chat_deltas("llama", lines)]
    assert [(d.content, d.finish_reason) for d in deltas] == [("Hi", None), (None, "stop")]


async def test_a_finish_reason_ends_a_stream_without_done() -> None:
    lines = _lines('data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"length"}]}', "")
    deltas = [d async for d in iter_chat_deltas("llama", lines)]
    assert [(d.content, d.finish_reason) for d in deltas] == [("Hi", "length")]


async def test_a_stream_that_just_stops_was_cut_off() -> None:
    lines = _lines('data: {"choices":[{"delta":{"content":"The answer is"}}]}', "")
    seen: list[str | None] = []
    with pytest.raises(EngineInterruptedError, match=r"before finishing the answer \(after 1"):
        async for delta in iter_chat_deltas("llama", lines):
            seen.append(delta.content)
    assert seen == ["The answer is"]  # what came is still delivered first


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ('data: {"error":{"code":500,"message":"KV cache full"}}', "KV cache full"),
        ("data: {not json", "malformed"),
        ("data: [1, 2]", "unexpected"),
    ],
)
async def test_chat_stream_errors(line: str, match: str) -> None:
    with pytest.raises(EngineError, match=match):
        async for _ in iter_chat_deltas("llama", _lines(line, "")):
            pass


def test_parse_chunk_with_everything() -> None:
    delta = parse_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "content": "a",
                        "reasoning_content": "thinking",
                        "tool_calls": [
                            {"index": 1, "id": "c1", "type": "function", "function": {"name": "f"}}
                        ],
                    },
                    "finish_reason": None,
                }
            ],
            "usage": {"completion_tokens": 3},
            "timings": {"predicted_n": 3},
        }
    )
    assert delta.content == "a"
    assert delta.reasoning == "thinking"
    assert delta.tool_calls == (ToolCallFragment(index=1, id="c1", name="f", arguments=""),)
    assert delta.usage == {"completion_tokens": 3, "timings": {"predicted_n": 3}}


def test_parse_chunk_without_choices() -> None:
    assert parse_chunk({"usage": {"total_tokens": 9}}) == ChatDelta(usage={"total_tokens": 9})


def test_assembler_merges_fragments() -> None:
    asm = MessageAssembler()
    asm.add(ChatDelta(reasoning="hmm "))
    asm.add(ChatDelta(content="Sure, "))
    asm.add(ChatDelta(tool_calls=(ToolCallFragment(index=0, id="c0", name="gen"),)))
    # llama.cpp re-sends the full name when its incremental parse grows.
    asm.add(ChatDelta(tool_calls=(ToolCallFragment(index=0, id="c0", name="generate_image"),)))
    asm.add(ChatDelta(tool_calls=(ToolCallFragment(index=0, arguments='{"prompt": '),)))
    asm.add(ChatDelta(tool_calls=(ToolCallFragment(index=1, name="other", arguments="{}"),)))
    asm.add(ChatDelta(tool_calls=(ToolCallFragment(index=0, arguments='"a cat"}'),)))
    asm.add(ChatDelta(content="done", finish_reason="tool_calls", usage={"a": 1}))
    asm.add(ChatDelta(usage={"b": 2}))

    assert asm.text == "Sure, done"
    assert asm.reasoning == "hmm "
    assert asm.finish_reason == "tool_calls"
    assert asm.usage == {"a": 1, "b": 2}
    calls = asm.tool_calls()
    assert calls[0] == {
        "id": "c0",
        "type": "function",
        "function": {"name": "generate_image", "arguments": '{"prompt": "a cat"}'},
    }
    assert calls[1]["function"] == {"name": "other", "arguments": "{}"}
    assert calls[1]["id"].startswith("call_")  # generated when the engine sent none
    assert asm.tool_calls() == calls  # stable ids


def test_empty_assembler() -> None:
    asm = MessageAssembler()
    assert (asm.text, asm.reasoning, asm.tool_calls()) == ("", None, [])
