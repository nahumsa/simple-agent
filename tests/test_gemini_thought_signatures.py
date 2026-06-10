from frameworks.barebones.context import InMemoryContext
from frameworks.barebones.llms import _to_openai_message, _to_tool_call
from agent_core.types import LLMResult, ToolCall


def test_tool_call_parser_preserves_gemini_thought_signature() -> None:
    raw_call = {
        "id": "function-call-1",
        "type": "function",
        "extra_content": {"google": {"thought_signature": "signature-a"}},
        "function": {
            "name": "read_file",
            "arguments": '{"path":"071-challenge-wheel.md"}',
        },
    }

    parsed = _to_tool_call(raw_call)

    assert parsed.extra_content == {"google": {"thought_signature": "signature-a"}}
    assert parsed.arguments == '{"path":"071-challenge-wheel.md"}'


def test_tool_call_parser_treats_none_fields_as_missing() -> None:
    parsed = _to_tool_call(
        {"id": None, "function": {"name": None, "arguments": None}}
    )

    assert parsed.id == ""
    assert parsed.name == ""
    assert parsed.arguments == "{}"


def test_assistant_tool_call_history_replays_gemini_thought_signature() -> None:
    context = InMemoryContext()
    context.add_assistant_tool_calls(
        LLMResult(
            content=None,
            tool_calls=[
                ToolCall(
                    id="function-call-1",
                    name="read_file",
                    args={"path": "071-challenge-wheel.md"},
                    raw_arguments='{"path":"071-challenge-wheel.md"}',
                    extra_content={"google": {"thought_signature": "signature-a"}},
                )
            ],
        )
    )

    openai_message = _to_openai_message(context.messages()[0])

    assert openai_message == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "function-call-1",
                "type": "function",
                "extra_content": {"google": {"thought_signature": "signature-a"}},
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"071-challenge-wheel.md"}',
                },
            }
        ],
    }
