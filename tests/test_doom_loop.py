from frameworks.barebones.doom_loop import check_for_doom_loop


def test_doom_loop_detects_repeated_user_messages() -> None:
    prompt = check_for_doom_loop(
        [
            {"role": "user", "content": "same question"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "same question"},
            {"role": "assistant", "content": "answer two"},
            {"role": "user", "content": "same question"},
        ]
    )

    assert prompt is not None
    assert "user has repeated the same message" in prompt


def test_doom_loop_detects_repeated_assistant_messages() -> None:
    prompt = check_for_doom_loop(
        [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "same answer"},
            {"role": "user", "content": "question two"},
            {"role": "assistant", "content": "same answer"},
            {"role": "user", "content": "question three"},
            {"role": "assistant", "content": "same answer"},
            {"role": "user", "content": "question four"},
        ]
    )

    assert prompt is not None
    assert "You have repeated the same response" in prompt


def test_doom_loop_ignores_non_consecutive_repeated_chat_messages() -> None:
    prompt = check_for_doom_loop(
        [
            {"role": "user", "content": "same question"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "different question"},
            {"role": "assistant", "content": "answer two"},
            {"role": "user", "content": "same question"},
        ]
    )

    assert prompt is None


def test_doom_loop_still_detects_repeated_tool_calls() -> None:
    prompt = check_for_doom_loop(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "name": "read_file", "arguments": {"path": "a.md"}}
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "same result"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "2", "name": "read_file", "arguments": {"path": "a.md"}}
                ],
            },
            {"role": "tool", "tool_call_id": "2", "content": "same result"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "3", "name": "read_file", "arguments": {"path": "a.md"}}
                ],
            },
            {"role": "tool", "tool_call_id": "3", "content": "same result"},
        ]
    )

    assert prompt is not None
    assert "read_file" in prompt
