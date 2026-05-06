from agent.context import InMemoryContext


def test_messages_returns_shallow_copy() -> None:
    context = InMemoryContext()
    context.add_user_message("hello")

    messages = context.messages()
    messages.append({"role": "assistant", "content": "injected outside context API"})

    assert messages is not context.messages()
    assert context.messages() == [{"role": "user", "content": "hello"}]
