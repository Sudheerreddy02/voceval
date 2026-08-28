from voceval.pipeline.live.openai_llm import to_openai_messages
from voceval.types import Message, ToolCall


def test_plain_messages_pass_through():
    history = [Message("system", "be brief"), Message("user", "hi")]
    assert to_openai_messages(history) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_tool_call_and_result_round_trip():
    history = [
        Message("assistant", "", tool_calls=[ToolCall("call_1", "book", {"size": 2})]),
        Message("tool", '{"ok": true}', tool_call_id="call_1"),
    ]
    out = to_openai_messages(history)

    assert out[0]["tool_calls"][0]["id"] == "call_1"
    assert out[0]["tool_calls"][0]["function"]["name"] == "book"
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"size": 2}'
    assert out[1] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}
