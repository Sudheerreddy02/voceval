from __future__ import annotations

import json
from collections.abc import AsyncIterator

from voceval.pipeline.base import LLM
from voceval.types import LLMDelta, Message, ToolCall


def to_openai_messages(history: list[Message]) -> list[dict]:
    out: list[dict] = []
    for m in history:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


class _PartialCall:
    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments = ""

    def assemble(self) -> ToolCall | None:
        if not self.name:
            return None
        try:
            args = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        return ToolCall(self.id or self.name, self.name, args)


class OpenAILLM(LLM):
    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncIterator[LLMDelta]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=to_openai_messages(messages),
            tools=tools or None,
            stream=True,
        )
        partials: dict[int, _PartialCall] = {}

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield LLMDelta(text=delta.content)
            for call in delta.tool_calls or []:
                partial = partials.setdefault(call.index, _PartialCall())
                if call.id:
                    partial.id = call.id
                if call.function and call.function.name:
                    partial.name = call.function.name
                if call.function and call.function.arguments:
                    partial.arguments += call.function.arguments

        for partial in partials.values():
            assembled = partial.assemble()
            if assembled:
                yield LLMDelta(tool_call=assembled)
        yield LLMDelta(done=True)
