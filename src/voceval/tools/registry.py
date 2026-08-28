from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass

from voceval.types import ToolCall, ToolResult


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable


def tool(name: str, description: str, parameters: dict) -> Callable[[Callable], Tool]:
    def wrap(fn: Callable) -> Tool:
        return Tool(name, description, parameters, fn)

    return wrap


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self._tools[t.name] = t

    def add(self, t: Tool) -> None:
        self._tools[t.name] = t

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    async def call(self, call: ToolCall) -> ToolResult:
        t = self._tools.get(call.name)
        if t is None:
            return ToolResult(call.id, f"unknown tool: {call.name}")
        try:
            result = t.fn(**call.arguments)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # tools face untrusted args, keep the call alive
            return ToolResult(call.id, f"error: {exc}")
        if not isinstance(result, str):
            result = json.dumps(result)
        return ToolResult(call.id, result)
