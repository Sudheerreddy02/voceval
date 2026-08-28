from __future__ import annotations

import importlib.util
from pathlib import Path

from voceval.agent import VoiceAgent
from voceval.config import Settings


def load_agent(entrypoint: str, settings: Settings) -> VoiceAgent:
    path = Path(entrypoint)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agent from {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "build_agent"):
        return module.build_agent(settings)
    if hasattr(module, "agent") and isinstance(module.agent, VoiceAgent):
        return module.agent
    raise AttributeError(f"{entrypoint} needs a build_agent(settings) or an agent")
