"""This project needs no LLM - generate and judge are injected by the caller."""
from __future__ import annotations


def respond(messages: list[dict]) -> str:
    return "BoundedReflexion does not call an LLM directly."
