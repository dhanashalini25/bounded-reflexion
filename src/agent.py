"""BoundedReflexion - the reusable pattern extracted for upstream contribution.

Pulled out of projects 03 (hard bounds) and 10 (judge-driven revision), with
every project-local import removed so it can be dropped into LangGraph, CrewAI
or AutoGen unchanged. `generate` and `judge` are injected, so the pattern is
framework- and provider-agnostic and testable without a network.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol


class Generator(Protocol):
    def __call__(self, task: str, constraints: list[str]) -> str: ...


class Judge(Protocol):
    def __call__(self, task: str, output: str) -> tuple[float, list[str]]:
        """Return (score in 0..1, constraints for the next attempt)."""


DEMO = "Explain backpressure to a junior engineer."


@dataclass
class Attempt:
    output: str
    score: float
    ms: int


@dataclass
class Outcome:
    best: str
    score: float
    attempts: list[Attempt] = field(default_factory=list)
    stopped_because: str = "revision_cap"

    @property
    def improvement(self) -> float:
        return round(self.score - self.attempts[0].score, 4) if self.attempts else 0.0


@dataclass
class BoundedReflexion:
    """Regenerate while a judge score keeps improving, under hard bounds.

    Args:
        generate: (task, constraints) -> output
        judge: (task, output) -> (score, constraints)
        max_revisions: hard cap on regenerations after the first attempt
        min_delta: stop when improvement falls below this
        max_seconds: wall-clock bound; the loop never outlives it

    Example:
        >>> refine = BoundedReflexion(
        ...     generate=lambda task, cons: "42" if cons else "forty-two",
        ...     judge=lambda task, out: (1.0, []) if out == "42" else (0.4, ["Use digits."]),
        ... )
        >>> refine("What is six times seven?").best
        '42'
    """

    generate: Generator
    judge: Judge
    max_revisions: int = 3
    min_delta: float = 0.05
    max_seconds: float = 120.0
    on_attempt: Callable[[Attempt], None] | None = None

    def __post_init__(self) -> None:
        if self.max_revisions < 0:
            raise ValueError("max_revisions must be >= 0")
        if not 0 < self.min_delta < 1:
            raise ValueError("min_delta must be between 0 and 1")

    def __call__(self, task: str) -> Outcome:
        started = time.monotonic()
        constraints: list[str] = []
        attempts: list[Attempt] = []
        stopped = "revision_cap"

        for i in range(self.max_revisions + 1):
            if i and time.monotonic() - started > self.max_seconds:
                stopped = "timeout"
                break

            t0 = time.perf_counter()
            output = self.generate(task, constraints)
            score, constraints = self.judge(task, output)
            attempt = Attempt(output=output, score=score, ms=int((time.perf_counter() - t0) * 1000))
            attempts.append(attempt)

            if self.on_attempt:
                self.on_attempt(attempt)

            if i:
                delta = score - attempts[-2].score
                if delta < 0:
                    stopped = "regression"
                    break
                if delta < self.min_delta:
                    stopped = "plateau"
                    break

        best = max(attempts, key=lambda a: a.score)
        return Outcome(best=best.output, score=best.score, attempts=attempts,
                       stopped_because=stopped)

    def as_node(self, task_key: str = "task", output_key: str = "output"):
        """A LangGraph-compatible node: state dict in, state dict out."""

        def node(state: dict) -> dict:
            outcome = self(state[task_key])
            return {
                **state,
                output_key: outcome.best,
                "reflexion": {
                    "score": outcome.score,
                    "attempts": len(outcome.attempts),
                    "stopped_because": outcome.stopped_because,
                    "improvement": outcome.improvement,
                },
            }

        return node


def run(prompt: str) -> str:
    """Demo with a toy generator and judge - no LLM, no network."""
    drafts = iter([
        "Backpressure is when a system slows down.",
        "Backpressure is a consumer telling a producer to slow down, usually by bounding a "
        "queue: once it is full the producer blocks or sheds load, so memory stays flat "
        "instead of the service falling over.",
    ])

    def generate(task: str, constraints: list[str]) -> str:
        return next(drafts, "Backpressure bounds a queue so producers cannot outrun consumers.")

    def judge(task: str, output: str) -> tuple[float, list[str]]:
        score = min(0.3 + len(output) / 300, 1.0)
        return score, ["Name the mechanism that applies the pressure."] if score < 0.8 else []

    outcome = BoundedReflexion(generate=generate, judge=judge)(prompt)
    return (
        f"{outcome.best}\n\n[{len(outcome.attempts)} attempts | score {outcome.score:.2f} "
        f"| improvement {outcome.improvement:+.2f} | stopped: {outcome.stopped_because}]"
    )
