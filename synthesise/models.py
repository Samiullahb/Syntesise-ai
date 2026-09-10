from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Problem:
    domain: str
    topic: str
    statement: str
    context: str = ""
    difficulty: str = "intermediate"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    passed: bool
    score: float
    correctness: float
    clarity: float
    relevance: float
    completeness: float
    feedback: str
    issues: list[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    problem: Problem
    solution: str
    checker: CheckResult
    refinement_count: int = 0
    model_trace: dict[str, str] = field(default_factory=dict)
