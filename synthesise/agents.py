from __future__ import annotations

from typing import Any

from .config import ModelConfig
from .gemini import GeminiClient
from .models import CheckResult, Problem


class EngineerAgent:
    def __init__(self, config: ModelConfig):
        self.client = GeminiClient(config.model_name, config.temperature, config.max_output_tokens)

    def create_problem(self, topic: str, domain: str, index: int = 1) -> Problem:
        data = self.client.generate_json(
            f"""Create one rigorous {domain} learning problem about: {topic}

This is item {index}. Make it self-contained, unambiguous, educational, and solvable.
Return JSON with exactly these keys:
statement, context, difficulty, learning_objective
""",
            system_instruction=(
                "You are the Engineer Agent in a multi-agent educational synthesis system. "
                "Design high-quality original tasks. Do not include the answer in the problem."
            ),
        )
        return Problem(
            domain=domain,
            topic=topic,
            statement=str(data.get("statement", "")).strip(),
            context=str(data.get("context", "")).strip(),
            difficulty=str(data.get("difficulty", "intermediate")).strip(),
            metadata={"learning_objective": data.get("learning_objective", "")},
        )


class CheckerAgent:
    def __init__(self, config: ModelConfig):
        self.client = GeminiClient(config.model_name, config.temperature, config.max_output_tokens)

    def check(self, problem: Problem, solution: str) -> CheckResult:
        data = self.client.generate_json(
            f"""Evaluate this problem and proposed solution independently.

DOMAIN: {problem.domain}
TOPIC: {problem.topic}
PROBLEM: {problem.statement}
CONTEXT: {problem.context}
PROPOSED SOLUTION:
{solution}

Return JSON exactly with:
passed (boolean), score (0 to 1), correctness (0 to 1), clarity (0 to 1),
relevance (0 to 1), completeness (0 to 1), feedback (string), issues (array of strings).
Be strict about factual and mathematical correctness. If an issue exists, explain it clearly.
""",
            system_instruction=(
                "You are the independent Checker Agent. Never assume the target answer is correct. "
                "Recompute or reason through the solution before scoring it."
            ),
        )
        return CheckResult(
            passed=bool(data.get("passed", False)),
            score=float(data.get("score", 0)),
            correctness=float(data.get("correctness", 0)),
            clarity=float(data.get("clarity", 0)),
            relevance=float(data.get("relevance", 0)),
            completeness=float(data.get("completeness", 0)),
            feedback=str(data.get("feedback", "")),
            issues=[str(x) for x in data.get("issues", [])],
        )


class TargetAgent:
    def __init__(self, config: ModelConfig):
        self.client = GeminiClient(config.model_name, config.temperature, config.max_output_tokens)

    def solve(self, problem: Problem, checker_feedback: str = "") -> str:
        return self.client.generate(
            f"""Solve the following problem carefully.

DOMAIN: {problem.domain}
TOPIC: {problem.topic}
PROBLEM: {problem.statement}
CONTEXT: {problem.context}

Independent checker feedback from a previous attempt:
{checker_feedback or "No previous feedback; produce the first solution."}

Give a self-contained final solution. Show reasoning where useful, state assumptions, and verify the conclusion.
Do not mention this multi-agent instruction in the answer.
""",
            system_instruction=(
                "You are the Target Agent. Produce the best final answer for the user. "
                "Use checker feedback to correct errors rather than defending the previous answer."
            ),
        )
