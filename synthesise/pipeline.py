from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from .agents import CheckerAgent, EngineerAgent, TargetAgent
from .config import AppConfig
from .models import SynthesisResult


class SynthesisPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.engineer = EngineerAgent(config.engineer_model)
        self.checker = CheckerAgent(config.checker_model)
        self.target = TargetAgent(config.target_model)

    def run_one(self, topic: str, domain: str, index: int = 1) -> SynthesisResult:
        problem = self.engineer.create_problem(topic, domain, index)
        solution = self.target.solve(problem)
        refinement_count = 0
        check = self.checker.check(problem, solution)

        while (
            (not check.passed or check.score < self.config.quality.min_score)
            and refinement_count < self.config.quality.max_refinement_attempts
        ):
            refinement_count += 1
            feedback = check.feedback
            if check.issues:
                feedback += "\nIssues:\n- " + "\n- ".join(check.issues)
            solution = self.target.solve(problem, feedback)
            check = self.checker.check(problem, solution)

        return SynthesisResult(
            problem=problem,
            solution=solution,
            checker=check,
            refinement_count=refinement_count,
            model_trace={
                "engineer": self.config.engineer_model.model_name,
                "checker": self.config.checker_model.model_name,
                "target": self.config.target_model.model_name,
            },
        )

    def run_batch(self, topic: str, domain: str, count: int) -> list[SynthesisResult]:
        count = max(1, count)
        results: list[SynthesisResult | None] = [None] * count
        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, count)) as pool:
            futures = {
                pool.submit(self.run_one, topic, domain, i + 1): i
                for i in range(count)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [result for result in results if result is not None]

    def save(self, results: Iterable[SynthesisResult], batch_id: str | None = None) -> Path:
        batch_id = batch_id or self.config.default_batch_id
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{batch_id}.json"
        payload = []
        for result in results:
            payload.append({
                "problem": {
                    "domain": result.problem.domain,
                    "topic": result.problem.topic,
                    "statement": result.problem.statement,
                    "context": result.problem.context,
                    "difficulty": result.problem.difficulty,
                    "metadata": result.problem.metadata,
                },
                "solution": result.solution,
                "checker": {
                    "passed": result.checker.passed,
                    "score": result.checker.score,
                    "correctness": result.checker.correctness,
                    "clarity": result.checker.clarity,
                    "relevance": result.checker.relevance,
                    "completeness": result.checker.completeness,
                    "feedback": result.checker.feedback,
                    "issues": result.checker.issues,
                },
                "refinement_count": result.refinement_count,
                "model_trace": result.model_trace,
            })
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
