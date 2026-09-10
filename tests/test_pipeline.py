from __future__ import annotations

from dataclasses import replace

from synthesise.agents import CheckerAgent, EngineerAgent, TargetAgent
from synthesise.config import load_config
from synthesise.pipeline import SynthesisPipeline


class FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def generate_json(self, prompt, **kwargs):
        if "Evaluate" in prompt:
            return {
                "passed": True,
                "score": 0.95,
                "correctness": 0.98,
                "clarity": 0.94,
                "relevance": 0.95,
                "completeness": 0.93,
                "feedback": "Good solution.",
                "issues": [],
            }
        return {
            "statement": "What is 2 + 2?",
            "context": "Basic arithmetic.",
            "difficulty": "beginner",
            "learning_objective": "Addition",
        }

    def generate(self, prompt, **kwargs):
        return "2 + 2 = 4."


def test_config_loads():
    config = load_config()
    assert config.engineer_model.provider == "gemini"
    assert config.checker_model.model_name == "gemini-2.5-flash"


def test_engineer_checker_target(monkeypatch):
    monkeypatch.setattr("synthesise.agents.GeminiClient", FakeClient)
    config = load_config()
    problem = EngineerAgent(config.engineer_model).create_problem("addition", "mathematics")
    solution = TargetAgent(config.target_model).solve(problem)
    check = CheckerAgent(config.checker_model).check(problem, solution)
    assert problem.statement
    assert solution == "2 + 2 = 4."
    assert check.passed is True
    assert check.score > 0.9


def test_pipeline_offline(monkeypatch, tmp_path):
    monkeypatch.setattr("synthesise.agents.GeminiClient", FakeClient)
    config = load_config()
    quality = replace(config.quality, max_refinement_attempts=0)
    config = replace(config, output_dir=str(tmp_path), quality=quality)
    pipeline = SynthesisPipeline(config)
    results = pipeline.run_batch("addition", "mathematics", 2)
    path = pipeline.save(results, "test_batch")
    assert len(results) == 2
    assert path.exists()
