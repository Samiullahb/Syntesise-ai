from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .config import DEFAULT_GEMINI_MODEL, load_config
from .gemini import GeminiError
from .pipeline import SynthesisPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SynThesisAI Gemini 3.1 Pro Preview pipeline")
    parser.add_argument("--topic", required=True, help="Topic or task to synthesize")
    parser.add_argument("--domain", default="general", help="Domain such as mathematics or technology")
    parser.add_argument("--count", type=int, default=None, help="Number of generated items")
    parser.add_argument("--batch-id", default=None, help="Output batch filename without .json")
    parser.add_argument("--config", default="config/settings.yaml", help="YAML config path")
    parser.add_argument("--engineer-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--checker-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_GEMINI_MODEL)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        config = replace(
            config,
            engineer_model=replace(config.engineer_model, model_name=args.engineer_model),
            checker_model=replace(config.checker_model, model_name=args.checker_model),
            target_model=replace(config.target_model, model_name=args.target_model),
        )

        count = args.count or config.num_problems
        if count < 1:
            raise ValueError("--count must be at least 1")

        pipeline = SynthesisPipeline(config)
        results = pipeline.run_batch(args.topic, args.domain, count)
        output = pipeline.save(results, args.batch_id)
        passed = sum(1 for item in results if item.checker.passed)
        print(f"Generated {len(results)} item(s); checker passed {passed}/{len(results)}.")
        print(f"Saved: {output}")
        return 0
    except GeminiError as exc:
        print(f"Gemini error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Application error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
