from __future__ import annotations

import argparse
import sys

from .config import load_config
from .gemini import GeminiError
from .pipeline import SynthesisPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SynThesisAI Gemini pipeline")
    parser.add_argument("--topic", required=True, help="Topic or task to synthesize")
    parser.add_argument("--domain", default="general", help="Domain such as mathematics or technology")
    parser.add_argument("--count", type=int, default=None, help="Number of generated items")
    parser.add_argument("--batch-id", default=None, help="Output batch filename without .json")
    parser.add_argument("--config", default="config/settings.yaml", help="YAML config path")
    parser.add_argument("--engineer-model", default=None)
    parser.add_argument("--checker-model", default=None)
    parser.add_argument("--target-model", default=None)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        # CLI overrides are intentionally applied immutably through dataclasses.replace.
        from dataclasses import replace
        if args.engineer_model:
            config = replace(config, engineer_model=replace(config.engineer_model, model_name=args.engineer_model))
        if args.checker_model:
            config = replace(config, checker_model=replace(config.checker_model, model_name=args.checker_model))
        if args.target_model:
            config = replace(config, target_model=replace(config.target_model, model_name=args.target_model))

        count = args.count or config.num_problems
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
