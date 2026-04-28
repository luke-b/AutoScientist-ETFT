"""
etft/agent_cli.py — ``etft-agent`` CLI command.

Runs a single research task via the fully-equipped research agent
(``create_research_agent``).

Usage
-----
    etft-agent --task "Improve the batch normalisation component."
    etft-agent --task "..." --context-file ./context.json --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoScientist-ETFT Research Agent CLI — run a single agent task."
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Natural-language task for the research agent.",
    )
    parser.add_argument(
        "--context-file",
        default=None,
        help="Optional JSON file containing additional context passed to the agent.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root data directory (overrides config.yaml defaults).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the agent result JSON to this file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from etft.agent_factory import create_research_agent
    from etft.config import load_config

    cfg = load_config(args.config)

    context: dict = {}
    if args.context_file:
        try:
            with open(args.context_file) as f:
                context = json.load(f)
        except Exception as exc:
            logger.error("Failed to load context file %r: %s", args.context_file, exc)
            sys.exit(1)

    agent = create_research_agent(cfg, data_root=args.data_root)
    logger.info("Running task: %r", args.task[:120])
    result = agent.run_task(task=args.task, context=context)

    output_dict = {
        "output": result.output,
        "success": result.success,
        "skills_invoked": result.skills_invoked,
        "steps": len(result.steps),
        "error": result.error,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_dict, f, indent=2)
        logger.info("Result written to %s", args.output)
    else:
        print(json.dumps(output_dict, indent=2))

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
