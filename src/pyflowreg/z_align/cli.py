"""
Command-line interface for z-alignment workflows.

Provides the ``pyflowreg-z-align`` command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pyflowreg.z_align.config import ZAlignConfig
from pyflowreg.z_align.pipeline import (
    run_all_stages,
    run_stage1,
    run_stage2,
    run_stage3,
)


def _parse_value(raw: str) -> Any:
    """Parse CLI override values."""
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass

    # Optional JSON parsing for lists/dicts
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _parse_overrides(params: Optional[list[str]]) -> Dict[str, Any]:
    """Parse KEY=VALUE CLI overrides."""
    overrides: Dict[str, Any] = {}
    if not params:
        return overrides

    for item in params:
        if "=" not in item:
            print(f"Warning: ignoring malformed override '{item}' (expected KEY=VALUE)")
            continue
        key, value = item.split("=", 1)
        overrides[key] = _parse_value(value)
    return overrides


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the ``run`` subcommand."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: configuration file not found: {config_path}")
        sys.exit(1)

    config = ZAlignConfig.from_file(config_path)
    overrides = _parse_overrides(args.of_params)

    if args.stage == "1":
        run_stage1(config, overrides or None)
        return

    if args.stage == "2":
        run_stage2(config)
        return

    if args.stage == "3":
        run_stage3(config)
        return

    run_all_stages(config, overrides or None)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PyFlowReg z-shift alignment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full z-align workflow
  pyflowreg-z-align run --config z_align.toml

  # Run only z-shift estimation/correction
  pyflowreg-z-align run --config z_align.toml --stage 2

  # Override stage-1 OFOptions from CLI
  pyflowreg-z-align run --config z_align.toml --of-params alpha=8 quality_setting=balanced
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    run_parser = subparsers.add_parser("run", help="Run z-align processing")
    run_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to z-align config file (.toml/.yaml/.yml)",
    )
    run_parser.add_argument(
        "--stage",
        "-s",
        choices=["1", "2", "3"],
        help="Run only one stage (default: run all applicable stages)",
    )
    run_parser.add_argument(
        "--of-params",
        nargs="*",
        metavar="KEY=VALUE",
        help="Override stage-1 OFOptions parameters",
    )
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
