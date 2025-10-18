"""
Command-line interface for session-level processing.

Provides `pyflowreg-session` command for multi-recording workflows.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from pyflowreg.session.config import SessionConfig, get_array_task_id
from pyflowreg.session.stage1_compensate import run_stage1, run_stage1_array
from pyflowreg.session.stage2_between_avgs import run_stage2
from pyflowreg.session.stage3_valid_mask import run_stage3


def run_all_stages(config: SessionConfig, of_options_override: Optional[dict] = None):
    """
    Run all stages sequentially.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    of_options_override : dict, optional
        Override OFOptions parameters
    """
    print("=" * 60)
    print("STAGE 1: Per-recording motion correction")
    print("=" * 60)
    run_stage1(config, of_options_override)

    print("\n" + "=" * 60)
    print("STAGE 2: Inter-sequence displacement computation")
    print("=" * 60)
    middle_idx, center_file, displacement_fields = run_stage2(config)

    print("\n" + "=" * 60)
    print("STAGE 3: Valid mask alignment and final mask")
    print("=" * 60)
    run_stage3(config, middle_idx, displacement_fields)

    print("\n" + "=" * 60)
    print("SESSION PROCESSING COMPLETE")
    print("=" * 60)


def cmd_run(args):
    """Handle 'run' command."""
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = SessionConfig.from_file(config_path)

    # Parse OF options override if provided
    of_override = {}
    if args.of_params:
        for param in args.of_params:
            try:
                key, value = param.split("=", 1)
                # Try to parse as int, float, or keep as string
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        # Keep as string, handle booleans
                        if value.lower() in ["true", "false"]:
                            value = value.lower() == "true"
                of_override[key] = value
            except ValueError:
                print(f"Warning: Invalid OF parameter format: {param}. Use key=value")

    # Handle scheduler modes
    if args.array or config.scheduler == "array":
        # Array mode
        if args.stage:
            # Array job for specific stage
            if args.stage == "1":
                run_stage1_array(config, of_override)
            elif args.stage == "3":
                # Stage 3 doesn't really benefit from array parallelization
                # but we can run it normally
                run_stage3(config)
            else:
                # Stage 2 is not parallelizable across recordings
                print(f"Stage {args.stage} does not support array mode")
                sys.exit(1)
        else:
            # Run all stages with array for Stage 1
            task_id = get_array_task_id()
            if task_id is None:
                print("Error: Array mode requested but no task ID found in environment")
                sys.exit(1)

            # In array mode for full run, each task does Stage 1 only
            # Then a single task (or manual step) runs Stages 2 and 3
            print(f"Array task {task_id}: Running Stage 1 only")
            run_stage1_array(config, of_override)
            print(
                "\nNote: After all array tasks complete, run stages 2 and 3 manually:"
            )
            print(f"  pyflowreg-session run --config {args.config} --stage 2")
            print(f"  pyflowreg-session run --config {args.config} --stage 3")

    else:
        # Local or Dask mode
        if args.stage:
            # Run specific stage
            if args.stage == "1":
                run_stage1(config, of_override, task_index=args.index)
            elif args.stage == "2":
                run_stage2(config)
            elif args.stage == "3":
                run_stage3(config)
            else:
                print(f"Error: Invalid stage: {args.stage}. Use 1, 2, or 3")
                sys.exit(1)
        else:
            # Run all stages
            run_all_stages(config, of_override)


def cmd_dask(args):
    """Handle 'dask' subcommand."""
    try:
        from dask_jobqueue import SLURMCluster, PBSCluster, SGECluster, LSFCluster
        from dask.distributed import Client
    except ImportError:
        print("Error: dask-jobqueue not installed. Install with:")
        print("  pip install dask-jobqueue")
        sys.exit(1)

    # Load configuration
    config_path = Path(args.config)
    config = SessionConfig.from_file(config_path)

    # Select cluster type
    cluster_map = {
        "slurm": SLURMCluster,
        "pbs": PBSCluster,
        "sge": SGECluster,
        "lsf": LSFCluster,
    }

    if args.scheduler.lower() not in cluster_map:
        print(f"Error: Unknown scheduler: {args.scheduler}")
        print(f"Available: {', '.join(cluster_map.keys())}")
        sys.exit(1)

    ClusterClass = cluster_map[args.scheduler.lower()]

    # Create cluster
    # These are defaults - users should configure via dask config files
    cluster = ClusterClass(
        cores=args.cores,
        memory=args.memory,
        walltime=args.walltime,
    )

    cluster.scale(jobs=args.jobs)

    print(f"Created {args.scheduler.upper()} cluster:")
    print(f"  Cores per job: {args.cores}")
    print(f"  Memory per job: {args.memory}")
    print(f"  Walltime: {args.walltime}")
    print(f"  Number of jobs: {args.jobs}")
    print(f"\nDashboard: {cluster.dashboard_link}")

    # Connect client
    with Client(cluster) as client:
        print(f"\nClient connected: {client}")
        print("\nRunning Stage 1 in parallel using Dask...")

        # Import here to use within Dask context
        from pyflowreg.session.stage1_compensate import (
            discover_input_files,
            compensate_single_recording,
        )

        input_files = discover_input_files(config)

        # Map compensation tasks
        def compensate_task(idx):
            return compensate_single_recording(input_files[idx], config)

        futures = client.map(compensate_task, range(len(input_files)))
        results = client.gather(futures)

        print(f"\nStage 1 complete for {len(results)} recordings")

        # Stages 2 and 3 run sequentially (not parallelizable)
        print("\nRunning Stage 2...")
        middle_idx, center_file, displacement_fields = run_stage2(config)

        print("\nRunning Stage 3...")
        run_stage3(config, middle_idx, displacement_fields)

        print("\nAll stages complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PyFlowReg session-level multi-recording processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages with local scheduler
  pyflowreg-session run --config session.toml

  # Run only Stage 1
  pyflowreg-session run --config session.toml --stage 1

  # Run Stage 1 in array job mode (auto-detects task ID)
  pyflowreg-session run --config session.toml --stage 1 --array

  # Run all stages with Dask on SLURM
  pyflowreg-session dask --config session.toml --scheduler slurm

  # Override OFOptions parameters
  pyflowreg-session run --config session.toml --of-params alpha=10 iterations=50
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run session processing")
    run_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to session configuration file (.toml or .yaml)",
    )
    run_parser.add_argument(
        "--stage",
        "-s",
        choices=["1", "2", "3"],
        help="Run only specified stage (default: all stages)",
    )
    run_parser.add_argument(
        "--array",
        action="store_true",
        help="Run in HPC array job mode (auto-detect task ID from environment)",
    )
    run_parser.add_argument(
        "--index",
        "-i",
        type=int,
        help="Process only recording at this index (for manual array jobs, 0-based)",
    )
    run_parser.add_argument(
        "--of-params",
        nargs="*",
        metavar="KEY=VALUE",
        help="Override OFOptions parameters (e.g., alpha=10 iterations=50)",
    )
    run_parser.set_defaults(func=cmd_run)

    # 'dask' command
    dask_parser = subparsers.add_parser("dask", help="Run with Dask distributed")
    dask_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to session configuration file",
    )
    dask_parser.add_argument(
        "--scheduler",
        choices=["slurm", "pbs", "sge", "lsf"],
        default="slurm",
        help="HPC scheduler type (default: slurm)",
    )
    dask_parser.add_argument(
        "--cores", type=int, default=4, help="Cores per job (default: 4)"
    )
    dask_parser.add_argument(
        "--memory", default="8GB", help="Memory per job (default: 8GB)"
    )
    dask_parser.add_argument(
        "--walltime", default="02:00:00", help="Walltime per job (default: 02:00:00)"
    )
    dask_parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=10,
        help="Number of jobs to launch (default: 10)",
    )
    dask_parser.set_defaults(func=cmd_dask)

    # Parse and execute
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
