"""Command-line interface for the ZEBRA-Prop package."""

import argparse
import sys


def main():
    """CLI entry point for running ZEBRA-Prop workflows."""
    # Keep the CLI surface minimal: training + preprocessing toolkit.
    parser = argparse.ArgumentParser(prog="zebra-prop")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("train", help="Run training with Hydra overrides")
    subparsers.add_parser("toolkit", help="Run zebra-tool-kit subcommands")

    args, unknown = parser.parse_known_args()

    if args.command in (None, "train"):
        # Forward unknown CLI overrides directly to Hydra.
        sys.argv = [sys.argv[0]] + unknown
        from .train import main_lightning

        main_lightning()
        return

    if args.command == "toolkit":
        from zebra_tool_kit.cli import main as toolkit_main

        raise SystemExit(toolkit_main(unknown))

    # Show help for unsupported commands.
    parser.print_help()
