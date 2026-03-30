"""Command line interface for zebra_tool_kit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .description import generate_description_from_csv
from .featurizer import AVAILABLE_FEATURIZERS, generate_feature_csvs
from .gui import run_gui


def _load_data_dir_from_config(config_path: Path) -> Path:
    """Read `data_dir` from config file. Fallback to `./data` on failure."""
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        data_dir = cfg.get("data_dir", "./data")
        return Path(str(data_dir))
    except Exception:
        return Path("./data")


def _resolve_data_dir(data_dir: str | None, config_path: str) -> Path:
    config = Path(config_path)
    resolved = Path(data_dir) if data_dir else _load_data_dir_from_config(config)
    return resolved.expanduser().resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zebra-tool-kit",
        description=(
            "Utilities for descriptor generation from CIF and "
            "description CSV generation."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    featurize = subparsers.add_parser(
        "featurize",
        help="Generate matminer descriptors from CIF files.",
    )
    featurize.add_argument(
        "--config-path",
        default="config/config.yaml",
        help="Config path used to resolve default data_dir.",
    )
    featurize.add_argument(
        "--data-dir",
        default=None,
        help="Override data_dir from config.",
    )
    featurize.add_argument(
        "--cif-dir",
        default=None,
        help="Directory containing .cif files (default: <data_dir>/cif).",
    )
    featurize.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save descriptor CSV files (default: <data_dir>/descriptor).",
    )
    featurize.add_argument(
        "--featurizer",
        default="all",
        choices=["all", *AVAILABLE_FEATURIZERS],
        help="Descriptor type to generate.",
    )

    describe = subparsers.add_parser(
        "describe",
        help="Generate text description from one descriptor CSV.",
    )
    describe.add_argument(
        "--config-path",
        default="config/config.yaml",
        help="Config path used to resolve default data_dir.",
    )
    describe.add_argument(
        "--data-dir",
        default=None,
        help="Override data_dir from config.",
    )
    describe.add_argument(
        "--input-csv",
        required=True,
        help="Descriptor CSV path.",
    )
    describe.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: <data_dir>/description/sample/description.csv).",
    )
    describe.add_argument(
        "--template",
        default=None,
        help="Inline template. Use placeholders like {{formula}}, {{PymatgenData mean row}}.",
    )
    describe.add_argument(
        "--template-file",
        default=None,
        help="Text file containing template string.",
    )
    describe.add_argument(
        "--include-zero",
        action="store_true",
        help="Keep zero-valued descriptors when no template is provided.",
    )
    describe.add_argument(
        "--formula-simplified",
        action="store_true",
        help="Apply formula simplification to all rows before sentence generation.",
    )
    describe.add_argument(
        "--integerize",
        action="store_true",
        help="Apply integerize scaling to all numeric labels before sentence generation.",
    )

    gui = subparsers.add_parser(
        "gui",
        help="Launch GUI template editor for description generation.",
    )
    gui.add_argument(
        "--config-path",
        default="config/config.yaml",
        help="Config path used to resolve default data_dir.",
    )
    gui.add_argument(
        "--data-dir",
        default=None,
        help="Override data_dir from config.",
    )
    gui.add_argument(
        "--input-csv",
        default=None,
        help="Optional descriptor CSV to preload.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run zebra_tool_kit CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "featurize":
        data_dir = _resolve_data_dir(args.data_dir, args.config_path)
        cif_dir = Path(args.cif_dir).expanduser().resolve() if args.cif_dir else data_dir / "cif"
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else data_dir / "descriptor"
        )

        saved = generate_feature_csvs(
            cif_dir=cif_dir,
            output_dir=output_dir,
            featurizer=args.featurizer,
        )
        print(f"[zebra-tool-kit] completed: {len(saved)} file(s)")
        return 0

    if args.command == "describe":
        data_dir = _resolve_data_dir(args.data_dir, args.config_path)
        output_csv = (
            Path(args.output_csv).expanduser().resolve()
            if args.output_csv
            else (data_dir / "description" / "sample" / "description.csv")
        )

        output = generate_description_from_csv(
            input_csv=Path(args.input_csv),
            output_csv=output_csv,
            template=args.template,
            template_file=Path(args.template_file) if args.template_file else None,
            include_zero=args.include_zero,
            formula_simplified=args.formula_simplified,
            integerize=args.integerize,
        )
        print(f"[zebra-tool-kit] saved: {output}")
        return 0

    if args.command == "gui":
        data_dir = _resolve_data_dir(args.data_dir, args.config_path)
        run_gui(data_dir=data_dir, input_csv=args.input_csv)
        return 0

    parser.print_help()
    return 1
