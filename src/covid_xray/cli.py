"""The ``covid-xray`` command-line interface.

Sub-commands are thin: they parse arguments, apply any overrides to the loaded
configuration, and delegate to the library. Importing torch costs a couple of
seconds, so the heavy modules are imported inside the handlers that need them
and ``covid-xray --help`` stays responsive.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from covid_xray import __version__
from covid_xray.config import ExperimentConfig
from covid_xray.logging_config import configure_logging

LOGGER: Final = logging.getLogger("covid_xray")

DEFAULT_RESULTS_DIR: Final = Path("results")
DEFAULT_RAW_DIR: Final = Path("data/raw")
DEFAULT_PROCESSED_DIR: Final = Path("data/processed")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags every sub-command understands."""
    parser.add_argument("-v", "--verbose", action="store_true", help="emit debug logging")


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by the train and evaluate sub-commands."""
    parser.add_argument(
        "--config", type=Path, required=True, help="path to the experiment YAML file"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="device override, e.g. cuda, cuda:1, mps or cpu (default: best available)",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="disable tqdm progress bars"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="covid-xray",
        description="Train and evaluate chest X-ray classifiers.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train a model from a config file")
    _add_run_arguments(train)
    _add_common_arguments(train)
    train.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    train.add_argument("--seed", type=int, default=None, help="override training.seed")
    train.add_argument(
        "--checkpoint", type=Path, default=None, help="override training.checkpoint_path"
    )
    train.set_defaults(handler=_handle_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint on the test split")
    _add_run_arguments(evaluate)
    _add_common_arguments(evaluate)
    evaluate.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint to evaluate (default: training.checkpoint_path from the config)",
    )
    evaluate.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"where to write metrics and figures (default: {DEFAULT_RESULTS_DIR}/<run name>)",
    )
    evaluate.add_argument(
        "--max-misclassified",
        type=int,
        default=9,
        help="number of misclassified examples to plot (default: 9)",
    )
    evaluate.set_defaults(handler=_handle_evaluate)

    split = subparsers.add_parser("split", help="split the raw dataset into train/val/test")
    _add_common_arguments(split)
    split.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    split.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    split.add_argument("--train-ratio", type=float, default=0.70)
    split.add_argument("--val-ratio", type=float, default=0.15)
    split.add_argument("--test-ratio", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument(
        "--link", action="store_true", help="symlink images instead of copying them"
    )
    split.add_argument(
        "--overwrite", action="store_true", help="replace an existing, non-empty output directory"
    )
    split.set_defaults(handler=_handle_split)

    plot = subparsers.add_parser("plot-history", help="plot curves from saved history files")
    _add_common_arguments(plot)
    plot.add_argument(
        "history", type=Path, nargs="+", help="one or more *_history.json files to overlay"
    )
    plot.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    plot.set_defaults(handler=_handle_plot_history)

    check = subparsers.add_parser("check-env", help="report the detected PyTorch runtime")
    _add_common_arguments(check)
    check.set_defaults(handler=_handle_check_env)

    return parser


def _apply_train_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Return a copy of ``config`` with any command-line overrides applied."""
    training = config.training
    updated = training
    if args.epochs is not None:
        LOGGER.info("Override: training.epochs=%s", args.epochs)
        updated = dataclasses.replace(updated, epochs=args.epochs)
    if args.seed is not None:
        LOGGER.info("Override: training.seed=%s", args.seed)
        updated = dataclasses.replace(updated, seed=args.seed)
    if args.checkpoint is not None:
        LOGGER.info("Override: training.checkpoint_path=%s", args.checkpoint)
        updated = dataclasses.replace(updated, checkpoint_path=args.checkpoint)
    if updated is training:
        return config
    return dataclasses.replace(config, training=updated)


def _handle_train(args: argparse.Namespace) -> int:
    """Run the ``train`` sub-command."""
    from covid_xray.runtime import resolve_device
    from covid_xray.training import fit

    config = _apply_train_overrides(ExperimentConfig.from_yaml(args.config), args)
    device = resolve_device(args.device)
    fit(config, device, show_progress=not args.no_progress)
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    """Run the ``evaluate`` sub-command."""
    from covid_xray.evaluation import evaluate, write_artifacts
    from covid_xray.runtime import resolve_device

    config = ExperimentConfig.from_yaml(args.config)
    checkpoint_path = args.checkpoint or config.training.checkpoint_path
    output_dir = args.output_dir or DEFAULT_RESULTS_DIR / config.run_name

    device = resolve_device(args.device)
    result = evaluate(config, checkpoint_path, device, show_progress=not args.no_progress)
    LOGGER.info("Test set: %s", result.metrics.format_summary())
    LOGGER.info("Per-class report:\n%s", result.report)
    write_artifacts(result, output_dir, max_misclassified=args.max_misclassified)
    return 0


def _handle_split(args: argparse.Namespace) -> int:
    """Run the ``split`` sub-command."""
    from covid_xray.datasplit import SplitRatios, split_dataset

    ratios = SplitRatios(train=args.train_ratio, val=args.val_ratio, test=args.test_ratio)
    split_dataset(
        args.raw_dir,
        args.output_dir,
        ratios=ratios,
        seed=args.seed,
        link=args.link,
        overwrite=args.overwrite,
    )
    LOGGER.info("Split written to %s", args.output_dir)
    return 0


def _handle_plot_history(args: argparse.Namespace) -> int:
    """Run the ``plot-history`` sub-command."""
    from covid_xray.history import TrainingHistory
    from covid_xray.plotting import plot_f1_curves, plot_loss_curves

    histories = [TrainingHistory.from_json(path) for path in args.history]
    plot_loss_curves(histories, args.output_dir / "loss_curves.png")
    plot_f1_curves(histories, args.output_dir / "f1_curves.png")
    return 0


def _handle_check_env(args: argparse.Namespace) -> int:
    """Run the ``check-env`` sub-command."""
    del args
    import torch

    from covid_xray.runtime import describe_device, resolve_device

    LOGGER.info("covid-xray %s", __version__)
    LOGGER.info("torch %s (CUDA build: %s)", torch.__version__, torch.version.cuda or "none")
    LOGGER.info(
        "CUDA available: %s (%d device(s))",
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    LOGGER.info("MPS available: %s", torch.backends.mps.is_available())
    LOGGER.info("Selected device: %s", describe_device(resolve_device()))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``covid-xray`` console script.

    Returns:
        ``0`` on success, ``1`` when a known error is raised, ``130`` on Ctrl-C.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    handler = args.handler
    try:
        exit_code: int = handler(args)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted.")
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        # ConfigError and HistoryError derive from ValueError; CheckpointError,
        # DatasetError and SplitError derive from RuntimeError.
        LOGGER.error("%s", exc)
        LOGGER.debug("Traceback:", exc_info=True)
        return 1
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
