from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from covid_xray import __version__
from covid_xray.cli import build_parser, main

from .conftest import CLASS_NAMES, write_image_tree


def test_version_flag_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_missing_subcommand_is_an_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "command", ["train", "evaluate", "split", "plot-history", "check-env"]
)
def test_every_subcommand_is_registered(command: str) -> None:
    parser = build_parser()
    args = parser.parse_args([command, *_minimal_args(command)])
    assert args.command == command
    assert callable(args.handler)


def _minimal_args(command: str) -> list[str]:
    if command in {"train", "evaluate"}:
        return ["--config", "configs/resnet50.yaml"]
    if command == "plot-history":
        return ["results/history/resnet50_augmented.json"]
    return []


def test_train_reports_a_bad_config_without_a_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"model": {"name": "simple"}}), encoding="utf-8")
    assert main(["train", "--config", str(bad)]) == 1


def test_train_reports_a_missing_config(tmp_path: Path) -> None:
    assert main(["train", "--config", str(tmp_path / "absent.yaml")]) == 1


def test_evaluate_reports_a_missing_checkpoint(tmp_path: Path, config_yaml: Path) -> None:
    exit_code = main(
        ["evaluate", "--config", str(config_yaml), "--checkpoint", str(tmp_path / "no.pth")]
    )
    assert exit_code == 1


def test_check_env_succeeds() -> None:
    assert main(["check-env"]) == 0


def test_split_end_to_end(tmp_path: Path, raw_root: Path) -> None:
    output = tmp_path / "processed"
    assert main(["split", "--raw-dir", str(raw_root), "--output-dir", str(output)]) == 0
    assert sorted(p.name for p in (output / "train").iterdir()) == sorted(CLASS_NAMES)


def test_split_reports_a_missing_raw_directory(tmp_path: Path) -> None:
    assert main(["split", "--raw-dir", str(tmp_path / "absent")]) == 1


def test_split_rejects_ratios_that_do_not_sum_to_one(tmp_path: Path, raw_root: Path) -> None:
    exit_code = main(
        [
            "split",
            "--raw-dir",
            str(raw_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--train-ratio",
            "0.9",
        ]
    )
    assert exit_code == 1


def test_plot_history_writes_both_figures(tmp_path: Path) -> None:
    history = tmp_path / "run_history.json"
    history.write_text(
        json.dumps(
            {
                "run_name": "run",
                "records": [
                    {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_f1": 0.7},
                    {"epoch": 2, "train_loss": 0.3, "val_loss": 0.35, "val_f1": 0.8},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "figures"
    assert main(["plot-history", str(history), "--output-dir", str(output)]) == 0
    assert (output / "loss_curves.png").is_file()
    assert (output / "f1_curves.png").is_file()


def test_plot_history_reports_a_malformed_file(tmp_path: Path) -> None:
    history = tmp_path / "broken.json"
    history.write_text("{not json", encoding="utf-8")
    assert main(["plot-history", str(history)]) == 1


def test_train_runs_end_to_end(tmp_path: Path, config_yaml: Path) -> None:
    """One real epoch through the CLI, checkpoint and history included."""
    assert main(["train", "--config", str(config_yaml), "--epochs", "1", "--no-progress"]) == 0
    assert (tmp_path / "models" / "run.pth").is_file()
    assert (tmp_path / "models" / "run_history.json").is_file()


def test_evaluate_runs_end_to_end(tmp_path: Path, config_yaml: Path) -> None:
    assert main(["train", "--config", str(config_yaml), "--epochs", "1", "--no-progress"]) == 0
    output = tmp_path / "results"
    exit_code = main(
        [
            "evaluate",
            "--config",
            str(config_yaml),
            "--output-dir",
            str(output),
            "--no-progress",
        ]
    )
    assert exit_code == 0
    assert (output / "metrics.json").is_file()


@pytest.fixture
def config_yaml(tmp_path: Path) -> Path:
    """A YAML config pointing at synthetic splits, ready for the CLI to consume."""
    for split in ("train", "val", "test"):
        write_image_tree(tmp_path / "data" / split, images_per_class=4, size=(16, 16))
    payload = {
        "data": {
            "train_dir": str(tmp_path / "data" / "train"),
            "val_dir": str(tmp_path / "data" / "val"),
            "test_dir": str(tmp_path / "data" / "test"),
            "img_size": [16, 16],
        },
        "model": {"name": "simple", "pretrained": False},
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "lr": 0.001,
            "checkpoint_path": str(tmp_path / "models" / "run.pth"),
            "seed": 0,
        },
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path
