"""The training loop: class weighting, mixed precision, early stopping, checkpointing."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from covid_xray.checkpoints import Checkpoint, save_checkpoint
from covid_xray.config import ClassWeighting, ExperimentConfig, MonitorMetric
from covid_xray.data import (
    Batch,
    XRayDataset,
    build_dataloader,
    build_eval_transform,
    build_train_transform,
)
from covid_xray.early_stopping import EarlyStopping
from covid_xray.history import EpochRecord, TrainingHistory
from covid_xray.metrics import ClassificationMetrics, compute_metrics
from covid_xray.models import count_parameters, create_model
from covid_xray.runtime import amp_is_supported, autocast_context, describe_device, seed_everything

LOGGER: Final = logging.getLogger(__name__)


def compute_class_weights(counts: Sequence[int], strategy: ClassWeighting) -> list[float] | None:
    """Return per-class loss weights, or ``None`` when weighting is disabled.

    Args:
        counts: Number of training images per class.
        strategy: ``"none"`` for an unweighted loss, ``"inverse"`` for
            ``total / count``, or ``"balanced"`` for scikit-learn's
            ``total / (n_classes * count)``. ``"inverse"`` scales the loss by a
            factor of ``n_classes`` relative to ``"balanced"``, which in
            practice acts like a larger learning rate.

    Raises:
        ValueError: If ``counts`` is empty.
    """
    if not counts:
        raise ValueError("Cannot compute class weights from an empty count sequence.")
    if strategy == "none":
        return None
    total = sum(counts)
    if strategy == "inverse":
        return [total / count if count else 0.0 for count in counts]
    return [total / (len(counts) * count) if count else 0.0 for count in counts]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Batch],
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: OneCycleLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    grad_clip: float | None = None,
    show_progress: bool = True,
) -> float:
    """Run one training epoch and return the mean loss per sample."""
    model.train()
    use_amp = scaler.is_enabled()
    running_loss = 0.0
    seen = 0

    for images, labels in tqdm(loader, desc="train", leave=False, disable=not show_progress):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, enabled=use_amp):
            outputs: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(outputs, labels)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # A decreased scale means the step was skipped because of inf/NaN
        # gradients; advancing the schedule then would desynchronise the
        # one-cycle LR from the optimiser.
        if scaler.get_scale() >= scale_before:
            scheduler.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size

    return running_loss / max(seen, 1)


def validate(
    model: nn.Module,
    loader: DataLoader[Batch],
    criterion: nn.Module,
    device: torch.device,
    *,
    show_progress: bool = True,
) -> tuple[float, ClassificationMetrics]:
    """Evaluate on the validation split and return the mean loss and metrics."""
    model.eval()
    use_amp = amp_is_supported(device)
    running_loss = 0.0
    seen = 0
    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val", leave=False, disable=not show_progress):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast_context(device, enabled=use_amp):
                outputs: torch.Tensor = model(images)
                loss: torch.Tensor = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return running_loss / max(seen, 1), compute_metrics(targets, predictions)


def _monitored_value(
    val_loss: float, metrics: ClassificationMetrics, monitor: MonitorMetric
) -> float:
    """Return the value of the metric that early stopping and checkpointing track."""
    if monitor == "val_loss":
        return val_loss
    if monitor == "val_accuracy":
        return metrics.accuracy
    return metrics.f1


def fit(
    config: ExperimentConfig,
    device: torch.device,
    *,
    show_progress: bool = True,
) -> TrainingHistory:
    """Train a model end to end and return its per-epoch history.

    The best epoch, judged by ``training.monitor``, is written to
    ``training.checkpoint_path``. The history is written after every epoch so an
    interrupted run still leaves a usable record.
    """
    training = config.training
    seed_everything(training.seed)
    LOGGER.info("Device: %s | seed: %d", describe_device(device), training.seed)

    train_dataset = XRayDataset(config.data.train_dir, build_train_transform(config.data))
    # Pin the validation split to the training class order so the label indices
    # cannot drift between the two directories.
    val_dataset = XRayDataset(
        config.data.val_dir,
        build_eval_transform(config.data),
        classes=train_dataset.classes,
    )
    train_loader = build_dataloader(train_dataset, training, device, shuffle=True)
    val_loader = build_dataloader(val_dataset, training, device, shuffle=False)

    classes = train_dataset.classes
    counts = train_dataset.class_counts()
    LOGGER.info(
        "Classes (%d): %s",
        len(classes),
        ", ".join(f"{name}={count}" for name, count in zip(classes, counts, strict=True)),
    )
    LOGGER.info("Samples: %d train, %d val", len(train_dataset), len(val_dataset))

    model = create_model(config.model.name, len(classes), pretrained=config.model.pretrained)
    model.to(device)
    LOGGER.info(
        "Model '%s' with %s trainable parameters",
        config.model.name,
        f"{count_parameters(model):,}",
    )

    weights = compute_class_weights(counts, training.class_weighting)
    weight_tensor = (
        None if weights is None else torch.tensor(weights, dtype=torch.float32, device=device)
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = Adam(model.parameters(), lr=training.lr, weight_decay=training.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=training.lr,
        steps_per_epoch=len(train_loader),
        epochs=training.epochs,
        pct_start=training.pct_start,
        anneal_strategy=training.anneal_strategy,
    )
    use_amp = amp_is_supported(device)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    if not use_amp:
        LOGGER.info("Mixed precision is disabled on %s; training in float32.", device.type)

    stopper = EarlyStopping(patience=training.patience, mode=training.monitor_mode)
    history = TrainingHistory(run_name=config.run_name)
    history_path = training.resolved_history_path

    for epoch in range(1, training.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            grad_clip=training.grad_clip,
            show_progress=show_progress,
        )
        val_loss, metrics = validate(
            model, val_loader, criterion, device, show_progress=show_progress
        )

        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_f1=metrics.f1,
                val_accuracy=metrics.accuracy,
                val_precision=metrics.precision,
                val_recall=metrics.recall,
                learning_rate=learning_rate,
            )
        )
        history.to_json(history_path)

        improved = stopper.update(_monitored_value(val_loss, metrics, training.monitor))
        LOGGER.info(
            "Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | %s%s",
            epoch,
            training.epochs,
            train_loss,
            val_loss,
            metrics.format_summary(),
            " | best" if improved else "",
        )

        if improved:
            save_checkpoint(
                Checkpoint(
                    epoch=epoch,
                    model_name=config.model.name,
                    classes=classes,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    metrics={"val_loss": val_loss, **metrics.as_dict(prefix="val_")},
                ),
                training.checkpoint_path,
            )

        if stopper.should_stop:
            LOGGER.info(
                "Early stopping: %s has not improved for %d epoch(s).",
                training.monitor,
                stopper.epochs_without_improvement,
            )
            break

    best = history.best(training.monitor, training.monitor_mode)
    if best is not None:
        LOGGER.info(
            "Best epoch %d with %s=%.4f. Checkpoint: %s | history: %s",
            best.epoch,
            training.monitor,
            best.value_of(training.monitor),
            training.checkpoint_path,
            history_path,
        )
    return history
