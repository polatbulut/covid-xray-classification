import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def save_checkpoint(state: dict, filename: str):
    """Save model + optimizer state to disk."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """Load state dict into model (and optimizer if provided)."""
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None and 'optim_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optim_state'])
    return checkpoint.get('epoch', None)


def compute_metrics(preds, targets, average='macro'):
    """Compute basic classification metrics."""
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average=average, zero_division=0)
    rec = recall_score(targets, preds, average=average, zero_division=0)
    f1 = f1_score(targets, preds, average=average, zero_division=0)
    return acc, prec, rec, f1
