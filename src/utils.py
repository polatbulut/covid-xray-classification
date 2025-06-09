import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(path, model, optimizer=None):
    chk = torch.load(path, map_location='cpu')
    model.load_state_dict(chk['model_state'])
    if optimizer and 'optim_state' in chk:
        optimizer.load_state_dict(chk['optim_state'])
    return chk.get('epoch', None)

def compute_metrics(preds, labels, average='macro'):
    return (
        accuracy_score(labels, preds),
        precision_score(labels, preds, average=average, zero_division=0),
        recall_score(labels, preds, average=average, zero_division=0),
        f1_score(labels, preds, average=average, zero_division=0)
    )
