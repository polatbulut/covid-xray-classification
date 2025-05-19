import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import XRayDataset
from model import SimpleCNN, get_transfer_model
from utils import load_checkpoint, compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model checkpoint')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test transforms
    test_tf = transforms.Compose([
        transforms.Resize(cfg['data']['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['data']['mean'], cfg['data']['std']),
    ])

    # Dataset & loader
    test_ds = XRayDataset(cfg['data']['test_dir'], transform=test_tf)
    test_loader = DataLoader(test_ds,
                             batch_size=cfg['training']['batch_size'],
                             shuffle=False,
                             num_workers=cfg['training']['num_workers'])

    # Model
    num_classes = len(test_ds.classes)
    if cfg['model']['type'].lower() == 'simple':
        model = SimpleCNN(num_classes)
    else:
        model = get_transfer_model(cfg['model']['type'], num_classes)
    model.to(device)

    # Load weights
    load_checkpoint(args.weights, model)
    model.eval()

    # Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Compute metrics
    acc, prec, rec, f1 = compute_metrics(all_preds, all_labels)
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"F1 Score      : {f1:.4f}")

if __name__ == '__main__':
    main()
