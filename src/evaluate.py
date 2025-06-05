import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import XRayDataset
from src.model import SimpleCNN, get_transfer_model
from src.utils import load_checkpoint, compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Path to YAML config file")
    parser.add_argument('--weights', required=True, help="Path to trained model checkpoint")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build test transforms (no augmentation)
    tf_test = transforms.Compose([
        transforms.Resize(cfg['data']['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['data']['mean'], cfg['data']['std'])
    ])

    # Load test dataset & DataLoader
    ds_test = XRayDataset(cfg['data']['test_dir'], tf_test)
    loader = DataLoader(
        ds_test,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers']
    )

    # Instantiate the same model architecture, then load weights
    num_classes = len(ds_test.classes)
    if cfg['model']['type'].lower() == 'simple':
        model = SimpleCNN(num_classes)
    else:
        model = get_transfer_model(cfg['model']['type'], num_classes)

    model = model.to(device)
    load_checkpoint(args.weights, model)
    model.eval()

    # Run inference on test set
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())

    # Compute & print metrics
    acc, prec, rec, f1 = compute_metrics(all_preds, all_labels)
    print(f"Test  Acc: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

if __name__ == '__main__':
    main()
