import os
from torch.utils.data import Dataset
from PIL import Image

class XRayDataset(Dataset):
    """
    A PyTorch Dataset for loading chest X-ray images organized in class-specific folders.
    Folder structure:
        root_dir/
            ClassA/
                img1.png
                img2.png
            ClassB/
                img3.png
    """
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(cls_folder, fname)
                    self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label