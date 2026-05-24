from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T
import numpy as np


class HistologyDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_paths = sorted(
            [p for p in self.image_dir.glob("*.png")
             if "-labelled" not in p.name]
        )

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        mask_name = image_path.stem + "-labelled.png"
        mask_path = self.mask_dir / mask_name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = np.array(image)
        mask = np.array(mask)

        # Convert image to tensor
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1) / 255.0

        # Convert mask to tensor
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask