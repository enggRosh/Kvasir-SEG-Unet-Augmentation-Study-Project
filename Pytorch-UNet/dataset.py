import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np

class KvasirSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale for binary segmentation

        if self.transform is not None:
            image = self.transform(image)
            mask = transforms.Resize((128, 128))(mask)
            mask = torch.from_numpy((np.array(mask) > 0).astype('float32')).unsqueeze(0).contiguous()

        return {'image': image, 'mask': mask}
