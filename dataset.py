import os
from torch.utils.data import Dataset
from PIL import Image


# Dataset class to test unlabelled images
class UnlabelledImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_paths = [
            os.path.join(img_dir, img_fname) for img_fname in os.listdir(img_dir)
            if img_fname.lower().endswith(('.png', '.jpeg', '.jpg'))
        ]
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path