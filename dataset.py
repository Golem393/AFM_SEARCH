import os
from torch.utils.data import Dataset
from PIL import Image

class UnlabelledImageDataset(Dataset):
    """Dataset Class for Unlabelled Images
    
    Class to be used with PyTorch to create dataset with unlabelled images

    Attributes:
        img_dir: Path to the directory containing the images
        transform: Apply transformations for data augmentation
        preprocess: Apply preprocessing step for use with foundation models
        model: Indicate which model you are using to use the right preprocess mode
    """
    
    def __init__(self, img_dir, transform=None, preprocess=None, model="standard"):
        self.img_dir = img_dir
        self.image_paths = [
            os.path.join(img_dir, img_fname) for img_fname in os.listdir(img_dir)
            if img_fname.lower().endswith(('.png', '.jpeg', '.jpg'))
        ]
        self.transform = transform
        self.preprocess = preprocess
        self.model = model
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
       
        if self.model == "standard" and self.preprocess:
            image = self.preprocess(image)
        elif self.model == "git" and self.preprocess:
            image = self.preprocess(image = image, return_tensors="pt")
        else:
            raise ValueError

        return image, image_path