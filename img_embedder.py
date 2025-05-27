"""Iamge Embedder

This image embedder embeds images using OpenAI's CLIP model. The embeddings
are stored in pandas Dataframe and exported in .h5 format

"""

import clip
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os
import argparse
import time

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-image_dir', type=str, help='Path to image folder')
parser.add_argument('--device', type=str, default='cpu', help='Device on which the model should run')
args = parser.parse_args()

img_dir = args.image_dir

# Load individual path of all images
print("Collect image paths:...", end='\r')
image_paths = [
            os.path.join(img_dir, img_fname) for img_fname in os.listdir(img_dir)
            if img_fname.lower().endswith(('.png', '.jpeg', '.jpg'))
            ]
print("Collect image paths: success")

# Load CLIP model and set to inference mode
print("Load CLIP model:...", end=' \r')
clip_model, preprocess_clip = clip.load("ViT-L/14@336px", device=args.device)
clip_model.eval()
print("Load CLIP model: success")


# Generate CLIP image embedding for every picture
embs = np.empty(shape=(len(image_paths), 768), dtype=np.float32) # allocate empty array for embeddings
paths_list = [] # list with image paths

def generate_clip_emb(path, device):
    """Generates image embeddings

    Args:
        path: str path to image
        device: str device to use
    """
    with torch.no_grad():
        img = Image.open(path).convert("RGB")
        img = preprocess_clip(img)
        img = img.unsqueeze(0).to(device)
        emb = clip_model.encode_image(img)
        emb /= emb.norm(dim=1, keepdim=True) 
        return emb

print("Generate image embeddings:...")
for idx, path in enumerate(image_paths):
    start = time.perf_counter()
    emb = generate_clip_emb(path, args.device)
    embs[idx] = emb.cpu().numpy()
    paths_list.append(path)
    progress = (idx+1) / len(image_paths)
    end = time.perf_counter()
    delta_t = end-start
    print(f"{np.round(progress*100,2)}% {np.round(delta_t,3)}s/img", end=' \r')

print("Generate image embeddings: success")

# Save image embeddings with image path
print("Save image embeddings:...", '\r')
df_embs = pd.DataFrame([paths_list, embs]).transpose()
df_embs.to_hdf("img_embeddings.h5", key="df_embs", mode="w")
print("Save image embeddings: success")
