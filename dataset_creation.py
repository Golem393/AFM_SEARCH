import subprocess
import os
import json
import pandas as pd
import numpy as np
from matching_algorithms import MeanMatcher

FLICKR8K_IMAGES = "/Users/benjaminkasper/Documents/Uni/RCI/Module/in2106/dataset/Flicker8k_Dataset"
FLICKR8K_CAPTIONS = "/Users/benjaminkasper/Documents/Uni/RCI/Module/in2106/dataset/Flickr8k_text/Flickr8k.token.txt"
FLICKR8K_EMB = "tests/flickr8k/"

HAWAII_IMAGES = 'imgs/hawaii'

def generate_flickr8k_emb(img_path, img_emb_path):
    
    file_path = os.path.join(img_emb_path, "img_embeddings.h5")
    if not os.path.isfile(file_path):
        # embedd all images
        subprocess.run(["python", "img_embedder.py", 
                        "-image_dir", img_path, 
                        "-target_dir", img_emb_path,
                        "--device", "mps"], 
                        check=True, 
                        capture_output=False)
    else:
        print("Embeddings already exist!")

def flickr8k_captions(source_path, target_path):
    image_caption_dict = {}

    with open(source_path, 'r', encoding='utf-8') as file:
        for line in file:
            image_id_caption, caption = line.strip().split('\t')
            image_id = image_id_caption.split('#')[0]
            caption = caption.strip().split(' .')[0]

            if image_id not in image_caption_dict:
                image_caption_dict[image_id] = []

            image_caption_dict[image_id].append(caption)

    target_path = os.path.join(target_path, 'flickr8k_captions.json')

    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(image_caption_dict, f, indent=2, ensure_ascii=False)

if  __name__ == '__main__':
    flickr8k_captions(FLICKR8K_CAPTIONS, 'tests/flickr8k/')
    generate_flickr8k_emb(FLICKR8K_IMAGES, FLICKR8K_EMB)

