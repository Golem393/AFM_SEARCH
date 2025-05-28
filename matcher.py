"""Matcher Algorithm

This file contains the matching algorithm to get the best query-image(s) matches
based on embeddings. 
Use:
    python matcher.py -img_emb "example.h5" -key_emb "example.npy"

TODO: This implementation is preliminary and requires future rework
"""
import numpy as np
import pandas as pd
import argparse
import csv

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('-img_emb', type=str, help='Path to file containing image embeddings (.h5 file)')
parser.add_argument('-key_emb', type=str, help='Path to file containing keyword embeddings (.npy file)')
args = parser.parse_args()

# Load image embeddings
df_emb = pd.read_hdf(args.img_emb)
img_embs = df_emb[1].to_numpy()
img_embs = np.array(img_embs.tolist())

# Load query embeddings
query_emb = np.load(args.key_emb)

# Get paths to images
paths = df_emb[0].tolist()

# Compute similarities
print("[INFO] Computing similarities:...", end='\r')
similarities = img_embs @ query_emb.T
print("[INFO] Computing similarities: success")

def mean_matcher(similarities, paths):
    """Matcher Algorithm based on mean over keywords.

    Args:
        similarities: np.array Array containing similarity scores
        paths: List List of same lenght as similarites containing image paths
    """
    means = np.mean(similarities, axis=1)
    criterion = np.percentile(means, 95)
    print(f"[DEBUG] similarity threshold: {criterion}")
    k = len(means[means >= criterion])
    print(f"[INFO] Found {k} matches")
    idcs = np.argsort(means)[::-1][:k]
    return [paths[idx] for idx in idcs]

selected_imgs = mean_matcher(similarities, paths)

with open('matches.csv', mode="w") as file:
    writer = csv.writer(file)
    writer.writerow(selected_imgs)