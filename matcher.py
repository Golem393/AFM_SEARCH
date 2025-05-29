"""Matcher Algorithm

This file contains the matching algorithm to get the best query-image(s) matches
based on embeddings. 
Use:
    python matcher.py -img_emb "example.h5" -key_emb "example.npy"

TODO: This implementation is preliminary and requires future rework with better matching algorithms
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

def pareto_dominance(s):
    """Calculates dominance matrix for Pareto Front Matching

    Calculates domination of all images to each other for Pareto Front.

    Args:
        s: np.array similarities matrix of shape (n_img, n_keys)

    Returns:
        domination_mat: np.array domination matrix
    """
    # Compute all possible dominations_
    # Criterion 1: i dominates j if s[i][k] >= s[j][k] for all k
    crit1 = np.all(s[:,:,np.newaxis].repeat(s.shape[0], axis=-1) >= s.T, axis=1)

    # Criterion 2: i dominates j if s[i][k] > s[j][k] for at least one k
    crit2 = np.any(s[:,:,np.newaxis].repeat(s.shape[0], axis=-1) > s.T, axis=1)

    # (crit1 and crit2) = True
    domination_mat = np.all(np.array([crit1,crit2]), axis=0)

    return domination_mat

def get_pareto_front(mat):
    """Fetches most recent Pareto Front
    
    """
    # If one column only includes False the image with the column idx is not dominated by any other image
    nondominated = np.all(mat==False, axis=0)
    # Return idx of nondominated images
    idx = np.where(nondominated)[0]
    # Set rows of nondominated values to False in preparation of rerun
    mat[idx] = False
    return idx, mat

mat = pareto_dominance(similarities)
fronts = []
for i in range(3):
    idx, mat = get_pareto_front(mat=mat)
    fronts.append(idx)

# Compute mathces
selected_imgs = [paths[idx] for idx in fronts[0]]
print(fronts[0])
#selected_imgs = mean_matcher(similarities, paths)

# Wirte as .csv
with open('matches.csv', mode="w") as file:
    writer = csv.writer(file)
    writer.writerow(selected_imgs)