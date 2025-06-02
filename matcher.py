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
from matching_algorithms import MeanMatcher, ParetoFrontMatcher

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

matcher = MeanMatcher(img_embs, query_emb)
selected_imgs = matcher.match(paths)

# Wirte as .csv
with open('matches.csv', mode="w") as file:
    writer = csv.writer(file)
    writer.writerow(selected_imgs)