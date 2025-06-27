#!/usr/bin/env python3
"""
Utility script for generating caption embeddings of ALL COCO Images.
(Initial h5py test file)

>>> python gen_emb_caption.py
"""
#%%
from caption_embedder import save_embeddings, load_embeddings, find_similar_images
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import time

PATH_EMBEDDINGS = Path("embeddings/")
PATH_RESULTS = Path("benchmark/")
FILE_CAPTIONS = Path("embeddings/flickr8k_captions.json")


embedder = SentenceTransformer("all-mpnet-base-v2") #best
# model2 = SentenceTransformer("all-MiniLM-L6-v2") #5 times faster according to the internet
# embeddings = embedder.encode(sentences)
# similarities = embedder.similarity(embeddings, embeddings)

with open(FILE_CAPTIONS, 'r') as f:
    file_name_captions_dict =  json.load(f)
      

#%%
start = time.time()

save_embeddings(file_name_captions_dict, embedder, PATH_EMBEDDINGS)

print(f"Time taken: {time.time() - start:.2f} seconds")

# %%
# start = time.time()
# all_embeddings = load_embeddings(PATH_EMBEDDINGS)
# print(f"Loaded {len(all_embeddings)} total embeddings.")
# print(f"Time taken: {time.time() - start:.2f} seconds")