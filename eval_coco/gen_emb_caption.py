#!/usr/bin/env python3
"""
Utility script for generating caption embeddings of ALL COCO Images.
(Initial h5py test file)

>>> python gen_emb_caption.py
"""
#%%
from caption_embedder import save_embeddings, load_embeddings, find_similar_images
from sentence_transformers import SentenceTransformer
from coco_extractor import COCOCaptionExtractor
from pathlib import Path
import time

PATH_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
PATH_EMBEDDINGS = Path("/embeddings")
PATH_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
PATH_RESULTS = Path("/benchmark")

embedder = SentenceTransformer("all-mpnet-base-v2") #best
# model2 = SentenceTransformer("all-MiniLM-L6-v2") #5 times faster according to the internet
# embeddings = embedder.encode(sentences)
# similarities = embedder.similarity(embeddings, embeddings)

extractor = COCOCaptionExtractor(PATH_ANNOTATIONS, PATH_IMAGES)    
captions = extractor.get_all_captions()#[:1000]
#%%
start = time.time()

save_embeddings(extractor, embedder, PATH_EMBEDDINGS)

print(f"Time taken: {time.time() - start:.2f} seconds")

# %%
# start = time.time()
# all_embeddings = load_embeddings(PATH_EMBEDDINGS)
# print(f"Loaded {len(all_embeddings)} total embeddings.")
<<<<<<< HEAD
# print(f"Time taken: {time.time() - start:.2f} seconds")
=======
# print(f"Time taken: {time.time() - start:.2f} seconds")
>>>>>>> eval_video
