"""
h5py test with simplified data structure
"""
#%%
from caption_embedder import save_embeddings, load_embeddings, find_similar_images
from sentence_transformers import SentenceTransformer
from coco_extractor import COCOCaptionExtractor
from typing import List, Optional, Union, Dict
from pathlib import Path
import numpy as np
import h5py

PATH_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
PATH_EMBEDDINGS = Path("/embeddings")
PATH_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
PATH_RESULTS = Path("/benchmark")
#%%

# %%
# test it 
#%%
embedder = SentenceTransformer("all-mpnet-base-v2") #best
# model2 = SentenceTransformer("all-MiniLM-L6-v2") #5 times faster according to the internet
# embeddings = embedder.encode(sentences)
# similarities = embedder.similarity(embeddings, embeddings)

extractor = COCOCaptionExtractor(PATH_ANNOTATIONS, PATH_IMAGES)    
captions = extractor.get_all_captions()#[:1000]
#%%
import time
start = time.time()

save_embeddings(extractor, embedder, PATH_EMBEDDINGS)

print(f"Time taken: {time.time() - start:.2f} seconds")

# %%
start = time.time()
all_embeddings = load_embeddings(PATH_EMBEDDINGS)
print(f"Loaded {len(all_embeddings)} total embeddings.")
print(f"Time taken: {time.time() - start:.2f} seconds")
#%%

# some_images_to_load = [
#     'COCO_train2014_000000000009.jpg',
#     'data/images/COCO_train2014_000000000025.jpg', # Works even with full paths
#     'non_existent_file.jpg'
# ]
# specific_embeddings = load_embeddings_flat_hdf5(PATH_EMBEDDINGS, image_paths=some_images_to_load)
# # %%
