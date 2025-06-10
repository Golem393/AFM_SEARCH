#%%

from sentence_transformers import SentenceTransformer
from coco_extractor import COCOCaptionExtractor
# from pipeline import CLIPLLaVAPipeline
# from datetime import datetime
from pathlib import Path
import numpy as np
# from clip_matcher import CLIPMatcher
# from git_matcher import GitMatcher

PATH_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
PATH_EMBEDDINGS = Path("/usr/prakt/s0115/AFM_SEARCH/eval_coco/embeddings")
PATH_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
PATH_RESULTS = Path("eval_coco/benchmark")

# %%
embedder = SentenceTransformer("all-mpnet-base-v2") #best
# model2 = SentenceTransformer("all-MiniLM-L6-v2") #5 times faster
# embeddings = embedder.encode(sentences)
# similarities = embedder.similarity(embeddings, embeddings)

extractor = COCOCaptionExtractor(PATH_ANNOTATIONS, PATH_IMAGES)    
captions = extractor.get_all_captions()

#%%
i = 0
caption_embeddings = {}
for path in extractor.get_all_filepaths():
    for caption in captions[Path(path).name]:
        i+=1
        if path not in caption_embeddings:
            caption_embeddings[Path(path).name] = []
        caption_embeddings[Path(path).name].append(embedder.encode(caption)) 
    if i+1%10000 == 0:
        print(f"Processed {i} captions")

for path in caption_embeddings:
    caption_embeddings[path] = np.array(caption_embeddings[path])


#%%


#%%
np.save(PATH_EMBEDDINGS.joinpath("caption_embeddings32"), caption_embeddings)


#%%

# loaded = np.load(PATH_EMBEDDINGS.joinpath("caption_embeddings.npy"), allow_pickle=True)
# caption_embeddings_loaded = loaded.item()
# # %%
# caption_embeddings_loaded 

# %%
