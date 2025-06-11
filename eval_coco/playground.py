#%% imports & constants
from sentence_transformers import SentenceTransformer
from coco_extractor import COCOCaptionExtractor
# from pipeline import CLIPLLaVAPipeline
# from datetime import datetime
from pathlib import Path
import numpy as np
import h5py
# from clip_matcher import CLIPMatcher
# from git_matcher import GitMatcher

PATH_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
PATH_EMBEDDINGS = Path("/usr/prakt/s0115/AFM_SEARCH/eval_coco/embeddings")
PATH_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
PATH_RESULTS = Path("eval_coco/benchmark")

#%% h5py functions
def save_embeddings_hdf5(extractor, embedder, save_path):
    """Save embeddings using HDF5 for memory efficiency"""
    
    captions = extractor.get_all_captions()
    filename = save_path / "caption_embeddings.h5"
    
    with h5py.File(filename, 'w') as f:
        for i, path in enumerate(extractor.get_all_filepaths()):
            if i % 1000 == 0:
                print(f"Processed {i} images...")
            
            # Process this image's captions
            embeddings = []
            for caption in captions[Path(path).name]:
                embeddings.append(embedder.encode(caption))
            
            # Save directly to HDF5 (no memory accumulation)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            f.create_dataset(str(path), data=embeddings_array, compression='gzip')
    print(f"Saved to {filename}")
       

def load_embeddings_hdf5(save_path, image_paths=None):
    """Load embeddings from HDF5"""
    filename = save_path / "caption_embeddings.h5"
    
    if image_paths is None:
        # Load all
        result = {}
        with h5py.File(filename, 'r') as f:
            for key in f.keys():
                result[key] = f[key]
        return result
    else:
        # Load specific paths
        result = {}
        with h5py.File(filename, 'r') as f:
            for path in image_paths:
                if str(Path(path).name) in f:
                    result[Path(path).name] = f[str(Path(path).name)][:]
        return result

# %%
embedder = SentenceTransformer("all-mpnet-base-v2") #best
# model2 = SentenceTransformer("all-MiniLM-L6-v2") #5 times faster
# embeddings = embedder.encode(sentences)
# similarities = embedder.similarity(embeddings, embeddings)

extractor = COCOCaptionExtractor(PATH_ANNOTATIONS, PATH_IMAGES)    
captions = extractor.get_all_captions()
#%%
save_embeddings_hdf5(extractor, embedder, PATH_EMBEDDINGS)

#%%
# Later load specific embeddings
test = load_embeddings_hdf5(PATH_EMBEDDINGS, None)
#%%



#%% init test, not working memory & space issues
# i = 0
# caption_embeddings = {}
# for path in extractor.get_all_filepaths():
#     for caption in captions[Path(path).name]:
#         i+=1
#         if path not in caption_embeddings:
#             caption_embeddings[Path(path).name] = []
#         caption_embeddings[Path(path).name].append(embedder.encode(caption)) 
#     if i+1%100 == 0:
#         print(f"Processed {i} captions")
#         break

# for path in caption_embeddings:
#     caption_embeddings[path] = np.array(caption_embeddings[path])


# #%%
# np.save(PATH_EMBEDDINGS.joinpath("caption_embeddings32"), caption_embeddings)

# #%%

# loaded = np.load(PATH_EMBEDDINGS.joinpath("caption_embeddings.npy"), allow_pickle=True)
# caption_embeddings_loaded = loaded.item()
# # %%
# caption_embeddings_loaded 

# %%
def load_embeddings_hdf5_recursive(save_path, image_paths=None):
    """
    Load embeddings from HDF5, recursively searching for datasets within the file.
    This handles the complex & nested file structures.
    
    > storage (Group)
        > group (Group)
            > dataset_mirrors (Group)
                > old_common_datasets (Group)
                    > coco (Group)
                        > images (Group)
                            > train2014 (Group)
                                - COCO_train2014_000000000009.jpg (Dataset: shape=(5, 768), dtype=float32)
                                - COCO_train2014_000000000025.jpg (Dataset: shape=(5, 768), dtype=float32)
    """
    filename = save_path / "caption_embeddings.h5"
    
    if not Path(filename).exists():
        raise FileNotFoundError(f"HDF5 file not found at: {filename}")

    result = {}
    with h5py.File(filename, 'r') as f:
        if image_paths is None:
            
            # This is the callback function that will be executed for each item.
            def collect_datasets(name, obj):
                # 'name' is the full path inside the HDF5 file (e.g., 'storage/group/...')
                # 'obj' is the h5py object itself (Group or Dataset)
                if isinstance(obj, h5py.Dataset):
                    # We use the filename as the key, not the full HDF5 internal path.
                    filename_key = Path(name).name
                    result[filename_key] = obj[:] # Read the data into memory

            f.visititems(collect_datasets)
            
        else:
            # Create a set of desired filenames for fast lookups
            target_filenames = {Path(p).name for p in image_paths}
            
            def find_specific_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    filename_key = Path(name).name
                    if filename_key in target_filenames:
                        # Find the original path 'p' from the user's list that matches the filename.
                        # This maintains consistency between input paths and output keys.
                        original_path = next((p for p in image_paths if Path(p).name == filename_key), filename_key)
                        result[original_path] = obj[:]
            
            f.visititems(find_specific_datasets)
            
            # Check for any requested images that were not found
            found_keys = {Path(k).name for k in result.keys()}
            not_found = set(target_filenames) - found_keys
            if not_found:
                print(f"Warning: Could not find embeddings for {len(not_found)} images. First 5 missing: {list(not_found)[:5]}")

    print(f"Successfully loaded {len(result)} embeddings into memory.")
    return result

# --- USAGE ---
# This code should now run without a NameError.
# PATH_EMBEDDINGS = Path("/path/to/your/embeddings_folder")
all_embeddings = load_embeddings_hdf5_recursive(PATH_EMBEDDINGS)
print(f"Total embeddings found and loaded: {len(all_embeddings)}")

# %%
