"""
h5py test with simplified data structure
"""
#%%
from sentence_transformers import SentenceTransformer
from coco_extractor import COCOCaptionExtractor
from typing import List, Optional, Union
from pathlib import Path
import numpy as np
import h5py

PATH_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
PATH_EMBEDDINGS = Path("/usr/prakt/s0115/AFM_SEARCH/eval_coco/embeddings")
PATH_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
PATH_RESULTS = Path("eval_coco/benchmark")
#%%
def save_embeddings(extractor: COCOCaptionExtractor, 
                    embedder:SentenceTransformer, 
                    embedding_file:Path,
                    )->None:
    """
    Saves caption embeddings to a HDF5 file (flat structure).
    
    Args:        
        extractor (COCOCaptionExtractor): An instance of the COCOCaptionExtractor to get captions and
        embedder (SentenceTransformer): An instance of SentenceTransformer to encode captions.
        save_path (Path): The directory where the HDF5 file will be saved.      
        
    Returns:
        None: The function saves the embeddings to a h5 file and does not return anything.
    """
    all_captions = extractor.get_all_captions()
    
    filename = embedding_file / "caption_embeddings.h5"
    
    print(f"Starting to save embeddings to file: {filename}")
    with h5py.File(filename, 'w') as f:
        # Iterate through all image filepaths provided by the extractor
        for i, path_obj in enumerate(extractor.get_all_filepaths()):
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} images...")

            # The key for captions = filename 
            caption_key = Path(path_obj).name 
            
            # Encode captions of current image
            embeddings = [embedder.encode(caption) for caption in all_captions[caption_key]]
            
            # Convert to a NumPy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Use the filename as the dataset key, ensuring a flat structure.
            f.create_dataset(caption_key, data=embeddings_array, compression='gzip')
    
    print(f"\nFinished saving. All embeddings keyed by the imagename are in {filename}")
    
    
    
def load_embeddings(embedding_file_path:Path, 
                    image_paths:List[Path]=None,
                    )->Dict[str, np.ndarray]:
    """
    Loads embeddings from a flat HDF5 file.
    
    Args:
        save_path (Path): Path of the HDF5 file.
        image_paths (list, optional): A list of specific image paths/filenames to load.
                                      If None, all embeddings are loaded. Defaults to None.
    
    Returns:
        dict: A dictionary mapping image filenames to their embedding arrays.
    """
    
    if not embedding_file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found at: {embedding_file_path}")

    result = {}
    with h5py.File(embedding_file_path, 'r') as f:
        if image_paths is None:
            # Load all embeddings
            print(f"Loading all {len(f.keys())} images with each 5 embeddings from {embedding_file_path}...")
            for key in f.keys():
                result[key] = f[key][:] # Read data into memory
        else:
            # Load only the specified embeddings
            print(f"Loading {len(image_paths)} specific images with each 5 embeddings from {embedding_file_path}...")
            for path in image_paths:
                key = Path(path).name  # Extract filename to use as the key
                if key in f:
                    result[key] = f[key][:]
                else:
                    print(f"Warning: Key for image '{key}' not found in HDF5 file.")
    
    print(f"Successfully loaded {len(result)} images with each 5 embeddings into memory.")
    return result

# %%
# test it 
#%%
embedder = SentenceTransformer("all-mpnet-base-v2") #best
# model2 = SentenceTransformer("all-MiniLM-L6-v2") #5 times faster according to the internet
# embeddings = embedder.encode(sentences)
# similarities = embedder.similarity(embeddings, embeddings)

extractor = COCOCaptionExtractor(PATH_ANNOTATIONS, PATH_IMAGES)    
captions = extractor.get_all_captions()

save_embeddings(extractor, embedder, PATH_EMBEDDINGS)
# %%
import time
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
