
#!/usr/bin/env python3
"""
Utility functions for saving, loading and find similar caption embedding of the COCO dataset.
"""

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from coco_extractor import COCOCaptionExtractor
from typing import List, Optional, Dict
from pathlib import Path
import numpy as np
import h5py


def save_embeddings(extractor: COCOCaptionExtractor, 
                    embedder: SentenceTransformer, 
                    embedding_folder: Path,
                    name_addition: str = None,
                    subset: List[Path]=None,
                    )->None:
    """
    Saves caption embeddings to a HDF5 file.
    
    Args:
        extractor (COCOCaptionExtractor): An instance of COCOCaptionExtractor to extract captions.
        embedder (SentenceTransformer): A SentenceTransformer model to encode captions.
        embedding_folder (Path): Directory where the embeddings will be saved.
        name_addition (str, optional): Additional string to append to the filename. Defaults to None.
        subset (List[Path], optional): A list of specific image paths to process.
                                       If None, processes all images from the extractor.    
                                       
    Returns:
        None: The function saves the embeddings to a file in the specified directory.
    """
    all_captions = extractor.get_all_captions()
    if name_addition is not None:
        filename = embedding_folder / f"caption_embeddings_{name_addition}.h5"
    else:
        filename = embedding_folder / "caption_embeddings.h5"
    
    print(f"Starting to save embeddings to file: {filename}")
    with h5py.File(filename, 'w') as f:
        if subset is None:
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
        else:
            for i, path_obj in enumerate(subset):
                if (i + 1) % 500 == 0:
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
                    )->Dict[Path, np.ndarray]:
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


def find_similar_images(
    query_image_paths: List[str],
    all_embeddings: Dict[str, np.ndarray],
    search_in_paths: Optional[List[str]] = None,
    aggregation: str = 'median', # 'mean' or 'median'
    threshold: Optional[float] = 0.35,
    top_percent: Optional[float] = None,
    return_scores: bool = True,
) -> List[str]|List[tuple[str, float]]:
    """
    Finds similar images from a pre-loaded dictionary of embeddings.

    Args:
        query_image_paths (List[str]): List of image paths/filenames to use as the query.
        all_embeddings (Dict[str, np.ndarray]): The pre-loaded dictionary mapping image
                                                 filenames to their (captions, dim) embedding arrays.
        search_in_paths (Optional[List[str]], optional): A specific list of image filenames
            to search within. If None, searches all images in the `all_embeddings` dict.
        aggregation (str, optional): How to aggregate similarity scores ('mean' or 'median').
        threshold (Optional[float], optional): If set, returns images with a score above this.
        top_percent (Optional[float], optional): Returns the top X percent of images.
        return_scores (bool, optional): If True, returns a list of tuples (filename, score).

    Returns:
        (List[str] | List[tuple[str, float]]): 
            A list of filenames or tuples (filename, score) of similar images based on the specified criteria.
    """
    CAPTIONS_PER_IMAGE = 5
    if top_percent is not None and threshold is not None:
        print("Warning: Both top_percent and threshold are set. top_percent will be used.")
    
    if aggregation not in ['mean', 'median']:
        raise ValueError("aggregation must be 'mean' or 'median'")

    # Separate the pre-loaded data into query and search arrays 
    query_filenames = {Path(p).name for p in query_image_paths}
    
    # Get query embeddings from the pre-loaded dictionary
    query_embeddings_list = []
    for q_filename in query_filenames:
        if q_filename in all_embeddings:
            query_embeddings_list.append(all_embeddings[q_filename][:5])
        else:
            print(f"Warning: Query image '{q_filename}' not found in pre-loaded embeddings. Skipping.")
    
    if not query_embeddings_list:
        print("Error: No valid query images were found in the provided embeddings.")
        return []
    
    # Shape: (num_queries * 5, dim)
    query_embeddings = np.vstack(query_embeddings_list)

    # Get the search space 
    if search_in_paths is None:
        search_pool_filenames = set(all_embeddings.keys())
    else:
        search_pool_filenames = {Path(p).name for p in search_in_paths}

    # # Avoid self-comparison
    search_filenames = search_pool_filenames #- query_filenames

    # Get search embeddings from the pre-loaded dictionary
    search_embeddings_list = []
    valid_search_paths = []
    for s_filename in search_filenames:
        if s_filename in all_embeddings:
            # load the first 5 embeddings for each search image!!! SOME HAVE MORE THAN 5!!!!!! PAIN
            search_embeddings_list.append(all_embeddings[s_filename][:5])
            valid_search_paths.append(s_filename)

    if not search_embeddings_list:
        print("Error: No valid images found in the search space.")
        return []

    # Create the giant search matrix: (num_images * 5, dim)
    search_embeddings = np.vstack(search_embeddings_list)
    
    # Shape: (num_queries * 5, num_searches * 5)
    all_sims = cosine_similarity(query_embeddings, search_embeddings)

    # Reshape and Aggregate
    num_search_images = len(valid_search_paths)
        
    sims_reshaped = all_sims.T
    
    if aggregation == 'mean':
        image_scores = np.mean(np.mean(sims_reshaped.reshape(num_search_images, CAPTIONS_PER_IMAGE, CAPTIONS_PER_IMAGE), 
                               axis=2), 
                               axis=1)
        
        # print(image_scores.shape)
    else: # median
        image_scores = np.median(np.median(sims_reshaped.reshape(num_search_images, CAPTIONS_PER_IMAGE, CAPTIONS_PER_IMAGE), 
                                 axis=2),
                                 axis=1)
    
    # Filter and Sort Results 
    path_score_pairs = sorted(zip(valid_search_paths, image_scores), key=lambda x: x[1], reverse=True)

    if threshold is not None:
        filtered_pairs = [(path, sim) for path, sim in path_score_pairs if sim >= threshold]
    else: # Use top_percent just for testing & development
        num_to_keep = max(1, int(len(path_score_pairs) * top_percent / 100))
        filtered_pairs = path_score_pairs[:num_to_keep]

    return filtered_pairs if return_scores else [path for path, _ in filtered_pairs] 