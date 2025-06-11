from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Union
from pathlib import Path
import numpy as np
import h5py

# Assuming you have these functions from your original post
# def save_embeddings_hdf5(...): ...
# def load_embeddings_hdf5(...): ...


def find_similar_images_by_score_aggregation(
    query_image_paths: List[str],
    h5_file_path: Union[str, Path],
    search_in_paths: Optional[List[str]] = None,
    aggregation: str = 'mean',  # 'mean' or 'median'
    threshold: Optional[float] = None,
    top_percent: Optional[float] = None,
    top_k: Optional[int] = None,
    captions_per_image: int = 5
) -> List[str]:
    """
    Finds similar images by first calculating all cross-caption similarities
    and then aggregating these scores.

    Args:
        query_image_paths (List[str]): List of image paths to use as the query.
        h5_file_path (Union[str, Path]): Path to the HDF5 file with embeddings.
        search_in_paths (Optional[List[str]], optional): List of image paths to search in. 
            If None, searches all images found via the extractor. Defaults to None.
        extractor (optional): Your extractor object, required to get all filepaths if 
            search_in_paths is None. Defaults to None.
        aggregation (str, optional): How to aggregate the similarity scores. 
            'mean' or 'median'. Defaults to 'mean'.
        threshold (Optional[float], optional): If top_percent is None, returns images with a
            score above this threshold. Defaults to None.
        top_percent (Optional[float], optional): Returns the top X percent of images. 
            This takes precedence over threshold. Defaults to None.
        top_k (Optional[int], optional): Limits the final number of returned image paths.
        captions_per_image (int, optional): Number of captions for each image. Defaults to 5.

    Returns:
        List[str]: A sorted list of the most similar image paths (filenames).
    """
    if top_percent is not None and threshold is not None:
        print("Warning: Both top_percent and threshold are set. top_percent will be used.")
    
    if aggregation not in ['mean', 'median']:
        raise ValueError("aggregation must be 'mean' or 'median'")

    # Load embeddings form the HDF5 file
    with h5py.File(h5_file_path, 'r') as f:
        # Determine the search space
        if search_in_paths is None:
            # Use all keys from the H5 file as the search space
            all_h5_keys = list(f.keys())
            search_filenames = all_h5_keys
        else:
            search_filenames = [Path(p).name for p in search_in_paths]

        # Load query embeddings
        query_embeddings_list = []
        for path in query_image_paths:
            if path in f:
                query_embeddings_list.append(f[filename][:])
            else:
                print(f"Warning: Query image {filename} not found in HDF5 file. Skipping.")
        
        if not query_embeddings_list:
            return [] # No valid query images found
        
        # Consolidate all query embeddings into one large array
        query_embeddings = np.vstack(query_embeddings_list) # Shape: (num_queries * 5, embedding_dim)

        # Load search embeddings
        search_embeddings_list = []
        valid_search_paths = []
        for filename in search_filenames:
            if filename in f:
                # To avoid comparing an image to itself in the search results
                if filename not in [Path(p).name for p in query_image_paths]:
                    search_embeddings_list.append(f[filename][:])
                    valid_search_paths.append(filename)

        if not search_embeddings_list:
            return [] # No valid search images found

        # Consolidate all search embeddings into one large array
        search_embeddings = np.vstack(search_embeddings_list) # Shape: (num_searches * 5, embedding_dim)

    # --- 2. Calculate Similarity ---
    # This is the core change. Calculate similarity between ALL individual captions.
    # The result is a large matrix where cell (i, j) is the similarity between
    # the i-th query caption and the j-th search caption.
    # Shape: (num_queries * 5, num_searches * 5)
    all_sims = cosine_similarity(query_embeddings, search_embeddings)

    # --- 3. Aggregate Scores per Image ---
    # We now reshape the matrix to group scores by search image and aggregate.
    num_search_images = len(valid_search_paths)
    
    # Reshape to (num_query_captions, num_search_images, captions_per_image)
    sims_reshaped = all_sims.reshape(query_embeddings.shape[0], num_search_images, captions_per_image)

    # Now, aggregate across the query captions (axis 0) and the search image's own captions (axis 2)
    # to get a single score per search image.
    if aggregation == 'mean':
        image_scores = np.mean(sims_reshaped, axis=(0, 2))
    else: # median
        # np.median doesn't accept a tuple of axes, so we reshape and calculate
        # This is equivalent to taking the median of all 25 (or M*5*5) scores for each search image.
        image_scores = np.median(sims_reshaped.reshape(-1, num_search_images), axis=0)

    # --- 4. Filter and Sort Results ---
    path_score_pairs = sorted(zip(valid_search_paths, image_scores), key=lambda x: x[1], reverse=True)

    # Filter based on top_percent or threshold
    if top_percent is not None:
        num_results = max(1, int(len(path_score_pairs) * top_percent / 100))
        filtered_pairs = path_score_pairs[:num_results]
    elif threshold is not None:
        filtered_pairs = [(path, sim) for path, sim in path_score_pairs if sim >= threshold]
    else:
        filtered_pairs = path_score_pairs

    # Extract paths and apply top_k limit
    result_paths = [path for path, _ in filtered_pairs]

    if top_k is not None:
        return result_paths[:top_k]
    
    return result_paths

# A simplified helper for the common case of a single query image
def find_similar_images_single_query(
    query_image_path: str,
    h5_file_path: Union[str, Path],
    **kwargs
) -> List[str]:
    """Helper function for a single query image."""
    return find_similar_images_by_score_aggregation([query_image_path], h5_file_path, **kwargs)


# --- USAGE EXAMPLES ---
"""
# Assume these are defined:
# PATH_EMBEDDINGS = Path("./embeddings")
# extractor = ... # your data extractor object

h5_path = PATH_EMBEDDINGS / "caption_embeddings.h5"

# Example 1: Find the top 10 most similar images to 'query.jpg' using the mean of similarity scores
similar_images = find_similar_images_single_query(
    query_image_path="query.jpg",
    h5_file_path=h5_path,
    extractor=extractor,
    aggregation='mean',
    top_k=10
)
print(f"Top 10 similar images (mean score): {similar_images}")


# Example 2: Find images with a median similarity score greater than 0.75
similar_images = find_similar_images_single_query(
    query_image_path="another_query.jpg",
    h5_file_path=h5_path,
    extractor=extractor,
    aggregation='median',
    threshold=0.75,
    top_k=50  # Limit to a max of 50 results even if many pass the threshold
)
print(f"Images with median score > 0.75: {similar_images}")


# Example 3: Search using multiple query images, finding the top 2% of matches
search_subset = ["image_001.jpg", "image_002.jpg", ..., "image_500.jpg"]
similar_images = find_similar_images_by_score_aggregation(
    query_image_paths=["query1.jpg", "query2.jpg"],
    h5_file_path=h5_path,
    search_in_paths=search_subset, # Search only within this subset
    aggregation='mean',
    top_percent=2,
    top_k=20
)
print(f"Top 2% of matches from subset: {similar_images}")

"""
