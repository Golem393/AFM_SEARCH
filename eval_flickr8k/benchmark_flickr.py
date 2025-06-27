"""
Benchmarking script for evaluating the models of the project on to the flickr8k dataset.

>>> python eval_coco/benchmark_flickr8k.py --port 5000 
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from caption_embedder import load_embeddings, find_similar_images
from paligemma_runner import PaliGemmaVerifier
from clip_matcher import CLIPMatcher
from pprint import pprint
import argparse
import json
import os

def open_json_file(file_path)->dict:
    """Open a JSON file and return its content.
    Format:
    all_models_progress = {
        "last_processed_index": -1,
        "total_captions_processed": 0,
        "clip": {
            "recall@": {"1": 0,"5": 0,"10": 0},
            "precision@": {"1": 0.0,"5": 0.0,"10": 0.0}
            },
            ...,
    }}
    """
    model_names = ["clip", "clip+paligemma"]
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            all_models_progress = json.load(f)
    else:
        all_models_progress = {
            "last_processed_index": 0,
            "total_captions_processed": 0,
             **{model_name : {
                "recall@": {"1": 0, "5": 0, "10": 0},
                "precision@": {"1": 0.0, "5": 0.0, "10": 0.0}
            } for model_name in model_names}
        }
        
    return all_models_progress


def save_json_file(filepath, data):
    """Saves data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
        
def precision_at_k(groundtruth: list, results: list, k: int) -> float:
    """
    Calculates precision at k.
    
    Args:
        groundtruth (list): The list of all relevant, correct item identifiers.
        results (list): The list of predicted item identifiers, ordered by confidence score.
        k (int): The cutoff for the top k results to consider.
        
    Returns:
        float: The precision score @ k.
    """
    groundtruth_set = set(groundtruth)
    top_k_results = results[:k]
    hits = sum(1 for item in top_k_results if item in groundtruth_set)
    return hits / k

def load_list_from_txt(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        data = [line.strip() for line in f]
    return data

def main():
    
    parser = argparse.ArgumentParser(description="choose a subset of the coco dataset to benchmark")
    parser.add_argument(
    "--port", type=int, default=5000, help="Port to run the server on"
    )
    args = parser.parse_args()
    port = args.port
    
               
    # init paths, coco extractor and pipeline
    FOLDER_IMAGES = Path("/usr/prakt/s0122/afm/dataset/flickr8k/Flicker8k_Dataset")
    FOLDER_EMBEDDINGS = Path("embeddings/")
    
    FILE_CAPTIONS = Path("embeddings/flickr8k_captions.json")
    # init caption embedding for precision ground truth 
    TOP_K = 30
    
    with open(FILE_CAPTIONS, 'r') as f:
        file_name_captions_dict =  json.load(f)
        
    FILE_PROGRESS = Path(f'benchmarks/eval_progress_pali.json')
    FILE_EMBEDDING = Path("embeddings/caption_embeddings.h5")
    preloaded_caption_embeddings = load_embeddings(FILE_EMBEDDING)
    files = sorted([f for f in FOLDER_IMAGES.iterdir() if f.is_file()])
    
    clip_matcher = CLIPMatcher(
        image_folder=FOLDER_IMAGES,
        embedding_folder=FOLDER_EMBEDDINGS,
        top_k=TOP_K,
        print_progress=False,
        port=port,
    )
   
    paligemma = PaliGemmaVerifier(port=port)
    
    results_dict = open_json_file(FILE_PROGRESS)
    start_index_img = results_dict.get("last_processed_index", 0)
   
    for index_img in range(start_index_img, len(files)):
        img_name = Path(files[index_img]).name
        captions = file_name_captions_dict.get(img_name, [])
        
        ground_truth = find_similar_images(
                query_image_paths=[img_name],
                all_embeddings=preloaded_caption_embeddings,
                aggregation='median',
                threshold=0.35,
                return_scores=False,
            )
        
        for caption in captions:
            try:
                matches_clip, _ = clip_matcher.find_top_matches(caption)
                
                results_dict["clip"]["recall@"]["1"] += img_name in matches_clip[:1]
                results_dict["clip"]["recall@"]["5"] += img_name in matches_clip[:5]
                results_dict["clip"]["recall@"]["10"] += img_name in matches_clip[:10]
                
                results_dict["clip"]["precision@"]["1"] += precision_at_k(ground_truth, matches_clip, 1)
                results_dict["clip"]["precision@"]["5"] +=  precision_at_k(ground_truth, matches_clip, 5)
                results_dict["clip"]["precision@"]["10"] +=  precision_at_k(ground_truth, matches_clip, 10)

                matches_clip_path = [str(Path(FOLDER_IMAGES).joinpath(Path(str(name)))) for name in matches_clip]
                
                results_paligemma = paligemma.verify_batch(matches_clip_path, caption)

                confirmed, rejected, _ = paligemma.corssref_results(results_paligemma, matches_clip)

                # Log results:
                LOG_PATH = Path("benchmarks/log.txt")
                with open(LOG_PATH, 'a') as file:
                    line = f"{caption.strip()} target:{img_name} confirmed:{confirmed} rejected:{rejected} clip:{matches_clip}\n"
                    file.write(line)
                
                results_paligemma_dict = dict(zip(matches_clip_path, results_paligemma))
                
                matches_paligemma = [Path(k).name for k, v in results_paligemma_dict.items() if str(v).strip().lower() == 'yes']
                
                results_dict["clip+paligemma"]["recall@"]["1"] += img_name in matches_paligemma[:1]
                results_dict["clip+paligemma"]["recall@"]["5"] += img_name in matches_paligemma[:5]
                results_dict["clip+paligemma"]["recall@"]["10"] += img_name in matches_paligemma[:10]
                
                results_dict["clip+paligemma"]["precision@"]["1"] += precision_at_k(ground_truth,   matches_paligemma, 1)
                results_dict["clip+paligemma"]["precision@"]["5"] +=  precision_at_k(ground_truth,  matches_paligemma, 5)
                results_dict["clip+paligemma"]["precision@"]["10"] +=  precision_at_k(ground_truth, matches_paligemma, 10)
                
                results_dict["total_captions_processed"] += 1
                
            except Exception as e:
                print(f"Error processing image {img_name} with caption '{caption}': {e}")
                continue
            
        results_dict["last_processed_index"] += 1
        save_json_file(FILE_PROGRESS, results_dict)
        
if __name__ == "__main__":
    main()