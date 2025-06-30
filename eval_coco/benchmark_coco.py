"""
Benchmarking script for evaluating the models of the project on to the COCO dataset.

>>> python eval_coco/benchmark_coco.py --port 5000 --subset 10000
### git supported for subsets 1000, 5000, and 10000
### clip supported for subsets 1000, 5000, 10000, and 0 (all)
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from coco_extractor import COCOCaptionExtractor
from caption_embedder import load_embeddings, find_similar_images
from llava_runner import LLaVAVerifier
from clip_matcher import CLIPMatcher
from git_matcher import GitMatcher
import argparse
import json
import os

#TODO: use LLaVA with top 30+ results of Clip until top_k is achieved-> no more!
#TODO: PaliGemma as LLaVA alternative?

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
    model_names = ["clip", "git", "clip+git", "clip+llava", "git+llava", "clip+git+llava"]
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
        "--subset", type=int, default=1000, help="Subset size to benchmark (1000, 5000, 10000, 0 for all)"
    )
    parser.add_argument(
    "--port", type=int, default=5000, help="Port to run the server on"
    )
    args = parser.parse_args()
    subset = args.subset
    port = args.port
    
               
    # init paths, coco extractor and pipeline
    FOLDER_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
    FOLDER_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
    FOLDER_EMBEDDINGS = Path("embeddings/")
    
    # init caption embedding for precision ground truth 
    top_k_clip = 30
    
    
    # init coco extractor
    coco_extractor = COCOCaptionExtractor(FOLDER_ANNOTATIONS, FOLDER_IMAGES)
    
    if subset != 0:
        FILE_PROGRESS = Path(f'benchmarks/eval_progress_{subset}.json')
        FILE_EMBEDDING_SUBSET = Path(f"embeddings/caption_embeddings_{subset}.h5")
        preloaded_caption_embeddings = load_embeddings(FILE_EMBEDDING_SUBSET)
        files = sorted(load_list_from_txt(f"subsets/subset_{subset}.txt"))
        
        clip_matcher = CLIPMatcher(
            image_folder=FOLDER_IMAGES,
            embedding_folder=FOLDER_EMBEDDINGS,
            top_k=top_k_clip,
            print_progress=False,
            port=port,
            subset=files,
        )
          
        git_matcher = GitMatcher(
            image_folder=FOLDER_IMAGES,
            embedding_folder= FOLDER_EMBEDDINGS,
            top_k=top_k_clip,
            print_progress=False,
            port=port,
            subset=files,
        )
        
    else:
        FILE_PROGRESS = Path(f'benchmarks/eval_progress_all.json')
        FILE_EMBEDDING = Path("embeddings/caption_embeddings.h5")
        preloaded_caption_embeddings = load_embeddings(FILE_EMBEDDING)
        files = sorted(list(coco_extractor.get_all_filepaths()))
        clip_matcher = CLIPMatcher(
            image_folder=FOLDER_IMAGES,
            embedding_folder=FOLDER_EMBEDDINGS,
            top_k=top_k_clip,
            print_progress=False,
            port=port,
        )

    
    llava = LLaVAVerifier(port=port)
    
    results_dict = open_json_file(FILE_PROGRESS)
    start_index = results_dict.get("last_processed_index", 0)
   
    for index in range(start_index, len(files)):
        img_name = Path(files[index]).name
        captions = coco_extractor.get_captions_for_image(img_name)
        
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
                matches_git, _ = git_matcher.find_top_matches(caption)
                
                matches_both = []
                for i in range(top_k_clip):
                    matches_both.append(matches_clip[i])
                    matches_both.append(matches_git[i])

                results_dict["clip"]["recall@"]["1"] += img_name in matches_clip[:1]
                results_dict["clip"]["recall@"]["5"] += img_name in matches_clip[:5]
                results_dict["clip"]["recall@"]["10"] += img_name in matches_clip[:10]
                
                results_dict["clip"]["precision@"]["1"] += precision_at_k(ground_truth, matches_clip, 1)
                results_dict["clip"]["precision@"]["5"] +=  precision_at_k(ground_truth, matches_clip, 5)
                results_dict["clip"]["precision@"]["10"] +=  precision_at_k(ground_truth, matches_clip, 10)

                
                results_dict["git"]["recall@"]["1"] += img_name in matches_git[:1]
                results_dict["git"]["recall@"]["5"] += img_name in matches_git[:5]
                results_dict["git"]["recall@"]["10"] += img_name in matches_git[:10]
                
                results_dict["git"]["precision@"]["1"] += precision_at_k(ground_truth, matches_git, 1)
                results_dict["git"]["precision@"]["5"] +=  precision_at_k(ground_truth, matches_git, 5)
                results_dict["git"]["precision@"]["10"] +=  precision_at_k(ground_truth, matches_git, 10)

                
                results_dict["clip+git"]["recall@"]["1"] += img_name in matches_both[:1]
                results_dict["clip+git"]["recall@"]["5"] += img_name in matches_both[:5]
                results_dict["clip+git"]["recall@"]["10"] += img_name in matches_both[:10]
                
                results_dict["clip+git"]["precision@"]["1"] += precision_at_k(ground_truth, matches_both, 1)
                results_dict["clip+git"]["precision@"]["5"] +=  precision_at_k(ground_truth, matches_both, 5)
                results_dict["clip+git"]["precision@"]["10"] +=  precision_at_k(ground_truth, matches_both, 10)
                
                matches_llava = llava.verify_images(FOLDER_IMAGES, 
                                    matches_both, 
                                    f"Does this image show a {caption}? (answer only with 'yes' or 'no' and nothing else!)")
                
                matches_llava_both = [Path(k).name for k, v in dict(matches_llava).items() if str(v).strip().lower() == 'yes']
                
                matches_llava_clip = [m for m in matches_llava_both if m in matches_clip]
                
                matches_llava_git = [m for m in matches_llava_both if m in matches_git]
                
                results_dict["clip+llava"]["recall@"]["1"] += img_name in matches_llava_clip[:1]
                results_dict["clip+llava"]["recall@"]["5"] += img_name in matches_llava_clip[:5]
                results_dict["clip+llava"]["recall@"]["10"] += img_name in matches_llava_clip[:10]
                
                results_dict["clip+llava"]["precision@"]["1"] += precision_at_k(ground_truth, matches_llava_clip, 1)
                results_dict["clip+llava"]["precision@"]["5"] +=  precision_at_k(ground_truth, matches_llava_clip, 5)
                results_dict["clip+llava"]["precision@"]["10"] +=  precision_at_k(ground_truth, matches_llava_clip, 10)
                
                
                results_dict["git+llava"]["recall@"]["1"] += img_name in matches_llava_git[:1]
                results_dict["git+llava"]["recall@"]["5"] += img_name in matches_llava_git[:5]
                results_dict["git+llava"]["recall@"]["10"] += img_name in matches_llava_git[:10]
                
                results_dict["git+llava"]["precision@"]["1"] += precision_at_k(ground_truth, matches_llava_git, 1)
                results_dict["git+llava"]["precision@"]["5"] +=  precision_at_k(ground_truth, matches_llava_git, 5)
                results_dict["git+llava"]["precision@"]["10"] +=  precision_at_k(ground_truth, matches_llava_git, 10)
                
                
                results_dict["clip+git+llava"]["recall@"]["1"] += img_name in matches_llava_both[:1]
                results_dict["clip+git+llava"]["recall@"]["5"] += img_name in matches_llava_both[:5]
                results_dict["clip+git+llava"]["recall@"]["10"] += img_name in matches_llava_both[:10]
                
                results_dict["clip+git+llava"]["precision@"]["1"] += precision_at_k(ground_truth, matches_llava_both, 1)
                results_dict["clip+git+llava"]["precision@"]["5"] +=  precision_at_k(ground_truth, matches_llava_both, 5)
                results_dict["clip+git+llava"]["precision@"]["10"] +=  precision_at_k(ground_truth, matches_llava_both, 10)
                
                results_dict["total_captions_processed"] += 1
                save_json_file(FILE_PROGRESS, results_dict)
                
            except Exception as e:
                print(f"Error processing image {img_name} with caption '{caption}': {e}")
                continue
            
        results_dict["last_processed_index"] += 1
        
if __name__ == "__main__":
    main()