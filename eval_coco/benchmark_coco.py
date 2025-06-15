import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline import CLIPLLaVAPipeline
from coco_extractor import COCOCaptionExtractor
from caption_embedder import load_embeddings, find_similar_images
from pipeline import CLIPLLaVAPipeline
from llava_runner import LLaVAVerifier
from clip_matcher import CLIPMatcher
from git_matcher import GitMatcher
from datetime import datetime
from pathlib import Path
import json
import os

#TODO: use LLaVA with top 30+ results of Clip until top_k is achieved-> no more!
#TODO: PaliGemma as LLaVA alternative?
#TODO: rewrite pipeline, clip_matcher, llava
#FIXME: check function prints!
#TODO: parallelize the LLaVA through multiple models -> check memory usage of llava and use max

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
    model_names = ["clip", "clip+git", "clip+llava", "clip+git+llava"]
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            all_models_progress = json.load(f)
    else:
        all_models_progress = {
            "last_processed_index": -1,
            "total_samples_processed": 0,
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



def main():
    # init paths, coco extractor and pipeline
    FOLDER_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
    FOLDER_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
    FOLDER_BENCHMARKS = Path("benchmarks/")
    FOLDER_EMBEDDINGS = Path("embeddings/")
        
    FILE_EMBEDDING = Path("embeddings/caption_embeddings.h5")
    FILE_TEST_EMBEDDING = Path("embeddings/caption_embeddings_1000_test.h5")#
    FILE_PROGRESS = Path('benchmarks/eval_progress.json')

    # init coco extractor
    coco_extractor = COCOCaptionExtractor(FOLDER_ANNOTATIONS, FOLDER_IMAGES)
    
    # init caption embedding for precision ground truth 
    preloaded_caption_embeddings = load_embeddings(FILE_EMBEDDING)
    top_k_clip = 30
    top_k_llava = 20
        
    clip_matcher = CLIPMatcher(
        image_folder=FOLDER_IMAGES,
        embedding_folder=FOLDER_EMBEDDINGS,
        top_k=top_k_clip
        )
    
    git_matcher = GitMatcher(
        image_folder=FOLDER_IMAGES,
        embedding_folder= FOLDER_EMBEDDINGS,
        top_k=top_k_clip,
        )
    
    llava = LLaVAVerifier()
       
    files = list(coco_extractor.get_all_filepaths())
    
    results_dict = open_json_file(FILE_PROGRESS)
    last_processed_index = results_dict.get("last_processed_index", -1)
    start_index = last_processed_index + 1
    
    
    # for coco_pair in coco_extractor.iter_image_captions():
    for index in range(start_index, len(files)):
        img_path = files[index]
        img_name = Path(img_path).name
        captions = coco_extractor.get_captions_for_image(img_path)
        
        ground_truth = find_similar_images(
                query_image_paths=[img_name],
                all_embeddings=preloaded_caption_embeddings,
                aggregation='median',
                threshold=0.35,
                return_scores=False,
            )
        
        for caption in captions:
            matches_clip, scores_clip = clip_matcher.find_top_matches(caption)
            matches_git, scores_git = git_matcher.find_top_matches(caption)
            merged_dict = {**dict(zip(matches_clip, scores_clip)), 
                           **dict(zip(matches_git, scores_git))}
            
            matches_both = [filename[0] for filename in sorted(merged_dict.items(), key=lambda item: item[1], reverse=True)]
            matches_diff_to_clip = list(set(matches_both) - set(matches_clip))
            
            results_dict["clip"]["recall@"]["1"] += img_name in matches_clip[:1]
            results_dict["clip"]["recall@"]["5"] += img_name in matches_clip[:5]
            results_dict["clip"]["recall@"]["10"] += img_name in matches_clip[:10]
            
            results_dict["clip"]["precision@"]["1"] += precision_at_k(ground_truth, matches_clip, 1)
            results_dict["clip"]["precision@"]["5"] +=  precision_at_k(ground_truth, matches_clip, 5)
            results_dict["clip"]["precision@"]["10"] +=  precision_at_k(ground_truth, matches_clip, 10)
            
            results_dict["clip+git"]["recall@"]["1"] += img_name in matches_both[:1]
            results_dict["clip+git"]["recall@"]["5"] += img_name in matches_both[:5]
            results_dict["clip+git"]["recall@"]["10"] += img_name in matches_both[:10]
            
            results_dict["clip+git"]["precision@"]["1"] += precision_at_k(ground_truth, matches_both, 1)
            results_dict["clip+git"]["precision@"]["5"] +=  precision_at_k(ground_truth, matches_both, 5)
            results_dict["clip+git"]["precision@"]["10"] +=  precision_at_k(ground_truth, matches_both, 10)
            
            results_dict["total_captions_processed"] += 1
            
            matches_llava = llava.verify_images(FOLDER_IMAGES, 
                                matches_both, 
                                f"Does this image show a {caption}? (answer only with 'yes' or 'no' and nothing else!)")
            
            

        results_dict["last_processed_index"] = index
            
        

    
        
if __name__ == "__main__":
    main()