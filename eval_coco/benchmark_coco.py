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

#TODO: use single steps of pipeline for benchmarking
#TODO: use LLaVA with top 30+ results of Clip until top_k is achieved-> no more!
#TODO: parallelize to speed up?
#TODO: PaliGemma as LLaVA alternative?
#TODO: Clip vs. Clip&Git
#TODO: Precision & Recall @k 1/5/10
#TODO: Caption embedding trashold -> just use something
#TODO: rewrite pipeline, clip_matcher, llava
#TODO: OPTIONAL: use the elbow method to get best threshold  

def open_json_file(file_path):
    """Open a JSON file and return its content.
    
    Format:
    
    all_models_progress = {
        "last_processed_index": -1,
        "total_samples_processed": 0,
        "models": {
            "clip": {
                "recall@": {"1": 0,"5": 0,"10": 0},
                "precision@": {"1": 0.0,"5": 0.0,"10": 0.0}
            },
            "clip+git": {
                "recall@": {"1": 0, "5": 0, "10": 0},
                "precision@": {"1": 0.0, "5": 0.0, "10": 0.0}
            },
            "clip+llava": {
                "recall@": {"1": 0, "5": 0, "10": 0},
                "precision@": {"1": 0.0, "5": 0.0, "10": 0.0}
            },
            "clip+git+llava": {
                "recall@": {"1": 0, "5": 0, "10": 0},
                "precision@": {"1": 0.0, "5": 0.0, "10": 0.0}
            }
        }
    }
    
    """
    
    model_names = ["clip", "clip+git", "clip+llava", "clip+git+llava"]
    # Try to load existing progress
    if os.path.exists(file_path):
        print(f"Resuming from saved progress in {file_path}")
        with open(file_path, 'r') as f:
            all_models_progress = json.load(f)
    else:
        print("No saved progress found. Starting a new evaluation.")
        # Initialize the structure if the file doesn't exist
        all_models_progress = {
            "last_processed_index": -1,
            "total_samples_processed": 0,
            "models": { 
                model_name :{
                    "recall@": {"1": 0, "5": 0, "10": 0},
                    "precision@": {"1": 0.0, "5": 0.0, "10": 0.0}
               } for model_name in model_names
            }
        }


def main():
    # FIXME: check function prints!
    
    # init paths, coco extractor and pipeline
    FOLDER_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
    FOLDER_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
    FOLDER_BENCHMARKS = Path("eval_coco/benchmarks")
    FOLDER_EMBEDDINGS = Path("eval_coco/embeddings")
        
    FILE_EMBEDDING = Path("embeddings/caption_embeddings.h5")
    FILE_TEST_EMBEDDING = Path("embeddings/caption_embeddings_1000_test.h5")#
    FILE_PROGRESS = Path('eval_coco/benchmarks/eval_progress.json')

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
    
    
    #FIXME: continue here!
    
    # init time, indices and recall values
    test_idx_limiter = 1
    time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    idx_img, idx_caption = 0, 0
    r1, r5, r10 = 0, 0, 0 
       
    for coco_pair in coco_extractor.iter_image_captions():
        img_path = coco_pair["image_path"]
        captions = coco_pair["captions"]
        img_name = Path(img_path).name
        idx_img+=1
        for caption in captions:
            print(f"time: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
            print(f"pipline caption: {caption}")
            print(f"Processing image {img_name}")
            
            
            
            matches = pipeline.run(caption)
            
            # R @ 1
            r1 += img_name in matches["confirmed"][:1]
            # R @ 5
            r5 += img_name in matches["confirmed"][:5]
            # R @ 10
            r10 += img_name in matches["confirmed"][:10]
            
            idx_caption += 1

        if ((idx_img+1) % 10000) == 0:
            print(r1/idx_caption, r5/idx_caption, r10/idx_caption)
        if idx_img >= test_idx_limiter:
            break
    # write results to file
    results_path = FOLDER_BENCHMARKS.joinpath(f'recall@k_{time}.txt') 
    with open(results_path, 'a') as file:
        file.write(f"""\nCoco Benchmark Results at {time} for {idx_img} images with {idx_caption} captions:\nr1: {r1/idx_caption}\nr5: {r5/idx_caption}\nr10: {r10/idx_caption}\n""")

    
        
if __name__ == "__main__":
    main()