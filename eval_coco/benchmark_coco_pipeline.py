# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

import sys
from coco_extractor import COCOCaptionExtractor
from datetime import datetime
from pathlib import Path

# Add the parent directory (AFM_SEARCH) to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline import CLIPLLaVAPipeline
from pathlib import Path

    

def main():
    # init paths, coco extractor and pipeline
    PATH_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
    PATH_EMBEDDINGS = Path("/usr/prakt/s0115/AFM_SEARCH/eval_coco/embeddings")
    PATH_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
    PATH_RESULTS = Path("eval_coco/benchmark")

    coco_extractor = COCOCaptionExtractor(PATH_ANNOTATIONS, PATH_IMAGES)
    
    pipeline = CLIPLLaVAPipeline(
    image_folder=PATH_IMAGES,
    clip_model="ViT-L/14@336px",
    top_k=30
    )   

    # init time, indices and recall values
    test_idx_limiter = 1
    time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    idx_img, idx_caption = 0, 0
    r1, r5, r10 = 0, 0, 0 
       
    for coco_pair in coco_extractor.iter_image_captions():
        idx_img+=1
        img_path = coco_pair["image_path"]
        captions = coco_pair["captions"]
        img_name = Path(img_path).name
        print(f"captions: {captions}")
        for caption in captions:
            idx_caption += 1
            print(f"\n\ntime: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
            print(f"pipline caption: {caption}")
            print(f"Processing image {img_name}")
            
            
            
            matches = pipeline.run(caption)
            
            print(f"matches found: {matches['confirmed']}")
            print(f"matches found: {matches['confirmed'][:1]}")
            # R @ 1
            r1 += 1 if img_name in matches["confirmed"][:1] else 0
            print(f"R@1: {r1}")
            # R @ 5
            r5 += 1 if img_name in matches["confirmed"][:5] else 0
            print(f"R@5: {r5}")
            # R @ 10
            r10 += 1 if img_name in matches["confirmed"][:10] else 0
            print(f"R@10: {r10}")
            

        if ((idx_img+1) % 10000) == 0:
            print(r1/idx_caption, r5/idx_caption, r10/idx_caption)
        if idx_img >= test_idx_limiter:
            break
    # write results to file
    results_path = PATH_RESULTS.joinpath(f'recall@k_{time}.txt') 
    with open(results_path, 'a') as file:
        file.write(f"""\nCoco Benchmark Results at {time} for {idx_img} images with {idx_caption} captions:\nr1: {r1/idx_caption}\nr5: {r5/idx_caption}\nr10: {r10/idx_caption}\n""")

    
        
if __name__ == "__main__":
    main()