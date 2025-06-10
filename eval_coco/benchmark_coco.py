from coco_extractor import COCOCaptionExtractor
from pipeline import CLIPLLaVAPipeline
from llava_runner import LLaVAVerifier
from clip_matcher import CLIPMatcher
from git_matcher import GitMatcher
from datetime import datetime
from pathlib import Path

#TODO: use single steps of pipeline for benchmarking
#TODO: use LLaVA with top 30+ results of Clip until top_k is achieved-> no more!
#TODO: parallelize to speed up?
#TODO: PaliGemma as LLaVA alternative?
#TODO: Clip vs. Clip&Git
#TODO: Precision & Recall @k 1/5/10
#TODO: Validation based on caption embedding similarity -> median (outlier robust)
#TODO: Caption embedding trashold
#TODO: rewrite pipeline, clip_matcher, llava


 

def main():
    # init paths, coco extractor and pipeline
    PATH_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
    PATH_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
    PATH_EMBEDDINGS = Path("/usr/prakt/s0115/AFM_SEARCH/eval_coco/embeddings")
    PATH_RESULTS = Path("eval_coco/benchmarks")

    coco_extractor = COCOCaptionExtractor(PATH_ANNOTATIONS, PATH_IMAGES)
    
    # pipeline = CLIPLLaVAPipeline(
    # image_folder=IMAGES_PATH,
    # clip_model="ViT-L/14@336px",
    # top_k=10
    # )   
    clip_matcher = CLIPMatcher(
    image_folder=PATH_IMAGES,
    embedding_folder=PATH_EMBEDDINGS,
    top_k=self.top_k
    )

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
    results_path = PATH_RESULTS.joinpath(f'recall@k_{time}.txt') 
    with open(results_path, 'a') as file:
        file.write(f"""\nCoco Benchmark Results at {time} for {idx_img} images with {idx_caption} captions:\nr1: {r1/idx_caption}\nr5: {r5/idx_caption}\nr10: {r10/idx_caption}\n""")

    
        
if __name__ == "__main__":
    main()