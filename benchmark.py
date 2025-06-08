"""
    Performs several automated benchmarks using the current pipeline
"""
from pipeline import CLIPLLaVAPipeline
import json
import os
from datetime import datetime

IMAGE_FOLDER = ""
CAPTIONS_FILE = ""
BENCHMARK_FOLDER = "benchmarks/"

# Recall@K benchmark
def benchmark_rk(captions_file, print_every_n):

    # utc time as unique experiment id
    experiment_id = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') 

    # Load captions .json
    with open(captions_file, 'r', encoding='utf-8') as file:
        captions_dict = json.load(file)

    # get query of every image
    imgs = list(captions_dict.keys())
    n_samples = len(imgs)

    r1, r5, r10 = 0, 0, 0    

    # Iterate through every image in the dataset and get it's caption
    for i, img in enumerate(imgs):
        # get the first caption of the image
        caption = captions_dict[img][0] 

        # get matches top 10 for the ground truth 
        matches = pipeline.run(caption)[:10]

        # R @ 1
        r1 += img in matches[:1]
        # R @ 5
        r5 += img in matches[:5]
        # R @ 10
        r10 += img in matches[:10]

        # print every n samples:
        if ((i+1) % print_every_n) == 0:
            print(r1/i, r5/i, r10/i)

    # write results to file
    results_path = os.path.join(BENCHMARK_FOLDER, 'recall_at_k.txt')
    with open(results_path, 'a') as file
        file.write(experiment_id + "r1:" + str(r1/n_samples) + " r5:" + str(r5/n_samples) + " r10:" + str(r10/n_samples) + "\n")

if __name__ == "__main__":
    if not os.path.isdir(BENCHMARK_FOLDER):
        try:
            os.makedirs(BENCHMARK_FOLDER)
            print(f"Benchmark folder created at {BENCHMARK_FOLDER}")
        except OSError as error:
            print(f"Failed to create folder: {error}")
    
    pipeline = CLIPLLaVAPipeline(
        image_folder=IMAGE_FOLDER,
        verification_prompt=f"Does this image show a {prompt}? (answer with 'yes' or 'no')",
        clip_model="ViT-L/14@336px",
        top_k=30
    )
    
    # launch benchmark
    benchmark_rk(CAPTIONS_FILE, 500)