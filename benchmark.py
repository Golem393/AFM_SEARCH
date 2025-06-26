"""
    Performs several automated benchmarks using the current pipeline
"""
from pipeline2 import CLIPLLaVAPipeline
import json
import os

IMAGE_FOLDER = "/usr/prakt/s0122/afm/dataset/flickr8k/Flicker8k_Dataset"
CAPTIONS_FILE = "/usr/prakt/s0122/afm/dataset/flickr8k/flickr8k_captions.json"
BENCHMARK_FOLDER = "benchmarks/"
PRINT_EVERY_N = 50

# Recall@K benchmark
def benchmark_rk(name):

    # Load captions .json
    with open(CAPTIONS_FILE, 'r', encoding='utf-8') as file:
        captions_dict = json.load(file)

    # get query of every image
    imgs = list(captions_dict.keys())

    r1_clip, r5_clip, r10_clip = 0, 0, 0
    r1, r5, r10 = 0, 0, 0    

    log_path = os.path.join(BENCHMARK_FOLDER, f"log_{name}.txt")
    results_path = os.path.join(BENCHMARK_FOLDER, f"recall_at_k_{name}.txt")

    # get matches top 30 for the ground truth
    pipeline = CLIPLLaVAPipeline(
        image_folder=IMAGE_FOLDER,
        verification_prompt=None,
        top_k_clip_matches=30
    )

    # Iterate through every image in the dataset and get it's caption
    for i, img in enumerate(imgs):
        # get the caption of the image
        caption = captions_dict[img][0]

        matches = pipeline.run(caption)

        with open(log_path, 'a') as file:
            line = f"{caption.strip()} target:{img} confirmed:{matches['confirmed']} rejected:{matches['rejected']} clip:{matches['clip_matches']}\n"
            file.write(line)

        clip_matches = matches['clip_matches']

        matches = matches['confirmed']

        # R @ k
        r1 += img in matches[:1]
        r5 += img in matches[:5]
        r10 += img in matches[:10]

        r1_clip += img in clip_matches[:1]
        r5_clip += img in clip_matches[:5]
        r10_clip += img in clip_matches[:10]

        n = i+1

        print(f"{n} clip+gemma r1: {r1/n:.3f}, r5: {r5/n:.3f}, r10: {r10/n:.3f} | clip r1: {r1_clip/n:.3f}, r5: {r5_clip/n:.3f}, r10: {r10_clip/n:.3f}")

        # print to file every
        if n % PRINT_EVERY_N == 0:
            with open(results_path, "a") as file:
                file.write(f"{n} clip+gemma r1: {r1/n:.3f}, r5: {r5/n:.3f}, r10: {r10/n:.3f} | clip r1: {r1_clip/n:.3f}, r5: {r5_clip/n:.3f}, r10: {r10_clip/n:.3f}\n")

    # write last "incomplete batch"
    with open(results_path, 'a') as file:
        file.write(f"{n} clip+gemma r1: {r1/n:.3f}, r5: {r5/n:.3f}, r10: {r10/n:.3f} | clip r1: {r1_clip/n:.3f}, r5: {r5_clip/n:.3f}, r10: {r10_clip/n:.3f}\n")

if __name__ == "__main__":
    if not os.path.isdir(BENCHMARK_FOLDER):
        try:
            os.makedirs(BENCHMARK_FOLDER)
            print(f"Benchmark folder created at {BENCHMARK_FOLDER}")
        except OSError as error:
            print(f"Failed to create folder: {error}")
    
    # launch benchmark
    benchmark_rk("flickr8k_pgemma")
