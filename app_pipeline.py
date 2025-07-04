from clip_matcher import CLIPMatcher
from paligemma_runner import PaliGemmaVerifier
import os
from pprint import pprint
import json
import time

class CLIPPaliGemmaPipeline:
    def __init__(self, 
                 gallery, 
                 top_k_clip_matches, 
                 video_embedder_type:str="keyframe_k_frames", 
                 frames_per_video_clip_max:int=20,
                 port:int=5000
                 ):
        
        self.gallery = gallery
        self.top_k_clip_matches = top_k_clip_matches
        self.video_embedder_type = video_embedder_type
        self.frames_per_video_clip_max = frames_per_video_clip_max
        self.port = port

        self.clip_matcher = CLIPMatcher(
            image_video_folder=self.gallery,
            embedding_folder='/usr/prakt/s0122/afm/dataset/flickr8k/',
            top_k=self.top_k_clip_matches,
            port=self.port,
            video_embedder_type=self.video_embedder_type,
            frames_per_video_clip_max=self.frames_per_video_clip_max,
        )
    
    def run(self, prompt):

        # Step 1: Run CLIP matcher to find top matches
        print("Running CLIP matcher...")
        top_files, top_scores = self.clip_matcher.find_top_matches(prompt)
        
        # top_files are just filenames, PaliGemma requires file_paths
        gallery = self.clip_matcher.image_video_folder
        top_filepaths = [os.path.join(gallery, file) for file in top_files]
        
        # Step 2: Verify matches with PaliGemma
        print("Verifying matches with PaliGemma")
        verifier = PaliGemmaVerifier(port=self.port)
        verdict = verifier.verify_batch(top_filepaths, prompt)
        
        # cross reference vlm verdict with clip top matches
        confirmed_matches, rejected_matches, unclear_matches = verifier.corssref_results(verdict, top_files)    
        
        # Print summary
        print("\n=== Results Summary ===")
        print(f"Total matches from CLIP: {len(top_files)}")
        print(f"Confirmed by PaliGemma: {len(confirmed_matches)}")
        print(f"Rejected by PaliGemma: {len(rejected_matches)}")
        print(f"Unclear results: {len(unclear_matches)}")

        return {
            "confirmed": confirmed_matches,
            "rejected": rejected_matches,
            "unclear": unclear_matches,
            "clip_matches": top_files,
        }
        
if __name__ == "__main__":

    pipeline = CLIPPaliGemmaPipeline(
        gallery="/usr/prakt/s0122/afm/dataset/flickr8k/Flicker8k_Dataset/",
        top_k_clip_matches=30, 
        video_embedder_type = "keyframe_k_frames",  #"keyframe_k_frames", "uniform_k_frames", "keyframe_average", "uniform_average", "optical_flow" "clip_k_frames"
        frames_per_video_clip_max = 20,
        port=5004
    )
    print("Pipeline running!")

    file_path = "tmp/gradio_tmp/search_requests.json"

    while True:
        if not os.path.exists(file_path):
            time.sleep(1)
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                time.sleep(0.1)
                continue  # file might be mid-write
        
        # Look for the first key (query) with an empty list as its value (no results)
        pending_query = next((k for k, v in data.items() if v == []), None)

        if pending_query:
            print(f"Process query: {pending_query}")
            results = pipeline.run(pending_query)

            # Update the file with the new results
            results_path = [os.path.join(pipeline.gallery, result) for result in results['confirmed']]
            data[pending_query] = results_path
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        else:
            time.sleep(0.1)
