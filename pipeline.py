import os
from clip_matcher import CLIPMatcher
from git_matcher import GitMatcher
from llava_runner import LLaVAVerifier
import matplotlib.pyplot as plt
from PIL import Image
import math

class CLIPLLaVAPipeline:
    def __init__(self, image_video_folder,  
                 clip_model="ViT-L/14@336px", 
                 top_k=10, 
                 llava_model="liuhaotian/llava-v1.5-7b",
                 video_embedder_type="keyframe_k_frames",
                 frames_per_video_clip_max=20):
        self.image_video_folder = image_video_folder
        self.git_model = "microsoft/git-large"
        self.clip_model = clip_model
        self.top_k = top_k
        self.llava_model = llava_model
        self.video_embedder_type = video_embedder_type
        self.frames_per_video_clip_max = frames_per_video_clip_max
        
    def run(self, prompt):
        # Step 1: Run CLIP matcher to find top matches
        print("Running CLIP matcher...")
        clip_matcher = CLIPMatcher(
            image_video_folder=self.image_video_folder,
            embedding_folder="eval_coco/embeddings",
            video_embedder_type = self.video_embedder_type,
            frames_per_video_clip_max = self.frames_per_video_clip_max,
            # prompt=prompt,
            # model=self.clip_model,
            top_k=self.top_k
        )
        # git_matcher = GitMatcher(
        #     image_folder=self.image_folder,
        #     prompt=self.git_prompt,
        #     model=self.git_model,
        #     top_k=self.top_k
        # )
        top_files, top_scores = clip_matcher.find_top_matches(prompt)
        # top_files, top_scores = git_matcher.find_top_matches(prompt)
        # output_folder = clip_matcher.output_folder
        
        # Step 2: Verify matches with LLaVA
        print("\nVerifying matches with LLaVA...")
        llava = LLaVAVerifier()
        verification_results = llava.verify_images(
            img_path=self.image_video_folder,
            images=top_files,
            prompt=f"Does this image show a {prompt}? (answer only with 'yes' or 'no' and nothing else!)"
        )
        
        # Step 3: Analyze results
        confirmed_matches = []
        rejected_matches = []
        unclear_matches = []
        
        for filename, output in verification_results.items():
            verdict = LLaVAVerifier.extract_verdict(output)
            if verdict is True:
                confirmed_matches.append(filename)
            elif verdict is False:
                rejected_matches.append(filename)
            else:
                unclear_matches.append(filename)
               
        return {
            "confirmed": confirmed_matches,
            "rejected": rejected_matches,
            "unclear": unclear_matches,
            "all_results": verification_results
        }

if __name__ == "__main__":
    prompt = "elephant"
    # Example usage
    pipeline = CLIPLLaVAPipeline(
        image_video_folder="Thailand/image_video",
        clip_model="ViT-L/14@336px",
        top_k=15,
        video_embedder_type = "keyframe_k_frames",  #"keyframe_k_frames", "uniform_k_frames", "keyframe_average", "uniform_average", "optical_flow" "clip_k_frames"
        frames_per_video_clip_max = 20
    )
    
    results = pipeline.run(prompt)