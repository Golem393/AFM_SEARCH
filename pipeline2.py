from clip_matcher import CLIPMatcher
from paligemma_runner import PaliGemmaVerifier
import os
from pprint import pprint

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
    
    def run(self, prompt):

        # Step 1: Run CLIP matcher to find top matches
        print("Running CLIP matcher...")
        clip_matcher = CLIPMatcher(
            image_video_folder=self.gallery,
            embedding_folder='/usr/prakt/s0122/afm/dataset/cc3m/',
            top_k=self.top_k_clip_matches,
            port=self.port,
            video_embedder_type=self.video_embedder_type,
            frames_per_video_clip_max=self.frames_per_video_clip_max,
        )

        top_files, top_scores = clip_matcher.find_top_matches(prompt)
        
        # top_files are just filenames, PaliGemma requires file_paths
        gallery = clip_matcher.image_video_folder
        top_filepaths = [os.path.join(gallery, file) for file in top_files]
        
        # Step 2: Verify matches with PaliGemma
        print("Verifying matches with PaliGemma")
        verifier = PaliGemmaVerifier(port=self.port)
        verdict = verifier.verify_batch(top_filepaths, prompt)
        
        print("PaliGemma verification results:")
        print(verdict)
        
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
    prompt = "dogs playing at the park"
    # Example usage
    pipeline = CLIPPaliGemmaPipeline(
        gallery="/usr/prakt/s0122/afm/dataset/cc3m/cc3m_0000/",
        top_k_clip_matches=30, 
        video_embedder_type = "keyframe_k_frames",  #"keyframe_k_frames", "uniform_k_frames", "keyframe_average", "uniform_average", "optical_flow" "clip_k_frames"
        frames_per_video_clip_max = 20,
        port=5004
    )
    
    results = pipeline.run(prompt)
    print(results)