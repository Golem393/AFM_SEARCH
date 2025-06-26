from clip_matcher import CLIPMatcher
from paligemma_runner import PaliGemmaVerifier
import os

class CLIPLLaVAPipeline:
    def __init__(self, image_folder, verification_prompt, top_k_clip_matches):
        self.image_folder = image_folder
        self.verification_prompt = verification_prompt #TODO: currently implemented on the server
        self.top_k_clip_matches = top_k_clip_matches
        
    def run(self, prompt):

        # Step 1: Run CLIP matcher to find top matches
        print("Running CLIP matcher...")
        clip_matcher = CLIPMatcher(
            image_folder=self.image_folder,
            prompt=prompt,
            top_k=self.top_k_clip_matches
        )

        top_files, top_scores = clip_matcher.find_top_matches()
        image_folder = clip_matcher.image_folder

        image_paths = [os.path.join(image_folder, file) for file in top_files]
        
        # Step 2: Verify matches with PaliGemma
        print("Verifying matches with PaliGemma")
        verifier = PaliGemmaVerifier()
        verdict = verifier.verify_batch(image_paths, prompt)

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
    pipeline = CLIPLLaVAPipeline(
        image_folder="/usr/prakt/s0122/afm/dataset/cc3m/cc3m_0000/",
        verification_prompt=None,
        top_k_clip_matches=30
    )
    
    results = pipeline.run(prompt)
    print(results)