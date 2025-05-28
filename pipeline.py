import os
from clip_matcher import CLIPMatcher
from llava_runner import LLaVAVerifier

class CLIPLLaVAPipeline:
    def __init__(self, image_folder, clip_prompt, verification_prompt, 
                 clip_model="ViT-L/14@336px", top_k=10, llava_model="liuhaotian/llava-v1.5-7b"):
        self.image_folder = image_folder
        self.clip_prompt = clip_prompt
        self.verification_prompt = verification_prompt
        self.clip_model = clip_model
        self.top_k = top_k
        self.llava_model = llava_model
        
    def run(self):
        # Step 1: Run CLIP matcher to find top matches
        print("Running CLIP matcher...")
        clip_matcher = CLIPMatcher(
            image_folder=self.image_folder,
            prompt=self.clip_prompt,
            model=self.clip_model,
            top_k=self.top_k
        )
        top_files, top_scores = clip_matcher.find_top_matches()
        output_folder = clip_matcher.output_folder
        
        # Step 2: Verify matches with LLaVA
        print("\nVerifying matches with LLaVA...")
        llava = LLaVAVerifier(model_path=self.llava_model)
        verification_results = llava.verify_images(
            image_folder=output_folder,
            prompt=self.verification_prompt
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
        
        # Print summary
        print("\n=== Results Summary ===")
        print(f"Total matches from CLIP: {len(top_files)}")
        print(f"Confirmed by LLaVA: {len(confirmed_matches)}")
        print(f"Rejected by LLaVA: {len(rejected_matches)}")
        print(f"Unclear results: {len(unclear_matches)}")
        
        # Create subfolders for confirmed and rejected matches
        confirmed_folder = os.path.join(output_folder, "confirmed")
        rejected_folder = os.path.join(output_folder, "rejected")
        unclear_folder = os.path.join(output_folder, "unclear")
        
        os.makedirs(confirmed_folder, exist_ok=True)
        os.makedirs(rejected_folder, exist_ok=True)
        os.makedirs(unclear_folder, exist_ok=True)
        
        # Move files to appropriate folders
        for filename in confirmed_matches:
            src = os.path.join(output_folder, filename)
            dst = os.path.join(confirmed_folder, filename)
            os.rename(src, dst)
        
        for filename in rejected_matches:
            src = os.path.join(output_folder, filename)
            dst = os.path.join(rejected_folder, filename)
            os.rename(src, dst)
            
        for filename in unclear_matches:
            src = os.path.join(output_folder, filename)
            dst = os.path.join(unclear_folder, filename)
            os.rename(src, dst)
        
        return {
            "confirmed": confirmed_matches,
            "rejected": rejected_matches,
            "unclear": unclear_matches,
            "all_results": verification_results
        }

if __name__ == "__main__":
    prompt = "flag of Thailand"
    # Example usage
    pipeline = CLIPLLaVAPipeline(
        image_folder="Thailand/image",
        clip_prompt=prompt,
        verification_prompt=f"Does this image show a {prompt}? (answer with 'yes' or 'no')",
        clip_model="ViT-L/14@336px",
        top_k=10
    )
    
    results = pipeline.run()