from clip_matcher2 import CLIPMatcher
from paligemma_runner import PaliGemmaVerifier
import os
import json
import time

class CLIPPaliGemmaPipeline:
    def __init__(self, 
                 gallery, 
                 config_path,
                 frames_per_video_clip_max:int=20,
                 port:int=5000
                 ):
        
        self.gallery = gallery
        self.config_path = config_path
        self.frames_per_video_clip_max = frames_per_video_clip_max
        self.port = port

        # get confgurable settings from config and set self. variables (e.g. keyframe extractor)
        print("Load settings...")
        self.get_settings() 

        self.clip_matcher = CLIPMatcher(
            image_video_folder=self.gallery,
            embedding_folder='/usr/prakt/s0122/afm/dataset/demo',
            top_k=self.top_n,
            port=self.port,
            video_embedder_type=self.keyframe_extractor,
            frames_per_video_clip_max=self.frames_per_video_clip_max,
        )

        self.verifier = PaliGemmaVerifier(port=self.port)
    
    def run(self, prompt):

        print("Updating settings before run...")
        self.get_settings()

        # Step 1: Run CLIP matcher to find top matches
        print("Running CLIP matcher...")
        top_files, top_scores = self.clip_matcher.find_top_matches(
            prompt, 
            self.top_n,
            self.retr_imgs,
            self.retr_vids
        )

        # top_files are just filenames, PaliGemma requires file_paths
        gallery = self.clip_matcher.image_video_folder
        top_filepaths = [os.path.join(gallery, file) for file in top_files]
        
        if self.use_vlm:
            # Verify matches with PaliGemma
            prompt = self.verification_prompt.split("<query>")[0] + prompt + self.verification_prompt.split("<query>")[1]
            print("Verifying matches with PaliGemma")
            verdict = self.verifier.verify_batch(top_filepaths, prompt)
        
            # cross reference vlm verdict with clip top matches
            confirmed_matches, rejected_matches, unclear_matches = self.verifier.corssref_results(verdict, top_files)  
            # get scores for confirmed matches
            confirmed_scores = [top_scores[top_files.index(match)] for match in confirmed_matches]
        
        else:
            confirmed_matches = top_files
            rejected_matches = []
            unclear_matches = []
            confirmed_scores = top_scores
        
        # Print summary
        print("\n=== Results Summary ===")
        print(f"Total matches from CLIP: {len(top_files)}")
        print(f"Confirmed by PaliGemma: {len(confirmed_matches)}")
        print(f"Rejected by PaliGemma: {len(rejected_matches)}")
        print(f"Unclear results: {len(unclear_matches)}")
        print("\n")

        return {
            "confirmed": confirmed_matches,
            "rejected": rejected_matches,
            "unclear": unclear_matches,
            "clip_matches": top_files,
            "confirmed_scores": confirmed_scores
        }
    
    def get_settings(self):
        # Load settings from config and set object attributes
        try:
            with open(self.config_path, "r") as cfg_file:
                cfg = json.load(cfg_file)
            top_n = cfg["active"]["NUM_CLIPMATCHES"] if cfg["active"]["NUM_CLIPMATCHES"] is not None else 30
            use_vlm = bool(cfg["active"]["VERIFY"]) if cfg["active"]["VERIFY"] is not None else True
            retr_imgs = bool(cfg["active"]["RETR_IMGS"]) if cfg["active"]["RETR_IMGS"] is not None else True
            retr_vids = bool(cfg["active"]["RETR_VIDS"]) if cfg["active"]["RETR_VIDS"] is not None else True
            keyframe_extractor = cfg["active"]["VIDEO_EMB"] if cfg["active"]["VIDEO_EMB"] is not None else "keyframe_k_frames"
            verification_prompt = cfg["active"]["VERIFIC_PROMPT"] if cfg["active"]["VERIFIC_PROMPT"] is not None else "Is <query> a fitting description of the image? Answer only with yes or no!"
            
        except Exception as e:
            print(f"Couldn't find config file {e}. Use default config")
            top_n = 30
            use_vlm = True
            retr_imgs = True
            retr_vids = True
            keyframe_extractor = "keyframe_k_frames" #"keyframe_k_frames", "uniform_k_frames", "keyframe_average", "uniform_average", "optical_flow" "clip_k_frames"
            verification_prompt = "Is <query> a fitting description of the image? Answer only with yes or no!"

        self.top_n = top_n
        self.keyframe_extractor = keyframe_extractor
        self.use_vlm = use_vlm
        self.retr_imgs = retr_imgs
        self.retr_vids = retr_vids
        self.verification_prompt = verification_prompt

        
if __name__ == "__main__":

    pipeline = CLIPPaliGemmaPipeline(
        gallery="/usr/prakt/s0122/afm/dataset/demo/cc3m_0000_0003",
        config_path="config.json",
        frames_per_video_clip_max = 20,
        port=5004
    )
    print("Pipeline running!")

    # Get current search requests from file
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
        
        # Look for the first key (query) with None as its value (no results)
        pending_query = next((k for k, v in data.items() if v is None), None)

        if pending_query:
            print(f"Process query: {pending_query}")
            results = pipeline.run(pending_query)

            # Update the file with the new results
            data[pending_query] = results
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        else:
            time.sleep(0.1)
