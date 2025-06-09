import sys
import os
import requests
import time
from typing import Dict

class LLaVAVerifier:
    def __init__(self):
        self.server_url = "http://localhost:5000"
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session = requests.Session()
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
      
    def verify_images(self, image_folder, prompt):
        results = {}
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                response = requests.post(
                    f"{self.server_url}/llava/verify",
                    json={
                        "image_path": image_path,
                        "prompt": prompt
                    }
                )
                results[filename] = response.json()['result']
                # print(response.json()['result'])
        return results
    
    def verify_images_batch(self, image_folder: str, prompt: str) -> Dict[str, str]:
        """Batch processing - send all images at once"""
        image_paths = []
        filenames = []
        
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                image_paths.append(image_path)
                filenames.append(filename)
        
        if not image_paths:
            return {}
        
        print(f"Processing {len(image_paths)} images in batch...")
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.server_url}/llava/verify_batch",
                json={
                    "image_paths": image_paths,
                    "prompt": prompt
                },
                timeout=600  # 10 minute timeout for batch
            )
            response.raise_for_status()
            results = response.json()['results']
            
            # Map back to filenames
            filename_results = {}
            for i, filename in enumerate(filenames):
                if image_paths[i] in results:
                    filename_results[filename] = results[image_paths[i]]
                    # print(f"{filename}: {results[image_paths[i]]}")
            
            elapsed = time.time() - start_time
            print(f"Batch processing completed in {elapsed:.2f} seconds")
            print(f"Average time per image: {elapsed/len(image_paths):.2f} seconds")
            
            return filename_results
            
        except requests.exceptions.RequestException as e:
            print(f"Batch request failed: {e}")
            # Fallback to individual processing
            return self.verify_images_concurrent(image_folder, prompt)
        
    @staticmethod
    def extract_verdict(text):
        # print(text)
        if text is None:
            return None
        text = text.lower()
        if "yes" in text and "no" not in text:
            return True
        if "no" in text and "yes" not in text:
            return False
        return None  # unclear answer