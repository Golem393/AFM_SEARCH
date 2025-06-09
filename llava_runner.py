import sys
import os
import requests

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

class LLaVAVerifier:
    def __init__(self, model_path="liuhaotian/llava-v1.5-7b"):
        self.server_url = "http://localhost:5000"
        self.model_path = model_path
        self.model_name = get_model_name_from_path(model_path)
      
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
                print(response.json()['result'])
        return results
        
    @staticmethod
    def extract_verdict(text):
        print(text)
        if text is None:
            return None
        text = text.lower()
        if "yes" in text and "no" not in text:
            return True
        if "no" in text and "yes" not in text:
            return False
        return None  # unclear answer