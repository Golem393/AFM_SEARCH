import sys
import os

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

class LLaVAVerifier:
    def __init__(self, model_path="liuhaotian/llava-v1.5-7b"):
        self.model_path = model_path
        self.model_name = get_model_name_from_path(model_path)
        
    def verify_images(self, image_folder, prompt):
        results = {}
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                args = type('Args', (), {
                    "model_path": self.model_path,
                    "model_base": None,
                    "model_name": self.model_name,
                    "query": prompt,
                    "conv_mode": None,
                    "image_file": image_path,
                    "sep": ",",
                    "temperature": 0,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 512
                })()
                
                output = eval_model(args)
                results[filename] = output

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