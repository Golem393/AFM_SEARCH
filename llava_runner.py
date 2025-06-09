import sys
import os
import requests

class LLaVAVerifier:
    def __init__(self):
        self.server_url = "http://localhost:5000"
      
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