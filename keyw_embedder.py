from transformers import pipeline
import torch
import numpy as np

class KeywordEmbedder:
    def __init__(self, prompt: str, device: str = 'cpu'):
        self.prompt = prompt
        self.device = device
        self.system_prompt = (
            "You extract important keywords or combinations of two keywords max from a given query for an image retrieval task. "
            "Return a comma separated list. Under no circumstances add words to the list that are not given by the user!"
        )
        
        # Load Gemma pipeline
        #print("[INFO] Load Gemma model:...", end='\r')
        self.pipe = pipeline(
            "text-generation",
            #model="google/gemma-3-1b-it",
            model = "google/flan-t5-base",
            device=self.device,
            torch_dtype=torch.bfloat16,
            token = "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"
        )
        print("[INFO] Load Gemma model: success")

    def extract_keywords(self):
        messages = [[
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": self.prompt}]}
        ]]

        output = self.pipe(messages, max_new_tokens=50)
        kw_string = output[0][0]['generated_text'][2]['content']
        print(f"[INFO] Extracted keywords: {kw_string}")
        return self.parse_keywords(kw_string)
    
    def parse_keywords(self, keyword_string: str):
        """Extracts list of keyword strings from comma-seperated string of keywords

        args:
            keyword: str string with comma-separated keywords
        """
        return [keyword.strip() for keyword in keyword_string.split(',') if keyword.strip()]
