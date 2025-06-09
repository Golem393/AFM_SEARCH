import torch
from PIL import Image
import os
import numpy as np
import shutil
from matching_algorithms import MeanMatcher, ParetoFrontMatcher
from keyw_embedder import KeywordEmbedder
import requests

class CLIPMatcher:
    def __init__(self, 
                 image_folder, 
                 embedding_folder, 
                 prompt, 
                 top_k=10
                 
                 ):
        self.image_folder = image_folder
        self.embedding_folder = embedding_folder
        self.prompt = prompt
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server_url = "http://localhost:5000"
        """
        # Validate model
        available_models = clip.available_models()
        if self.model_name not in available_models:
            raise ValueError(f"Model '{self.model_name}' is not available. Choose from: {available_models}")
        
        # Load model
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)"""
        
        # Prepare file and folder names
        self.prompt_name_safe = self.prompt.replace(" ", "_")
        self.base_folder = os.path.join(self.image_folder, "..")

        model_name = requests.get("http://localhost:5000/clip/model_name").json()['clip_model_name']
        model_name_safe = model_name.replace("/", "_")
        self.embedding_file = os.path.join(self.embedding_folder, f"{model_name_safe}_embeddings.npy")
        self.filename_file = os.path.join(self.embedding_folder, f"{model_name_safe}_filenames.npy")
        self.output_folder = os.path.join(self.embedding_folder, f"{self.prompt_name_safe}_{model_name_safe}_top_matches")
        
        # Load or compute embeddings
        if os.path.exists(self.embedding_file) and os.path.exists(self.filename_file):
            self.image_embeddings, self.image_filenames = self.load_embeddings()
        else:
            self.image_embeddings, self.image_filenames = self.compute_embeddings()
        

    def get_image_embedding(self, image_path):
        response = requests.post(
            f"{self.server_url}/clip/embed_image",
            json={"image_path": image_path}
        )
        return np.array(response.json()['result'])

    def get_text_features(self):
        response = requests.post(
            f"{self.server_url}/clip/embed_text",
            json={"text": self.prompt}
        )
        return np.array(response.json()['result'])
    
    def compute_embeddings(self):
        print("Computing image embeddings...")
        image_embeddings = []
        image_filenames = []

        for filename in os.listdir(self.image_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(self.image_folder, filename)
                try:
                    image_feature = self.get_image_embedding(image_path)
                    image_embeddings.append(image_feature)
                    image_filenames.append(filename)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

        image_embeddings = np.vstack(image_embeddings)
        image_filenames = np.array(image_filenames)

        np.save(self.embedding_file, image_embeddings)
        np.save(self.filename_file, image_filenames)
        return image_embeddings, image_filenames
    
    def load_embeddings(self):
        print("Loading cached embeddings...")
        return np.load(self.embedding_file), np.load(self.filename_file)
    
    def find_top_matches(self):
        matcher = MeanMatcher(self.image_embeddings, self.get_text_features())
        selected_imgs = matcher.match(self.image_filenames)

        ''' Get text features
        text_features = self.get_text_features()
        
        # Compute similarities
        similarities = image_embeddings @ text_features.T
        similarities = similarities.squeeze()
        
        # Get top indices
        top_indices = np.argsort(similarities)[::-1][:self.top_k] '''
        
        similarities = np.zeros(len(selected_imgs))
        return [selected_imgs[idx] for idx in range(len(selected_imgs))], [similarities[idx] for idx in range(len(selected_imgs))]