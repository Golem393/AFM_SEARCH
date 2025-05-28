import torch
import clip
from PIL import Image
import os
import numpy as np
import shutil

class CLIPMatcher:
    def __init__(self, image_folder, prompt, model="ViT-L/14@336px", top_k=10):
        self.image_folder = image_folder
        self.prompt = prompt
        self.model_name = model
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate model
        available_models = clip.available_models()
        if self.model_name not in available_models:
            raise ValueError(f"Model '{self.model_name}' is not available. Choose from: {available_models}")
        
        # Load model
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        
        # Prepare file and folder names
        self.model_name_safe = self.model_name.replace("/", "_")
        self.prompt_name_safe = self.prompt.replace(" ", "_")
        self.base_folder = os.path.join(self.image_folder, "..")
        
        self.embedding_file = os.path.join(self.base_folder, f"{self.model_name_safe}_embeddings.npy")
        self.filename_file = os.path.join(self.base_folder, f"{self.model_name_safe}_filenames.npy")
        self.output_folder = os.path.join(self.base_folder, f"{self.prompt_name_safe}_{self.model_name_safe}_top_matches")
        
    def compute_embeddings(self):
        print("Computing image embeddings...")
        image_embeddings = []
        image_filenames = []

        for filename in os.listdir(self.image_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(self.image_folder, filename)
                try:
                    image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        image_feature = self.model.encode_image(image).float()
                        image_feature /= image_feature.norm(dim=-1, keepdim=True)
                        image_embeddings.append(image_feature.cpu().numpy())
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
    
    def get_text_features(self):
        text_tokens = clip.tokenize([self.prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()
    
    def find_top_matches(self):
        # Load or compute embeddings
        if os.path.exists(self.embedding_file) and os.path.exists(self.filename_file):
            image_embeddings, image_filenames = self.load_embeddings()
        else:
            image_embeddings, image_filenames = self.compute_embeddings()
        
        # Get text features
        text_features = self.get_text_features()
        
        # Compute similarities
        similarities = image_embeddings @ text_features.T
        similarities = similarities.squeeze()
        
        # Get top indices
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        # Print results
        print(f"\nTop {self.top_k} most similar images for: \"{self.prompt}\" using {self.model_name}\n")
        for i, idx in enumerate(top_indices):
            print(f"{i+1:2d}. {image_filenames[idx]:30s} | Similarity: {similarities[idx]:.4f}")
        
        # Create output folder
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)
        
        # Copy top images
        for idx in top_indices:
            src = os.path.join(self.image_folder, image_filenames[idx])
            dst = os.path.join(self.output_folder, image_filenames[idx])
            shutil.copy(src, dst)
        
        # Save scores
        score_file = os.path.join(self.output_folder, "scores.txt")
        with open(score_file, "w") as f:
            for idx in top_indices:
                fname = image_filenames[idx]
                score = similarities[idx]
                f.write(f"{fname}\t{score:.4f}\n")
        
        return [image_filenames[idx] for idx in top_indices], [similarities[idx] for idx in top_indices]