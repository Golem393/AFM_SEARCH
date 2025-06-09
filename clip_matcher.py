import torch
import clip
from PIL import Image
import os
import numpy as np
import shutil
from matching_algorithms import MeanMatcher, ParetoFrontMatcher
from keyword_extractor import KeyWordExtractor

class CLIPMatcher:
    def __init__(self, image_folder, keywextractor, top_k, model="ViT-L/14@336px"):
        self.image_folder = image_folder
        self.model_name = model
        self.keywextractor = keywordextractor # Can be none to just use the prompt w/o keyword extraction
        self.top_k = top_k # only display top k matches
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate model
        available_models = clip.available_models()
        if self.model_name not in available_models:
            raise ValueError(f"Model '{self.model_name}' is not available. Choose from: {available_models}")
        
        # Load model
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        
        # Prepare file and folder names
        self.model_name_safe = self.model_name.replace("/", "_")
        self.base_folder = os.path.join(self.image_folder, "..")
        
        self.embedding_file = os.path.join(self.base_folder, f"{self.model_name_safe}_embeddings.npy")
        self.filename_file = os.path.join(self.base_folder, f"{self.model_name_safe}_filenames.npy")
        self.output_folder = os.path.join(self.base_folder, f"{self.model_name_safe}_top_matches")

        # Check if image embeddings already exist:
        # If exist: Load embeddings from disk to memory to improve runtime in benchmarks. (Otherwise have to be loaded to disk for every retrieval request)
        if os.path.exists(self.embedding_file) and os.path.exists(self.filename_file):
            self.image_embeddings, self.filepaths = self.load_embeddings()
        # If not exist: compute embedding ans 
        else:
            self.image_embeddings, self.filepaths = self.compute_embeddings()
        
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
    
    def get_text_features(self, prompt):
        
        # If no keyword extractor is given to the CLIPMatcher instance, the whole prompt will be used for retrieval
        if self.keywordextractor is not None:
            keywordExtractor = self.keywordextractor
            prompt = keywordExtractor.extract(prompt, device=self.device)

        text_tokens = clip.tokenize(prompt).to(self.device) 
        # text_tokens = clip.tokenize([self.prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()
    
    def find_top_matches(self, prompt):
        matcher = MeanMatcher(self.image_embeddings, self.get_text_features(prompt))
        selected_imgs = matcher.match(self.filepaths)[:self.top_k]
        
        # Print results
        print(f"\nTop {self.top_k} most similar images for: \"{prompt}\" using {self.model_name}\n")
        for i in range(len(selected_imgs)):
            #print(f"{i+1:2d}. {selected_imgs[i]:30s} | Similarity: {similarities[i]:.4f}")
            print(f"{i+1:2d}. {selected_imgs[i]:30s}")

        # Create output folder
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)
        
        # Copy top images
        for idx in range(len(selected_imgs)):
            src = os.path.join(self.image_folder, selected_imgs[idx])
            dst = os.path.join(self.output_folder, selected_imgs[idx])
            shutil.copy(src, dst)
        
        # Save scores 
        #TODO: matching algorithm do not yet support this feature
        score_file = os.path.join(self.output_folder, "scores.txt")
        similarities = np.zeros(len(selected_imgs))
        with open(score_file, "w") as f:
            for idx in range(len(selected_imgs)):
                fname = selected_imgs[idx]
                score = 0 # TODO:
                f.write(f"{fname}\t{score:.4f}\n")
        
        # TODO: Just return selected images, maybe datatype check (List vs. numpy array)?
        return [selected_imgs[idx] for idx in range(len(selected_imgs))], [similarities[idx] for idx in range(len(selected_imgs))]