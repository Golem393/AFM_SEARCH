import os
import torch
import numpy as np
from pathlib import Path
import shutil
from PIL import Image
from matching_algorithms import MeanMatcher, ParetoFrontMatcher
import requests

class GitMatcher:
    def __init__(self, 
                 image_folder: Path, 
                 embedding_folder: Path, 
                 top_k: int,
                 print_progress: bool = False,
                 port: int = 5000,
                 ):
        self.image_folder = image_folder
        self.top_k = top_k
        self.port = port
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.print_progress = print_progress
        print(f"initializing GitMatcher with device: {self.device}") if self.print_progress else None

        # Prepare file and folder names
        model_name = requests.get(f"http://localhost:{self.port}/git/model_name").json()['git_model_name']
        model_name_safe = model_name.replace("/", "_")
        # self.prompt_name_safe = self.prompt.replace(" ", "_")
        self.base_folder = embedding_folder

        self.embedding_file = os.path.join(self.base_folder, f"{model_name_safe}_embeddings.npy")
        self.filename_file = os.path.join(self.base_folder, f"{model_name_safe}_filenames.npy")
        # self.output_folder = os.path.join(self.base_folder, f"{self.prompt_name_safe}_{model_name_safe}_top_matches")
        print(f"Done") if self.print_progress else None

    def compute_caption_from_server(self, image_path):
        try:
            response = requests.post(f"http://localhost:{self.port}/git/caption", json={
                "image_path": image_path
            })
            response.raise_for_status()
            return response.json()['result']
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return None

    def compute_embeddings(self):
        print(f"Compute embedds") if self.print_progress else None

        image_embeddings = []
        image_filenames = []
        i = 0
        
        for filename in os.listdir(self.image_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(self.image_folder, filename)
                try:
                    caption = self.compute_caption_from_server(image_path)
                    if caption is not None:
                        image_embeddings.append(caption)
                        image_filenames.append(filename)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
                    
                i += 1
                if i % 1000 == 0 and self.print_progress:
                    print(f"Processed {i} images...")
        np.save(self.embedding_file, image_embeddings)
        np.save(self.filename_file, image_filenames)
        
        print(f"Done") if self.print_progress else None
    
        return image_embeddings, image_filenames

    def load_embeddings(self):
        print("Loading cached captions...")
        return np.load(self.embedding_file, allow_pickle=True), np.load(self.filename_file, allow_pickle=True)

    def get_text_features(self):
        return self.prompt

    def find_top_matches(self, prompt):
        # Load or compute embeddings
        if os.path.exists(self.embedding_file) and os.path.exists(self.filename_file):
            image_embeddings, image_filenames = self.load_embeddings()
        else:
            image_embeddings, image_filenames = self.compute_embeddings()

        # Compute similarities
        similarities = []
        for caption in image_embeddings:
            similarity = self.compute_similarity(prompt, caption)
            similarities.append(similarity)

        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        selected_imgs = [image_filenames[i] for i in top_indices]
        selected_scores = [similarities[i] for i in top_indices]

        # # Print results
        # print(f"\nTop {self.top_k} most similar images for: \"{prompt}\" using GIT\n")
        # for i, (img, score) in enumerate(zip(selected_imgs, selected_scores)):
        #     print(f"{i+1:2d}. {img:30s} | Similarity: {score:.4f}")

        # # Create output folder
        # if os.path.exists(self.output_folder):
        #     shutil.rmtree(self.output_folder)
        # os.makedirs(self.output_folder)

        # # Copy top images
        # for img in selected_imgs:
        #     src = os.path.join(self.image_folder, img)
        #     dst = os.path.join(self.output_folder, img)
        #     shutil.copy(src, dst)

        # # Save scores
        # score_file = os.path.join(self.output_folder, "scores.txt")
        # with open(score_file, "w") as f:
        #     for img, score in zip(selected_imgs, selected_scores):
        #         f.write(f"{img}\t{score:.4f}\n")

        return selected_imgs, selected_scores

    def compute_similarity(self, text1, text2):
        # Simple similarity based on common words
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        return len(set1 & set2) / len(set1 | set2)
