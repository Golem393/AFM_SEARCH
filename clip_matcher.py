from matching_algorithms import MeanMatcher
from PIL import Image
import numpy as np
import requests
from video.clip_video_embedder import CLIPVideoEmbedder
from pathlib import Path
import requests
import torch
import shutil
import os

class CLIPMatcher:
    def __init__(self, 
                 image_video_folder, 
                 embedding_folder, 
                 top_k=10,
                 print_progress: bool = False,
                 port:int=5000,
                 subset=None,
                 frames_per_video_clip_max=None,
                 video_embedder_type=None
                 ):
        self.image_video_folder = image_video_folder
        self.video_embedder_type = video_embedder_type
        self.embedding_folder = embedding_folder
        self.top_k = top_k
        self.subset = subset
        self.port = port
        self.frames_per_video_clip_max = frames_per_video_clip_max
        self.print_progress = print_progress
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server_url = f"http://localhost:{self.port}"
        """
        # Validate model
        available_models = clip.available_models()
        if self.model_name not in available_models:
            raise ValueError(f"Model '{self.model_name}' is not available. Choose from: {available_models}")
        
        # Load model
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)"""
        print(f"initializing ClipMatcher with device: {self.device}") if self.print_progress else None
        
        model_name = requests.get(f"http://localhost:{self.port}/clip/model_name").json()['clip_model_name']
        model_name_safe = model_name.replace("/", "_")
        self.clip_video_embedder = CLIPVideoEmbedder(self.video_embedder_type, frames_per_video_clip_max, self.server_url)
        if self.subset is None:
            self.embedding_file = os.path.join(self.embedding_folder, f"{model_name_safe}_embeddings.npy")
            self.video_embedding_file = os.path.join(self.embedding_folder, f"{model_name_safe}_{self.video_embedder_type}_embeddings.npy")
            self.filename_file = os.path.join(self.embedding_folder, f"{model_name_safe}_filenames.npy")
            self.video_timestamp_file = os.path.join(self.embedding_folder, f"{model_name_safe}_{self.video_embedder_type}_timestamps.npy")
        else:
            self.embedding_file = os.path.join(self.embedding_folder, f"{model_name_safe}_embeddings_{len(self.subset)}.npy")
            self.video_embedding_file = os.path.join(self.embedding_folder, f"{model_name_safe}_{self.video_embedder_type}_embeddings_{len(self.subset)}.npy")
            self.filename_file = os.path.join(self.embedding_folder, f"{model_name_safe}_filenames_{len(self.subset)}.npy")
            self.video_timestamp_file = os.path.join(self.embedding_folder, f"{model_name_safe}_{self.video_embedder_type}_timestamps_{len(self.subset)}.npy")
        # Load or compute embeddings
        if os.path.exists(self.embedding_file) and os.path.exists(self.filename_file):
            self.image_embeddings, self.image_filenames = self.load_embeddings()
        else:
            self.image_embeddings, self.image_filenames = self.compute_embeddings()

        if os.path.exists(self.video_embedding_file) and os.path.exists(self.video_timestamp_file):
            self.video_embeddings, self.video_timestamps = self.load_video_embeddings()
        else:
            self.video_embeddings, self.video_timestamps = self.compute_video_embeddings()

        self.all_embeddings = np.concatenate([self.image_embeddings, self.video_embeddings], axis=0)
        self.all_filenames_time_stamps = list(self.image_filenames) + list(self.video_timestamps)
        print(f"Done") if self.print_progress else None

    def get_image_embedding(self, image_path):
        response = requests.post(
            f"{self.server_url}/clip/embed_image",
            json={"image_path": image_path}
        )
        return np.array(response.json()['result'])

    def get_text_features(self, prompt):
        response = requests.post(
            f"{self.server_url}/clip/embed_text",
            json={"text": prompt}
        )
        return np.array(response.json()['result'])
    
    def compute_embeddings(self):
        print("Computing image embeddings...")
        image_embeddings = []
        image_filenames = []
        i = 0

        if self.subset is None:
            list_dir = os.listdir(self.image_video_folder)
        else:
            list_dir = self.subset
            print(f"compute embs for subset")

        for filename in list_dir:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(self.image_video_folder, filename)
                try:
                    image_feature = self.get_image_embedding(image_path)
                    image_embeddings.append(image_feature)
                    image_filenames.append(filename)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
                i += 1
                if i % 1000 == 0 and self.print_progress:
                    print(f"Processed {i} images...")

        image_embeddings = np.vstack(image_embeddings)
        image_filenames = np.array(image_filenames)

        np.save(self.embedding_file, image_embeddings)
        np.save(self.filename_file, image_filenames)
        print(f"Done") if self.print_progress else None

        return image_embeddings, image_filenames
    
    def compute_video_embeddings(self):
        print("Computing video embeddings...")
        video_embeddings = []
        video_codes = []

        if self.subset is None:
            list_dir = os.listdir(self.image_video_folder)
        else:
            list_dir = self.subset
            print(f"compute embs for subset")

        for filename in list_dir:
            if filename.lower().endswith((".mp4")):
                video_path = os.path.join(self.image_video_folder, filename)
                video_features, video_timestamps = self.clip_video_embedder.get_video_embedding_and_timestamps(video_path)
                for i in range(len(video_features)):
                    video_embeddings.append(video_features[i])
                    video_codes.append(video_timestamps[i])

        video_embeddings = np.vstack(video_embeddings)
        video_codes = np.array(video_codes)

        np.save(self.video_embedding_file, video_embeddings)
        np.save(self.video_timestamp_file, video_codes)

        return video_embeddings, video_codes

    def load_embeddings(self):
        print("Loading cached embeddings...")
        return np.load(self.embedding_file), np.load(self.filename_file)
    
    def load_video_embeddings(self):
        print("Loading cached embeddings...")
        return np.load(self.video_embedding_file), np.load(self.video_timestamp_file)
    
    def find_top_matches(self, prompt):
        matcher = MeanMatcher(self.all_embeddings, self.get_text_features(prompt))
        selected_imgs_vid, similarities = matcher.match(self.all_filenames_time_stamps, self.top_k)

        # # Get text features
        # text_features = self.get_text_features()
        
        # # Compute similarities
        # similarities = self.image_embeddings @ text_features.T
        # similarities = similarities.squeeze()
        
        # # Get top indices
        # top_indices = np.argsort(similarities)[::-1][:self.top_k] 
        
        return selected_imgs_vid, similarities
