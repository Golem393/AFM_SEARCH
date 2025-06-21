import torch
from PIL import Image
import os
import numpy as np
import shutil
from matching_algorithms import MeanMatcher, ParetoFrontMatcher
import requests
from clip_video_embedder import CLIPVideoEmbedder
from pathlib import Path

class CLIPMatcher:
    def __init__(self, image_video_folder, prompt, video_embedder_type, top_k, frames_per_video_clip_max):
        self.image_video_folder = image_video_folder
        self.video_embedder_type = video_embedder_type
        self.prompt = prompt
        self.top_k = top_k
        self.frames_per_video_clip_max = frames_per_video_clip_max
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server_url = "http://localhost:5000"
        
        # Prepare file and folder names
        self.prompt_name_safe = self.prompt.replace(" ", "_")
        self.base_folder = os.path.join(self.image_video_folder, "..")

        model_name = requests.get("http://localhost:5000/clip/model_name").json()['clip_model_name']
        model_name_safe = model_name.replace("/", "_")
        self.embedding_file = os.path.join(self.base_folder, f"{model_name_safe}_embeddings.npy")
        self.video_embedding_file = os.path.join(self.base_folder, f"{model_name_safe}_{self.video_embedder_type}_embeddings.npy")
        self.filename_file = os.path.join(self.base_folder, f"{model_name_safe}_filenames.npy")
        self.video_filename_file = os.path.join(self.base_folder, f"{model_name_safe}_{self.video_embedder_type}_filenames.npy")
        self.output_folder = os.path.join(self.base_folder, f"{self.prompt_name_safe}_{model_name_safe}_top_matches")

        self.clip_video_embedder = CLIPVideoEmbedder(self.video_embedder_type, frames_per_video_clip_max)

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

        for filename in os.listdir(self.image_video_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(self.image_video_folder, filename)
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
    
    def compute_video_embeddings(self):
        print("Computing video embeddings...")
        video_embeddings = []
        video_filenames = []

        for filename in os.listdir(self.image_video_folder):
            if filename.lower().endswith((".mp4")):
                video_path = os.path.join(self.image_video_folder, filename)
                #try:
                video_features, video_paths = self.clip_video_embedder.get_video_embedding_and_paths(video_path)
                for i in range(len(video_features)):
                    video_feature = video_features[i]
                    video_frame_path = video_paths[i]
                    video_embeddings.append(video_feature)
                    video_filenames.append(video_frame_path)
                #except Exception as e:
                    #print(f"Failed to process {filename}: {e}")
        video_embeddings = np.vstack(video_embeddings)
        video_filenames = np.array(video_filenames)

        np.save(self.video_embedding_file, video_embeddings)
        np.save(self.video_filename_file, video_filenames)

        return video_embeddings, video_filenames
    
    def load_embeddings(self):
        print("Loading cached embeddings...")
        return np.load(self.embedding_file), np.load(self.filename_file)
    
    def load_video_embeddings(self):
        print("Loading cached embeddings...")
        return np.load(self.video_embedding_file), np.load(self.video_filename_file)
    
    def find_top_matches(self):
        # Load or compute embeddings
        if os.path.exists(self.embedding_file) and os.path.exists(self.filename_file):
            image_embeddings, image_filenames = self.load_embeddings()
        else:
            image_embeddings, image_filenames = self.compute_embeddings()

        if os.path.exists(self.video_embedding_file) and os.path.exists(self.video_filename_file):
            video_embeddings, video_filenames = self.load_video_embeddings()
        else:
            video_embeddings, video_filenames = self.compute_video_embeddings()

        all_embeddings = np.concatenate([image_embeddings, video_embeddings], axis=0)
        all_filenames = list(image_filenames) + list(video_filenames)


        matcher = MeanMatcher(all_embeddings, self.get_text_features())
        selected_data, cos_similarities = matcher.match(all_filenames, self.top_k)#[:self.top_k]
        
        # Print results
        print(f"\nTop {self.top_k} most similar images / videos for: \"{self.prompt}\" using CLIP\n")
        for i in range(len(selected_data)):
            print(f"{i+1:2d}. {selected_data[i]:30s}")

        # Create output folder
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

        # Copy top images / videos 
        for idx in range(len(selected_data)):
            src = os.path.join(self.image_video_folder, selected_data[idx])
            selected_data[idx] = selected_data[idx].replace(os.sep, "_")
            dst = os.path.join(self.output_folder, selected_data[idx])
            shutil.copy(src, dst)
        
        # Save scores
        score_file = os.path.join(self.output_folder, "scores.txt")
        similarities = np.zeros(len(selected_data))
        with open(score_file, "w") as f:
            for idx in range(len(selected_data)):
                fname = selected_data[idx]
                score = cos_similarities[idx]
                f.write(f"{fname}\t{score:.4f}\n")
        
        return [selected_data[idx] for idx in range(len(selected_data))], [similarities[idx] for idx in range(len(selected_data))]