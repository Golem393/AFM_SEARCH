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
        
        # Compute embeddings of existing img/videos in gallery or load embeddings from existing file
        if os.path.exists(self.embedding_file) and os.path.exists(self.filename_file):
            self.image_embeddings, self.image_filenames = self.load_embeddings()
        else:
            self.image_embeddings, self.image_filenames = self.compute_embeddings(self.get_all_filepaths())

        if os.path.exists(self.video_embedding_file) and os.path.exists(self.video_timestamp_file):
            self.video_embeddings, self.video_timestamps = self.load_video_embeddings()
        else:
            self.video_embeddings, self.video_timestamps = self.compute_video_embeddings( self.get_all_filepaths())

        self.save() # save embeddings to file

        self.embedded_filepaths = self.get_embedded_filepaths() # get paths of all files that are embedded
    
    def get_all_filepaths(self):
        """Get paths to all files in image_video folder"""
        return [
            f for f in os.listdir(self.image_video_folder) 
            if os.path.isfile(os.path.join(self.image_video_folder, f))
            and f.lower().endswith((".png", ".jpeg", ".png", ".mp4"))
        ]

    def get_embedded_filepaths(self):
        """Get set of paths to all files that are currently embedded."""
        emb_img_filepaths = set(self.image_filenames) if self.image_filenames is not None else set()
        emb_vid_filepaths = [f"{video.split('.mp4_')[0]}.mp4" for video in self.video_timestamps] if self.video_timestamps is not None else set()
        return emb_img_filepaths | emb_vid_filepaths

    def concat_image_video_emb(self):
        """Concatenates image and video embeddings into one all embeddings file"""
        # Check what type of embeddings are avilable: img + video OR only image/video OR none
        if self.image_embeddings is not None and self.video_embeddings is not None:
            # video embeddings AND image embeddings
            self.all_embeddings = np.concatenate([self.image_embeddings, self.video_embeddings], axis=0)
            self.all_filenames_time_stamps = list(self.image_filenames) + list(self.video_timestamps)
            print("Loaded video AND image embeddings")
        elif self.image_embeddings is not None or self.video_embeddings is not None:
            # video embeddings OR image embeddings
            self.all_embeddings = self.image_embeddings if self.image_embeddings is not None else self.video_embeddings
            self.all_filenames_time_stamps = self.image_filenames if self.image_embeddings is not None else self.video_timestamps
            print("Loaded ONLY image embeddings") if self.image_embeddings is not None else print("Loaded ONLY video embeddings")
        else:
            # NO video embeddings AND image embeddings
            self.all_embeddings = None
            self.all_filenames_time_stamps = None

    def save(self):
        """Save image and video embeddings to file"""
        if self.image_embeddings is not None:
            np.save(self.embedding_file, self.image_embeddings)
            np.save(self.filename_file, self.image_filenames)
            print(f"Saved image embeddings") if self.print_progress else None
        if self.video_embeddings is not None:
            np.save(self.video_embedding_file, self.video_embeddings)
            np.save(self.video_timestamp_file, self.video_timestamps)
            print(f"Saved video embeddings") if self.print_progress else None

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
    
    def compute_embeddings(self, filepaths):
        """Computes image embeddings for images provided in filepaths"""
        print("Computing image embeddings...")
        image_embeddings = []
        image_filenames = []
        i = 0

        for filename in filepaths:
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

        # Check if images were found and corresponding embeddings were computed:
        if image_filenames:
            image_embeddings = np.vstack(image_embeddings)
            image_filenames = np.array(image_filenames)          
            print(f"Computed {i} image embeddings") if self.print_progress else None

        else:
            image_embeddings, image_filenames = None, None
            print(f"No images found") if self.print_progress else None

        return image_embeddings, image_filenames
    
    def compute_video_embeddings(self, filepaths):
        """Compute embeddings of keyframes from videos provided in filepaths"""
        print("Computing video embeddings...")
        video_embeddings = []
        video_codes = []

        for filename in filepaths:
            if filename.lower().endswith((".mp4")):
                video_path = os.path.join(self.image_video_folder, filename)
                video_features, video_timestamps = self.clip_video_embedder.get_video_embedding_and_timestamps(video_path)
                for i in range(len(video_features)):
                    video_embeddings.append(video_features[i])
                    video_codes.append(video_timestamps[i])

        # Check if videos were found and corresponding embeddings were computed:
        if video_codes:
            # video embeddings were computed
            video_embeddings = np.vstack(video_embeddings)
            video_codes = np.array(video_codes)
            print(f"Saved video embeddings") if self.print_progress else None

        else:
            # if no video embeddings were computed
            video_embeddings, video_codes = None, None
            print(f"No videos found") if self.print_progress else None

        return video_embeddings, video_codes

    def load_embeddings(self):
        print("Loading saved embeddings from file...")
        return np.load(self.embedding_file), np.load(self.filename_file)
    
    def load_video_embeddings(self):
        print("Loading saved embeddings from file...")
        return np.load(self.video_embedding_file), np.load(self.video_timestamp_file)

    def add_emebddings(self, new_img_emb, new_img_filenames, new_video_emb, new_video_timestamps):
        """Adds new embeddings and saves all embeddings to file"""
        if new_img_emb is not None: # added files contain at least one image
            if self.image_embeddings is not None: # image embeddings already exist --> append new embeddings
                self.image_embeddings = np.concatenate([self.image_embeddings, new_img_emb], axis=0)
                self.image_filenames = np.array(list(self.image_filenames) + list(new_img_filenames))
            else: # no image embeddings exist yet
                self.image_embeddings = new_img_emb
                self.image_filenames = new_img_filenames

        if new_video_emb is not None: # added files contain at least one video
            if self.video_embeddings is not None: # video embeddings alread exist
                self.video_embeddings = np.concatenate([self.video_embeddings, new_video_emb], axis = 0)
                self.video_timestamps = np.array(list(self.video_timestamps) + list(new_video_timestamps))  
            else: # no video embeddings exist yet
                self.video_embeddings = new_video_emb
                self.video_timestamps = new_video_timestamps
        
        self.save() # save embeddings to file

    def rm_embeddings(self, deleted_filepaths):
        """Removes embeddings of deleted files"""
        deleted_img_filepaths = [d for d in deleted_filepaths if d.lower().endswith((".jpg", ".jpeg", ".png"))]
        deleted_video_filepaths = [d for d in deleted_filepaths if d.lower().endswith((".mp4"))]

        if deleted_filepaths: # only delete if there are images to delete
        # image indices to be removed
            img_indices = [
                list(self.image_filenames).index(deleted_img) 
                for deleted_img in deleted_img_filepaths
            ]
            self.image_embeddings = np.delete(self.image_embeddings, img_indices, axis=0) # delete emb at index
            self.image_filenames = np.delete(self.image_filenames, img_indices) # delete filenames at index
        
        if deleted_video_filepaths:
            vid_indices = [
                i for i, timestamp in enumerate(list(self.video_timestamps))
                if any(timestamp.startswith(videopath + "_") for videopath in deleted_video_filepaths)
            ]
            self.video_embeddings = np.delete(self.video_embeddings, vid_indices, axis=0) # delete all keyframe emb for videos
            self.video_timestamps = np.delete(self.video_timestamps, vid_indices) # delete all timestamps for videos

        self.save() # save embeddings to file

    def find_top_matches(self, prompt):
        # check if items were added/removed to/from the gallery 
        added = set(self.get_all_filepaths()) - self.embedded_filepaths
        removed = self.embedded_filepaths - set(self.get_all_filepaths())
        
        if len(added): # Files were added --> compute new embeddings and add
            added_image_emb, added_img_filenames = self.compute_embeddings(list(added)) 
            added_video_emb, added_video_timestamps = self.compute_video_embeddings(list(added))
            self.add_emebddings(added_image_emb, added_img_filenames, added_video_emb, added_video_timestamps)
            self.embedded_filepaths = self.get_embedded_filepaths()
            
        if len(removed): # Files were deleted --> rm embeddings
            self.rm_embeddings(list(removed))
            self.embedded_filepaths = self.get_embedded_filepaths()

        self.concat_image_video_emb() # concat embeddings to allow search for images and videos
        
        # If there are no embeddings and return None to avoid crash
        if self.all_embeddings is None:
            return None, None

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