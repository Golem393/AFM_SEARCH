import os
import torch
import numpy as np
import shutil
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from matching_algorithms import MeanMatcher, ParetoFrontMatcher

class GitMatcher:
    def __init__(self, image_folder, prompt, model = "microsoft/git-large", top_k=10):
        self.image_folder = image_folder
        self.prompt = prompt
        self.model_name = model
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

        # Prepare file and folder names
        self.model_name_safe = self.model_name.replace("/", "_")
        self.prompt_name_safe = self.prompt.replace(" ", "_")
        self.base_folder = os.path.join(self.image_folder, "..")

        self.embedding_file = os.path.join(self.base_folder, f"{self.model_name_safe}_embeddings.npy")
        self.filename_file = os.path.join(self.base_folder, f"{self.model_name_safe}_filenames.npy")
        self.output_folder = os.path.join(self.base_folder, f"{self.prompt_name_safe}_{self.model_name_safe}_top_matches")

    def compute_embeddings(self):
        print("Computing image captions...")
        image_embeddings = []
        image_filenames = []

        for filename in os.listdir(self.image_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(self.image_folder, filename)
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    generated_ids = self.model.generate(**inputs, max_length=50)
                    caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    image_embeddings.append(caption)
                    image_filenames.append(filename)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

        np.save(self.embedding_file, image_embeddings)
        np.save(self.filename_file, image_filenames)
        return image_embeddings, image_filenames

    def load_embeddings(self):
        print("Loading cached captions...")
        return np.load(self.embedding_file, allow_pickle=True), np.load(self.filename_file, allow_pickle=True)

    def get_text_features(self):
        return self.prompt

    def find_top_matches(self):
        # Load or compute embeddings
        if os.path.exists(self.embedding_file) and os.path.exists(self.filename_file):
            image_embeddings, image_filenames = self.load_embeddings()
        else:
            image_embeddings, image_filenames = self.compute_embeddings()

        # Compute similarities
        similarities = []
        for caption in image_embeddings:
            similarity = self.compute_similarity(self.prompt, caption)
            similarities.append(similarity)

        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        selected_imgs = [image_filenames[i] for i in top_indices]
        selected_scores = [similarities[i] for i in top_indices]

        # Print results
        print(f"\nTop {self.top_k} most similar images for: \"{self.prompt}\" using {self.model_name}\n")
        for i, (img, score) in enumerate(zip(selected_imgs, selected_scores)):
            print(f"{i+1:2d}. {img:30s} | Similarity: {score:.4f}")

        # Create output folder
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

        # Copy top images
        for img in selected_imgs:
            src = os.path.join(self.image_folder, img)
            dst = os.path.join(self.output_folder, img)
            shutil.copy(src, dst)

        # Save scores
        score_file = os.path.join(self.output_folder, "scores.txt")
        with open(score_file, "w") as f:
            for img, score in zip(selected_imgs, selected_scores):
                f.write(f"{img}\t{score:.4f}\n")

        return selected_imgs, selected_scores

    def compute_similarity(self, text1, text2):
        # Simple similarity based on common words
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        return len(set1 & set2) / len(set1 | set2)
