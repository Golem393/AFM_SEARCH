import torch
import clip
from PIL import Image
import os
import numpy as np
import shutil
import argparse

# ==== Argument Parser ====
parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default="Thailand/image", help='Folder with input images')
parser.add_argument('--prompt', type=str, default="monkey", help='Text prompt to search for')
parser.add_argument('--model', type=str, default="ViT-L/14@336px", help='CLIP model to use')
parser.add_argument('--top_k', type=int, default=10, help='Top K similar images to retrieve')
args = parser.parse_args()

# ==== Device Setup ====
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==== Model Validation ====
available_models = clip.available_models()
if args.model not in available_models:
    raise ValueError(f"Model '{args.model}' is not available. Choose from: {available_models}")

model, preprocess = clip.load(args.model, device=device)

# ==== File Naming ====
model_name_safe = args.model.replace("/", "_")
prompt_name_safe = args.prompt.replace(" ", "_")
base_folder = os.path.join(args.image_folder, "..")

embedding_file = os.path.join(base_folder, f"{model_name_safe}_embeddings.npy")
filename_file = os.path.join(base_folder, f"{model_name_safe}_filenames.npy")
output_folder = os.path.join(base_folder, f"{prompt_name_safe}_{model_name_safe}_top_matches")

# ==== Load or Compute Image Embeddings ====
if os.path.exists(embedding_file) and os.path.exists(filename_file):
    print("Loading cached embeddings...")
    image_embeddings = np.load(embedding_file)
    image_filenames = np.load(filename_file)
else:
    print("Computing image embeddings...")
    image_embeddings = []
    image_filenames = []

    for filename in os.listdir(args.image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(args.image_folder, filename)
            try:
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_feature = model.encode_image(image).float()
                    image_feature /= image_feature.norm(dim=-1, keepdim=True)
                    image_embeddings.append(image_feature.cpu().numpy())
                    image_filenames.append(filename)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    image_embeddings = np.vstack(image_embeddings)
    image_filenames = np.array(image_filenames)

    np.save(embedding_file, image_embeddings)
    np.save(filename_file, image_filenames)

# ==== Encode the Text ====
text_tokens = clip.tokenize([args.prompt]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().numpy()

# ==== Compute Similarities ====
similarities = image_embeddings @ text_features.T
similarities = similarities.squeeze()

# ==== Get Top-K Matches ====
top_indices = np.argsort(similarities)[::-1][:args.top_k]
print(f"\nTop {args.top_k} most similar images for: \"{args.prompt}\" using {args.model}\n")
for i, idx in enumerate(top_indices):
    print(f"{i+1:2d}. {image_filenames[idx]:30s} | Similarity: {similarities[idx]:.4f}")

# ==== Copy Top-K Images ====
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

for idx in top_indices:
    src = os.path.join(args.image_folder, image_filenames[idx])
    dst = os.path.join(output_folder, image_filenames[idx])
    shutil.copy(src, dst)

# ==== Save Confidence Metrics ====
score_file = os.path.join(output_folder, "scores.txt")
with open(score_file, "w") as f:
    for idx in top_indices:
        fname = image_filenames[idx]
        score = similarities[idx]
        f.write(f"{fname}\t{score:.4f}\n")

