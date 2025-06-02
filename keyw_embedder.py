"""Keyword Extractor and Embedder for Queries

This keyword embedder extracts important keywords from the original query
using Google's Gemma3 1B Parameter LLM with a system prompt. The extracted
keywords are then embedded using OpenAI's CLIP model.
Use:
    python keyw_embedder.py -query "lorem ipsum" --device "mps"

"""

from transformers import pipeline
import torch
import argparse
import clip
from utils import parse_keywords
import numpy as np

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('-query', type=str, help='Query for images you want to retrieve')
parser.add_argument('--device', type=str, default='cpu', help='Device on which the model should run')
args = parser.parse_args()

# Use HuggingFace transformer pipeline to feed prompt to Google Gemma 1B LLM
print("[INFO] Load Gemma model:...", end=' \r')
pipe = pipeline("text-generation", 
                model="google/gemma-3-1b-it", 
                device=args.device, 
                torch_dtype=torch.bfloat16)
print("[INFO] Load Gemma model: success")

# System propmpt for Gemma model. 
# TODO: To be optimized (max. 32K Token context)
system_prompt = "You extract important keywords or combinations of two keywords" \
" max from a given query for an image retrieval task. Return a comma separated list." \
"Under no circumstances add words to the list that are not given by the user!"

query = args.query

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": query},]
        },
    ],
]

output = pipe(messages, max_new_tokens=50)

# Extract output string
kw_string = output[0][0]['generated_text'][2]['content']
print(f"[INFO] Extracted keywords: {kw_string}")

# Load CLIP model and set to inference mode
print("[INFO] Load CLIP model:...", end=' \r')
clip_model, preprocess_clip = clip.load("ViT-L/14@336px", device=args.device)
clip_model.eval()
print("[INFO] Load CLIP model: success")

# Generate CLIP text embeddings
def generate_text_emb(query, device):
    """Generates text embeddings of query

    Args:
        query: str or list[str] keyword(s)
        device: str device to use
    """
    query = clip.tokenize(query)
    with torch.no_grad():
        query = query.to(device)
        emb = clip_model.encode_text(query)
        emb /= emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy()
        return emb

kw_list = parse_keywords(kw_string) # get list of keywords

query_emb = generate_text_emb(query=kw_list, device=args.device)

# Save keyword embeddings
print("[INFO] Save keyword embeddings:...", end='\r')
np.save("query_embeddings.npy", query_emb)
print("[INFO] Save keyword embeddings: success")