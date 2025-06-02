#!/bin/sh

QUERY="lorem ipsum"
DEVICE="cpu"
IMAGE_DIR="imgs/.../"
IMAGE_EMB_FILE="..."
KEY_EMB_FILE="..."

python img_embedder.py -image_dir "$IMAGE_DIR" --device "$DEVICE"
python keyw_embedder.py -query "$QUERY" --device "$DEVICE"
python matcher.py -img_emb "$IMAGE_EMB_FILE" -key_emb "$KEY_EMB_FILE"
