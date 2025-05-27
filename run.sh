#!/bin/sh

QUERY="lorem ipsum"
DEVICE="cpu"
IMAGE_DIR="imgs/.../"

python img_embedder.py -image_dir "$IMAGE_DIR" --device "$DEVICE"
python keyw_embedder.py -query "$QUERY" --device "$DEVICE"
