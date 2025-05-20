#!/bin/bash

PROMPT="beach"
IMAGE_FOLDER="Thailand/image"
TOP_K=10

MODELS=("RN50" "RN101" "RN50x4" "RN50x16" "RN50x64" "ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-L/14@336px")

for MODEL in "${MODELS[@]}"; do
    echo "Running for model: $MODEL"
    python clip_matcher.py \
        --image_folder "$IMAGE_FOLDER" \
        --prompt "$PROMPT" \
        --model "$MODEL" \
        --top_k "$TOP_K"
done