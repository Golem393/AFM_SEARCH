import decord  # Efficient video reader
import ffmpeg  # For keyframe extraction
import numpy as np
from PIL import Image

from typing import List
import base64
from io import BytesIO
from PIL import Image
import requests
import io
import torch
import cv2
import subprocess
import re

def get_num_frames(video_path, max_frames):
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / fps
        num_frames = max(1, min(max_frames, int(duration / 5)))  # 1 frame every ~5s
        return num_frames

def extract_at_timestamps(video_path: str, timestamps: List[float]) -> List[Image.Image]:
        """Helper to extract frames at specific timestamps"""
        frames = []
        for ts in timestamps:
            out, _ = (
                ffmpeg.input(video_path, ss=ts)
                .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            frames.append(Image.open(io.BytesIO(out)))
        return frames

def embed_frames(server_url, frames: List[Image.Image]) -> torch.Tensor:
    """Generate CLIP embeddings for a list of frames"""
    with torch.no_grad():
        images = [pil_to_base64(frame) for frame in frames]
        # Send to server
        response = requests.post(
            f'{server_url}/clip/embed_images',
            json={"images": images}
        )
        frame_embeddings = np.array(response.json()['result'])
    return frame_embeddings

def pil_to_base64(img, format='JPEG'):
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    return base64.b64encode(img_bytes).decode('utf-8')

def compute_optical_flow(prev, next, scale=0.25):
    prev_small = cv2.resize(prev, (0, 0), fx=scale, fy=scale)
    next_small = cv2.resize(next, (0, 0), fx=scale, fy=scale)

    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(next_small, cv2.COLOR_BGR2GRAY),
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Resize flow back to original size if needed
    flow = cv2.resize(flow, (prev.shape[1], prev.shape[0]))
    flow *= 1.0 / scale  # Adjust vector magnitude
    return flow

def detect_scene_changes_direct(video_path: str, threshold: float = 0.05):
    """Detect scene changes with their scores using ffmpeg metadata print filter."""
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-filter:v', 'select=gt(scene\\,0),metadata=print',
            '-an', '-f', 'null', '-'
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()

        scene_changes = []
        scene_thresholds = []
        lines = output.splitlines()
        for i in range(len(lines) - 1):
            if "pts_time" in lines[i] and "lavfi.scene_score" in lines[i + 1]:
                time_match = re.search(r'pts_time:(\d+(\.\d+)?)', lines[i])
                score_match = re.search(r'scene_score=(\d+(\.\d+)?)', lines[i + 1])
                if time_match and score_match:
                    time = float(time_match.group(1))
                    score = float(score_match.group(1))
                    if score > threshold:
                        scene_changes.append(time)
                        scene_thresholds.append(score)

        return scene_changes, scene_thresholds

    except subprocess.CalledProcessError as e:
        print(f"Scene detection failed: {e.output.decode()}")
        return []

def select_keyframes_hybrid(timestamps, scores, duration, max_k=None,
                        score_thresh=0.2, std_thresh=0.03, min_k=1):
    # Fallback to uniform sampling if no keyframes at all
    if not timestamps or not scores or len(timestamps) == 0 or len(scores) == 0:
        fallback_k = max(min_k, int(duration / 15))  # one every ~15 seconds
        return np.linspace(0, duration, fallback_k + 2)[1:-1].tolist()

    timestamps = np.array(timestamps)
    scores = np.array(scores)

    # Normalize scores
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # Compute base number of keyframes depending on duration
    base_k = int(duration / 5)  # 1 keyframe every ~15s
    base_k = max(base_k, min_k)

    # Influence from score quality and number of keyframes
    quality_boost = int(np.clip(norm_scores.mean() * len(timestamps), 0, 5))
    diversity_bonus = int(np.clip(np.std(timestamps) / duration * 10, 0, 5))

    # Final k (adaptive)
    k = base_k + quality_boost + diversity_bonus
    if max_k is not None:
        k = min(k, max_k)
    k = min(k, len(timestamps))  # don't exceed available keyframes
    k = max(k, min_k)

    # If scores are too uniformly bad → fallback to uniform sampling
    if norm_scores.max() < score_thresh or norm_scores.std() < std_thresh:
        return np.linspace(0, duration, k + 2)[1:-1].tolist()

    # Score-based greedy selection with diversity
    selected = []
    while len(selected) < k:
        best_idx = -1
        best_score = -np.inf

        for i, t in enumerate(timestamps):
            if i in selected:
                continue
            score = norm_scores[i]

            # Diversity penalty
            penalty = 0
            if selected:
                dists = np.abs(t - timestamps[selected])
                min_dist = dists.min()
                penalty = 1 - (min_dist / duration)  # closer = more penalty

            combined = score - 0.6 * penalty  # increase weight for more diversity
            if combined > best_score:
                best_score = combined
                best_idx = i

        if best_idx == -1:
            break
        selected.append(best_idx)

    return sorted(timestamps[i] for i in selected)