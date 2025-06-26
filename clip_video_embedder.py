import torch
import numpy as np
from PIL import Image
import decord  # Efficient video reader
import ffmpeg  # For keyframe extraction
from typing import List
import base64
from io import BytesIO
from PIL import Image
import requests
import subprocess
import json
import io
import os
import shutil
import re

class CLIPVideoEmbedder:
    def __init__(self, video_embedder_type, frames_per_video_clip_max):
        self.video_embedder_type = video_embedder_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server_url = "http://localhost:5000"
        self.frames_per_video_clip_max = frames_per_video_clip_max

    def get_video_embedding_and_timestamps(self, video_path):
        video_name = os.path.basename(video_path)
        
        if self.video_embedder_type == "uniform_average":
            frames, main_keyframe_time, _ = self.extract_uniform_frames(video_path, max_frames=self.frames_per_video_clip_max)
            embeddings = self.embed_frames(frames)
            video_key_frame_code = video_name + "_" + str(main_keyframe_time)
            return [embeddings.mean(axis=0)], [video_key_frame_code]

        elif self.video_embedder_type == "keyframe_average":
            frames, main_keyframe_time, _ = self.extract_keyframes_ffmpeg(video_path, max_frames=self.frames_per_video_clip_max)
            embeddings = self.embed_frames(frames)
            video_key_frame_code = video_name + "_" + str(main_keyframe_time)
            return [embeddings.mean(axis=0)], [video_key_frame_code]

        elif self.video_embedder_type == "uniform_k_frames":
            frames, _, timestamps = self.extract_uniform_frames(video_path, max_frames=self.frames_per_video_clip_max)
            embeddings = self.embed_frames(frames)
            video_key_frame_codes = []
            for i, _ in enumerate(timestamps):
                video_key_frame_codes.append(video_name + "_" + str(timestamps[i]))
            return embeddings, video_key_frame_codes
        
        elif self.video_embedder_type == "keyframe_k_frames":
            frames, _, timestamps = self.extract_keyframes_ffmpeg(video_path, max_frames=self.frames_per_video_clip_max)
            embeddings = self.embed_frames(frames)
            video_key_frame_codes = []
            for i, _ in enumerate(timestamps):
                video_key_frame_codes.append(video_name + "_" + str(timestamps[i]))
            return embeddings, video_key_frame_codes

        #elif self.video_embedder_type == "optical_flow":
            
        
        else:
            raise ValueError(f"Unknown type: {self.video_embedder_type}")
    
    def extract_uniform_frames(self, video_path: str, max_frames: int = 16) -> List[Image.Image]:
        """Extract uniformly spaced frames using decord, based on video duration"""
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / fps

        # Choose number of frames to extract (every 7 seconds or max_frames)
        num_frames = max(1, min(max_frames, int(duration / 7)))  # 1 frame every ~7s

        if num_frames == 1:
            frame_indices = [total_frames // 2]
        else:
            # Avoid starting exactly at 0. Space samples at segment centers
            segment_length = total_frames / num_frames
            frame_indices = [int((i + 0.5) * segment_length) for i in range(num_frames)]
            frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]  # clamp

        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in frame_indices]
        timestamps = [idx / fps for idx in frame_indices]

        # Middle timestamp (best keyframe)
        main_keyframe_time = timestamps[len(timestamps) // 2]

        return frames, main_keyframe_time, timestamps


    def _detect_scene_changes_direct(self, video_path: str, threshold: float = 0.05):
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

    def select_keyframes_hybrid(self, timestamps, scores, duration, max_k=None,
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

    def extract_keyframes_ffmpeg(self, video_path: str, max_frames: int = 16) -> List[Image.Image]:
        """Intelligently extracts keyframes up to max_frames based on content complexity"""
        try:
            # Get video metadata
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            #print("Video", video_path)
            #print("Duration", duration)
            
            # Detect all scene changes
            scene_changes, scene_thresholds = self._detect_scene_changes_direct(video_path)
            if len(scene_changes) != 0:
                best_idx = int(np.argmax(scene_thresholds))
                main_keyframe_time = scene_changes[best_idx]
            else:
                main_keyframe_time = duration / 2.0
           # print(len(scene_changes))
            #print(scene_thresholds, "Thresholds")

            selected_timestamps = self.select_keyframes_hybrid(scene_changes, scene_thresholds, duration, max_frames)
            print(len(selected_timestamps), "Selected timestamps")


            return self._extract_at_timestamps(video_path, selected_timestamps), main_keyframe_time, selected_timestamps

        except Exception as e:
            print(f"Keyframe extraction failed: {e}")
            return self.extract_uniform_frames(video_path, min(3, max_frames))

    def _extract_at_timestamps(self, video_path: str, timestamps: List[float]) -> List[Image.Image]:
        """Helper to extract frames at specific timestamps"""
        frames = []
        for ts in timestamps:
            out, _ = (
                ffmpeg.input(video_path, ss=ts)
                .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
                .run(capture_stdout=True, quiet=True)
            )
            frames.append(Image.open(io.BytesIO(out)))
        return frames

    def embed_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        """Generate CLIP embeddings for a list of frames"""
        with torch.no_grad():
            images = [self.pil_to_base64(frame) for frame in frames]
            # Send to server
            response = requests.post(
                f'{self.server_url}/clip/embed_images',
                json={"images": images}
            )
            frame_embeddings = np.array(response.json()['result'])
        return frame_embeddings
    
    def pil_to_base64(self, img, format='JPEG'):
        buffer = BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        img_bytes = buffer.read()
        return base64.b64encode(img_bytes).decode('utf-8')

