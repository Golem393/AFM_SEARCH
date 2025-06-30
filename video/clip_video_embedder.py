import torch

import os
from video.key_frame_extractors import (
    extract_keyframes_optical_flow, extract_keyframes_ffmpeg, 
    extract_uniform_frames, extract_keyframes_optical_flow,
    extract_keyframes_clip
)

from video.video_helper import (
    get_num_frames, extract_at_timestamps, embed_frames
)

class CLIPVideoEmbedder:
    def __init__(self, video_embedder_type, frames_per_video_clip_max, server_url):
        self.video_embedder_type = video_embedder_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server_url = server_url
        self.frames_per_video_clip_max = frames_per_video_clip_max

    def get_video_embedding_and_timestamps(self, video_path):
        video_name = os.path.basename(video_path)

        num_frames = get_num_frames(video_path, self.frames_per_video_clip_max)
        frames = None

        if self.video_embedder_type == "uniform_average":
            main_keyframe_time, _ = extract_uniform_frames(video_path, num_frames=num_frames)

        elif self.video_embedder_type == "keyframe_average":
            main_keyframe_time, _ = extract_keyframes_ffmpeg(video_path, max_frames=self.frames_per_video_clip_max)

        elif self.video_embedder_type == "uniform_k_frames":
            _, timestamps = extract_uniform_frames(video_path, num_frames=num_frames)
        
        elif self.video_embedder_type == "keyframe_k_frames":
            _, timestamps = extract_keyframes_ffmpeg(video_path, max_frames=self.frames_per_video_clip_max)

        elif self.video_embedder_type == "optical_flow":
            timestamps, frames = extract_keyframes_optical_flow(video_path, num_keyframes=num_frames)

        elif self.video_embedder_type == "clip_k_frames":
            timestamps, frames = extract_keyframes_clip(video_path, self.server_url, num_keyframes=num_frames)

        else:
            raise ValueError(f"Unknown type: {self.video_embedder_type}")
        
        if "average" in self.video_embedder_type:
            frames = extract_at_timestamps(video_path, [main_keyframe_time])
            embeddings = embed_frames(self.server_url, frames)
            video_key_frame_code = video_name + "_" + str(main_keyframe_time)
            return [embeddings.mean(axis=0)], [video_key_frame_code]
        else:
            print(len(timestamps), "Selected timestamps")
            if frames is None:
                frames = extract_at_timestamps(video_path, timestamps)
            embeddings = embed_frames(self.server_url, frames)
            video_key_frame_codes = []
            for i, _ in enumerate(timestamps):
                video_key_frame_codes.append(video_name + "_" + str(timestamps[i]))
            return embeddings, video_key_frame_codes

    

