import cv2
import numpy as np
import decord
import ffmpeg
from PIL import Image
from typing import List
import requests
from video.video_helper import (
    detect_scene_changes_direct,
    compute_optical_flow, select_keyframes_hybrid,
    extract_at_timestamps, embed_frames
)

def extract_uniform_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    """Extract uniformly spaced frames using decord, based on video duration"""
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    if num_frames == 1:
        frame_indices = [total_frames // 2]
    else:
        # Avoid starting exactly at 0. Space samples at segment centers
        segment_length = total_frames / num_frames
        frame_indices = [int((i + 0.5) * segment_length) for i in range(num_frames)]
        frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]  # clamp

    timestamps = [idx / fps for idx in frame_indices]

    # Middle timestamp (best keyframe)
    main_keyframe_time = timestamps[len(timestamps) // 2]

    return main_keyframe_time, timestamps

def extract_keyframes_ffmpeg(video_path: str, max_frames: int = 16, num_frames = None) -> List[Image.Image]:
    """Intelligently extracts keyframes up to max_frames based on content complexity"""
    try:
        # Get video metadata
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        
        # Detect all scene changes
        scene_changes, scene_thresholds = detect_scene_changes_direct(video_path)
        if len(scene_changes) != 0:
            best_idx = int(np.argmax(scene_thresholds))
            main_keyframe_time = scene_changes[best_idx]
        else:
            main_keyframe_time = duration / 2.0

        if num_frames is not None:
            selected_timestamps = select_keyframes_hybrid(scene_changes, scene_thresholds, duration, max_k = num_frames, min_k= num_frames)
        else:
            selected_timestamps = select_keyframes_hybrid(scene_changes, scene_thresholds, duration, max_k = max_frames)


        return main_keyframe_time, selected_timestamps

    except Exception as e:
        print(f"Keyframe extraction failed: {e}")
        return extract_uniform_frames(video_path, min(3, max_frames))

def extract_keyframes_optical_flow(video_path, num_keyframes=10):
    cap = cv2.VideoCapture(video_path)
    motion_scores = []

    ret, prev_frame = cap.read()
    if not ret:
        print("Could not read video.")
        return [], []

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        flow = compute_optical_flow(prev_frame, curr_frame)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = np.mean(mag)
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        # Store motion score, timestamp, and current frame
        motion_scores.append((motion_score, timestamp, Image.fromarray(curr_frame)))
        prev_frame = curr_frame

    cap.release()

    # Sort by motion score descending and take top N entries
    motion_scores.sort(reverse=True, key=lambda x: x[0])
    top_keyframes = motion_scores[:num_keyframes]

    # Sort by timestamp to preserve chronological order
    top_keyframes.sort(key=lambda x: x[1])

    keyframe_times = [t for _, t, _ in top_keyframes]
    keyframe_frames = [f for _, _, f in top_keyframes]

    return keyframe_times, keyframe_frames

def extract_keyframes_clip(video_path: str, server_url, num_keyframes):
    num_uniform_frames = num_keyframes * 4
    _, time_stamps = extract_uniform_frames(video_path, num_uniform_frames)

    frames = extract_at_timestamps(video_path, time_stamps)
    embeddings = embed_frames(server_url, frames)

    # Compute cosine similarity matrix
    sim_matrix = embeddings @ embeddings.T

    # Greedy frame selection: pick most dissimilar frames
    selected = [0]
    for _ in range(1, num_keyframes):
        remaining = list(set(range(num_uniform_frames)) - set(selected))
        min_sim = [(i, sim_matrix[i][selected].mean().item()) for i in remaining]
        next_frame = sorted(min_sim, key=lambda x: x[1])[0][0]
        selected.append(next_frame)

    selected_frames = [frames[i] for i in selected]
    selected_timestamps = [time_stamps[i] for i in selected]

    return selected_timestamps, selected_frames

def extract_keyframes_clip_improved(video_path: str, server_url, num_keyframes):
    num_uniform_frames = num_keyframes * 4
    _, time_stamps = extract_uniform_frames(video_path, num_uniform_frames)
    frames = extract_at_timestamps(video_path, time_stamps)

    filtered_frames_data = [] # Store (original_index, frame, timestamp)
    for i, (frame, timestamp) in enumerate(zip(frames, time_stamps)):
        # Convert PIL Image to OpenCV format for processing
        cv_frame = np.array(frame)
        cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_RGB2GRAY)

        # 1. Check for pure black/white frames (simple check)
        mean_intensity = cv_frame.mean()
        if mean_intensity < 10 or mean_intensity > 245: # Example thresholds
            continue # Skip very dark or very bright uniform frames

        # 2. Check for blurriness/lack of detail (Laplacian Variance)
        laplacian_var = cv2.Laplacian(cv_frame, cv2.CV_64F).var()
        if laplacian_var < 50: # Example threshold, adjust as needed
            continue # Skip blurry or low-detail frames

        # If it passes the checks, add it to our filtered list
        filtered_frames_data.append((i, frame, timestamp))

    if not filtered_frames_data:
        # Fallback if all frames are filtered out (unlikely for most videos)
        print("Warning: All uniform frames filtered out. Falling back to uniform extraction.")
        _, time_stamps = extract_uniform_frames(video_path, num_keyframes)
        frames = extract_at_timestamps(video_path, time_stamps)
        return time_stamps, frames


    # Now, process the filtered frames
    filtered_frames = [data[1] for data in filtered_frames_data]
    filtered_original_indices = [data[0] for data in filtered_frames_data]
    filtered_timestamps = [data[2] for data in filtered_frames_data]

    if not filtered_frames: # Should be caught by the above check, but for safety
        return [], [] # No valid frames found

    embeddings = embed_frames(server_url, filtered_frames)

    # Compute cosine similarity matrix
    sim_matrix = embeddings @ embeddings.T

    # Greedy frame selection: pick most dissimilar frames from the FILTERED set
    selected_filtered_indices = []
    if len(filtered_frames) > 0:
        selected_filtered_indices.append(0) # Start with the first valid frame
    
    for _ in range(1, min(num_keyframes, len(filtered_frames))):
        remaining = list(set(range(len(filtered_frames))) - set(selected_filtered_indices))
        if not remaining:
            break
        
        min_sim = [(i, sim_matrix[i][selected_filtered_indices].mean().item()) for i in remaining]
        next_frame_in_filtered_set = sorted(min_sim, key=lambda x: x[1])[0][0]
        selected_filtered_indices.append(next_frame_in_filtered_set)

    # Map back to original uniform_timestamps and uniform_frames
    final_selected_timestamps = [filtered_timestamps[i] for i in selected_filtered_indices]
    final_selected_frames = [filtered_frames[i] for i in selected_filtered_indices]
    
    # Sort by timestamp for chronological order
    combined = sorted(zip(final_selected_timestamps, final_selected_frames), key=lambda x: x[0])
    final_selected_timestamps = [t for t, _ in combined]
    final_selected_frames = [f for _, f in combined]

    return final_selected_timestamps, final_selected_frames
