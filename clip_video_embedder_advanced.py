import cv2
import numpy as np

def extract_frames_by_flow(video_path: str, target_frames: int = 16, target_size: tuple = (224, 224)) -> np.ndarray:
    """Extract frames using optical flow motion analysis"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # First pass: calculate flow energies for all frames
    prev_frame = None
    flow_energies = []
    all_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame to target size
        frame = cv2.resize(frame, target_size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_frames.append(frame)
        
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, gray, None, 
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
            flow_energies.append(np.mean(magnitude))
            
        prev_frame = gray
    
    cap.release()
    
    if not flow_energies:
        raise ValueError("Not enough frames to calculate optical flow")
    
    # Ensure we have flow energies for all frames (first frame gets zero)
    flow_energies.insert(0, 0)
    
    # Select key frames based on flow energy peaks
    selected_indices = select_peaks(flow_energies, target_frames)
    selected_frames = [all_frames[i] for i in selected_indices]
    
    # Convert to numpy array (T, H, W, C)
    return np.array(selected_frames)

def select_peaks(energies, num_frames):
    """Select frames with highest flow energy peaks"""
    # Simple peak detection
    peaks = []
    for i in range(1, len(energies)-1):
        if energies[i] > energies[i-1] and energies[i] > energies[i+1]:
            peaks.append((i, energies[i]))
    
    # Sort by energy and select top frames
    peaks.sort(key=lambda x: x[1], reverse=True)
    selected = [p[0] for p in peaks[:num_frames]]
    
    # If not enough peaks, add remaining from highest energies
    if len(selected) < num_frames:
        remaining = num_frames - len(selected)
        all_indices = np.argsort(energies)[-remaining:]
        selected.extend(all_indices)
    
    # Ensure we have exactly num_frames, sorted chronologically
    selected = sorted(list(set(selected)))[:num_frames]
    
    # Pad if needed (shouldn't happen with the above logic)
    while len(selected) < num_frames:
        selected.append(len(energies)-1)  # Add last frame
    
    return selected