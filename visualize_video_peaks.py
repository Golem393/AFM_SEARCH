import ffmpeg
import re
import matplotlib.pyplot as plt

def extract_scene_scores(video_path):
    try:
        out, err = (
            ffmpeg.input(video_path)
            .filter('showinfo')  # no select here
            .output('null', f='null')
            .global_args('-loglevel', 'debug')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        times = []
        scores = []

        for line in err.decode().split('\n'):
            # Match both time and scene score
            time_match = re.search(r'pts_time:(\d+\.\d+)', line)
            scene_match = re.search(r'scene:(\d+\.\d+)', line)
            if time_match and scene_match:
                times.append(float(time_match.group(1)))
                scores.append(float(scene_match.group(1)))

        return times, scores

    except ffmpeg.Error as e:
        print("Scene score extraction failed:", e.stderr.decode())
        return [], []

def plot_scene_scores(times, scores, threshold=0.005, output_path='scene_scores.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(times, scores, label='Scene Change Score', color='blue')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Time (s)')
    plt.ylabel('Scene Change Score')
    plt.title('Scene Change Scores Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Scene score plot saved to: {output_path}")

# === USAGE ===
video_path = 'Thailand/video/VID-20240830-WA0003.mp4'
times, scores = extract_scene_scores(video_path)
plot_scene_scores(times, scores, threshold=0.005)