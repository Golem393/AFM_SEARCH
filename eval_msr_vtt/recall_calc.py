import json
import time
import os
from tqdm import tqdm
from pipeline2 import CLIPPaliGemmaPipeline

def load_dataset(json_path):
    with open(json_path) as f:
        content = f.read()
        return json.loads(content) if content.strip() else []


def load_existing_results(output_path):
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return []

def evaluate_recall(pipeline, dataset, k_values=[1, 5, 10], video_embedder_type = "uniform_average"):
    output_path=f"recall_results_{video_embedder_type}_with_gemma.json"
    insufficient_counts = {}
    zero_match_counts = {}
    summary_log_path = f"recall_edge_case_summary_{video_embedder_type}.json"
    if os.path.exists(summary_log_path):
        with open(summary_log_path) as f:
            previous_summary = json.load(f)
            insufficient_counts = previous_summary.get("insufficient_counts", {})
            zero_match_counts = previous_summary.get("zero_match_counts", {})
    recall = {k: 0 for k in k_values}
    total_queries = 0
    total_time = 0

    all_results = load_existing_results(output_path)
    already_evaluated_ids = {video["video_id"]: video for video in all_results}

    start_time = time.time()

    print("Using every video and every 3rd caption.")
    filtered_dataset = dataset[::8]

    for video_data in tqdm(filtered_dataset, desc="Evaluating"):
        video_id = video_data["video_id"]
        video_file = video_data["video"]

        if video_id in already_evaluated_ids:
            cached = already_evaluated_ids[video_id]
            for caption_result in cached.get("caption_recalls", []):
                hits = caption_result.get("hits", {})
                for k in k_values:
                    if hits.get(f"recall@{k}", 0):
                        recall[k] += 1
                total_queries += 1
            total_time += cached.get("average_caption_time_sec", 0) * len(cached.get("caption_recalls", []))
            continue

        captions = video_data["caption"][::12]  # Every 3rd caption

        video_metrics = {
            "video_id": video_id,
            "video_file": video_file,
            "caption_recalls": [],
            "average_caption_time_sec": 0.0
        }

        video_caption_times = []

        for caption in captions:
            caption_result = {
                "caption": caption,
                "hits": {}
            }
            try:
                caption_start = time.time()
                results = pipeline.run(caption)
                caption_end = time.time()

                time_taken = caption_end - caption_start
                video_caption_times.append(time_taken)

                top_matches = results["confirmed"]
                for k in k_values:
                    actual_k = min(k, len(top_matches))
                    if actual_k == 0:
                        # No candidates at all — can't compute recall
                        zero_match_counts[k] = zero_match_counts.get(k, 0) + 1
                        insufficient_counts[k] = insufficient_counts.get(k, 0) + 1
                        print(f"No top matches available for recall@{k}. Skipping metric.")
                        caption_result["hits"][f"recall@{k}"] = 0.0  # or 0.0 or "N/A"
                        continue
                    if len(top_matches) < k:
                        insufficient_counts[k] = insufficient_counts.get(k, 0) + 1
                        print(f"Only {len(top_matches)} matches available for recall@{k} (expected {k})")
                    hit = any(video_file in match for match in top_matches[:k])
                    if hit:
                        recall[k] += 1
                    caption_result["hits"][f"recall@{k}"] = int(hit) / actual_k

                total_queries += 1
                total_time += time_taken

            except Exception as e:
                caption_result["error"] = str(e)

            video_metrics["caption_recalls"].append(caption_result)

        if video_caption_times:
            video_metrics["average_caption_time_sec"] = sum(video_caption_times) / len(video_caption_times)

        all_results.append(video_metrics)

        # Save after each video to avoid data loss
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        summary_log = {
            "insufficient_counts": insufficient_counts,
            "zero_match_counts": zero_match_counts
        }
        with open(f"recall_edge_case_summary_{video_embedder_type}.json", "w") as f:
            json.dump(summary_log, f, indent=2)

    # Final average recall
    avg_recall = {f"recall@{k}": (recall[k] / total_queries) if total_queries else 0 for k in k_values}
    elapsed_time = time.time() - start_time
    avg_caption_time = (total_time / total_queries) if total_queries else 0

    summary = {
        "average_recall": avg_recall,
        "total_queries": total_queries,
        "elapsed_time_sec": elapsed_time,
        "average_caption_time_sec": avg_caption_time
    }

    summary_log = {
    "insufficient_counts": insufficient_counts,
    "zero_match_counts": zero_match_counts
    }

    with open(f"recall_edge_case_summary_{video_embedder_type}.json", "w") as f:
        json.dump(summary_log, f, indent=2)

    with open(f"recall_summary_{video_embedder_type}_with_gemma.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary

# Example usage
if __name__ == "__main__":
    dataset = load_dataset("eval_msr_vtt/msrvtt_train_7k.json")

    video_embedder_type = "keyframe_k_frames"#"clip_k_frames""  # or "keyframe_k_frames"

    pipeline = CLIPPaliGemmaPipeline(
        gallery="eval_msr_vtt/video/",
        top_k_clip_matches=20,
        video_embedder_type=video_embedder_type,
        frames_per_video_clip_max=20,
        port=5000
    )

    metrics = evaluate_recall(pipeline, dataset, video_embedder_type=video_embedder_type)
    print("Evaluation Results:")
    for metric, value in metrics["average_recall"].items():
        print(f"{metric}: {value:.4f}")
    print(f"Total queries: {metrics['total_queries']}")
    print(f"Elapsed time (sec): {metrics['elapsed_time_sec']:.2f}")
    print(f"Average caption time (sec): {metrics['average_caption_time_sec']:.4f}")
