import json
import time
import os
import numpy as np
import pickle # To load the saved embeddings
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers import util # For cosine similarity
from pipeline2 import CLIPPaliGemmaPipeline # Assuming this is your retrieval pipeline
import torch

# --- Load Embeddings Global Setup ---
# You need to define the path to your saved embeddings file
EMBEDDINGS_FILE_PATH = 'eval_msr_vtt/embed/msrvtt_caption_embeddings.pkl'
all_caption_embeddings_data = {} # Will store video_id -> numpy array of its caption embeddings
sentence_embedding_model = None # Will load the model used for embedding captions

def load_all_caption_embeddings():
    """Loads all pre-computed caption embeddings into memory."""
    global all_caption_embeddings_data
    global sentence_embedding_model # We need this model to embed the *query*
                                   # unless the query is directly from the dataset.

    if not os.path.exists(EMBEDDINGS_FILE_PATH):
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE_PATH}. "
                                "Please run the embedding script first.")
    try:
        with open(EMBEDDINGS_FILE_PATH, 'rb') as f:
            loaded_data = pickle.load(f)
        for entry in loaded_data:
            all_caption_embeddings_data[entry['video_id']] = entry['captions_embeddings']
        print(f"Loaded embeddings for {len(all_caption_embeddings_data)} videos.")

        # Also load the SentenceTransformer model used for embedding
        # This is crucial for embedding the *query* caption during evaluation
        # Make sure the model name here matches the one used in the embedding script
        model_name_for_query_embedding = 'all-MiniLM-L6-v2' # MUST MATCH EMBEDDING SCRIPT
        print(f"Loading SentenceTransformer model for query embedding: {model_name_for_query_embedding}...")
        sentence_embedding_model = SentenceTransformer(model_name_for_query_embedding)
        print("SentenceTransformer model for query embedding loaded.")

    except Exception as e:
        print(f"Error loading embeddings or SentenceTransformer model: {e}")
        # Consider exiting or handling gracefully if embeddings are crucial
        exit()

def get_video_caption_embeddings(video_id):
    """Retrieves all caption embeddings for a given video_id."""
    return all_caption_embeddings_data.get(f'{video_id}', None)

# --- Original Functions (slightly modified or reused) ---
def load_dataset(json_path):
    with open(json_path) as f:
        content = f.read()
        return json.loads(content) if content.strip() else []

def load_existing_results(output_path):
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {output_path}. Starting fresh.")
            return []
    return []

# --- Modified Evaluation Function ---
def evaluate_retrieval(pipeline, dataset, k_values=[1, 5, 10], video_embedder_type="uniform_average", similarity_threshold=0.75):
    """
    Evaluates recall and weighted precision using semantic similarity.

    Args:
        pipeline (CLIPPaliGemmaPipeline): Your video retrieval pipeline.
        dataset (list): Loaded MSR-VTT dataset.
        k_values (list): List of k values for evaluation (e.g., [1, 5, 10]).
        video_embedder_type (str): Type of video embedder used in the pipeline.
        similarity_threshold (float): Minimum cosine similarity to consider a retrieved
                                      video's caption as "relevant" to the query, if it's
                                      not the exact ground truth video. Values range from 0 to 1.
    """
    output_path = f"retrieval_results_{video_embedder_type}_with_gemma.json"
    summary_log_path = f"retrieval_edge_case_summary_{video_embedder_type}.json"
    similarity_bins = {f"{i*0.05:.2f}-{(i+1)*0.05:.2f}": 0 for i in range(21)} # Bins from 0.00-0.05 up to 1.00-1.05
    
    # Initialize metrics
    recall = {k: 0 for k in k_values}
    weighted_precision = {k: 0.0 for k in k_values}
    total_queries = 0
    total_time = 0
    insufficient_counts = {k: 0 for k in k_values}
    zero_match_counts = {k: 0 for k in k_values}

    # Load existing results for resuming
    all_results = load_existing_results(output_path)
    # Reconstruct already_evaluated_ids to map video_id -> dictionary of its *last* caption_recall
    # This assumes that if a video_id is in all_results, all its captions have been processed
    # If you only save partially processed videos, this needs adjustment.
    already_evaluated_video_ids = set()
    for video_entry in all_results:
        already_evaluated_video_ids.add(video_entry.get("video_id"))


    # Load edge case summaries
    if os.path.exists(summary_log_path):
        with open(summary_log_path) as f:
            previous_summary = json.load(f)
            insufficient_counts = previous_summary.get("insufficient_counts", {})
            zero_match_counts = previous_summary.get("zero_match_counts", {})

    start_global_time = time.time()

    print("Using every 8th video and every 12th caption for evaluation.")
    filtered_dataset = dataset[::8] # Process every 8th video

    # Ensure embeddings are loaded before starting evaluation
    load_all_caption_embeddings()

    for video_data in tqdm(filtered_dataset, desc="Evaluating"):
        video_id = video_data["video_id"]
        # The true video file name, e.g., "video204.mp4"
        true_video_filename = video_data["video"]

        """if video_id in already_evaluated_video_ids:
            # If the video_id was already processed and saved, we need to re-aggregate its metrics
            # from `all_results` to update overall recall/precision.
            # This part needs careful handling to correctly sum up metrics from cached results.
            # For simplicity, if a video_id is in `already_evaluated_video_ids`, we skip re-running
            # the pipeline, but we must *add its previously computed metrics* to `recall` and `weighted_precision`.
            # A more robust way might be to store the overall `recall` and `weighted_precision` counters
            # directly in the summary file and just load them.
            # For now, let's assume `all_results` contains the *final* computed metrics for that video_id.
            # We'll iterate through `all_results` once at the end to sum up.
            continue # Skip processing if already evaluated"""

        captions_for_video = video_data["caption"][::12] # Every 12th caption for this video

        video_metrics_entry = {
            "video_id": video_id,
            "true_video_filename": true_video_filename,
            "caption_evaluations": [], # Renamed to be more general
            "average_caption_processing_time_sec": 0.0
        }

        current_video_caption_times = []

        for query_caption_text in captions_for_video:
            caption_eval_result = {
                "query_caption": query_caption_text,
                "recall_hits": {},
                "weighted_precision_scores": {}
            }
            #try:
            caption_start = time.time()
            # Run your retrieval pipeline
            results = pipeline.run(query_caption_text)
            caption_end = time.time()
            time_taken = caption_end - caption_start
            current_video_caption_times.append(time_taken)

            retrieved_matches = results["confirmed"] # List of retrieved video filenames or IDs

            # Get embedding for the query caption
            query_embedding = sentence_embedding_model.encode(query_caption_text, convert_to_numpy=True)
            # Reshape to (1, embedding_dim) for cosine_similarity
            query_embedding = query_embedding.reshape(1, -1)

            for k in k_values:
                actual_k = min(k, len(retrieved_matches))
                
                if actual_k == 0:
                    zero_match_counts[k] = zero_match_counts.get(k, 0) + 1
                    insufficient_counts[k] = insufficient_counts.get(k, 0) + 1
                    print(f"[{video_id}] No top matches for recall/precision@{k}. Skipping.")
                    caption_eval_result["recall_hits"][f"recall@{k}"] = 0
                    caption_eval_result["weighted_precision_scores"][f"precision@{k}"] = 0.0
                    continue

                if len(retrieved_matches) < k:
                    insufficient_counts[k] = insufficient_counts.get(k, 0) + 1
                    # print(f"[{video_id}] Only {len(retrieved_matches)} matches for @{k}.")

                # --- Recall Calculation (still binary hit for the true video) ---
                # Check if the *true* video is in the top K retrieved
                true_video_in_top_k = any(true_video_filename in match for match in retrieved_matches[:k])
                if true_video_in_top_k:
                    recall[k] += 1 # Count as a hit for overall recall

                caption_eval_result["recall_hits"][f"recall@{k}"] = 1 if true_video_in_top_k else 0


                # --- Weighted Precision Calculation ---
                current_k_precision_score = 0.0
                for i, retrieved_video_path in enumerate(retrieved_matches[:k]):
                    # Extract video_id from retrieved_video_path (assuming 'videoXXX.mp4' format)
                    retrieved_video_id = os.path.splitext(os.path.basename(retrieved_video_path))[0]
                    retrieved_video_id = retrieved_video_id.split(".mp4")[0]
                    if retrieved_video_id == video_id:
                        # This is the ground truth video
                        relevance_weight = 1.0
                    else:
                        # It's not the ground truth video, so calculate semantic similarity
                        retrieved_video_caption_embeddings = get_video_caption_embeddings(retrieved_video_id)
                        if retrieved_video_caption_embeddings is not None:
                            # Calculate cosine similarity between query embedding and all captions of the retrieved video
                            # Take the maximum similarity as the relevance score
                            similarities = util.cos_sim(query_embedding, retrieved_video_caption_embeddings)
                            max_similarity = torch.max(similarities).item() # .item() to get Python float
                            bin_index = min(int(max_similarity / 0.05), 20) # Ensure index doesn't go out of bounds for 1.00
                            bin_key = f"{bin_index*0.05:.2f}-{(bin_index+1)*0.05:.2f}"
                            similarity_bins[bin_key] += 1
                            relevance_weight = max_similarity if max_similarity >= similarity_threshold else 0.0
                        else:
                            # No embeddings found for the retrieved video, treat as not relevant
                            relevance_weight = 0.0

                    current_k_precision_score += relevance_weight

                # Weighted Precision@K for this query
                if actual_k > 0:
                    caption_eval_result["weighted_precision_scores"][f"precision@{k}"] = current_k_precision_score / actual_k
                else:
                    caption_eval_result["weighted_precision_scores"][f"precision@{k}"] = 0.0

            total_queries += 1
            total_time += time_taken

            #except Exception as e:
                #caption_eval_result["error"] = str(e)
                #print(f"Error processing {query_caption_text} for video {video_id}: {e}")

            video_metrics_entry["caption_evaluations"].append(caption_eval_result)

        if current_video_caption_times:
            video_metrics_entry["average_caption_processing_time_sec"] = sum(current_video_caption_times) / len(current_video_caption_times)

        all_results.append(video_metrics_entry)

        # Save after each video to avoid data loss
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        summary_log = {
            "insufficient_counts": insufficient_counts,
            "zero_match_counts": zero_match_counts
        }
        with open(summary_log_path, "w") as f:
            json.dump(summary_log, f, indent=2)

        similarity_distribution_path = f"similarity_distribution_{video_embedder_type}.json"
        with open(similarity_distribution_path, "w") as f:
            json.dump(similarity_bins, f, indent=2)
        print(f"\nSimilarity distribution saved to: {similarity_distribution_path}")

    # --- Aggregate final metrics from all_results ---
    final_recall = {k: 0 for k in k_values}
    final_weighted_precision = {k: 0.0 for k in k_values}
    final_total_queries = 0
    final_total_time = 0.0

    for video_entry in all_results:
        for caption_eval in video_entry["caption_evaluations"]:
            final_total_queries += 1
            final_total_time += video_entry["average_caption_processing_time_sec"] # This assumes the time is per video per caption

            for k in k_values:
                # Aggregate recall
                if caption_eval["recall_hits"].get(f"recall@{k}", 0) == 1:
                    final_recall[k] += 1
                # Aggregate weighted precision
                final_weighted_precision[k] += caption_eval["weighted_precision_scores"].get(f"precision@{k}", 0.0)

    # Calculate final averages
    avg_recall = {f"recall@{k}": (final_recall[k] / final_total_queries) if final_total_queries else 0 for k in k_values}
    avg_weighted_precision = {f"weighted_precision@{k}": (final_weighted_precision[k] / final_total_queries) if final_total_queries else 0 for k in k_values}

    elapsed_global_time = time.time() - start_global_time
    avg_caption_time_overall = (final_total_time / final_total_queries) if final_total_queries else 0 # this avg is slightly off due to how time is logged.

    summary = {
        "average_recall": avg_recall,
        "average_weighted_precision": avg_weighted_precision,
        "total_queries_evaluated": final_total_queries,
        "elapsed_total_time_sec": elapsed_global_time,
        "average_pipeline_time_per_query_sec": avg_caption_time_overall
    }

    # Save summary
    with open(f"retrieval_summary_{video_embedder_type}_with_gemma.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary

# Example usage
if __name__ == "__main__":
    # Make sure your MSR-VTT JSON path is correct
    dataset_path = "eval_msr_vtt/msrvtt_train_7k.json"
    dataset = load_dataset(dataset_path)

    video_embedder_type = "uniform_average" # or "uniform_average", "clip_k_frames"

    # Initialize your pipeline. Ensure the 'gallery' path is correct.
    pipeline = CLIPPaliGemmaPipeline(
        gallery="eval_msr_vtt/video/", # Path to your video files
        top_k_clip_matches=20, # Max matches your pipeline returns for scoring
        video_embedder_type=video_embedder_type,
        frames_per_video_clip_max=20,
        port=5000
    )

    # Run the evaluation
    metrics = evaluate_retrieval(pipeline, dataset, video_embedder_type=video_embedder_type, similarity_threshold=0.70)
    print("\n--- Evaluation Results ---")
    print(f"Video Embedder Type: {video_embedder_type}")
    for metric_name, value_dict in metrics["average_recall"].items():
        print(f"{metric_name}: {value_dict:.4f}")
    for metric_name, value_dict in metrics["average_weighted_precision"].items():
        print(f"{metric_name}: {value_dict:.4f}")
    print(f"Total queries evaluated: {metrics['total_queries_evaluated']}")
    print(f"Total elapsed time (sec): {metrics['elapsed_total_time_sec']:.2f}")
    print(f"Average pipeline time per query (sec): {metrics['average_pipeline_time_per_query_sec']:.4f}")