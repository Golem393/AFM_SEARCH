import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

def create_and_save_embeddings(json_file_path, output_folder, model_name='all-MiniLM-L6-v2'):
    """
    Creates text embeddings for captions in a JSON file and saves them.

    Args:
        json_file_path (str): Path to your MSR-VTT JSON file.
        output_folder (str): Directory where the embeddings will be saved.
        model_name (str): Name of the sentence-transformer model to use.
                          'all-MiniLM-L6-v2' is a good balance of speed and performance.
                          Other options include 'all-mpnet-base-v2' (larger, better performance)
                          or 'distilbert-base-nli-mean-tokens'.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    print(f"Loading SentenceTransformer model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Model loaded.")

    print(f"Loading data from: {json_file_path}...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} video entries.")
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return

    all_embeddings_data = []

    for entry in data:
        video_id = entry.get('video_id')
        captions = entry.get('caption', [])

        if not video_id or not captions:
            print(f"Skipping entry due to missing video_id or captions: {entry}")
            continue

        print(f"Processing video_id: {video_id} with {len(captions)} captions...")

        # Create embeddings for all captions of the current video
        caption_embeddings = model.encode(captions, convert_to_numpy=True, show_progress_bar=False)

        # Store the video_id and its captions' embeddings
        all_embeddings_data.append({
            'video_id': video_id,
            'captions_embeddings': caption_embeddings
        })

        # Optionally, save embeddings for each video separately
        # This can be useful if your dataset is very large and you want to process in chunks
        # video_output_path = os.path.join(output_folder, f"{video_id}_embeddings.npy")
        # np.save(video_output_path, caption_embeddings)
        # print(f"Saved embeddings for {video_id} to {video_output_path}")

    # Save all collected embeddings into a single file for easier loading later
    # You can choose to save as a NumPy array (if all embeddings are of the same shape)
    # or as a pickled list of dictionaries for more flexibility.
    # For this structure (list of dicts), pickle is often more straightforward.

    final_output_path = os.path.join(output_folder, 'msrvtt_caption_embeddings.pkl')
    import pickle
    with open(final_output_path, 'wb') as f:
        pickle.dump(all_embeddings_data, f)
    print(f"\nAll embeddings saved to: {final_output_path}")
    print("Each entry in the saved file is a dictionary with 'video_id' and 'captions_embeddings' (a NumPy array).")


# --- How to use the script ---
if __name__ == "__main__":
    # IMPORTANT: Replace 'your_msrvtt_data.json' with the actual path to your JSON file
    # and 'embeddings_output' with your desired output folder name.
    json_data_path = 'eval_msr_vtt/msrvtt_train_7k.json'
    output_directory = 'eval_msr_vtt/embed'

    # Make sure to replace 'your_msrvtt_data.json' with the correct path to your data file.
    # Example: If your JSON is in the same directory as the script, just use its filename.
    # If it's in a 'data' folder, use 'data/your_msrvtt_data.json'.
    create_and_save_embeddings(json_data_path, output_directory)

    print("\nScript finished. Check the specified output folder for the embeddings.")
    print("To load the embeddings later (e.g., for calculating similarity):")
    print(f"  import pickle")
    print(f"  with open('{os.path.join(output_directory, 'msrvtt_caption_embeddings.pkl')}', 'rb') as f:")
    print(f"      loaded_embeddings_data = pickle.load(f)")
    print(f"  # loaded_embeddings_data will be a list of dictionaries, each containing 'video_id' and 'captions_embeddings'")