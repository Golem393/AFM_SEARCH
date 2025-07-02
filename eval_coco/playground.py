#%%
from caption_embedder import save_embeddings, load_embeddings, find_similar_images
import matplotlib.pyplot as plt
from importlib import reload
from pathlib import Path
from PIL import Image
import random


PATH_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
PATH_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
PATH_RESULTS = Path("./output")
PATH_EMBEDDINGS = Path("./embeddings") 

PATH_FILE_EMBEDDING = Path("embeddings/caption_embeddings.h5")
PATH_FILE_TEST_EMBEDDING = Path("embeddings/caption_embeddings_1000_test.h5")

#%%
# Helper functions
def get_random_image_filenames(image_path: Path, 
                               amount: int,
                               test_set: list = None, 
                               ) -> list[str]:
    if test_set is None:
        all_files = [f.name for f in image_path.iterdir() if f.is_file()]
    else:
        all_files = test_set
    return random.sample(all_files, amount)


def save_plot_for_query(query_image_filename: str, 
                        results_data: list,
                        name_addition: str = None,):
    """
    Generates a plot of the query image and its search results,
    and saves it to a file in the globally defined PATH_RESULTS directory.

    Args:
        query_image_filename (str): Filename of the query image.
        results_data (list): List of tuples (filename, score) for results.
                             Assumes results_data corresponds to the query_image_filename.
        test_set (list, optional): A list of filenames to filter results against.
    """
    # Using global paths defined in the script (PATH_IMAGES, PATH_RESULTS)
    images_base_path = PATH_IMAGES
    output_dir = PATH_RESULTS

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    result_image_filenames = [filename for filename, _ in results_data]
    scores = [score for _, score in results_data]

    images_to_plot = [query_image_filename] + result_image_filenames
    num_images = len(images_to_plot)

    if num_images == 0:
        print(f"No images to plot for query {query_image_filename}.")
        return

    # Determine the grid layout
    if num_images == 1:
        num_rows, num_cols = 1, 1
    else:
        num_cols = int(num_images**0.5)
        if num_cols == 0:  # Should not happen if num_images > 1
            num_cols = 1
        num_rows = (num_images + num_cols - 1) // num_cols

    fig, ax_array = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3), squeeze=False)
    # squeeze=False ensures ax_array is always a 2D numpy array, simplifying access
    axes = ax_array.flatten() # Flatten the 2D array of axes for easy iteration

    for i, img_filename in enumerate(images_to_plot):
        if i >= len(axes): # Safety break if more images than subplot cells
            print(f"Warning: Not enough subplot cells for all images. Skipping {img_filename}")
            break
        try:
            img_path = images_base_path / img_filename
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            
            short_img_name = Path(img_filename).stem.split('_')[-1] # Extract ID part of filename

            if i == 0: # Query image
                title_text = f"Query: {short_img_name}"
                axes[i].set_title(title_text, fontsize=8)
            else: # Result image
                # Scores list corresponds to result_image_filenames, so index is i-1
                current_score = scores[i-1] 
                title_text = f"Res {i}: {short_img_name}\nScore {current_score:.4f}"
                axes[i].set_title(title_text, fontsize=8)
        except FileNotFoundError:
            print(f"Warning: Image {img_filename} not found at {img_path}")
            axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=8)
            axes[i].axis('off')
        except Exception as e:
            print(f"Error loading/plotting {img_filename}: {e}")
            axes[i].text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=8)
            axes[i].axis('off')

    # Hide any unused subplots
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Construct output filename from the query image's name
    if name_addition is None:
        output_filename = f"{Path(query_image_filename).stem}.png"
    else:
        output_filename = f"{Path(query_image_filename).stem}_{name_addition}.png"
    output_filepath = output_dir / output_filename
    
    plt.savefig(output_filepath)
    print(f"Plot saved to {output_filepath}")
    plt.close(fig) # Close the figure to free memory


#%%
# Load all images This might take up to 300 seconds and use 1.4GB of RAM, but you only do it once.
print("Loading all embeddings into memory. This may take 300 seconds...")
preloaded_embeddings = load_embeddings(PATH_FILE_EMBEDDING)
print("...loading complete!")


#%%
reload(caption_embedder);from caption_embedder import find_similar_images
results_median = find_similar_images(
    query_image_paths=["COCO_train2014_000000000025.jpg"],
    all_embeddings=preloaded_embeddings, # Pass the pre-loaded data
    aggregation='median',
    top_percent= 10,
    return_scores=True,
)
print(f"Results: {results_median}")

#%%
reload(caption_embedder);from caption_embedder import find_similar_images
results_mean = find_similar_images(
    query_image_paths=["COCO_train2014_000000000025.jpg"],
    all_embeddings=preloaded_embeddings, # Pass the pre-loaded data
    aggregation='mean',
    top_percent= 10,
    return_scores=True,
)
print(f"Results: {results_mean}")

#%%
# compare mean vs median
save_plot_for_query(query_image_filename="COCO_train2014_000000000025.jpg", 
                    results_data=results_median, 
                    name_addition="median")
save_plot_for_query(query_image_filename="COCO_train2014_000000000025.jpg", 
                    results_data=results_mean, 
                    name_addition="mean")

#%%
# Iter 15 random images from data set to quality eval the reuslts

for img in get_random_image_filenames(PATH_IMAGES, 15, test_set=list(preloaded_embeddings.keys())):
    print(f"Querying for image: {img}")
    results = find_similar_images(
        query_image_paths=[img],
        all_embeddings=preloaded_embeddings, # Pass the pre-loaded data
        aggregation='median',
        top_percent= 10,
        return_scores=True,
    )
    save_plot_for_query(query_image_filename=img, results_data=results, name_addition="median")
    results = find_similar_images(
            query_image_paths=[img],
            all_embeddings=preloaded_embeddings, # Pass the pre-loaded data
            aggregation='mean',
            top_percent= 10,
            return_scores=True,
        )
    save_plot_for_query(query_image_filename=img, results_data=results, name_addition="mean")




#%%



#%%
#%%
from coco_extractor import COCOCaptionExtractor
import coco_extractor 
reload(coco_extractor)
import coco_extractor
FOLDER_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
FOLDER_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
FOLDER_BENCHMARKS = Path("benchmarks/")
FOLDER_EMBEDDINGS = Path("embeddings/")
    
FILE_EMBEDDING = Path("embeddings/caption_embeddings.h5")
FILE_TEST_EMBEDDING = Path("embeddings/caption_embeddings_1000_test.h5")#
FILE_PROGRESS = Path('benchmarks/eval_progress.json')
extractor = COCOCaptionExtractor(FOLDER_ANNOTATIONS, FOLDER_IMAGES)

# %%
reload(coco_extractor)
import coco_extractor
extractor.get_captions_for_image("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000000025.jpg")
# %%


# llava path: /storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000372535.jpg
# llava response: No
# llava filename: COCO_train2014_000000471480.jpg
# llava path: /storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000471480.jpg
# llava response: No
# llava filename: COCO_train2014_000000060599.jpg
# llava path: /storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000060599.jpg
# llava response: No
# llava filename: COCO_train2014_000000021253.jpg
# llava path: /storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000021253.jpg
# llava response: No
# llava filename: COCO_train2014_000000320478.jpg
# llava path: /storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000320478.jpg
# llava response: No
# llava filename: COCO_train2014_000000223091.jpg
# llava path: /storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000223091.jpg
#%%


def show_images(image_paths: list[Path]):
    """
    Displays a list of images.

    Args:
        image_paths (list[Path]): A list of Path objects to images.
    """
    num_images = len(image_paths)
    if num_images == 0:
        print("No images to display.")
        return

    # Determine the grid layout
    if num_images == 1:
        num_rows, num_cols = 1, 1
    else:
        num_cols = int(num_images**0.5)
        if num_cols == 0:
            num_cols = 1
        num_rows = (num_images + num_cols - 1) // num_cols

    fig, ax_array = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3), squeeze=False)
    axes = ax_array.flatten()

    for i, img_path in enumerate(image_paths):
        if i >= len(axes):
            break
        try:
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(Path(img_path).name, fontsize=8)
        except FileNotFoundError:
            axes[i].axis('off')
        except Exception as e:
            axes[i].axis('off')

    # Hide any unused subplots
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


img_paths = [
    
"/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000372535.jpg",
"/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000471480.jpg",
"/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000060599.jpg",
"/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000021253.jpg",
"/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000320478.jpg",
"/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000223091.jpg",
"/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014/COCO_train2014_000000076776.jpg",
]
print("A person standing on the snow next to a tree.")
show_images(img_paths)
# %%
import os
import json
from pprint import pprint

def open_json_file(file_path)->dict:
    """Open a JSON file and return its content.
    Format:
    all_models_progress = {
        "last_processed_index": -1,
        "total_captions_processed": 0,
        "clip": {
            "recall@": {"1": 0,"5": 0,"10": 0},
            "precision@": {"1": 0.0,"5": 0.0,"10": 0.0}
            },
            ...,
    }}
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            all_models_progress = json.load(f)
        
    return all_models_progress
def calc_metrics(results:dict)->tuple[dict, int]:
    captions = results["total_captions_processed"]
    images = results["last_processed_index"]
    
    for model in ["clip", "clip+paligemma"]:
        for metric_name, metric_value in results[model].items():
            if isinstance(metric_value, dict):
                for k, value in metric_value.items():
                    results[model][metric_name][k] = value / captions
            else:
                # Handle scalar values like "acc"
                results[model][metric_name] = metric_value / captions
    pprint("Metrics calculated:")
    pprint(results)
    return results, captions, images
#%%
normalized_results, samples, images = calc_metrics(open_json_file("benchmarks/eval_progress_10000.json"))                    
# %%
print(f"Metrics for {samples} captions")
print("Normalized Clip Results:")
pprint(normalized_results["clip"])
print("\n")
print("Normalized Clip+LLaVA Results:")
pprint(normalized_results["clip+llava"])
# %%
