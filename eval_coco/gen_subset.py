from pathlib import Path
from coco_extractor import COCOCaptionExtractor
from sentence_transformers import SentenceTransformer
from caption_embedder import save_embeddings, load_embeddings, find_similar_images
import random

PATH_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
PATH_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")


def get_random_image_filenames(image_path: Path, 
                               amount: int,
                               test_set: list = None, 
                               ) -> list[str]:
    if test_set is None:
        all_files = [f.name for f in image_path.iterdir() if f.is_file()]
    else:
        all_files = test_set
    return random.sample(all_files, amount)


subset_1000 = get_random_image_filenames(PATH_IMAGES, 1000)
subset_5000 = get_random_image_filenames(PATH_IMAGES, 5000)
subset_10000 = get_random_image_filenames(PATH_IMAGES, 10000)

def save_list_to_txt(data: list[str], filename: str):
    with open(filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

def load_list_from_txt(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        data = [line.strip() for line in f]
    return data

save_list_to_txt(subset_1000, "subsets/subset_1000.txt")
save_list_to_txt(subset_5000, "subsets/subset_5000.txt")
save_list_to_txt(subset_10000, "subsets/subset_10000.txt")


extractor = COCOCaptionExtractor(PATH_ANNOTATIONS, PATH_IMAGES)
embedder = SentenceTransformer("all-mpnet-base-v2") 

    
save_embeddings(
    extractor= extractor,
    embedder=embedder,
    embedding_folder=Path("embeddings"),
    name_addition="1000",
    subset=subset_1000,
)

save_embeddings(
    extractor= extractor,
    embedder=embedder,
    embedding_folder=Path("embeddings"),
    name_addition="5000",
    subset=subset_5000,
)

save_embeddings(
    extractor= extractor,
    embedder=embedder,
    embedding_folder=Path("embeddings"),
    name_addition="10000",
    subset=subset_10000,
)

