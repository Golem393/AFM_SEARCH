from coco_extractor import COCOCaptionExtractor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from git_matcher import GitMatcher
from clip_matcher import CLIPMatcher
import argparse

parser = argparse.ArgumentParser(description="Run the model server")
parser.add_argument(
    "--port", type=int, default=5000, help="Port to run the server on"
)
args = parser.parse_args()
port = args.port

def load_list_from_txt(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        data = [line.strip() for line in f]
    return data
    

PATH_FOLDER_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
PATH_FOLDER_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
PATH_FOLDER_BENCHMARKS = Path("benchmarks/")
PATH_FOLDER_EMBEDDINGS = Path("embeddings/")
    

git_matcher = GitMatcher(
    image_folder=PATH_FOLDER_IMAGES,
    embedding_folder= PATH_FOLDER_EMBEDDINGS,
    top_k=30,
    print_progress=True,
    port=port,
    subset=load_list_from_txt("subsets/subset_1000.txt"),
    )
git_matcher.compute_embeddings()

git_matcher = GitMatcher(
    image_folder=PATH_FOLDER_IMAGES,
    embedding_folder= PATH_FOLDER_EMBEDDINGS,
    top_k=30,
    print_progress=True,
    port=port,
    subset=load_list_from_txt("subsets/subset_5000.txt"),
    )
git_matcher.compute_embeddings()

git_matcher = GitMatcher(
    image_folder=PATH_FOLDER_IMAGES,
    embedding_folder= PATH_FOLDER_EMBEDDINGS,
    top_k=30,
    print_progress=True,
    port=port,
    subset=load_list_from_txt("subsets/subset_10000.txt"),
    )
git_matcher.compute_embeddings()
