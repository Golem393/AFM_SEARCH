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
    

PATH_FOLDER_ANNOTATIONS = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/annotations")
PATH_FOLDER_IMAGES = Path("/storage/group/dataset_mirrors/old_common_datasets/coco/images/train2014")
PATH_FOLDER_BENCHMARKS = Path("eval_coco/benchmarks")
PATH_FOLDER_EMBEDDINGS = Path("eval_coco/embeddings")
    
PATH_FILE_EMBEDDING = Path("embeddings/caption_embeddings.h5")
PATH_FILE_TEST_EMBEDDING = Path("embeddings/caption_embeddings_1000_test.h5")

git_matcher = GitMatcher(
    image_folder=PATH_FOLDER_IMAGES,
    embedding_folder= PATH_FOLDER_EMBEDDINGS,
    top_k=30,
    print_progress=True,
    port=port
    )
git_matcher.compute_embeddings()
