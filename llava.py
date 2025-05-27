
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


#from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.eval.run_llava import eval_model


# Set your prompt here
prompt = "Does this image show a girl with black hair"

# Folder path containing images
image_folder = "Thailand/girl_with_black_hair_ViT-L_14@336px_top_matches"

# Load the LLaVA model and processor
model_path = "liuhaotian/llava-v1.5-7b"



# Loop through all images in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": prompt,
            "conv_mode": None,
            "image_file": image_path,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()

        output = eval_model(args)
        print(output)
