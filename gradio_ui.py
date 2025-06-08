import gradio as gr
import os

from clip_model import CLIPModel
from keyword_extractor import KeyWordExtractor
from pipeline import Pipeline

# Variables
IMAGE_DIR = ""
IMG_EMB_PATH = ""
DEVICE = ""

def search_images(query):
    return pipeline.run(query, use_kw_extractor=False)

# Load all images initially
def load_all_images():
    return [os.path.join(IMAGE_DIR, img) for img in os.listdir(IMAGE_DIR)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Callback for search button
def handle_search(query):
    if not query.strip():
        return load_all_images()
    return search_images(query)

with gr.Blocks() as demo:
    gr.Markdown("## 🔍 AFM Search")

    with gr.Row():
        query_input = gr.Textbox(
            label="Search", 
            placeholder="Search Anything", 
            scale=4
        )
        search_btn = gr.Button(
            "Search", 
            scale=1, 
            size="lg"
        )


    gallery = gr.Gallery(label="Image Gallery", columns=5, height="auto")

    # Bind search button to function
    search_btn.click(fn=handle_search, inputs=query_input, outputs=gallery)

    # Show all images on load
    demo.load(fn=load_all_images, outputs=gallery)

if __name__ == "__main__":
    try:

        # set up pipeline and load to memory
        device = DEVICE
        embedding_database_path = IMG_EMB_PATH
        clipmodel = CLIPModel(device)
        keywordextractor = KeyWordExtractor('kw-extractor-tokenizer', 'kw-extactor-model')
        pipeline = Pipeline(clipmodel, keywordextractor, embedding_database_path, device)

        demo.launch()

    except KeyboardInterrupt:
        print("\nGradio app stopped by user.")