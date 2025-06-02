from PIL import Image
import gradio as gr
import os

# --- Config ---
IMAGE_FOLDER = "images"
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')

# loads all relevant image paths or image data initially.


def load_all_image_paths(folder_path: str):
    """
    Scans the given folder for images and returns a list of their full paths.
    """
    all_paths = []
    if not os.path.isdir(folder_path):
        print(f"Warning: Image folder '{folder_path}' not found.")
        return []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(SUPPORTED_EXTENSIONS):
            all_paths.append(os.path.join(folder_path, fname))
    if not all_paths:
        print(f"No images found in '{folder_path}'. Please add some images.")
    return sorted(all_paths)


# Load all image paths once at the start
ALL_AVAILABLE_IMAGE_PATHS = load_all_image_paths(IMAGE_FOLDER)

# serch algo placeholder


def search_function(search_query: str, all_image_paths: list):
    """
    This is where you'll plug in YOUR search logic.
    It takes a search query (string) and the list of all available image paths.
    It should return a LIST of image paths that match the query.
    """
    print(f"Searching for: '{search_query}'")

    # If search query is empty or just whitespace, show all images
    if not search_query or search_query.isspace():
        print("Empty query, returning all images.")
        return all_image_paths
    # TODO: impl
    ...

# Update Gradio func after search input


def update_image_gallery(search_query_from_textbox: str):
    """
    Updates the image gallery based on the search query of the textbox
    """
    # search aglo call
    filtered_image_paths = search_function(
        search_query_from_textbox, ALL_AVAILABLE_IMAGE_PATHS
    )

    return filtered_image_paths if filtered_image_paths is not None else ALL_AVAILABLE_IMAGE_PATHS


# actual Gradio app setup
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AFM Image Search App")
    gr.Markdown(
        "Enter a keyword to filter images.")

    with gr.Row():
        search_input = gr.Textbox(
            label="Search Query",
            placeholder="e.g., cat, landscape, castle",
            interactive=True,
            show_label=True,
            # elem_id="search-input-textbox" #CSS styling if needed
        )
        # maybe with a button to avoid live updates on text change??
        # search_button = gr.Button("Search")

    image_gallery = gr.Gallery(
        label="Image Results",
        value=ALL_AVAILABLE_IMAGE_PATHS,  # Initial all images are diesplayed
        # columns=[4, 5, 6],  # Responsive columns (dependend on screen size)
        columns=5,
        height="auto",  # 600,
        object_fit="fill",  # "contain" shows whole image, "cover" fills and crops
        preview=False,
        allow_preview=True,
        show_label=False,
        # elem_classes="image-gallery",  # CSS classes for custom styling
        # elem_id="gallery"  # css styling if needed
    )

    # Update gallery input changes (live search)
    search_input.change(
        fn=update_image_gallery,
        inputs=[search_input],
        outputs=[image_gallery]
    )

    # with search button:
    # search_button.click(
    #     fn=update_image_gallery,
    #     inputs=[search_input],
    #     outputs=[image_gallery]
    # )

    if not ALL_AVAILABLE_IMAGE_PATHS:
        gr.Warning(f"No images found in '{IMAGE_FOLDER}'.")


if __name__ == "__main__":
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(
            f"Created image folder: '{IMAGE_FOLDER}'. Please add your images there and restart the app.")

    demo.launch(share=False)
