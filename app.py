import gradio as gr
import os
import json
import time
import uuid

class ImageRetrievalApp:
    """Gradio UI for retrieval pipeline.
    This implements the Gradio UI for the image retrieval pipeline. 
    If used on a SLURM Cluster this app need to be run on an entry node.

    """
    def __init__(self, port:int=7860):
        
        # Create required tmp folder for gradio (gradio uses these for caching)
        try:
            tmp_dir = "tmp/gradio_tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            self._tmp_dir = tmp_dir
        except OSError as e:
            raise OSError("Failed to create a tmp folder that is needed for Gradio") from e
        
        os.environ["GRADIO_TEMP_DIR"] = self._tmp_dir
        os.environ["TMPDIR"] = self._tmp_dir
        os.environ["TEMP"] = self._tmp_dir
        os.environ["TMP"] = self._tmp_dir

        self.port = port # port the gradio ui can be accessed
        self.gallery_dir = "/usr/prakt/s0122/afm/dataset/flickr8k/Flicker8k_Dataset/"
        self._suporrted_formats = ('.png', '.jpeg', '.jpg')
        self._batchsize = 20 # number of images to load at once
        self._allowed_paths = [self._tmp_dir, self.gallery_dir] # paths gradio is allowed to access outside working dir

        # Paths to all items in the gallery
        self._all_paths = sorted([
            os.path.join(self.gallery_dir, item) 
            for item in os.listdir(self.gallery_dir) 
            if item.lower().endswith(self._suporrted_formats)
        ])

        self.app = self.build_interface()


    def _load_batch(self, batch_idx, current_items):
        """Returns a new batch of items and new batch index."""
        start = batch_idx * self._batchsize
        end = start + self._batchsize
        new_batch = self._all_paths[start:end]
        return current_items + new_batch, batch_idx + 1


    def _reset_gallery(self):
        """Method to reset the Gallery after cancelling a completed search."""
        initial_batch = self._all_paths[:self._batchsize]
        self.batch_index.value = 1
        gallery_title = "## 🎞️ Camera Roll"
        return initial_batch, self.batch_index.value, gallery_title
    

    def _request_pipeline(self, file_path, query):
        """Write prompt to .json file to request retrieved images from pipeline.
        
        This step write the prompt to a .json in the tmp/ folder in the usr/ 
        directory, so that the pipeline running on the SLURM Cluster can access it. 
        This function checks if results for this query are already computed and has
        access to the complete search history.
        """
        data = {}
        if os.path.exists(file_path): # if .json exists, load it
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

        # Check if this prompt has been queried before and instnatly obtain results 
        # TODO: Currently doesn't take changes in the gallery into account!
        if query in data:
            return data[query]
        
        # New search needs to be started and prompt is written to .json
        else: 
            data[query] = []
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2)
            return None

  
    def _wait_for_response(self, file_path, query, check_interval=0.1):
        """Waits for the results to be written into the json"""
        while True:
            if not os.path.exists(file_path):
                time.sleep(check_interval)
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    time.sleep(check_interval)
                    continue  # file might be mid-write
            
            # check if the key exists and has a non-empty value
            if query in data and data[query]:
                print(f"{len(data[query])} matches retrieved by pipeline for query '{query}'")
                return data[query]
            else:
                time.sleep(check_interval)  # wait 100ms before checking again


    def _search_items(self, prompt, current_items):
        """Central search function of the UI."""
        # construct path to json that handles UI, Pipeline communication
        json_file_path = os.path.join(self._tmp_dir, "search_requests.json")
        # Request matching images for query from pipeline
        retrieved_items = self._request_pipeline(json_file_path, prompt)
        print(f"Requested matches for query '{prompt}'")
        
        # use results from search history if prompt was queried before
        if retrieved_items is not None: 
            print(f"Prompt was queried before, obtain results from history.")
            current_items = retrieved_items
        else:
            # if new prompt is used wait for pipeline to write reults
            time.sleep(0.1)
            current_items = self._wait_for_response(json_file_path, prompt)

        gallery_title = f"## Found {len(current_items)} matches for '{prompt}'"
        # return matches to be displayed, (batch_idx), new title containing no. of matches
        return current_items, 1, gallery_title
    
    
    def _on_search_or_cancel(self, prompt, current_items, btn_label):
        """Switches functionality of the search button between search and cancel."""
        if btn_label == "Search":
            results, new_batch_idx, new_title = self._search_items(prompt, current_items)
            return results, new_batch_idx, new_title, gr.update(value="Cancel"), "Cancel"
        else:
            results, new_batch_idx, new_title = self._reset_gallery()
            # Optionally clear search input as well
            return results, new_batch_idx, new_title, gr.update(value="Search"), "Search"


    def build_search_tab(self):
        """Search/Gallery Tab UI implementation."""
        with gr.Tab("Search"):
            # Title to display: Displays number of matches after each search
            gallery_title = gr.Markdown("## 🎞️ Gallery")

            with gr.Row():
                self.search_input = gr.Textbox(
                    label="Search", 
                    placeholder="Search for anyting...", 
                    scale=4
                )
                self.search_btn = gr.Button(
                    "Search",
                    scale=1,
                    size="lg"
                )
            
            # Batch of images to display
            initial_batch = self._all_paths[:self._batchsize]
            self.batch_index = gr.State(1)  # start after first batch
            
            self.btn_label = gr.State("Search") # state for the search/cancel button

            self.gallery = gr.Gallery(
                label=" ",
                value=initial_batch,
                columns=4,
                height="auto"
            )

            self.load_more_btn = gr.Button("Load More Images")
            # Bind load more button
            self.load_more_btn.click(
                fn=self._load_batch,
                inputs=[self.batch_index, self.gallery],
                outputs=[self.gallery, self.batch_index]
            )   

            self.search_btn.click(
                fn=self._on_search_or_cancel,
                inputs=[self.search_input, self.gallery, self.btn_label],
                outputs=[
                    self.gallery, # update gallery
                    self.batch_index, # handle batch index TODO: relevant for results?
                    gallery_title, # update gallery title to display no. of matches
                    self.search_btn, # update text on the search button (search vs. cancel)
                    self.btn_label  # update state of the search/cancel button
                ]
            )

    def _handle_upload(self, uploaded_files):
        """Handles the upload logic of the UI"""
        if not uploaded_files:
            return "⚠️ No files selected."

        uploaded_paths = []

        for file in uploaded_files:
            filename = os.path.basename(file)
            name, fformat = os.path.splitext(filename)
            if fformat.lower() not in self._suporrted_formats:
                continue  # Skip unsupported formats

            target_path = os.path.join(self.gallery_dir, filename)
            
            if os.path.exists(target_path):
                uid = uuid.uuid4().hex[:8]
                filename = f"{name}_{uid}{fformat}"
                target_path = os.path.join(self.gallery_dir, filename)
        
            try:
                os.rename(file, target_path)  # move uploaded tmp file to gallery dir
                uploaded_paths.append(filename)
            except Exception as e:
                return f"❌ Failed to upload {filename}: {str(e)}"

        if uploaded_paths: # uploaded items successfully
            self._all_paths.extend(uploaded_paths)
            self._all_paths = sorted(self._all_paths)
            return f"✅ Uploaded: {len(uploaded_paths)} item."
        else:
            return "⚠️ No valid items uploaded."

    def build_upload_tab(self):
        """Upload tab UI implementation."""
        with gr.Tab("Upload"):
            gr.Markdown("## ⬆️ Upload Images to Gallery")

            uploader = gr.File(
                file_types=["image"],
                file_count="multiple",
                label="Select Images to Upload",
                interactive=True
            )
            upload_status = gr.Textbox(
                interactive=False,
                show_label=False,
                placeholder="Upload status will appear here..."
            )
            upload_btn = gr.Button("Upload Selected Files")

            upload_btn.click(
                fn=self._handle_upload,
                inputs=[uploader],
                outputs=[upload_status]
            ) 

    def build_settings_tab(self):
        with gr.Tab("Settings"):
            gr.Markdown("## ⚙️ Settings (to be implemented)")

    def build_interface(self):
        with gr.Blocks() as interface:
            gr.Markdown('# AFM_SEARCH')
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tabs():
                        self.build_search_tab()
                        self.build_upload_tab()
                        self.build_settings_tab()
        return interface

    def launch(self):
        self.app.launch(server_name="0.0.0.0", server_port=self.port, share=False, allowed_paths=self._allowed_paths)

if __name__ == "__main__":
    app = ImageRetrievalApp(port=7864)
    app.launch()