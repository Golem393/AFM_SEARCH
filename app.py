import gradio as gr
import os
import json
import time
import re

class ImageRetrievalApp:
    """Class for Gradio UI of the image/video retrieval pipeline.
    This class has a lot of extra functionality to allow for it to run on 
    a SLURM cluster with no port forwarding between compute and entry node.
    This app need to be run on an entry node!
    """
    def __init__(self, port:int=7860):
        
        # Create required tmp folder for gradio (gradio uses these for caching)
        try:
            tmp_dir = "tmp/gradio_tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            self._tmp_dir = tmp_dir
        except OSError as e:
            raise OSError("Failed to create a tmp folder that is needed for Gradio") from e

        # set env variables to tmp folder
        os.environ["GRADIO_TEMP_DIR"] = self._tmp_dir
        os.environ["TMPDIR"] = self._tmp_dir
        os.environ["TEMP"] = self._tmp_dir
        os.environ["TMP"] = self._tmp_dir

        self.port = port # port the gradio ui can be accessed
        self.gallery_dir = "/usr/prakt/s0122/afm/dataset/demo/cc3m_0000_0003" # directory where the gallery is located
        self._suporrted_formats = ('.png', '.jpeg', '.jpg', '.mp4') # list of supported file formats
        self._batchsize = 20 # number of images to load at once (adjust according to system)
        self._allowed_paths = [self._tmp_dir, self.gallery_dir] # paths gradio is allowed to access outside working dir
        
        # Paths to all items in the gallery
        self._all_paths = sorted([
            os.path.join(self.gallery_dir, item) 
            for item in os.listdir(self.gallery_dir) 
            if item.lower().endswith(self._suporrted_formats)
        ])

        self.app = self.build_interface() # build app

    # ======================================================================= #
    # This section contains methods for load more feature

    def _load_batch(self, batch_idx, current_items):
        """Returns a new batch of items and new batch index."""
        start = batch_idx * self._batchsize
        end = start + self._batchsize
        new_batch = self._all_paths[start:end]
        status = f"✅ Loaded {len(new_batch)} more items" if len(new_batch) > 0 else "ℹ️ All items are already loaded" 
        gr.Info(status)
        return current_items + new_batch, batch_idx + 1

    # ======================================================================= #
    # This section contains methods for the cancel search feature

    def _reset_gallery(self):
        """Reset the Gallery after cancelling a completed search."""
        initial_batch = self._all_paths[:self._batchsize]
        self.batch_index.value = 1
        gallery_title = "## 🎞️ Camera Roll"
        return initial_batch, self.batch_index.value, gallery_title
    
    def _on_search_or_cancel(self, prompt, current_items, btn_label):
        """Switches functionality of the search button between search and cancel."""
        if btn_label == "Search":
            results, new_batch_idx, new_title, scores_dict = self._search_items(prompt, current_items)
            
            # Triggered when an illegal query was given:
            if results is None: 
                initial_batch, new_batch_idx, new_title = self._reset_gallery()
                return initial_batch, new_batch_idx, new_title, gr.update(value="Search"), "Search", None

            results_filenames = [os.path.basename(p) for p in results] # get filenames of results to display along item
            return zip(results,results_filenames), new_batch_idx, new_title, gr.update(value="Cancel"), "Cancel", scores_dict
        else:
            initial_batch, new_batch_idx, new_title = self._reset_gallery()
            # TODO: clear search input as well
            return initial_batch, new_batch_idx, new_title, gr.update(value="Search"), "Search", None

    # ======================================================================= #
    # This section contains methods for the search feature.

    def _request_pipeline(self, file_path, query):
        """Write prompt to .json file to request retrieved images from pipeline.
        
        Writes the prompt to a .json in the tmp/ folder in the usr/ directory, so that 
        the pipeline running on the SLURM Cluster can access it. 
        This function checks if results for this query are already computed and has
        access to the complete search history.
        """
        if os.path.exists(file_path): # if .json exists, load it
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = {}

        # Set data for query to none to start search
        data[query] = None
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
            
            # check if the key exists and has a not None value
            if query in data and data[query] is not None:
                return data[query]
            else:
                time.sleep(check_interval)  # wait 100ms before checking again

    def _search_items(self, prompt, current_items):
        """Central search function of the UI."""
        if not bool(prompt): # check if string is empty
            return None, None, None, None
        if not prompt.isascii(): # check if non-ascii characters are used
            gr.Warning("⚠️ Please only use ASCII characters")
            return None, None, None, None
        if len(prompt) > 500: # check if no. character is less than 1500
            gr.Warning("⚠️ Please use a shorter search query")
            return None, None, None, None 
        if not bool(re.search(r'[A-Za-z]', prompt)):
            gr.Warning("⚠️ Please use at least one letter")
            return None, None, None, None 
        
        # construct path to json that handles UI, Pipeline communication
        json_file_path = os.path.join(self._tmp_dir, "search_requests.json")
        # Request matching images for query from pipeline
        retrieved_items = self._request_pipeline(json_file_path, prompt)
        print(f"Requested matches for query '{prompt}'")
        gr.Info("ℹ️ Requested matches from pipeline")
        
        # use results from search history if prompt was queried before
        if retrieved_items is not None: 
            print(f"Prompt was queried before, obtain results from history.")
            results_path = [os.path.join(self.gallery_dir, result) for result in retrieved_items['confirmed']]
            scores = retrieved_items['confirmed_scores']
            gr.Info("✅ Loaded matches from history")
        # if new prompt is used wait for pipeline to write reults
        else:
            time.sleep(0.1)
            results = self._wait_for_response(json_file_path, prompt)
            results_path = [os.path.join(self.gallery_dir, result) for result in results['confirmed']]
            scores = results['confirmed_scores']
            gr.Info(f"✅ Found {len(results_path)} matches")

        scores_dict = self._scores_dict(results_path, scores) # scores dict for clip confidence section
        
        matches = []
        for match in results_path:
            if ".mp4" in match: # video match
                filepath, _ = self._extract_videocodes(match)
                if filepath not in matches:
                    matches.append(filepath)
            else: # image match
                matches.append(match)
        
        if any(".mp4" in match for match in matches): # Info message for video matches
            gr.Info("ℹ️ Some matches are videos. See confidence section for more info on the exact timestamp") 
        
        gallery_title = f"## {len(results_path)} matches for '{prompt}'"
        
        return matches, 1, gallery_title, scores_dict
    
    def _scores_dict(self, results_path, scores) -> dict:
        """Compute the score dict to display clip confidence in confidence section"""
        scores_dict = {}
        for result, score in zip(results_path, scores):
            if ".mp4" in result: # video
                filepath, timestamp = self._extract_videocodes(result)
                minutes, seconds = divmod(int(float(timestamp)), 60)
                item = f"{os.path.basename(filepath)} at {minutes:02}:{seconds:02}"
                scores_dict[item] = score
            else: # image
                scores_dict[os.path.basename(result)] = score
        return scores_dict

    def _extract_videocodes(self, path) -> tuple:
        """Helper Function to extract filepath and timestamp from video codes"""
        assert ".mp4" in path, "Path must be a video!"
        filepath = f"{path.split('.mp4_')[0]}.mp4"
        timestamp = path.split(".mp4_")[1]
        return (filepath, timestamp)
    
    # ======================================================================= #
    # This section contains methods for the delete feature

    def _on_gallery_select(self, evt: gr.SelectData):
        """EventListener for item selection in the gallery. Important for delete function"""
        selected_item_path = evt.value
        print(f"Selected image: {selected_item_path}")
        return selected_item_path
       
    def _delete_selected_item(self, selected_item):
        """Deletes selected items from disk and updates the gallery"""
        selected_item_path = os.path.join(self.gallery_dir, selected_item['image']['orig_name']) # TODO: add video support

        if selected_item_path and os.path.exists(selected_item_path):
            try:
                os.remove(selected_item_path)
                self._all_paths.remove(selected_item_path)
                updated_batch = self._all_paths[:self._batchsize]
                gr.Info(f"✅ Deleted {os.path.basename(selected_item_path)}.")
                return updated_batch, 1
            except Exception as e:
                gr.Error(f"❌ Failed to delete: {str(e)}")
                return gr.update(), gr.update()
        gr.Warning("⚠️ No valid item selected.")
        return gr.update(), gr.update()
    
    # ======================================================================= #
    # This section contains methods for the upload feature

    def _handle_upload(self, uploaded_files):
        """Handles the upload logic of the UI"""
        if not uploaded_files:
            gr.Warning("⚠️ No files selected.")
            return gr.update(), gr.update()

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
                uploaded_paths.append(os.path.join(self.gallery_dir, filename))
            except Exception as e:
                gr.Error(f"❌ Failed to upload {filename}: {str(e)}")
                return gr.update(), gr.update()

        if uploaded_paths: # uploaded items successfully
            self._all_paths.extend(uploaded_paths)
            self._all_paths = sorted(self._all_paths)
            gr.Info(f"✅ Uploaded: {len(uploaded_paths)} item(s).")
            return self._all_paths[:self._batchsize], 1
        else:
            gr.Warning("⚠️ No valid items uploaded.")
            return self._all_paths[:self._batchsize], 1
      
    # ======================================================================= #
    # This section contains methods for the settings page     

    def _get_stats(self):
        """Returns number of images and videos alongside size on disk in bytes"""
        n_imgs, n_vids, s_imgs, s_vids = 0, 0, 0, 0
        for file in os.listdir(self.gallery_dir):
            filepath = os.path.join(self.gallery_dir, file)
            if not os.path.isfile(filepath):
                continue
            size = os.path.getsize(filepath)
            if file.lower().endswith(('.jpg','.jpeg','.png')):
                n_imgs += 1
                s_imgs += size
            elif file.lower().endswith(('.mp4')):
                n_vids += 1
                s_vids += size
        return {"imgs": [n_imgs, s_imgs], "vids": [n_vids, s_vids]}
    
    def _save_config(self, img, vid, verify, ver_prompt, n_clip, vid_emb):
        """Saves the current settings into the config.json."""
        
        if not img and not vid:
            gr.Warning(f"⚠️ You excluded image AND videos from your results. This causes no results.")

        config_path = "config.json"
        if not os.path.exists(config_path): # if .json exists, load it
            gr.Error(f"❌ Couldn't find config file at {config_path}")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)

            # Update config
            config.setdefault("active", {})  # ensure key exists
            config["active"].update({
                "RETR_IMGS": int(img),
                "RETR_VIDS": int(vid),
                "VERIFY": int(verify),
                "VERIFIC_PROMPT": ver_prompt,
                "NUM_CLIPMATCHES": n_clip,
                "VIDEO_EMB": vid_emb
            })

            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump(config, file, indent=4)
            gr.Info("✅ Settings saved successfuly")
        except (json.JSONDecodeError, IOError) as e:
            gr.Error(f"❌ Failed to save settings: {str(e)}")

    def _restore_defaults(self):
        """Restores all setting to their default values"""
        config_path = "config.json"

        if not os.path.exists(config_path):
            gr.Error(f"❌ Couldn't find config file at {config_path}")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)

            if "default" not in config:
                gr.Error("❌ No 'default' configuration found.")
                return

            # Overwrite active settings with default values
            config["active"] = config["default"].copy()

            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump(config, file, indent=4)

            gr.Info("✅ Defaults restored successfully")

        except (json.JSONDecodeError, IOError) as e:
            gr.Error(f"❌ Failed to restore defaults: {str(e)}")

    # ======================================================================= #
    # This section contains methods to build the Search, Upload and Settings page
    
    def build_search_tab(self):
        """Search/Gallery Tab UI implementation."""
        with gr.Tab("Search"):
            # Title to display: Displays number of matches after each search
            gallery_title = gr.Markdown("## 🎞️ Camera Roll")

            # Top row with Searchbox and Search button
            with gr.Row():

                self.search_input = gr.Textbox(
                    placeholder="Search for anyting...", 
                    show_label=False,
                    scale=4
                )

                self.search_btn = gr.Button(
                    "Search",
                    scale=1,
                    size="lg"
                )
            
            # Batch of images to display
            initial_batch = self._all_paths[:self._batchsize]
            
            # State variables
            self.batch_index = gr.State(1)  # start after first batch
            self.btn_label = gr.State("Search") # state for the search/cancel button
            self.selected_item = gr.State("") # state to collect selected item

            # Gallery
            self.gallery = gr.Gallery(
                label=" ",
                value=initial_batch,
                columns=4,
                height="auto",
                interactive = False
            )

            with gr.Accordion("See confindence", open=False):
                confidence_scores = gr.Label(show_label=False, show_heading=False)

            # Bottom row with load more and delete button
            with gr.Row():
                self.load_more_btn = gr.Button("Load More Images")
                self.delete_btn = gr.Button("Delete Selected Item")
            
            # Bind load more button
            self.load_more_btn.click(
                fn=self._load_batch,
                inputs=[self.batch_index, self.gallery],
                outputs=[self.gallery, self.batch_index]
            )   

            # Select item event listener
            self.gallery.select(
                fn=self._on_gallery_select,
                inputs=[],
                outputs=self.selected_item
            )

            # Bind delete button
            self.delete_btn.click(
                fn=self._delete_selected_item,
                inputs=[self.selected_item],
                outputs=[self.gallery, self.batch_index]
            )

            # Bind search/cancel button
            self.search_btn.click(
                fn=self._on_search_or_cancel,
                inputs=[self.search_input, self.gallery, self.btn_label],
                outputs=[
                    self.gallery, # update gallery
                    self.batch_index, # handle batch index TODO: relevant for results?
                    gallery_title, # update gallery title to display no. of matches
                    self.search_btn, # update text on the search button (search vs. cancel)
                    self.btn_label,  # update state of the search/cancel button
                    confidence_scores # update scores of matches to display
                ]
            )

    def build_upload_tab(self):
        """Upload tab UI implementation."""
        with gr.Tab("Upload"):
            gr.Markdown("## ⬆️ Upload Images and Videos")

            upload_mask = gr.File(
                file_types=["image", ".mp4"],
                file_count="multiple",
                label="Select Media to Upload",
                interactive=True
            )

            upload_btn = gr.Button("Upload Selected Files")

            upload_btn.click(
                fn=self._handle_upload,
                inputs=[upload_mask],
                outputs=[self.gallery, self.batch_index]
            ) 

    def build_settings_tab(self):
        with gr.Tab("Settings"):
            gr.Markdown("## ⚙️ Settings")

            gr.Markdown("### General")
            with gr.Row():
                retrieve_videos = gr.Checkbox(
                    value=True,
                    label="Retrieve Videos",
                    info="Deactivate to exclude videos from the search results.",
                    interactive=True
                )
                retrieve_images = gr.Checkbox(
                    value=True,
                    label="Retrieve Images",
                    info="Deactivate to exclude images from the search results.",
                    interactive=True
                )

            gr.Markdown("### VLM")
            vlm_checkbox = gr.Checkbox(
                value=True, 
                label="Verify with VLM", 
                info="A VLM is used to refine search results. Deactivating this" \
                " feature improves runtime but reduces quality of the results.", 
                interactive=True
            )
            verification_prompt = gr.Textbox(
                value="Is <query> a fitting description of the image? Answer only with yes or no!",
                placeholder="Use <query> as a placeholder for your search query",
                label="VLM Verification Prompt",
                info="The VLM verifies items using a verification prompt. You can set a custom one here.",
                interactive=True
            )
            
            gr.Markdown("### CLIP")
            top_k = gr.Slider(
                minimum=1, 
                maximum=100, 
                value=30, 
                step=1, 
                label="Number of CLIP Matches",
                info="CLIP retrieves a pre-defined number of items to be " \
                "verified by the VLM. Increasing this value negatively affects " \
                "runtime but might improve results.",
                interactive = True,
            ) 
            video_embedder = gr.Dropdown(
                choices=["keyframe_k_frames", "uniform_k_frames", "keyframe_average", "uniform_average", "optical_flow", "clip_k_frames"],
                value="keyframe_k_frames",
                multiselect=False,
                label="Video Embedder",
                info="To extract keyframes from videos different algorithms can be used.",
                interactive=True
            )

            with gr.Row():
                save_btn = gr.Button("Save Settings")
                restore_btn = gr.Button("Restore Defaults")

            gr.Markdown('### Stats')
            stats = self._get_stats()
            gr.Markdown(
                f"{stats['imgs'][0]} Image(s): {stats['imgs'][1]/1e6:.1f}MB <br>{stats['vids'][0]} Video(s): {stats['vids'][1]/1e6:.1f}MB", 
                container=True, 
                line_breaks=True
            )

            #Bind save button
            save_btn.click(
                fn=self._save_config,
                inputs=[retrieve_images, retrieve_videos, vlm_checkbox, verification_prompt, top_k, video_embedder],
                outputs=[]
            )
            # Bind reset button
            restore_btn.click(
                fn=self._restore_defaults,
                inputs=[],
                outputs=[]
            )

    def build_interface(self):
        with gr.Blocks() as interface:
            gr.Markdown('# 🔍 AFM SEARCH')
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tabs():
                        self.build_search_tab()
                        self.build_upload_tab()
                        self.build_settings_tab()
            gr.Markdown("<div style='text-align: center; color: gray;'>Made by Benjamin Kasper, Flavio Arrigoni and Simon Hartmann</div>")
        return interface

    def launch(self):
        self.app.launch(server_name="0.0.0.0", server_port=self.port, share=False, allowed_paths=self._allowed_paths)

if __name__ == "__main__":
    app = ImageRetrievalApp(port=7864)
    app.launch()