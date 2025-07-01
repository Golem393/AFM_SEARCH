import torch
import clip
from flask import Flask, request, jsonify
from concurrent.futures import Future, ThreadPoolExecutor
from transformers import AutoProcessor, AutoModelForVision2Seq

import argparse
import os
import base64
from io import BytesIO

import cv2

import threading
import queue
from PIL import Image

# Argparser to start the server on custom port
parser = argparse.ArgumentParser(description="Run the model server")
parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
args = parser.parse_args()

PORT = args.port # get port
TOKEN = os.environ.get("HF_TOKEN") # get token from environment

app = Flask(__name__)

# Global model variables
clip_model = None
clip_preprocess = None
clip_device = None
paligemma_processor = None
paligemma_model = None

# Request queues
clip_queue = queue.Queue(maxsize=100)
paligemma_queue = queue.Queue(maxsize=100)

def initialize_models():
    global clip_model, clip_preprocess, clip_device
    global paligemma_model, paligemma_processor, paligemma_device
    
    # Initialize CLIP
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-L/14@336px"
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load(model_name, device=clip_device)

    # Initialize PaliGemma
    paligemma_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading PaliGemma model")
    paligemma_processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224", token=TOKEN)
    paligemma_model = AutoModelForVision2Seq.from_pretrained(
        "google/paligemma-3b-mix-224",
        torch_dtype=torch.bfloat16, 
        token=TOKEN
    ).to(paligemma_device)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass

def clip_worker():
    while True:
        task = clip_queue.get()
        if task is None:  # Sentinel value to stop the thread
            break
        try:
            image_path, prompt = task['image_path'], task['prompt']
            image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(clip_device)
            with torch.no_grad():
                image_feature = clip_model.encode_image(image).float()
                text_tokens = clip.tokenize([prompt]).to(clip_device)
                text_features = clip_model.encode_text(text_tokens).float()
                similarity = (image_feature @ text_features.T).item()
            task['future'].set_result(similarity)
        except Exception as e:
            task['future'].set_exception(e)
        finally:
            clip_queue.task_done()

def paligemma_worker():
    while True:
        task = paligemma_queue.get()
        if task is None:
            break
        try:
            result = verify_batch(task["batch"])
            task["future"].set_result(result)
        except Exception as e:
            task["future"].set_exception(e)
        finally:
            paligemma_queue.task_done()

def verify_batch(batch):
    prompts = [f"Is '{item['prompt']}' a fitting description of the image? Answer only with yes or no!" for item in batch]
    
    # check if image or if video keyframe by inspecting path:
    images_keyframes = []
    # iterate through every item in list
    for item in batch:
        img_keyframe_path = item["image_path"]
        # item is image
        if img_keyframe_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            images_keyframes.append(Image.open(img_keyframe_path).convert("RGB"))
        
        # item is video keyframe
        elif ".mp4" in img_keyframe_path.lower():
            try:
                # keyframe items have the structure: <path>/<video_name>.mp4_timestamp
                video_path, timestamp_str = img_keyframe_path.rsplit(".mp4_", 1) 
                timestamp = float(timestamp_str)
                video_path = f"{video_path}.mp4"
                images_keyframes.append(extract_frame_at_timestamp(video_path, timestamp))
            
            except Exception as e:
                print(f"Failed to process video {video_path}: {e}")

    # preprocess
    inputs = paligemma_processor(images=images_keyframes, text=prompts, return_tensors="pt", padding=True).to(paligemma_device)
    
    # forward pass
    with torch.no_grad():
        outputs = paligemma_model.generate(**inputs, max_new_tokens=5)
    
    # decode results
    results = paligemma_processor.batch_decode(outputs, skip_special_tokens=True) 
    #paligemma returns original prompt with answer which is in a new line: <pormpt>\n<answer>
    return [res.strip() for res in results]

@app.route('/clip/embed_image', methods=['POST'])
def embed_image():
    data = request.json
    future = Future()
    try:
        image_path = data['image_path']
        image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(clip_device)
        with torch.no_grad():
            image_feature = clip_model.encode_image(image).float()
            image_feature /= image_feature.norm(dim=-1, keepdim=True)  # Normalize!
        future.set_result(image_feature.cpu().numpy().tolist())
    except Exception as e:
        future.set_exception(e)
    return jsonify({'result': future.result()})

@app.route('/clip/embed_images', methods=['POST'])
def embed_images():
    data = request.json
    future = Future()
    try:
        image_tensors = []

        if "image_paths" in data:
            for path in data["image_paths"]:
                image = Image.open(path)
                image_tensor = clip_preprocess(image).unsqueeze(0)
                image_tensors.append(image_tensor)

        elif "images" in data:
            for image_data in data["images"]:
                # If raw PIL images are sent via a frontend, adapt this as needed
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
                image_tensor = clip_preprocess(image).unsqueeze(0)
                image_tensors.append(image_tensor)

        else:
            raise ValueError("No valid image input found. Provide 'image_paths' or 'images'.")

        # Stack into a batch
        batch = torch.cat(image_tensors, dim=0).to(clip_device)

        with torch.no_grad():
            image_features = clip_model.encode_image(batch).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

        future.set_result(image_features.cpu().numpy().tolist())
    except Exception as e:
        future.set_exception(e)

    return jsonify({'result': future.result()})

@app.route('/clip/embed_text', methods=['POST'])
def embed_text():
    data = request.json
    future = Future()
    try:
        text = data['text']
        text_tokens = clip.tokenize([text]).to(clip_device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        future.set_result(text_features.cpu().numpy().tolist())
    except Exception as e:
        future.set_exception(e)
    return jsonify({'result': future.result()})

@app.route("/paligemma/verify_batch", methods=["POST"])
def pgemma_verify_batch():
    items = request.json.get("items", [])
    if not isinstance(items, list):
        return jsonify({"error": "items must be a list"}), 400

    future = Future()
    paligemma_queue.put({"batch": items, "future": future})

    try:
        result = future.result(timeout=60)
        return jsonify({"results": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def extract_frame_at_timestamp(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        raise RuntimeError(f"Could not read frame at {timestamp:.2f}s")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)
    
@app.route('/clip/model_name', methods=['GET'])
def get_clip_model_name():
    return jsonify({'clip_model_name': "ViT-L/14@336px"})

@app.route('/paligemma/model_name', methods=['GET'])
def get_paligemma_model_name():
    return jsonify({'clip_model_name': "paligemma-3b-mix-224"})

if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    if not isinstance(TOKEN, str): raise ValueError(f"No valid token found in environment. See README.md for help.")
    initialize_models() # init models
    for _ in range(2):
        threading.Thread(target=clip_worker, daemon=True).start()
    for _ in range(1):
        threading.Thread(target=paligemma_worker, daemon=True).start()
    
    app.run(host='0.0.0.0', port=PORT) # run server
    print(f"Successfully started model server on localhost:{PORT}")