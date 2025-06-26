import torch
import clip
from flask import Flask, request, jsonify
from concurrent.futures import Future, ThreadPoolExecutor
from transformers import AutoProcessor, AutoModelForVision2Seq

import argparse
import os

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
    images = [Image.open(item["image_path"]).convert("RGB") for item in batch]

    inputs = paligemma_processor(images=images, text=prompts, return_tensors="pt", padding=True).to(paligemma_device)

    with torch.no_grad():
        outputs = paligemma_model.generate(**inputs, max_new_tokens=5)
    
    # decode results
    results = paligemma_processor.batch_decode(outputs, skip_special_tokens=True) 
    
    #paligemma returns original prompt with answer which is in a new line
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