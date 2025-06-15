import torch
import clip
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model_with_loaded
from llava.model.builder import load_pretrained_model
from flask import Flask, request, jsonify
from concurrent.futures import Future, ThreadPoolExecutor
from transformers import AutoProcessor, AutoModelForCausalLM
import os
import argparse


import threading
import queue
from PIL import Image

app = Flask(__name__)

# Global model variables
clip_model = None
clip_preprocess = None
clip_device = None
llava_model = None
llava_tokenizer = None
llava_image_processor = None
llava_context_len = None
git_model = None
git_processor = None

# Request queues
clip_queue = queue.Queue(maxsize=100)
llava_queue = queue.Queue(maxsize=50)
git_queue = queue.Queue(maxsize=100)

llava_executor = ThreadPoolExecutor(max_workers=2) 

def initialize_models():
    global clip_model, clip_preprocess, clip_device, llava_model, llava_model_name, git_processor, git_model
    global llava_tokenizer, llava_model, llava_image_processor, llava_context_len
    
    # Initialize CLIP
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-L/14@336px"
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load(model_name, device=clip_device)
    
    # Initialize LLaVA
    llava_model_path = "liuhaotian/llava-v1.5-7b"
    print("Loading LLaVA model...")
    llava_model_name = get_model_name_from_path(llava_model_path)
    llava_tokenizer, llava_model, llava_image_processor, llava_context_len = load_pretrained_model(
        llava_model_path, None, llava_model_name
    )

    # # Initialize GIT
    print("Loading GIT model...")
    git_processor = AutoProcessor.from_pretrained("microsoft/git-large")
    git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large").to("cuda")

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

def llava_worker():
    while True:
        task = llava_queue.get()
        if task is None:
            break
        try:
            output = eval_model_with_loaded(task["prompt"], task["image_path"],
                                            llava_model, llava_tokenizer, llava_image_processor)
            task['future'].set_result(output)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            task['future'].set_exception(e)
        finally:
            llava_queue.task_done()

def git_worker():
    while True:
        task = git_queue.get()
        if task is None:
            break
        try:
            image_path = task['image_path']
            image = Image.open(image_path).convert("RGB")
            inputs = git_processor(images=image, return_tensors="pt").to("cuda")
            generated_ids = git_model.generate(**inputs, max_length=50)
            caption = git_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            task['future'].set_result(caption)
        except Exception as e:
            task['future'].set_exception(e)
        finally:
            git_queue.task_done()

@app.route('/git/caption', methods=['POST'])
def caption_image():
    data = request.json
    image_path = data['image_path']
    future = Future()
    git_queue.put({
        'image_path': image_path,
        'future': future
    })
    return jsonify({'result': future.result()})


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

@app.route('/llava/verify_batch', methods=['POST'])
def verify_images_batch():
    data = request.json
    image_paths = data.get("image_paths", [])
    prompt = data.get("prompt")
    
    if not image_paths or not prompt:
        return jsonify({"error": "Missing 'image_paths' or 'prompt' in request"}), 400
    
    futures = []
    results = {}
    
    # Submit all tasks
    for image_path in image_paths:
        future = Future()
        llava_queue.put({
            'image_path': image_path,
            'prompt': prompt,
            'future': future
        })
        futures.append((image_path, future))
    
    # Collect results
    for image_path, future in futures:
        try:
            result = future.result(timeout=120)
            results[image_path] = result
        except Exception as e:
            results[image_path] = f"Error: {str(e)}"
    
    return jsonify({'results': results})

@app.route('/llava/verify', methods=['POST'])
def verify_image():
    data = request.json
    image_path = data.get("image_path")
    prompt = data.get("prompt")

    if not image_path or not prompt:
        return jsonify({"error": "Missing 'image_path' or 'prompt' in request"}), 400

    future = Future()
    llava_queue.put({
        'image_path': image_path,
        'prompt': prompt,
        'future': future
    })

    try:
        result = future.result(timeout=60)  # Wait for up to 60 seconds
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'result': str(e)}), 500
    
@app.route('/clip/model_name', methods=['GET'])
def get_clip_model_name():
    return jsonify({'clip_model_name': "ViT-L/14@336px"})

@app.route('/llava/model_name', methods=['GET'])
def get_llava_model_name():
    return jsonify({'llava_model_name': llava_model_name})

@app.route('/git/model_name', methods=['GET'])
def get_git_model_name():
    return jsonify({'git_model_name': git_model.name_or_path})

if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    initialize_models()
    
    parser = argparse.ArgumentParser(description="Run the model server")
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on"
    )
    args = parser.parse_args()
    port = args.port
    
    for _ in range(2):
        threading.Thread(target=clip_worker, daemon=True).start()
    for _ in range(1):
        threading.Thread(target=llava_worker, daemon=True).start()
    for _ in range(2):
        threading.Thread(target=git_worker, daemon=True).start()
    
    app.run(host='0.0.0.0', port=port)