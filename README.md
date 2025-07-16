# AFM_SEARCH

Git repo of Group 5 for the Practical Course _Applied Foundation Models_

## 🔧 Setup 

**LLaVA Installation:**
1. Clone LLaVA Folder into same directory as files: `git clone https://github.com/haotian-liu/LLaVA.git`
2. `cd LLaVA`
3. `pip install --upgrade pip` (enables PEP 660 support)
4. `pip install -e .`
5. `pip install -e ".[train]"`
6. `pip install flash-attn --no-build-isolation` (optional)
7. `pip install -r requirements.txt`
8. At /LLaVA/llava/eval/run_llava.py insert function `eval_mode_with_loaded` from eval_mode_with_loaded.txt

**PaliGemma Installation:**
1. To create an environment cuda is required, ensure that cuda 12.1 is installed!
2. Create conda environment with provided env.yml `conda env create -f env.yml`

**Token:**

A Hugging Face token and access to the model `google/paligemma-3b-pt-224` is required. 
1. [Create a Hugging Face account](https://huggingface.co)
2. Go to the [model card](https://huggingface.co/google/paligemma-3b-pt-224) of `google/paligemma-3b-pt-224` and request access
3. Go to your acccount -> Settings -> Access Tokens
4. Create a token and save it somewhere safe!
5. Enter in your terminal: `export HF_TOKEN=<your_token>`
6. If you want to save your token for all terminal sessions: `source ~/.bashrc  # or source ~/.zshrc`

## 🚀 Use Application

**SLURM Cluster:**
1. `salloc --nodes=1 --cpus-per-task=4 --mem=32G --gres=gpu:1,VRAM:16G --time=hh:mm:ss --partition=PRACT--qos=practical_course`
2. `conda activate <env>`
3. Open second terminal and run `srun --jobid=<id> --pty bash`
4. In the first terminal `python3 model_server.py --port <e.g. 5000>`
5. Open second terminal and run `conda activate <env>`
6. If server is started successfully, in the second terminal run `python3 app_pipeline.py --port <e.g. 5000>`
7. Open a third terminal on an entry node `conda activate <env>`
8. In third terminal run `python3 app.py``
9. The app can be accessed on your localhost under http://localhost:7864

## 🧪 Benchmarking

**SLURM Cluster:**
1. Follow the first 5 steps outlined in the Use Application section.
2. If server is started successfully, in the second terminal run `python3 benchmark.py --port <e.g. 5000>`