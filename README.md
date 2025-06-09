# AFM_SEARCH

## Setup 🔧:

1. Clone LLaVA Folder into same directory as files: `git clone https://github.com/haotian-liu/LLaVA.git`
2. `cd LLaVA`
3. `pip install --upgrade pip` (enables PEP 660 support)
4. `pip install -e .`
5. `pip install -e ".[train]"`
6. `pip install flash-attn --no-build-isolation` (optional)
7. At /LLaVA/llava/eval/run_llava.py insert function `eval_mode_with_loaded` from eval_mode_with_loaded.txt

## Start Sever 🖥️:
1. `salloc --nodes=1 --ntasks=2 --cpus-per-task=4 --mem=32G --gres=gpu:1,VRAM:24G --time=02:00:00 --partition=PRACT--qos=practical_course`
2. `conda activate <env>`
3. Open second terminal `srun --jobid=<id> --pty bash`
4. In the first terminal `python3 model_server.py`
5. Second terminal `conda activate <env>`
6. If server is started, in the second terminal `python3 pipelin.py` or similar script to use pipelin