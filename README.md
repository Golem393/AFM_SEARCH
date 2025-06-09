# AFM_SEARCH
use python 3.10 in your venv/conda
#### Activate cuda & compiler: 
```
module load cuda/12.1.0
module load compiler/gcc-10.1
```
#### Install LLaVA in project root
```
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
optional: pip install flash-attn --no-build-isolation
```
#### Install requirements
```
cd ..
pip install -r requirements.txt
```

```
srun --jobid=1376466 --pty bash
```