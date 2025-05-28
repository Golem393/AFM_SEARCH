# AFM_SEARCH
LLaVA Folder righ next to the scripts

git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation