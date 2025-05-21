# example code to download a model

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# from huggingface_hub import login
# login()

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='meta-llama/Llama-3.2-1B',
    repo_type='model',
    local_dir='./models/llama3.2-1b',
    ignore_patterns=["*.h5", "*.msgpack", "*.pth", "*.gguf", "*.bin"],
    resume_download=True,
)