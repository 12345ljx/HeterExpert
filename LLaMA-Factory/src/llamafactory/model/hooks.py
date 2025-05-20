import os
import time
def grad_hook(module, grad_in, grad_out):
    print('========= register_backward_hook output:======== ')
    print(grad_out[0].size())
    print(grad_out[0])
    print(grad_out[0].dtype)
    log_path = "/usr/workdir/LLaMA-Factory/src/llmtuner/logs"
    os.makedirs(log_path, exist_ok=True)
    
    with open(os.path.join(log_path, f"grad_hook_10.log"), "a") as f:
        f.write(f"{grad_out[0].size()}\n")
        f.write(f"{time.asctime()}: {grad_in[0]}\n")
        f.write(f"{time.asctime()}: {grad_out[0]}\n")