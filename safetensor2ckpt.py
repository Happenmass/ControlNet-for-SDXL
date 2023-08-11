from safetensors.torch import load_file
import torch
import sys
import os

path = sys.argv[1]
if not os.path.exists(path):
    print('path not exists')
    sys.exit(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

weights = load_file(path, device=device)
weights["state_dict"] = weights


torch.save(weights, os.path.splitext(path)[0] + '.ckpt')
