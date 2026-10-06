# reproducibility_utils.py

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        cudnn.deterministic = True
        cudnn.benchmark = False