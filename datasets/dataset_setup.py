import torch.nn as nn
import math
import torch
import random
import numpy as np

def get_dataset_data_path():
    return 'replace with your own path'

def setup_seed(seed, only_torch = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    if only_torch:
        return
    np.random.seed(seed)
    random.seed(seed)

def init_model_para(model):
    # return
    setup_seed(1234, only_torch = True)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
        elif isinstance(layer, nn.Conv2d):
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
            nn.init.normal_(layer.weight, 0., math.sqrt(2. / n))

