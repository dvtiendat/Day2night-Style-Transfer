import torch
import torch.nn as nn
from torch.nn import init 
import albumentations as A
from albumentations import ToTensorV2
import yaml 
import numpy as np
import os 

def get_transform():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2()
    ],
    additional_targets={'image0': 'image'},
    is_check_shapes=False
    )

def init_weights(net, init_type='normal', init_gain=0.02):
    '''
    Initialize network weights

    Inputs: 
    - net: network to initialize
    - init_type: type of initialization
    - init_gain: gain for initialization

    Returns:
    - None: Network with initialized weights
    '''
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Initialization method {init_type} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f'Initialize network using {init_type} initialization')
    net.apply(init_func)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def save_checkpoint(model, optimizer, path="/kaggle/working/Ukiyo-e-style-transfer/checkpoints", filename="model.pth"):
    if not os.path.exists(path):
        os.makedirs(path)

    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join(path, filename))
    
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location='cuda', weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr