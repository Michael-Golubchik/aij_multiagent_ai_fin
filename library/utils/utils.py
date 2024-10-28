import torch
import numpy as np


def from_numpy(device, array, dtype=np.float32):
    array = array.astype(dtype)
    tensor = torch.from_numpy(array)
    return tensor.to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
